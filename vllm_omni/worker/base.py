"""Base worker class for vLLM-Omni with process-scoped GPU memory accounting."""

from __future__ import annotations

import gc
import json
import os
import threading
import time
from contextlib import AbstractContextManager, nullcontext

import torch
from vllm.logger import init_logger
from vllm.utils.mem_utils import format_gib, memory_profiling
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

from vllm_omni.diffusion.data import (
    OmniACK,
    OmniSleepTask,
    OmniWakeTask,
)
from vllm_omni.entrypoints.utils import detect_pid_host
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.gpu_memory_utils import (
    get_process_gpu_memory,
    is_process_scoped_memory_available,
)

logger = init_logger(__name__)


class OmniGPUWorkerBase(GPUWorker):
    """Base GPU worker for vLLM-Omni with process-scoped memory accounting.

    This class overrides determine_available_memory() to use per-process GPU
    memory tracking via pynvml, allowing multiple stages to initialize
    concurrently on the same GPU without memory accounting interference.

    It also replaces vLLM's TorchProfilerWrapper with OmniTorchProfilerWrapper
    for custom trace naming, background gzip, and trace path collection.
    """

    def load_model(self, *args, **kwargs):
        with self._maybe_get_memory_pool_context("weights"):
            res = super().load_model(*args, **kwargs)
            current_omni_platform.synchronize()
            gc.collect()
            return res

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._omni_cumem_woken_tags: set[str] = set()
        self._omni_cumem_sleeping_tags: set[str] = set()

        # Replace vLLM's profiler with platform-specific profiler
        profiler_config = self.vllm_config.profiler_config
        if profiler_config and profiler_config.profiler == "torch":
            from vllm_omni.profiler import create_omni_profiler

            stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
            worker_name = f"stage{stage_id}_rank{self.rank}"
            self.profiler = create_omni_profiler(
                profiler_config=profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
            )

    def _omni_worker_identity(self) -> str:
        model_cfg = getattr(self.vllm_config, "model_config", None)
        stage_id = getattr(model_cfg, "stage_id", 0)
        replica_id = getattr(model_cfg, "replica_id", getattr(self, "replica_id", "unknown"))
        return f"stage={stage_id} replica={replica_id} rank={self.rank} local_rank={self.local_rank}"

    @staticmethod
    def _omni_format_bytes(num_bytes: int) -> str:
        return f"{num_bytes / 1024**3:.2f}GiB"

    def _omni_cumem_state(self, allocator) -> dict[str, object]:
        by_tag: dict[str, dict[str, int]] = {}
        for data in allocator.pointer_to_data.values():
            tag = data.tag
            tag_state = by_tag.setdefault(tag, {"count": 0, "bytes": 0, "backed_up_bytes": 0})
            tag_state["count"] += 1
            tag_state["bytes"] += int(data.handle[1])
            if data.cpu_backup_tensor is not None:
                tag_state["backed_up_bytes"] += int(data.cpu_backup_tensor.numel() * data.cpu_backup_tensor.element_size())
        return {
            "current_tag": allocator.current_tag,
            "num_ptrs": len(allocator.pointer_to_data),
            "sleeping_tags": sorted(self._omni_cumem_sleeping_tags),
            "woken_tags": sorted(self._omni_cumem_woken_tags),
            "by_tag": {
                tag: {
                    "count": state["count"],
                    "bytes": self._omni_format_bytes(state["bytes"]),
                    "backed_up_bytes": self._omni_format_bytes(state["backed_up_bytes"]),
                }
                for tag, state in sorted(by_tag.items())
            },
        }

    def _omni_log_cumem_state(self, event: str, allocator, tags=None) -> None:
        logger.info(
            "[vllm_omni_cumem] %s %s tags=%s state=%s",
            event,
            self._omni_worker_identity(),
            tags,
            self._omni_cumem_state(allocator),
        )

    @staticmethod
    def _omni_debug_enabled(name: str) -> bool:
        return os.getenv(name, "0").lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _omni_tensor_debug_metadata(tensor) -> dict[str, object]:
        if not isinstance(tensor, torch.Tensor):
            return {"type": type(tensor).__name__}
        metadata: dict[str, object] = {
            "type": type(tensor).__name__,
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": int(tensor.numel()),
            "nbytes": int(tensor.nbytes),
            "stride": tuple(tensor.stride()),
            "storage_offset": int(tensor.storage_offset()),
            "is_contiguous": bool(tensor.is_contiguous()),
            "requires_grad": bool(tensor.requires_grad),
        }
        try:
            metadata["data_ptr"] = int(tensor.data_ptr())
        except Exception as exc:
            metadata["data_ptr_error"] = repr(exc)
        try:
            metadata["storage_data_ptr"] = int(tensor.untyped_storage().data_ptr())
        except Exception as exc:
            metadata["storage_data_ptr_error"] = repr(exc)
        return metadata

    @staticmethod
    def _omni_candidate_param_names(target_name: str) -> list[str]:
        candidates = [target_name]
        if target_name.startswith("thinker."):
            stripped = target_name[len("thinker.") :]
            candidates.append(stripped)
            candidates.append(f"language_model.{stripped}")
        if target_name.startswith("thinker.model."):
            candidates.append(f"language_model.model.{target_name[len('thinker.model.'):]}")
        if target_name.startswith("thinker.lm_head."):
            candidates.append(f"language_model.lm_head.{target_name[len('thinker.lm_head.'):]}")
        return list(dict.fromkeys(candidates))

    def _omni_find_debug_param(self, target_name: str):
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None)
        if model is None:
            return None, None
        try:
            named_params = dict(model.named_parameters(remove_duplicate=False))
        except TypeError:
            named_params = dict(model.named_parameters())
        except Exception as exc:
            return None, {"error": repr(exc)}

        for candidate in self._omni_candidate_param_names(target_name):
            param = named_params.get(candidate)
            if param is not None:
                return candidate, param
        suffixes = [candidate.split(".", 1)[-1] for candidate in self._omni_candidate_param_names(target_name) if "." in candidate]
        for name, param in named_params.items():
            if any(name.endswith(suffix) for suffix in suffixes):
                return name, param
        return None, None

    def _omni_target_cumem_metadata(self, allocator, tensor) -> dict[str, object]:
        if not isinstance(tensor, torch.Tensor):
            return {"type": type(tensor).__name__}
        try:
            data_ptr = int(tensor.data_ptr())
            storage_ptr = int(tensor.untyped_storage().data_ptr())
            matches: list[dict[str, object]] = []
            for ptr, data in allocator.pointer_to_data.items():
                size = int(data.handle[1])
                begin = int(ptr)
                end = begin + size
                if begin <= data_ptr < end or begin <= storage_ptr < end:
                    backup = data.cpu_backup_tensor
                    matches.append(
                        {
                            "ptr": begin,
                            "size_bytes": size,
                            "tag": data.tag,
                            "data_ptr_offset": data_ptr - begin,
                            "storage_ptr_offset": storage_ptr - begin,
                            "has_cpu_backup": backup is not None,
                            "backup_nbytes": (
                                int(backup.numel() * backup.element_size()) if backup is not None else 0
                            ),
                        }
                    )
            return {
                "current_tag": allocator.current_tag,
                "tracked_allocations": len(allocator.pointer_to_data),
                "matched": bool(matches),
                "matches": matches,
            }
        except Exception as exc:
            return {"error": repr(exc)}

    def _omni_dump_wake_debug(self, event: str, allocator, tags=None, exception: Exception | None = None) -> None:
        if not self._omni_debug_enabled("VERL_OMNI_WAKE_CUMEM_DEBUG"):
            return

        target_name = os.getenv("VERL_OMNI_WEIGHT_SYNC_TARGET_NAME", "thinker.audio_tower.conv2d1.bias")
        resolved_name, param = self._omni_find_debug_param(target_name)
        payload: dict[str, object] = {
            "event": event,
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            "identity": self._omni_worker_identity(),
            "tags": tags,
            "target_name": target_name,
            "resolved_name": resolved_name,
            "cumem_state": self._omni_cumem_state(allocator),
        }
        if exception is not None:
            payload["exception"] = repr(exception)
        if isinstance(param, torch.nn.Parameter):
            payload["param"] = self._omni_tensor_debug_metadata(param.data)
            payload["target_cumem"] = self._omni_target_cumem_metadata(allocator, param.data)
        elif param is not None:
            payload["param_lookup_error"] = param
        else:
            payload["param_status"] = "not_found"

        dump_dir = os.getenv("VERL_OMNI_WEIGHT_SYNC_DUMP_DIR")
        if not dump_dir:
            return
        try:
            os.makedirs(dump_dir, exist_ok=True)
            path = os.path.join(
                dump_dir,
                f"vllm_omni_cumem_{event}_pid{os.getpid()}_tid{threading.get_ident()}_{int(time.time() * 1000)}.json",
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            logger.info("[vllm_omni_cumem] dumped %s to %s", event, path)
        except Exception:
            logger.exception("[vllm_omni_cumem] failed to dump event=%s", event)

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        """Override to set trace filename before starting the profiler.

        Args:
            is_start: True to start profiling, False to stop.
            profile_prefix: Optional prefix for trace filename (vLLM compat).

        vLLM's profile() only passes is_start, so we generate a descriptive
        trace filename here before delegating to the profiler.
        """
        if self.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. For diffusion models, set --profiler-config via CLI. "
                "For omni models, add profiler_config to your stage config file."
            )
        if is_start:
            from vllm_omni.profiler import OmniTorchProfilerWrapper

            if isinstance(self.profiler, OmniTorchProfilerWrapper):
                # Include stage_id and rank in default filename to distinguish
                # traces from different stages profiling on the same local_rank
                stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
                filename = profile_prefix or f"stage{stage_id}_rank{self.rank}_{int(time.time())}"
                self.profiler.set_trace_filename(filename)
            self.profiler.start()
        else:
            self.profiler.stop()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Process-scoped GPU memory profiling for concurrent stage initialization.

        Algorithm:
            1. requested_memory = total_gpu_memory * gpu_memory_utilization
               (computed in init_device from cache_config)

            2. process_memory = memory used by THIS process only (via pynvml)
               - Uses nvmlDeviceGetComputeRunningProcesses to get per-PID memory
               - Supports CUDA_VISIBLE_DEVICES with indices, UUIDs, or MIG IDs

            3. available_kv_cache = requested_memory - process_memory

        Fallback:
            If NVML is unavailable, falls back to profiling data:
            available = requested - (weights + activations + non_torch)
        """
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            self.model_runner.profile_run()
            if current_omni_platform.is_rocm():
                torch.accelerator.synchronize()
            return kv_cache_memory_bytes

        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

        self.non_torch_memory = profile_result.non_torch_increase
        self.peak_activation_memory = profile_result.torch_peak_increase

        process_memory = (
            get_process_gpu_memory(self.local_rank)
            if is_process_scoped_memory_available() and detect_pid_host()
            else None
        )

        if process_memory is not None:
            # NVML available: use per-process memory
            self.available_kv_cache_memory_bytes = max(0, self.requested_memory - process_memory)
            logger.debug(
                "Process-scoped memory (PID %d, GPU %d): requested=%s, used=%s, available=%s",
                os.getpid(),
                self.local_rank,
                format_gib(self.requested_memory),
                format_gib(process_memory),
                format_gib(self.available_kv_cache_memory_bytes),
            )
            logger.info_once(
                "Available KV cache memory: %s GiB (process-scoped)",
                format_gib(self.available_kv_cache_memory_bytes),
                scope="local",
            )
        else:
            # NVML unavailable: use profiling data as conservative fallback
            profiled_usage = (
                int(self.model_runner.model_memory_usage)
                + profile_result.torch_peak_increase
                + profile_result.non_torch_increase
            )
            self.available_kv_cache_memory_bytes = max(0, self.requested_memory - profiled_usage)
            logger.debug(
                "Profiling fallback (PID %d, GPU %d): requested=%s, profiled=%s, available=%s",
                os.getpid(),
                self.local_rank,
                format_gib(self.requested_memory),
                format_gib(profiled_usage),
                format_gib(self.available_kv_cache_memory_bytes),
            )
            logger.info_once(
                "Available KV cache memory: %s GiB (profiling fallback)",
                format_gib(self.available_kv_cache_memory_bytes),
                scope="local",
            )

        return int(self.available_kv_cache_memory_bytes)

    # Provide memory pool context
    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        v1_config_enabled = False
        if hasattr(self, "vllm_config"):
            model_cfg = getattr(self.vllm_config, "model_config", None)
            v1_config_enabled = getattr(model_cfg, "enable_sleep_mode", False)

        is_sleep_enabled = v1_config_enabled or getattr(self.cache_config, "enable_sleep_mode", False)
        if is_sleep_enabled:
            current_omni_platform.synchronize()
            gc.collect()
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            logger.info(f"[LLM Worker {self.rank}] Sleep Mode ENABLED. Activating CuMem pool for tag: {tag}")
            return allocator.use_memory_pool(tag=tag)
        else:
            logger.warning(f"[LLM Worker {self.rank}] Sleep Mode DISABLED.")
            return nullcontext()

    def sleep(self, level: int = 1) -> bool:
        """
        Put the worker to sleep.
        Args:
            level: 1 (Offload weights to CPU), level: 2 (Total Discard).
        """
        from vllm.device_allocator.cumem import CuMemAllocator

        mem_before = current_omni_platform.get_current_memory_usage(self.device)
        offload_tags = ("weights",) if level == 1 else tuple()
        allocator = CuMemAllocator.get_instance()
        known_tags = {data.tag for data in allocator.pointer_to_data.values()}
        self._omni_log_cumem_state("sleep_before", allocator, tags=offload_tags)
        self._omni_dump_wake_debug("sleep_before", allocator, tags=offload_tags)
        allocator.sleep(offload_tags=offload_tags)
        self._omni_cumem_woken_tags.clear()
        self._omni_cumem_sleeping_tags = set(known_tags)
        self._omni_log_cumem_state("sleep_after", allocator, tags=offload_tags)
        self._omni_dump_wake_debug("sleep_after_allocator", allocator, tags=offload_tags)
        current_omni_platform.empty_cache()
        current_omni_platform.synchronize()
        self._omni_dump_wake_debug("sleep_after_sync", allocator, tags=offload_tags)
        mem_after = current_omni_platform.get_current_memory_usage(self.device)
        freed = max(0, mem_before - mem_after)
        remaining_gb = mem_after / 1024**3
        logger.info(
            f"[LLM Worker {self.rank}] Level {level} Sleep: Freed "
            f"{freed / 1024**3:.2f} GiB. {remaining_gb:.2f}GiB memory "
            "is still in use."
        )
        return True

    def wake_up(self, tags: list[str] | None = None) -> bool:
        "Physical video memory reloading logic"
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        self._omni_log_cumem_state("wake_before", allocator, tags=tags)
        self._omni_dump_wake_debug("wake_before", allocator, tags=tags)
        known_tags = {data.tag for data in allocator.pointer_to_data.values()}
        if not self._omni_cumem_woken_tags and not self._omni_cumem_sleeping_tags:
            has_backed_up_tensor = any(data.cpu_backup_tensor is not None for data in allocator.pointer_to_data.values())
            if not has_backed_up_tensor:
                self._omni_cumem_woken_tags = set(known_tags)
        if tags is None:
            tags_to_wake = sorted(tag for tag in known_tags if tag not in self._omni_cumem_woken_tags)
        else:
            tags_to_wake = [tag for tag in tags if tag not in self._omni_cumem_woken_tags]

        if not tags_to_wake:
            logger.info(
                "[vllm_omni_cumem] wake_skip_already_warm %s requested_tags=%s woken_tags=%s",
                self._omni_worker_identity(),
                tags,
                sorted(self._omni_cumem_woken_tags),
            )
        else:
            allocator.wake_up(tags_to_wake)
            if self._omni_debug_enabled("VERL_OMNI_INJECT_WAKE_FAILURE_AFTER_ALLOCATOR"):
                raise RuntimeError("Injected wake failure after allocator.wake_up for CuMem state-machine probe")
        self._omni_dump_wake_debug("wake_after_allocator", allocator, tags=tags)
        self._omni_log_cumem_state("wake_after", allocator, tags=tags)
        try:
            current_omni_platform.synchronize()
        except Exception as exc:
            self._omni_dump_wake_debug("wake_sync_exception", allocator, tags=tags, exception=exc)
            raise
        if tags_to_wake:
            self._omni_cumem_woken_tags.update(tags_to_wake)
            self._omni_cumem_sleeping_tags.difference_update(tags_to_wake)
        self._omni_dump_wake_debug("wake_after_sync", allocator, tags=tags)
        logger.info(f"[LLM Worker {self.rank}] Wake-up complete.")
        return True

    def handle_sleep_task(self, task: OmniSleepTask) -> OmniACK:
        "Handle deterministic Sleep command from the main process"
        try:
            if isinstance(task, dict):
                task = OmniSleepTask(**task)
            logger.info(f"[LLM Worker {self.rank}] Handshake Received: Task {task.task_id}, Level {task.level}")
            if task.level == 2:
                if hasattr(self.model_runner, "graph_runners"):
                    self.model_runner.graph_runners.clear()
                    logger.info(f"[LLM Worker {self.rank}] LLM CUDA Graphs cleared.")
            mem_before = current_omni_platform.get_current_memory_usage(self.device)
            self.sleep(level=task.level)
            mem_after = current_omni_platform.get_current_memory_usage(self.device)
            rank_freed = max(0, mem_before - mem_after)
            if torch.distributed.is_initialized():
                t_freed = torch.tensor([float(rank_freed)], device=self.device)
                torch.distributed.all_reduce(t_freed)
                total_freed = int(t_freed.item())
                torch.distributed.barrier()
            else:
                total_freed = rank_freed
            current_stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
            ack = OmniACK(
                task_id=task.task_id,
                status="SUCCESS",
                stage_id=current_stage_id,
                rank=self.rank,
                freed_bytes=total_freed,
                metadata={
                    "source": "omni_platform_audit",
                    "total_freed_gib": f"{total_freed / 1024**3:.2f}",
                    "rank_freed_gib": f"{rank_freed / 1024**3:.2f}",
                    "rank_residual_gib": f"{mem_after / 1024**3:.2f}",
                },
            )
            if hasattr(self, "result_mq") and self.result_mq:
                self.result_mq.put(ack)
            logger.info(f"[LLM Worker {self.rank}] ACK emitted for Task {task.task_id}")
            return ack
        except Exception as e:
            logger.error(f"[LLM Worker {self.rank}] Sleep Task Failed: {e}", exc_info=True)
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier()
                except Exception:
                    pass
            return OmniACK(task_id=task.task_id, status="ERROR", error_msg=str(e))

    def handle_wake_task(self, task: OmniWakeTask) -> OmniACK:
        "Handle deterministic Wakeup command from the main process"
        try:
            if isinstance(task, dict):
                task = OmniWakeTask(**task)
            self.wake_up(tags=task.tags)
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            self._omni_dump_wake_debug("handle_wake_after_wake_up", allocator, tags=task.tags)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            gc.collect()
            current_omni_platform.synchronize()
            self._omni_dump_wake_debug("handle_wake_before_ack", allocator, tags=task.tags)
            usage_now = current_omni_platform.get_current_memory_usage(self.device)
            current_stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
            ack = OmniACK(
                task_id=task.task_id,
                status="SUCCESS",
                stage_id=current_stage_id,
                rank=self.rank,
                metadata={"state": "WARM", "current_vram_gib": f"{usage_now / 1024**3:.2f}"},
            )
            if hasattr(self, "result_mq") and self.result_mq:
                self.result_mq.put(ack)
            logger.info(f"[LLM Worker {self.rank}] Wake-up ACK emitted.")
            return ack
        except Exception as e:
            logger.error(f"[LLM Worker {self.rank}] Wake-up Failed: {e}", exc_info=True)
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier()
                except Exception:
                    pass
            tid = task.task_id if hasattr(task, "task_id") else "unknown"
            return OmniACK(task_id=tid, status="ERROR", error_msg=str(e))
