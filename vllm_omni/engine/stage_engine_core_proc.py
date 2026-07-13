"""
Stage Core Process for vLLM-Omni V1 architecture.

StageEngineCoreProc inherits from vLLM's EngineCoreProc and runs the engine core
busy loop in a subprocess, communicating with StageEngineCoreClient via ZMQ.
"""

from __future__ import annotations

import contextlib
import glob
import math
import os
import signal
import socket
import threading
import time
import traceback
import zlib
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Any

import msgspec
import zmq
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value,
)
from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.utils.system_utils import (
    decorate_logs,
    get_mp_context,
    set_process_title,
)
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    SignalCallback,
    get_engine_zmq_addresses,
)
from vllm.v1.utils import shutdown

from vllm_omni.distributed.omni_coordinator import OmniCoordClientForStage
from vllm_omni.engine.stage_init_utils import set_death_signal

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.executor import Executor

logger = init_logger(__name__)


_SIGNAL_EXIT_BASE = 128


def _stage_core_diag_dir() -> str | None:
    diag_dir = os.environ.get("VERL_OMNI_STAGE_CORE_DIAG_DIR")
    if not diag_dir:
        return None
    return os.path.abspath(diag_dir)


def _write_stage_core_crash_diagnostic() -> str | None:
    diag_dir = _stage_core_diag_dir()
    if diag_dir is None:
        return None
    try:
        os.makedirs(diag_dir, exist_ok=True)
        path = os.path.join(
            diag_dir,
            f"stage_core_crash_{socket.gethostname()}_{os.getpid()}_{int(time.time())}.log",
        )
        env_keys = (
            "CUDA_VISIBLE_DEVICES",
            "LOCAL_RANK",
            "RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "VLLM_PORT",
            "VLLM_HOST_IP",
            "NCCL_SOCKET_IFNAME",
            "VERL_OMNI_VLLM_PORT_SEED",
            "VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE",
            "VERL_OMNI_VLLM_STAGE_CORE_DIRECT_PORT_GAP",
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"hostname={socket.gethostname()}\n")
            f.write(f"pid={os.getpid()}\n")
            for key in env_keys:
                f.write(f"{key}={os.environ.get(key, '')}\n")
            f.write("\ntraceback:\n")
            f.write(traceback.format_exc())
        return path
    except Exception:
        logger.exception("Failed to write StageEngineCoreProc crash diagnostic.")
        return None


def _format_recent_stage_core_crash_diagnostics(limit: int = 3) -> str:
    diag_dir = _stage_core_diag_dir()
    if diag_dir is None:
        return ""
    try:
        paths = sorted(
            glob.glob(os.path.join(diag_dir, "stage_core_crash_*.log")),
            key=os.path.getmtime,
            reverse=True,
        )[:limit]
    except Exception:
        return f" Stage core crash diagnostic dir: {diag_dir} (failed to list files)."
    if not paths:
        return f" Stage core crash diagnostic dir: {diag_dir} (no crash files found)."
    snippets: list[str] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            tail = "".join(lines[-40:]).strip()
        except Exception as exc:
            tail = f"<failed to read diagnostic: {exc}>"
        snippets.append(f"\n--- {path} ---\n{tail}")
    return " Recent StageEngineCoreProc crash diagnostics:" + "".join(snippets)


def _configure_vllm_startup_handshake_timeout() -> int | None:
    """Propagate vLLM-Omni's stage startup budget into vLLM's core handshake."""
    timeout_seconds = os.environ.get("VERL_OMNI_VLLM_STARTUP_HANDSHAKE_TIMEOUT")
    timeout_minutes = os.environ.get("VERL_OMNI_VLLM_STARTUP_HANDSHAKE_TIMEOUT_MINS")
    if not timeout_seconds and not timeout_minutes:
        return None

    try:
        if timeout_minutes:
            handshake_timeout_mins = int(timeout_minutes)
        else:
            handshake_timeout_mins = int(math.ceil(int(timeout_seconds) / 60))
    except ValueError:
        logger.warning(
            "Invalid vLLM startup handshake timeout env: seconds=%r minutes=%r",
            timeout_seconds,
            timeout_minutes,
        )
        return None

    if handshake_timeout_mins <= 0:
        logger.warning(
            "Ignoring non-positive vLLM startup handshake timeout: %s minute(s)",
            handshake_timeout_mins,
        )
        return None

    import vllm.v1.engine.core as vllm_engine_core

    original_timeout = getattr(vllm_engine_core, "HANDSHAKE_TIMEOUT_MINS", None)
    vllm_engine_core.HANDSHAKE_TIMEOUT_MINS = handshake_timeout_mins
    logger.warning(
        "Configured vLLM startup handshake timeout: %s minute(s) (was %s)",
        handshake_timeout_mins,
        original_timeout,
    )
    return handshake_timeout_mins


def _resolve_stage_core_port_slice(
    actor_port_base: int,
    offset: int,
    stride: int,
    guard: int,
    spread: int = 0,
    min_tail: int = 1,
    seed: str | None = None,
) -> tuple[int, int, int]:
    """Return ``(slice_base, allocator_start, slice_end)`` for stage-core ports."""
    slice_base = actor_port_base + offset
    if stride > offset:
        slice_end = actor_port_base + stride - 1
    else:
        slice_end = min(slice_base + 63, 65535)
    slice_end = min(slice_end, 65535)

    allocator_start = slice_base + max(guard, 0)
    if allocator_start > slice_end:
        allocator_start = slice_base
    elif spread > 0 and seed:
        usable_start_count = slice_end - allocator_start - max(min_tail, 1) + 2
        if usable_start_count > 1:
            spread_count = min(spread, usable_start_count)
            allocator_start += zlib.crc32(seed.encode("utf-8")) % spread_count
    return slice_base, allocator_start, slice_end


def _stage_core_port_seed(
    *,
    omni_stage_id: int | None,
    omni_replica_id: int,
    dp_rank: int,
    local_dp_rank: int,
) -> str:
    return "|".join(
        [
            os.environ.get("VERL_OMNI_VLLM_STAGE_CORE_PORT_SEED")
            or os.environ.get("VERL_OMNI_VLLM_PORT_SEED")
            or os.environ.get("LUBAN_JOB_ID")
            or os.environ.get("AIP_JOB_ID")
            or os.environ.get("VC_JOB_ID")
            or os.environ.get("JOB_ID")
            or os.environ.get("APP_ID")
            or os.environ.get("K8S_APP_ID")
            or "verl_omni_stage_core",
            socket.gethostname(),
            str(os.getpid()),
            str(omni_stage_id if omni_stage_id is not None else "none"),
            str(omni_replica_id),
            str(dp_rank),
            str(local_dp_rank),
        ]
    )


def _is_tcp_port_bindable(port: int) -> bool:
    """Best-effort local listen probe before handing a port to torch TCPStore."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", port))
            sock.listen(1)
        return True
    except OSError:
        return False


def _next_bindable_tcp_port(start: int, end: int) -> int:
    for port in range(start, end + 1):
        if _is_tcp_port_bindable(port):
            return port
    raise RuntimeError(f"No bindable TCP port in stage-core slice {start}-{end}")


def _stage_core_allocator_cursor_start(vllm_port: int, port_end: int, direct_gap: int) -> int:
    gap = max(int(direct_gap), 1)
    cursor_start = vllm_port + gap
    if cursor_start > port_end:
        cursor_start = vllm_port + 1
    return cursor_start


def _configure_stage_core_port_allocator(
    *,
    omni_stage_id: int | None = None,
    omni_replica_id: int = 0,
    dp_rank: int = 0,
    local_dp_rank: int = 0,
) -> None:
    """Give the stage-core subprocess its own slice of the actor port range."""
    port_env = os.environ.get("VLLM_PORT")
    if not port_env:
        return

    try:
        actor_port_base = int(port_env)
        offset = int(os.environ.get("VERL_OMNI_VLLM_STAGE_CORE_PORT_OFFSET", "64"))
        stride = int(os.environ.get("VERL_OMNI_VLLM_PORT_STRIDE", "128"))
        guard = int(os.environ.get("VERL_OMNI_VLLM_STAGE_CORE_PORT_GUARD", "32"))
        spread = int(os.environ.get("VERL_OMNI_VLLM_STAGE_CORE_PORT_SPREAD", "16"))
        min_tail = int(os.environ.get("VERL_OMNI_VLLM_STAGE_CORE_PORT_MIN_TAIL", "8"))
        direct_gap = int(os.environ.get("VERL_OMNI_VLLM_STAGE_CORE_DIRECT_PORT_GAP", "8"))
    except ValueError:
        logger.warning("Invalid vLLM port allocator envs; leaving VLLM_PORT=%s unchanged", port_env)
        return

    if offset <= 0:
        return

    seed = _stage_core_port_seed(
        omni_stage_id=omni_stage_id,
        omni_replica_id=omni_replica_id,
        dp_rank=dp_rank,
        local_dp_rank=local_dp_rank,
    )
    slice_base, allocator_start, port_end = _resolve_stage_core_port_slice(
        actor_port_base=actor_port_base,
        offset=offset,
        stride=stride,
        guard=guard,
        spread=max(spread, 0),
        min_tail=max(min_tail, 1),
        seed=seed,
    )
    if slice_base > port_end or slice_base > 65535:
        logger.warning(
            "Invalid stage-core vLLM port slice: actor_base=%s offset=%s stride=%s",
            actor_port_base,
            offset,
            stride,
        )
        return

    vllm_port = _next_bindable_tcp_port(allocator_start, port_end)
    os.environ["VLLM_PORT"] = str(vllm_port)
    try:
        import vllm.envs as vllm_envs

        vllm_envs.disable_envs_cache()
    except Exception:
        pass

    import vllm.utils.network_utils as network_utils

    if getattr(network_utils, "_verl_omni_stage_core_port_allocator", False):
        return

    lock = threading.Lock()
    # vLLM has a few call sites that read VLLM_PORT directly, while others go
    # through get_open_port(). Keep those ownership domains disjoint: reserve
    # a small direct-user gap and start the patched allocator after it.
    allocator_cursor_start = _stage_core_allocator_cursor_start(
        vllm_port,
        port_end,
        direct_gap,
    )
    cursor = {"next": allocator_cursor_start}
    first_port: int | None = None
    if os.environ.get("VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE", "0") == "1":
        master_port = os.environ.get("MASTER_PORT")
        if master_port:
            try:
                candidate = int(master_port)
                if _is_tcp_port_bindable(candidate):
                    first_port = candidate
                else:
                    logger.warning(
                        "MASTER_PORT=%s is not bindable for stage-core TCPStore; "
                        "falling back to stage-core allocator slice %s-%s",
                        candidate,
                        allocator_start,
                        port_end,
                    )
            except ValueError:
                logger.warning("Invalid MASTER_PORT=%r; using stage-core allocator slice", master_port)
    original_get_open_port = network_utils._get_open_port

    def get_open_port() -> int:
        with lock:
            nonlocal first_port
            if first_port is not None:
                port = first_port
                first_port = None
                return port
            start = cursor["next"]
            if start > port_end:
                raise RuntimeError(
                    "vLLM stage-core port slice exhausted: "
                    f"start={allocator_cursor_start} end={port_end}"
                )
            port = _next_bindable_tcp_port(start, port_end)
            cursor["next"] = port + 1
            return port

    def get_open_ports_list(count: int = 5) -> list[int]:
        return [get_open_port() for _ in range(count)]

    network_utils.get_open_port = get_open_port
    network_utils.get_open_ports_list = get_open_ports_list
    network_utils._verl_omni_stage_core_port_allocator = True

    patch_targets = [
        ("vllm.v1.executor.multiproc_executor", "get_open_port", get_open_port),
        ("vllm.v1.executor.uniproc_executor", "get_open_port", get_open_port),
        ("vllm.v1.executor.ray_executor", "get_open_port", get_open_port),
        ("vllm.v1.executor.ray_executor_v2", "get_open_port", get_open_port),
        ("vllm.v1.engine.utils", "get_open_port", get_open_port),
        ("vllm.distributed.device_communicators.shm_broadcast", "get_open_port", get_open_port),
        ("vllm_omni.engine.stage_engine_startup", "get_open_ports_list", get_open_ports_list),
        ("vllm_omni.distributed.omni_coordinator.runtime", "get_open_ports_list", get_open_ports_list),
    ]
    for module_name, attr_name, replacement in patch_targets:
        try:
            module = __import__(module_name, fromlist=[attr_name])
        except Exception:
            continue
        if hasattr(module, attr_name):
            setattr(module, attr_name, replacement)

    logger.warning(
        "Installed stage-core vLLM port allocator: actor_base=%s slice_base=%s start=%s vllm_port=%s alloc_next=%s end=%s first_port=%s guard=%s spread=%s min_tail=%s direct_gap=%s",
        actor_port_base,
        slice_base,
        allocator_start,
        vllm_port,
        cursor["next"],
        port_end,
        first_port,
        guard,
        max(spread, 0),
        max(min_tail, 1),
        max(direct_gap, 1),
    )


def _signal_exit_code(signum: int) -> int:
    """Return the conventional process exit code for signal-driven exits."""
    return _SIGNAL_EXIT_BASE + signum


class StageEngineCoreProc(EngineCoreProc):
    """Stage-specific engine core process for vLLM-Omni.

    Inherits from EngineCoreProc and provides its own ``run_stage_core``
    entry point for launching in a subprocess.  Does **not** delegate to
    ``EngineCoreProc.run_engine_core()``.
    """

    @staticmethod
    def run_stage_core(
        *args: Any,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
        omni_coordinator_address: str | None = None,
        omni_stage_id: int | None = None,
        omni_replica_id: int = 0,
        **kwargs: Any,
    ) -> None:
        """Launch StageEngineCoreProc busy loop in background process.

        Omni-specific kwargs:
          - ``omni_coordinator_address``: ROUTER address of the head-side
            :class:`OmniCoordinator`. When provided, this subprocess
            instantiates an :class:`OmniCoordClientForStage` after the
            HELLO/INIT/READY handshake completes and reports its status +
            queue length via heartbeats. The hook is wired so each
            heartbeat refreshes ``queue_length`` from the live scheduler.
          - ``omni_stage_id``: logical stage id this replica belongs to.
            Required when ``omni_coordinator_address`` is provided.
          - ``omni_replica_id``: cluster-unique replica id within the
            stage (assigned by :class:`OmniMasterServer`). Used for
            logging / metrics only.
        """
        signal_callback: SignalCallback | None = None
        maybe_register_config_serialize_by_value()

        engine_core: StageEngineCoreProc | None = None
        coord_client: OmniCoordClientForStage | None = None
        try:
            # NOTE: previous revisions hardcoded data_parallel_size=1 here
            # (TODO referencing issue #984). The hardcoding has been removed
            # so the DP fields propagate through from the caller exactly
            # like upstream vLLM.

            stage_label = f"stage{omni_stage_id}" if omni_stage_id is not None else "noid"
            set_death_signal(signal.SIGTERM)
            set_process_title(f"StageEngineCoreProc_{stage_label}_replica{omni_replica_id}_DP{dp_rank}")
            decorate_logs()
            # Workaround for flashinfer/jit-cache version mismatch in CI.
            # The parent process handles this gracefully via ring_globals.py,
            # but the subprocess hits an unprotected import in TopKTopPSampler.
            # Setting this env var allows the same graceful fallback to work.
            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
            os.environ["VLLM_OMNI_REPLICA_ID"] = str(max(int(omni_replica_id), 0))
            _configure_vllm_startup_handshake_timeout()
            _configure_stage_core_port_allocator(
                omni_stage_id=omni_stage_id,
                omni_replica_id=omni_replica_id,
                dp_rank=dp_rank,
                local_dp_rank=local_dp_rank,
            )

            engine_core = StageEngineCoreProc(
                *args,
                engine_index=dp_rank,
                **kwargs,
            )

            # Each subprocess corresponds to exactly one omni replica with
            # its own OmniMasterServer allocation, so the heartbeat client
            # runs unconditionally — there is no dp_rank-based gating.
            if omni_coordinator_address is not None:
                if omni_stage_id is None:
                    raise ValueError("omni_stage_id must be provided when omni_coordinator_address is set")
                addresses: EngineZmqAddresses = engine_core.addresses
                if not addresses.inputs or not addresses.outputs:
                    raise RuntimeError(
                        "EngineCore handshake did not populate input/output addresses; "
                        "cannot start OmniCoordClientForStage"
                    )
                coord_client = OmniCoordClientForStage(
                    coord_zmq_addr=omni_coordinator_address,
                    input_addr=addresses.inputs[0],
                    output_addr=addresses.outputs[0],
                    stage_id=int(omni_stage_id),
                )

                def _refresh_queue_length() -> None:
                    """Pre-heartbeat hook: refresh queue_length from scheduler."""
                    scheduler = getattr(engine_core, "scheduler", None)
                    if scheduler is None:
                        return
                    try:
                        coord_client._queue_length = int(  # type: ignore[union-attr]
                            scheduler.get_num_unfinished_requests()
                        )
                    except Exception:
                        # Live scheduler stats are best-effort — heartbeats
                        # must not fail because of a stats lookup error.
                        pass

                coord_client._on_heartbeat = _refresh_queue_length

            def wakeup_engine() -> None:
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            signal_callback = SignalCallback(wakeup_engine)

            def signal_handler(signum: int, frame: Any) -> None:
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()
                raise SystemExit(_signal_exit_code(signum))

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("StageEngineCoreProc exiting.")
            raise
        except Exception:
            diag_path = _write_stage_core_crash_diagnostic()
            if diag_path is not None:
                logger.error("StageEngineCoreProc crash diagnostic written to %s", diag_path)
            if engine_core is None:
                logger.exception("StageEngineCoreProc failed to start.")
            else:
                logger.exception("StageEngineCoreProc encountered a fatal error.")
                engine_core._send_engine_dead()
            raise
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if coord_client is not None:
                with contextlib.suppress(RuntimeError):
                    coord_client.close()
            if engine_core is not None:
                engine_core.shutdown()


def spawn_stage_core(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool = False,
) -> tuple[EngineZmqAddresses, BaseProcess, str]:
    """Spawn a *StageEngineCoreProc* subprocess without performing the handshake.

    Must be called while the correct device env vars are set (e.g. under
    the stage-launch lock).  Call ``complete_stage_handshake`` afterwards.

    Returns ``(addresses, process, handshake_address)``.
    """
    addresses = get_engine_zmq_addresses(vllm_config)
    handshake_address = get_open_zmq_ipc_path()

    ctx = get_mp_context()
    proc = ctx.Process(
        target=StageEngineCoreProc.run_stage_core,
        name="StageEngineCoreProc",
        kwargs={
            "vllm_config": vllm_config,
            "local_client": True,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
            "dp_rank": 0,
            "local_dp_rank": 0,
        },
    )
    proc.start()
    return addresses, proc, handshake_address


def complete_stage_handshake(
    proc: BaseProcess,
    handshake_address: str,
    addresses: EngineZmqAddresses,
    vllm_config: VllmConfig,
    handshake_timeout: int,
) -> None:
    """Perform the HELLO/INIT/READY handshake with an already-spawned proc.

    On failure the process is terminated before re-raising.
    """
    try:
        _perform_handshake(proc, handshake_address, addresses, vllm_config, handshake_timeout)
    except Exception:
        shutdown([proc])
        raise


def _perform_handshake(
    proc: BaseProcess,
    handshake_address: str,
    addresses: EngineZmqAddresses,
    vllm_config: VllmConfig,
    handshake_timeout: int,
) -> None:
    """Run the HELLO / INIT / READY handshake with the subprocess."""
    with zmq_socket_ctx(handshake_address, zmq.ROUTER, bind=True) as handshake_socket:
        poller = zmq.Poller()
        poller.register(handshake_socket, zmq.POLLIN)
        poller.register(proc.sentinel, zmq.POLLIN)

        identity, msg = _recv(poller, handshake_socket, proc, "HELLO", handshake_timeout)
        if msg.get("status") != "HELLO":
            raise RuntimeError(f"Expected HELLO, got: {msg}")

        init_payload = EngineHandshakeMetadata(
            addresses=addresses,
            parallel_config={},
        )
        handshake_socket.send_multipart([identity, msgspec.msgpack.encode(init_payload)])

        identity, msg = _recv(poller, handshake_socket, proc, "READY", handshake_timeout)
        if msg.get("status") != "READY":
            raise RuntimeError(f"Expected READY, got: {msg}")
        num_gpu_blocks = msg.get("num_gpu_blocks")
        if num_gpu_blocks is not None:
            vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks


def _recv(
    poller: zmq.Poller,
    handshake_socket: zmq.Socket,
    proc: BaseProcess,
    expected: str,
    timeout_s: int = 600,
) -> tuple[bytes, dict]:
    """Wait for one handshake message; raise if the process dies first."""
    timeout_ms = timeout_s * 1000
    while True:
        events = dict(poller.poll(timeout=timeout_ms))
        if not events:
            raise TimeoutError(
                f"Timed out waiting for {expected} from StageEngineCoreProc after {timeout_s}s. "
                f"This typically indicates model loading or initialization is taking too long. "
                f"Consider increasing `stage_init_timeout` for large models."
            )
        if handshake_socket in events:
            identity, raw = handshake_socket.recv_multipart()
            return identity, msgspec.msgpack.decode(raw)
        if proc.exitcode is not None:
            diagnostic = _format_recent_stage_core_crash_diagnostics()
            raise RuntimeError(
                f"StageEngineCoreProc died during {expected} (exit code {proc.exitcode}).{diagnostic}"
            )
