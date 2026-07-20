"""Shared rendezvous helpers for multi-node vLLM-Omni launch."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from omegaconf import OmegaConf
from vllm.logger import init_logger

logger = init_logger(__name__)


def _as_optional_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"vLLM distributed rendezvous {name} must be an integer, got {value!r}") from exc


def _is_true_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _is_standalone_rollout_dp1(data_parallel_size: Any) -> bool:
    dp_size = _as_optional_int("data_parallel_size", data_parallel_size)
    if dp_size is None or dp_size > 1:
        return False
    return (
        _is_true_env("VERL_OMNI_FORCE_STANDALONE_ROLLOUT")
        or os.environ.get("VERL_OMNI_RESOURCE_SPLIT_IMPL") == "standalone_rollout"
    )


def inject_llm_stage_rendezvous_from_env(
    stage_configs: Sequence[Any],
    *,
    master_addr: Any = None,
    master_port: Any = None,
    data_parallel_size: Any = None,
    data_parallel_size_local: Any = None,
    data_parallel_start_rank: Any = None,
    data_parallel_address: Any = None,
    data_parallel_rpc_port: Any = None,
    node_rank: Any = None,
    nnodes: Any = None,
) -> None:
    """Apply an explicit LLM-stage torch.distributed rendezvous contract.

    The vLLM launcher may compute a DP rank-0 address/port dynamically and
    pass it through CLI args. Prefer those explicit values over the environment
    fallback, because Ray head is not necessarily the vLLM DP coordinator.
    When stage_configs_path is used, top-level EngineArgs are stripped before
    stage config resolution, so the full DP launch contract must be restored on
    each LLM stage config explicitly.
    """
    if _is_standalone_rollout_dp1(data_parallel_size):
        logger.warning(
            "[vLLM-Omni] Skip controlled vLLM rendezvous for standalone rollout "
            "DP1 stage replicas; preserving local TP-only launch."
        )
        return

    source = "explicit launch args"
    if master_addr is None and master_port is None:
        master_addr = os.environ.get("VERL_OMNI_VLLM_DIST_MASTER_ADDR")
        master_port = os.environ.get("VERL_OMNI_VLLM_DIST_MASTER_PORT")
        source = "environment"

    distributed_values = {
        "data_parallel_size": data_parallel_size,
        "data_parallel_size_local": data_parallel_size_local,
        "data_parallel_start_rank": data_parallel_start_rank,
        "data_parallel_address": data_parallel_address,
        "data_parallel_rpc_port": data_parallel_rpc_port,
        "node_rank": node_rank,
        "nnodes": nnodes,
    }

    if (
        not master_addr
        and not master_port
        and all(value is None for value in distributed_values.values())
    ):
        return
    if not master_addr or not master_port:
        raise ValueError(
            "vLLM distributed rendezvous master_addr and master_port must be set together"
        )

    master_addr = str(master_addr)
    try:
        master_port_int = int(master_port)
    except ValueError as exc:
        raise ValueError(
            "vLLM distributed rendezvous master_port must be an integer, "
            f"got {master_port!r}"
        ) from exc

    int_fields = {
        "data_parallel_size",
        "data_parallel_size_local",
        "data_parallel_start_rank",
        "data_parallel_rpc_port",
        "node_rank",
        "nnodes",
    }
    normalized_values: dict[str, Any] = {}
    for name, value in distributed_values.items():
        if value is None:
            continue
        if name in int_fields:
            normalized_values[name] = _as_optional_int(name, value)
        else:
            normalized_values[name] = str(value)

    for cfg in stage_configs:
        stage_type = getattr(cfg, "stage_type", "llm")
        if stage_type == "diffusion":
            continue
        if not hasattr(cfg, "engine_args") or cfg.engine_args is None:
            cfg.engine_args = OmegaConf.create({})
        cfg.engine_args.master_addr = master_addr
        cfg.engine_args.master_port = master_port_int
        for name, value in normalized_values.items():
            setattr(cfg.engine_args, name, value)
        logger.warning(
            "[vLLM-Omni] LLM stage %s uses controlled vLLM rendezvous %s:%s from %s"
            " with stage DP args %s",
            getattr(cfg, "stage_id", "?"),
            master_addr,
            master_port_int,
            source,
            sorted(normalized_values.keys()),
        )
