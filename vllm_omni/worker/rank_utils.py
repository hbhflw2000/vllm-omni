from __future__ import annotations

from typing import Any


def get_dp_local_rank(parallel_config: Any, visible_device_count: int) -> int:
    """Resolve DP rank within the current process' visible device set."""
    dp_local_rank = getattr(parallel_config, "data_parallel_rank_local", None)
    if dp_local_rank is not None:
        return int(dp_local_rank)

    dp_rank = int(getattr(parallel_config, "data_parallel_index", 0))
    tp_pp_world_size = int(getattr(parallel_config, "pipeline_parallel_size", 1)) * int(
        getattr(parallel_config, "tensor_parallel_size", 1)
    )
    local_dp_capacity = max(1, visible_device_count // max(1, tp_pp_world_size))
    return dp_rank % local_dp_capacity


def get_dp_adjusted_local_rank(
    base_local_rank: int,
    parallel_config: Any,
    visible_device_count: int,
) -> int:
    tp_pp_world_size = int(getattr(parallel_config, "pipeline_parallel_size", 1)) * int(
        getattr(parallel_config, "tensor_parallel_size", 1)
    )
    return int(base_local_rank) + (
        get_dp_local_rank(parallel_config, visible_device_count) * tp_pp_world_size
    )
