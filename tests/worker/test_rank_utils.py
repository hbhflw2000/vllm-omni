from types import SimpleNamespace

import pytest

from vllm_omni.worker.rank_utils import get_dp_adjusted_local_rank, get_dp_local_rank

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _parallel_config(
    *,
    dp_rank: int,
    dp_local_rank: int | None = None,
    tp: int = 1,
    pp: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        data_parallel_index=dp_rank,
        data_parallel_rank_local=dp_local_rank,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
    )


def test_global_dp_rank_folds_to_visible_devices_for_remote_split_replica() -> None:
    config = _parallel_config(dp_rank=3, tp=4)

    assert get_dp_local_rank(config, visible_device_count=4) == 0
    assert [
        get_dp_adjusted_local_rank(rank, config, visible_device_count=4)
        for rank in range(4)
    ] == [0, 1, 2, 3]


def test_global_dp_rank_maps_to_local_dp_capacity_when_multiple_dp_fit_per_node() -> None:
    visible_device_count = 8
    tp = 2

    assert [
        get_dp_local_rank(_parallel_config(dp_rank=dp_rank, tp=tp), visible_device_count)
        for dp_rank in range(4)
    ] == [0, 1, 2, 3]
    assert get_dp_adjusted_local_rank(
        1,
        _parallel_config(dp_rank=3, tp=tp),
        visible_device_count,
    ) == 7


def test_explicit_dp_local_rank_takes_precedence() -> None:
    config = _parallel_config(dp_rank=9, dp_local_rank=1, tp=4)

    assert get_dp_local_rank(config, visible_device_count=8) == 1
    assert get_dp_adjusted_local_rank(2, config, visible_device_count=8) == 6
