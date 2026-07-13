"""Unit tests for AsyncOmniEngine port ownership helpers."""

from __future__ import annotations

import socket

import pytest
from omegaconf import OmegaConf

from vllm_omni.engine.async_omni_engine import (
    _defer_tcp_zmq_endpoint,
    _inject_llm_stage_rendezvous_from_env,
)
from vllm_omni.engine.stage_engine_core_proc import _resolve_stage_core_port_slice
from vllm_omni.engine.stage_engine_core_proc import _stage_core_allocator_cursor_start
from vllm_omni.engine.stage_engine_core_proc import _configure_vllm_startup_handshake_timeout
from vllm_omni.engine.stage_engine_core_proc import _is_tcp_port_bindable

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_defer_tcp_zmq_endpoint_preserves_host_and_uses_ephemeral_port():
    assert _defer_tcp_zmq_endpoint("tcp://10.66.156.22:64076") == "tcp://10.66.156.22:0"
    assert _defer_tcp_zmq_endpoint("tcp://[::1]:64076") == "tcp://[::1]:0"


def test_defer_tcp_zmq_endpoint_leaves_ipc_paths_unchanged():
    address = "ipc:///tmp/vllm-test.sock"
    assert _defer_tcp_zmq_endpoint(address) == address


def test_inject_llm_stage_rendezvous_from_env(monkeypatch):
    monkeypatch.setenv("VERL_OMNI_VLLM_DIST_MASTER_ADDR", "10.66.151.39")
    monkeypatch.setenv("VERL_OMNI_VLLM_DIST_MASTER_PORT", "65395")
    stage_configs = [
        OmegaConf.create({"stage_id": 0, "stage_type": "llm", "engine_args": {}}),
        OmegaConf.create({"stage_id": 1, "stage_type": "diffusion", "engine_args": {}}),
    ]

    _inject_llm_stage_rendezvous_from_env(stage_configs)

    assert stage_configs[0].engine_args.master_addr == "10.66.151.39"
    assert stage_configs[0].engine_args.master_port == 65395
    assert "master_addr" not in stage_configs[1].engine_args
    assert "master_port" not in stage_configs[1].engine_args


def test_inject_llm_stage_rendezvous_prefers_launch_args(monkeypatch):
    monkeypatch.setenv("VERL_OMNI_VLLM_DIST_MASTER_ADDR", "10.66.174.46")
    monkeypatch.setenv("VERL_OMNI_VLLM_DIST_MASTER_PORT", "65155")
    stage_configs = [
        OmegaConf.create({"stage_id": 0, "stage_type": "llm", "engine_args": {}}),
    ]

    _inject_llm_stage_rendezvous_from_env(
        stage_configs,
        master_addr="10.66.157.11",
        master_port=32839,
    )

    assert stage_configs[0].engine_args.master_addr == "10.66.157.11"
    assert stage_configs[0].engine_args.master_port == 32839


def test_inject_llm_stage_rendezvous_skips_standalone_dp1(monkeypatch):
    monkeypatch.setenv("VERL_OMNI_FORCE_STANDALONE_ROLLOUT", "1")
    monkeypatch.setenv("VERL_OMNI_RESOURCE_SPLIT_IMPL", "standalone_rollout")
    monkeypatch.setenv("VERL_OMNI_VLLM_DIST_MASTER_PORT", "65258")
    stage_configs = [
        OmegaConf.create({"stage_id": 0, "stage_type": "llm", "engine_args": {}}),
    ]

    _inject_llm_stage_rendezvous_from_env(
        stage_configs,
        master_addr="127.0.0.1",
        master_port=29501,
        data_parallel_size="1",
        node_rank="3",
        nnodes="4",
    )

    llm_args = stage_configs[0].engine_args
    assert "master_addr" not in llm_args
    assert "master_port" not in llm_args
    assert "data_parallel_size" not in llm_args
    assert "node_rank" not in llm_args
    assert "nnodes" not in llm_args


def test_inject_llm_stage_rendezvous_restores_stage_dp_contract(monkeypatch):
    monkeypatch.setenv("VERL_OMNI_VLLM_DIST_MASTER_ADDR", "10.66.174.46")
    monkeypatch.setenv("VERL_OMNI_VLLM_DIST_MASTER_PORT", "65155")
    stage_configs = [
        OmegaConf.create({"stage_id": 0, "stage_type": "llm", "engine_args": {}}),
        OmegaConf.create({"stage_id": 1, "stage_type": "diffusion", "engine_args": {}}),
    ]

    _inject_llm_stage_rendezvous_from_env(
        stage_configs,
        master_addr="10.66.155.61",
        master_port="32803",
        data_parallel_size="4",
        data_parallel_size_local="1",
        data_parallel_start_rank="3",
        data_parallel_address="10.66.155.61",
        data_parallel_rpc_port="33541",
        node_rank="3",
        nnodes="4",
    )

    llm_args = stage_configs[0].engine_args
    assert llm_args.master_addr == "10.66.155.61"
    assert llm_args.master_port == 32803
    assert llm_args.data_parallel_size == 4
    assert llm_args.data_parallel_size_local == 1
    assert llm_args.data_parallel_start_rank == 3
    assert llm_args.data_parallel_address == "10.66.155.61"
    assert llm_args.data_parallel_rpc_port == 33541
    assert llm_args.node_rank == 3
    assert llm_args.nnodes == 4
    assert "data_parallel_size" not in stage_configs[1].engine_args


def test_stage_core_port_slice_skips_guard_band():
    slice_base, allocator_start, slice_end = _resolve_stage_core_port_slice(
        actor_port_base=61512,
        offset=64,
        stride=128,
        guard=32,
    )

    assert slice_base == 61576
    assert allocator_start == 61608
    assert slice_end == 61639


def test_stage_core_port_slice_can_spread_start_with_tail_reserved():
    args = dict(
        actor_port_base=61512,
        offset=64,
        stride=128,
        guard=32,
        spread=16,
        min_tail=8,
        seed="job-a|host-a|pid-a|stage0|replica0|dp0|local0",
    )

    slice_base, allocator_start, slice_end = _resolve_stage_core_port_slice(**args)
    _, repeated_start, _ = _resolve_stage_core_port_slice(**args)
    _, other_start, _ = _resolve_stage_core_port_slice(
        **(args | {"seed": "job-b|host-a|pid-b|stage0|replica0|dp0|local0"})
    )

    assert slice_base == 61576
    assert 61608 <= allocator_start <= 61623
    assert repeated_start == allocator_start
    assert other_start != allocator_start
    assert slice_end - allocator_start + 1 >= 8


def test_stage_core_allocator_cursor_skips_direct_vllm_port_gap():
    assert _stage_core_allocator_cursor_start(vllm_port=63145, port_end=63175, direct_gap=8) == 63153


def test_stage_core_allocator_cursor_falls_back_when_gap_exhausts_slice():
    assert _stage_core_allocator_cursor_start(vllm_port=63174, port_end=63175, direct_gap=8) == 63175


def test_stage_core_tcp_port_probe_rejects_listening_port():
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("", 0))
    port = blocker.getsockname()[1]
    blocker.listen(1)
    try:
        assert not _is_tcp_port_bindable(port)
    finally:
        blocker.close()

    assert _is_tcp_port_bindable(port)


def test_stage_core_startup_handshake_timeout_env_overrides_vllm_constant(monkeypatch):
    import vllm.v1.engine.core as vllm_engine_core

    monkeypatch.setattr(vllm_engine_core, "HANDSHAKE_TIMEOUT_MINS", 5)
    monkeypatch.setenv("VERL_OMNI_VLLM_STARTUP_HANDSHAKE_TIMEOUT", "1800")

    assert _configure_vllm_startup_handshake_timeout() == 30
    assert vllm_engine_core.HANDSHAKE_TIMEOUT_MINS == 30
