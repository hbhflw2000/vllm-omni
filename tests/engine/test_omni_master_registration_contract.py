"""Local contract tests for OmniMasterServer dynamic replica registration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import msgspec
import pytest
import zmq
from vllm.utils.network_utils import get_open_port

from vllm_omni.engine.stage_engine_startup import OmniMasterServer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _register_headless_llm_replica(master_port: int, dp_rank: int) -> dict:
    ctx = zmq.Context()
    try:
        sock = ctx.socket(zmq.DEALER)
        sock.connect(f"tcp://127.0.0.1:{master_port}")
        sock.send(
            msgspec.msgpack.encode(
                {
                    "stage_id": 0,
                    "replica_id": -1,
                    "stage_config": {"stage_id": 0, "stage_type": "llm"},
                    "replica_bind_address": "127.0.0.1",
                    "replica_handshake_port": get_open_port(),
                    "replica_input_port": get_open_port(),
                    "replica_output_port": get_open_port(),
                    "replica_binds_sockets": False,
                    "engine_start_index": dp_rank,
                    "engine_count": 1,
                }
            )
        )
        assert sock.poll(timeout=5_000)
        return msgspec.msgpack.decode(sock.recv())
    finally:
        sock.close(linger=0)
        ctx.term()


def test_concurrent_headless_llm_registrations_get_unique_ports_and_rank_ranges(monkeypatch):
    monkeypatch.setenv("VLLM_PORT", "45100")
    monkeypatch.setenv("VERL_OMNI_MASTER_ZMQ_PORT_BASE", "45200")
    master_port = get_open_port()
    server = OmniMasterServer(
        master_address="127.0.0.1",
        master_port=master_port,
        stage_ids=[0],
        stage_replica_counts={0: 0},
    )
    server.start()

    try:
        dp_ranks = [1, 3, 2]
        with ThreadPoolExecutor(max_workers=len(dp_ranks)) as pool:
            replies = list(pool.map(lambda rank: _register_headless_llm_replica(master_port, rank), dp_ranks))

        replica_ids = [int(reply["replica_id"]) for reply in replies]
        assert len(set(replica_ids)) == len(replica_ids)

        ports: list[int] = []
        registered_engine_starts: set[int] = set()
        for replica_id in replica_ids:
            alloc = server.get_allocation(0, replica_id=replica_id)
            ports.extend(
                [
                    int(alloc.handshake_bind_address.rsplit(":", 1)[1]),
                    int(alloc.input_bind_address.rsplit(":", 1)[1]),
                    int(alloc.output_bind_address.rsplit(":", 1)[1]),
                ]
            )
            assert alloc.engine_count == 1
            registered_engine_starts.add(int(alloc.engine_start_index))

            ctx = zmq.Context()
            try:
                sock = ctx.socket(zmq.ROUTER)
                sock.bind(alloc.handshake_bind_address)
            finally:
                sock.close(linger=0)
                ctx.term()

        assert len(ports) == 9
        assert len(set(ports)) == 9
        assert min(ports) >= 45200
        assert registered_engine_starts == set(dp_ranks)
    finally:
        server.stop()
