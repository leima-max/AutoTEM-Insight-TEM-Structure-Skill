from __future__ import annotations

import multiprocessing as mp

import pytest

from autotem.simulation.process_utils import collect_process_queue_payload


def _large_queue_payload_worker(queue) -> None:
    queue.put({"ok": True, "blob": "x" * 1_000_000})


def test_collect_process_queue_payload_drains_large_payload_before_join() -> None:
    ctx = mp.get_context("spawn")
    try:
        queue = ctx.Queue()
    except PermissionError as exc:
        pytest.skip(f"multiprocessing Queue unavailable in this sandbox: {exc}")
    process = ctx.Process(target=_large_queue_payload_worker, args=(queue,))
    process.start()

    status, payload = collect_process_queue_payload(
        process,
        queue,
        timeout_s=10.0,
        settle_s=2.0,
    )

    assert status in {"payload", "payload_terminated"}
    assert isinstance(payload, dict)
    assert payload["ok"] is True
    assert len(payload["blob"]) == 1_000_000
    assert not process.is_alive()
