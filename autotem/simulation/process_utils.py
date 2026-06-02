"""Small multiprocessing helpers for optional simulation adapters."""

from __future__ import annotations

import queue as queue_module
import time
from typing import Any


def collect_process_queue_payload(
    process: Any,
    queue: Any,
    *,
    timeout_s: float,
    settle_s: float = 2.0,
) -> tuple[str, Any | None]:
    """Read a worker queue while the process is alive.

    `multiprocessing.Queue` can block a child process at `put()` until the
    parent drains enough bytes. Joining the process before reading the queue can
    therefore turn a successful large result into an apparent timeout.
    """
    timeout = max(0.1, float(timeout_s))
    deadline = time.monotonic() + timeout
    payload: Any | None = None

    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            payload = queue.get(timeout=min(0.1, remaining))
            break
        except queue_module.Empty:
            if process.is_alive():
                continue
            process.join(0.1)
            try:
                payload = queue.get_nowait()
            except queue_module.Empty:
                return "empty", None
            break

    if payload is None:
        if process.is_alive():
            process.terminate()
            process.join(settle_s)
            return "timeout", None
        process.join(settle_s)
        try:
            payload = queue.get_nowait()
        except queue_module.Empty:
            return "empty", None

    process.join(settle_s)
    if process.is_alive():
        process.terminate()
        process.join(settle_s)
        return "payload_terminated", payload
    return "payload", payload
