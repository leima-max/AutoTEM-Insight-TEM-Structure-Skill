"""Run the installed, line-delimited JSON CLI and retain a compact local tool trace."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from autotem.harness.cli import PROTOCOL_VERSION, strict_json

from .evidence import require


class UnexpectedResponse(ValueError):
    """A delivered JSONL request received a different public result than required."""


def submit_and_wait(call, arguments):
    """Poll the public run status, including the scheduler's finalizing window."""
    state = call("submit_analysis", {**arguments, "asynchronous": True})
    run_id = state["run_id"]
    run_seconds = arguments["plan"].get("budgets", {}).get("run_seconds", 900)
    deadline = time.monotonic() + run_seconds + 60
    while state["status"] in {"preparing", "queued", "running", "finalizing"}:
        if time.monotonic() >= deadline:
            call("cancel_run", {"run_id": run_id})
            raise TimeoutError(f"Run {run_id} exceeded its budget and scheduler grace period")
        time.sleep(1)
        state = call("get_run_status", {"run_id": run_id})
    return state


def read_reply(replies, request, expired, timeout=180):
    tool, request_id = request["tool"], request["request_id"]
    deadline = time.monotonic() + timeout
    while True:
        try:
            line = replies.get(timeout=max(0, deadline - time.monotonic()))
        except queue.Empty as exc:
            expired[request_id] = tool
            raise TimeoutError(f"No JSONL response to {tool} within {timeout} seconds") from exc
        require(line, f"JSONL bridge exited during {tool}; see bridge.stderr.log")
        envelope = strict_json(line)
        received = envelope["request_id"]
        require(envelope["schema_version"] == PROTOCOL_VERSION, "Wrong JSONL envelope")
        if received in expired:
            require(envelope["tool"] == expired[received], "Wrong JSONL envelope")
            del expired[received]
            continue
        require(received == request_id and envelope["tool"] == tool, "Wrong JSONL envelope")
        return envelope


@contextmanager
def open_bridge(output: Path, trace: list, *, command: list[str] | None = None):
    environment = os.environ.copy()
    if sys.flags.isolated:
        environment.pop("PYTHONPATH", None)
    if command is None:
        command = [sys.executable, *(["-I"] if sys.flags.isolated else []),
                   "-m", "autotem.harness.cli", "--root", str(output / "harness"), "stdio"]
    replies = queue.Queue()
    expired = {}
    with (output / "bridge.stderr.log").open("wb") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            encoding="ascii",
            env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        def read_stdout():
            try:
                for line in process.stdout:
                    replies.put(line)
            finally:
                replies.put("")

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()

        def call(tool, arguments=None, *, expect_error=None):
            request = {"tool": tool, "arguments": arguments or {}, "request_id": len(trace) + 1}

            # Inputs here are generated under output. The exported trace uses relative paths.
            def relative(value):
                if isinstance(value, str) and Path(value).is_absolute():
                    return Path(value).relative_to(output).as_posix()
                if isinstance(value, dict):
                    return {k: relative(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [relative(v) for v in value]
                return value

            record = relative(request)
            trace.append(record)
            process.stdin.write(json.dumps(request, ensure_ascii=True, allow_nan=False) + "\n")
            process.stdin.flush()
            try:
                envelope = read_reply(replies, request, expired)
            except TimeoutError:
                record["transport_status"] = "timed_out"
                raise
            record["ok"] = envelope["ok"]
            error = envelope["error"] or {}
            if error:
                record["error_code"] = error.get("code")
            result = envelope["result"]
            if isinstance(result, dict):
                if isinstance(result.get("error"), dict):
                    record["run_error_code"] = result["error"].get("code")
                record.update(
                    {
                        k: result[k]
                        for k in ("run_id", "job_id", "status", "report_type")
                        if k in result
                    }
                )
            if expect_error:
                if envelope["ok"] is not False or error.get("code") != expect_error:
                    raise UnexpectedResponse(
                        f"{tool} expected {expect_error}, received {error.get('code') or 'success'}"
                    )
                return error
            require(envelope["ok"] is True, f"{tool} failed: {error.get('code')}")
            return result

        try:
            yield call
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            reader.join(timeout=2)
            process.stdout.close()
