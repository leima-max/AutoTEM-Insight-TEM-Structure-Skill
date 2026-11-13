"""Probe contracts in the installed private release interpreter and capture the raw observations."""

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mcp_probe(work, *, command=None):
    """Read the real SDK server's JSON-RPC replies over ordinary project-local stdio."""
    messages = [
        {
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "autotem-contract-freeze", "version": "1.0.0"},
            },
        },
        {"method": "notifications/initialized"},
        {"id": 2, "method": "tools/list"},
        {"id": 3, "method": "tools/call", "params": {"name": "list_capabilities", "arguments": {}}},
    ]
    with (
        (work / "mcp.stderr.log").open("wb") as log,
        subprocess.Popen(
            command
            if command is not None
            else [
                sys.executable,
                "-I",
                "-m",
                "autotem.harness.cli",
                "--root",
                str(work / "harness"),
                "mcp",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            cwd=work,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        ) as server,
    ):
        returned = queue.Queue()

        def receive():
            for _ in range(4):
                line = server.stdout.readline(8 * 1024 * 1024)
                returned.put(line)
                if not line:
                    break

        reader = threading.Thread(target=receive, daemon=True)
        reader.start()
        responses = {}
        try:
            for message in messages:
                server.stdin.write((json.dumps({"jsonrpc": "2.0", **message}) + "\n").encode())
                server.stdin.flush()
                if "id" in message:
                    response = json.loads(returned.get(timeout=90))
                    require(
                        response.get("id") == message["id"] and "result" in response,
                        "MCP server did not return the requested result",
                    )
                    responses[message["id"]] = response["result"]
        finally:
            server.stdin.close()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            reader.join(timeout=2)
    info, listing, called = (responses[i] for i in (1, 2, 3))
    require(not listing.get("nextCursor"), "Unexpected continuation in bounded nine-tool discovery")
    tools = {t["name"]: t["inputSchema"] for t in listing["tools"]}
    require(len(tools) == len(listing["tools"]), "Duplicate MCP tool")
    envelope = called["structuredContent"]
    require(
        not called.get("isError")
        and len(called["content"]) == 1
        and json.loads(called["content"][0]["text"]) == envelope,
        "MCP content differs from structured capability result",
    )
    return {
        "protocol": info["protocolVersion"],
        "server": info["serverInfo"]["name"],
        "tools": tools,
        "capabilities": envelope,
    }


def jsonl_probe(work):
    from autotem.harness.cli import MAX_REQUEST_BYTES

    requests = [
        ("unknown_tool", {"tool": "no_such_tool"}),
        ("top_level_field", {"tool": "list_capabilities", "extra": 0}),
        ("invalid_revision", {"tool": "list_capabilities", "schema_version": "2.0.0"}),
        (
            "negative_offset",
            {"tool": "read_artifact", "arguments": {"artifact_id": "x", "offset": -1}},
        ),
        (
            "oversize_page",
            {"tool": "read_artifact", "arguments": {"artifact_id": "x", "limit": 1001}},
        ),
        (
            "repeated_format",
            {"tool": "export_report", "arguments": {"run_id": "x", "formats": ["json", "json"]}},
        ),
        (
            "too_many_plugins",
            {"tool": "list_capabilities", "arguments": {"enabled_plugins": ["a"] * 17}},
        ),
        (
            "foreign_request_field",
            {"tool": "get_run_status", "arguments": {"run_id": "0" * 32, "extra": 1}},
        ),
        ("bad_artifact", {"tool": "read_artifact", "arguments": {"artifact_id": "0" * 32}}),
        ("missing_run", {"tool": "get_run_status", "arguments": {"run_id": "0" * 32}}),
        (
            "invalid_plan",
            {"tool": "validate_plan", "arguments": {"plan": {"input_path": "x", "steps": []}}},
        ),
        (
            "reserved_operation",
            {
                "tool": "validate_plan",
                "arguments": {
                    "plan": {
                        "input_path": "x",
                        "steps": [{"id": "reserved", "operation_id": "workbench.launch"}],
                    }
                },
            },
        ),
        (
            "redacted_parameter",
            {
                "tool": "validate_plan",
                "arguments": {
                    "plan": {
                        "input_path": "x",
                        "user_input": {"api_key": "autotem-probe-sentinel"},
                        "steps": [{"id": "load", "operation_id": "input.load"}],
                    }
                },
            },
        ),
    ]
    lines = [json.dumps({**value, "request_id": name}) for name, value in requests]
    raw = [
        ("duplicate_key", '{"tool":"list_capabilities","tool":"list_capabilities"}'),
        ("nonfinite", '{"tool":"read_artifact","arguments":{"limit":NaN}}'),
        ("oversize_line", " " + ("x" * MAX_REQUEST_BYTES)),
    ]
    names = [name for name, _ in requests] + [name for name, _ in raw]
    lines.extend(line for _, line in raw)
    with (work / "jsonl.stderr.log").open("wb") as log:
        proc = subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "autotem.harness.cli",
                "--root",
                str(work / "harness"),
                "stdio",
            ],
            input=("\n".join(lines) + "\n").encode(),
            stdout=subprocess.PIPE,
            stderr=log,
            cwd=work,
            timeout=180,
            check=True,
        )
    responses = [json.loads(line) for line in proc.stdout.decode("utf-8").splitlines()]
    require(
        len(responses) == len(names) and b"autotem-probe-sentinel" not in proc.stdout,
        "JSONL did not produce a redacted one-to-one response",
    )
    return dict(zip(names, responses, strict=True))


def selection_data():
    import numpy as np

    stack = np.arange(2 * 9 * 13, dtype=np.int16).reshape(2, 9, 13)
    color = np.arange(2 * 9 * 13 * 3, dtype=np.uint16).reshape(2, 9, 13, 3).astype(np.uint8)
    return stack, color


def inspect_selection(work):
    import numpy as np
    from PIL import Image

    from autotem.harness.cli import dispatch_request
    from autotem.harness.service import Harness

    folder = work / "fixtures"
    folder.mkdir()
    stack, color = selection_data()
    np.save(folder / "stack.npy", stack, allow_pickle=False)
    images = [Image.fromarray(frame) for frame in color]
    images[0].save(folder / "multichannel.tif", save_all=True, append_images=images[1:])
    h = Harness(work / "harness")
    cases = [
        ("npy_no_selection", "stack.npy", {}),
        ("npy_frame_roi", "stack.npy", {"frame": 1, "roi": [1, 6, 2, 9]}),
        ("tif_no_frame", "multichannel.tif", {}),
        ("tif_no_channel", "multichannel.tif", {"frame": 1}),
        (
            "tif_frame_channel_roi",
            "multichannel.tif",
            {"frame": 1, "channel": 2, "roi": [1, 6, 2, 9]},
        ),
        ("tif_bad_frame", "multichannel.tif", {"frame": 5}),
    ]
    rows = {}
    for name, filename, selection in cases:
        path = folder / filename
        envelope = dispatch_request(
            h,
            {
                "request_id": name,
                "tool": "inspect_input",
                "arguments": {"path": str(path), "selection": selection},
            },
        )
        require(envelope["ok"], f"Inspection tool did not return a run for {name}")
        result = envelope["result"]
        rows[name] = {
            "input": path.relative_to(work).as_posix(),
            "selection": selection,
            "run_id": result["run_id"],
            "job_id": result["job_id"],
            "status": result["status"],
            "snapshot_sha256": sha(path),
            "error_code": (result.get("error") or {}).get("code"),
            "input_requests": result["input_requests"],
        }
        if result["dataset"]:
            raw = next(a for a in result["artifacts"] if a["role"] == "raw")
            rows[name]["raw"] = {
                "artifact_id": raw["artifact_id"],
                "sha256": raw["sha256"],
                "path": Path(raw["path"]).relative_to(work).as_posix(),
                "shape": raw["metadata"]["shape"],
                "dtype": raw["metadata"]["dtype"],
                "array_sha256": raw["metadata"]["array_sha256"],
                "selection": result["dataset"]["metadata"]["selection"],
            }
    return rows


def collect(bundle, work, out):
    from autotem.acceptance.driver import runtime_record
    from autotem.harness.cli import MAX_REQUEST_BYTES
    from autotem.harness.contracts import HarnessPlan
    from autotem.harness.input_requests import ANSWER_UNITS, InputResponse
    from autotem.harness.service import TERMINAL

    require(not out.exists(), "Output already exists")
    require(
        not work.exists() or work.is_dir() and not any(work.iterdir()),
        "Use a fresh capture directory",
    )
    work.mkdir(parents=True, exist_ok=True)
    runtime = runtime_record(True, bundle)
    probe = {
        "schema_version": "1.0.0",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
        "collector_sha256": sha(Path(__file__)),
        "mcp": mcp_probe(work),
        "protocol_errors": jsonl_probe(work),
        "selections": inspect_selection(work),
        "models": {
            "plan": HarnessPlan.model_json_schema(),
            "input_response": InputResponse.model_json_schema(),
        },
        "answer_units": ANSWER_UNITS,
        "terminal_states": sorted(TERMINAL),
        "max_request_bytes": MAX_REQUEST_BYTES,
    }
    with out.open("x", encoding="utf-8") as stream:
        json.dump(probe, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["bundle", "work", "out"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    collect(args.bundle.resolve(), args.work.resolve(), args.out.resolve())
