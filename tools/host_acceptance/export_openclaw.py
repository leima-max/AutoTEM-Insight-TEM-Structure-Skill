"""Prepare observed-host-export turns from one OpenClaw v3 session JSONL for the field verifier."""

import argparse
import hashlib
import json
import runpy
import shutil
import uuid
from pathlib import Path

_verifier = runpy.run_path(str(Path(__file__).with_name("verify.py")))
TOOLS = _verifier["TOOLS"]
require = _verifier["require"]
unique_json = _verifier["unique_json"]
envelope = _verifier["envelope"]


def branch(data, project, leaf):
    records = [
        (i, unique_json(line))
        for i, line in enumerate(data.decode("utf-8-sig").splitlines(), 1)
        if line.strip()
    ]
    require(records and isinstance(records[0][1], dict), "missing_openclaw_header")
    header = records[0][1]
    require(
        header.get("type") == "session"
        and header.get("version") == 3
        and isinstance(header.get("id"), str)
        and header["id"],
        "unsupported_openclaw_session",
    )
    # OpenClaw can copy another project's conversation into a new session and cwd.
    require(not header.get("parentSession"), "openclaw_inherited_session_needs_parent")
    require(
        isinstance(header.get("cwd"), str)
        and Path(header["cwd"]).is_absolute()
        and Path(header["cwd"]).resolve() == project,
        "openclaw_workspace_mismatch",
    )
    entries, leaves = {}, set()
    for line, row in records[1:]:
        require(isinstance(row, dict) and row.get("type") != "session", "duplicate_openclaw_header")
        entry_id, parent = row.get("id"), row["parentId"]
        require(
            isinstance(entry_id, str) and entry_id and entry_id not in entries,
            "invalid_openclaw_entry_id",
        )
        require(
            parent is None or isinstance(parent, str) and parent in entries,
            "unknown_openclaw_parent",
        )
        entries[entry_id] = (line, row)
        leaves.discard(parent)
        leaves.add(entry_id)
    if leaf is None:
        require(len(leaves) == 1, "select_an_openclaw_branch_leaf")
        leaf = next(iter(leaves))
    require(leaf in leaves, "unknown_openclaw_leaf")
    chain = []
    cursor = leaf
    while cursor is not None:
        line, row = entries[cursor]
        chain.append((line, row))
        cursor = row["parentId"]
    return header["id"], leaf, reversed(chain)


def message_text(content):
    if isinstance(content, str):
        return content
    require(
        isinstance(content, list) and all(isinstance(block, dict) for block in content),
        "invalid_openclaw_text",
    )
    return "\n".join(block["text"] for block in content if block.get("type") == "text")


def read_session(data, project, leaf=None):
    session, selected, chain = branch(data, project.resolve(), leaf)
    turns, pending = [], {}
    seen_calls = set()
    source = "session.jsonl"
    for line, row in chain:
        if row["type"] != "message":
            continue
        message = row["message"]
        role = message["role"]
        if role == "user":
            require(
                not pending and (not turns or "answer" in turns[-1]), "openclaw_turn_unfinished"
            )
            prompt = message_text(message["content"])
            require(prompt.strip(), "missing_openclaw_user_text")
            turns.append(
                {
                    "host": "openclaw",
                    "session_id": session,
                    "prompt": prompt,
                    "user_line": line,
                    "calls": [],
                }
            )
        elif role == "assistant":
            require(turns, "openclaw_assistant_before_user")
            turn = turns[-1]
            require(not pending, "openclaw_turn_unfinished")
            reason = message.get("stopReason")
            require(reason in {"stop", "toolUse"}, "openclaw_assistant_incomplete")
            require(isinstance(message["content"], list), "invalid_openclaw_assistant_content")
            calls = [block for block in message["content"] if block.get("type") == "toolCall"]
            require(calls or reason != "toolUse", "openclaw_tool_use_without_call")
            if calls:
                require("answer" not in turn, "openclaw_call_after_answer")
                for call in calls:
                    name = call["name"]
                    require(
                        isinstance(name, str) and isinstance(call.get("arguments"), dict),
                        "invalid_openclaw_tool_call",
                    )
                    if name.startswith("autotem__"):
                        require(name.removeprefix("autotem__") in TOOLS, "unexpected_openclaw_tool")
                    call_id = call["id"]
                    require(
                        isinstance(call_id, str) and call_id and call_id not in seen_calls,
                        "duplicate_openclaw_call",
                    )
                    seen_calls.add(call_id)
                    pending[call_id] = (line, name, call["arguments"])
            elif reason == "stop":
                require(not pending and "answer" not in turn, "openclaw_final_answer_mismatch")
                answer = message_text(message["content"])
                require(answer.strip(), "missing_openclaw_final_answer")
                turn.update(answer=answer, answer_evidence=source, answer_line=line)
        elif role == "toolResult":
            name, call_id = message.get("toolName", ""), message.get("toolCallId")
            require(call_id in pending, "unmatched_openclaw_result")
            start, requested_name, arguments = pending.pop(call_id)
            require(
                name == requested_name and isinstance(message.get("content"), list),
                "openclaw_tool_name_mismatch",
            )
            if not name.startswith("autotem__"):
                continue
            tool = name.removeprefix("autotem__")
            try:
                response = envelope(tool, message)
            except json.JSONDecodeError as exc:
                if any(
                    "middle content omitted" in block.get("text", "")
                    for block in message["content"]
                ):
                    raise ValueError("openclaw_tool_result_truncated") from exc
                raise
            host_error = message.get("isError")
            require(
                response.get("request_id") == call_id
                and type(host_error) is bool
                and not (host_error and response["ok"]),
                "openclaw_result_identity_or_status_mismatch",
            )
            if not response["ok"]:
                # Pi can resolve a tool successfully while the Harness reports a business error.
                error = response["error"]
                require(
                    response.get("result") is None
                    and all(
                        isinstance(error.get(k), str) and error[k].strip()
                        for k in ("code", "message")
                    )
                    and type(error.get("retryable")) is bool,
                    "incomplete_openclaw_business_error",
                )
            turns[-1]["calls"].append(
                {
                    "server": "autotem",
                    "tool": tool,
                    "arguments": arguments,
                    "response": response,
                    "host_is_error": host_error,
                    "original_evidence": source,
                    "call_id": call_id,
                    "request_line": start,
                    "result_line": line,
                }
            )
    require(
        turns and not pending and all("answer" in turn for turn in turns),
        "openclaw_turn_unfinished",
    )
    require(any(turn["calls"] for turn in turns), "no_openclaw_autotem_exchange")
    return selected, turns


def write_capture(destination, data, selected, turns):
    (destination / "session.jsonl").write_bytes(data)
    manifest = {
        "capture_format": "observed-host-export",
        "session_id": turns[0]["session_id"],
        "leaf": selected,
        "session_sha256": hashlib.sha256(data).hexdigest(),
        "turns": [],
    }
    for number, turn in enumerate(turns, 1):
        prompt, trace = f"turn-{number}.txt", f"turn-{number}.json"
        (destination / prompt).write_bytes(turn.pop("prompt").encode())
        (destination / trace).write_text(
            json.dumps(turn, indent=2, ensure_ascii=False) + "\n", encoding="utf8"
        )
        manifest["turns"].append({"prompt": prompt, "trace": trace})
    (destination / "turns.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    return manifest


def export(session_file, project, destination, leaf=None):
    project = project.resolve(strict=True)
    destination = destination.resolve()
    require(
        project.is_dir() and not destination.is_relative_to(project),
        "keep_field_records_outside_project",
    )
    if destination.exists():
        raise FileExistsError(destination)
    data = session_file.read_bytes()
    selected, turns = read_session(data, project, leaf)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}-{uuid.uuid4().hex}.tmp")
    staging.mkdir()
    try:
        manifest = write_capture(staging, data, selected, turns)
        staging.rename(destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_jsonl", type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument(
        "--out", required=True, type=Path, help="New private field records directory"
    )
    parser.add_argument("--leaf", help="Native entry ID of the session branch to export")
    args = parser.parse_args()
    try:
        result = export(args.session_jsonl, args.project, args.out, args.leaf)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"OpenClaw export failed: {exc}\n")
    print(f"{len(result['turns'])} turns, original SHA-256 {result['session_sha256']}")
