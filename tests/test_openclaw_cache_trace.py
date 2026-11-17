import copy
import hashlib
import json
import runpy
from pathlib import Path

import pytest

reader = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/host_acceptance/export_openclaw.py")
)
marker = "\n\n⚠️ [... middle content omitted — showing head and tail ...]\n\n"


def lines(rows):
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode()


def fingerprints(record):
    def digest(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()

    record["messageFingerprints"] = [digest(m) for m in record["messages"]]
    record["messagesDigest"] = digest("|".join(record["messageFingerprints"]))


def fixture(project):
    tool = "autotem__list_capabilities"
    response = {
        "schema_version": "1.0.0",
        "request_id": "first",
        "tool": "list_capabilities",
        "ok": True,
        "result": {"description": "a" * 14000 + "😀" + "b" * 5000},
        "error": None,
    }
    text = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
    messages = [
        {"role": "user", "content": "Check capabilities.", "timestamp": 1000},
        {
            "role": "assistant",
            "stopReason": "toolUse",
            "timestamp": 1010,
            "content": [{"type": "toolCall", "name": tool, "id": "first", "arguments": {}}],
        },
        {
            "role": "toolResult",
            "toolName": tool,
            "toolCallId": "first",
            "isError": False,
            "timestamp": 1020,
            "content": [{"type": "text", "text": text}],
        },
        {
            "role": "assistant",
            "stopReason": "stop",
            "timestamp": 1030,
            "content": [{"type": "text", "text": "Checked."}],
        },
    ]
    snapshot = {
        "stage": "session:after",
        "sessionId": "test",
        "workspaceDir": str(project),
        "runId": "run",
        "sessionKey": "agent:main:main",
        "seq": 3,
        "ts": "2026-01-01T01:02:03Z",
        "messageCount": 4,
        "messageRoles": [m["role"] for m in messages],
        "messages": copy.deepcopy(messages),
    }
    fingerprints(snapshot)
    # The guard uses JS UTF-16 lengths, including a two-code-unit emoji in the unshown middle.
    length = len(text.encode("utf-16-le")) // 2
    limit = 7000
    budget = limit - len(f"[... {length - limit} more characters truncated]")
    tail = budget * 3 // 10
    preview = text[: budget - tail - 63] + marker + text[-tail:]
    messages[2]["content"][0]["text"] = (
        preview + f"[... {length - len(preview)} more characters truncated]"
    )
    rows = [{"type": "session", "version": 3, "id": "test", "cwd": str(project)}]
    rows.extend(
        {"type": "message", "id": str(i), "parentId": str(i - 1) if i else None, "message": m}
        for i, m in enumerate(messages)
    )
    return rows, snapshot


def test_after_snapshot_restores_one_result_and_retains_both_files(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    native, snapshot = fixture(project)
    session, cache, out = tmp_path / "raw", tmp_path / "trace", tmp_path / "review"
    session.write_bytes(lines(native))
    cache.write_bytes(lines([{"stage": "stream:context"}, snapshot]))
    with pytest.raises(ValueError, match="truncated"):
        reader["export"](session, project, out)
    result = reader["export"](session, project, out, cache_trace=cache)
    assert session.read_bytes() == (out / "session.jsonl").read_bytes()
    assert cache.read_bytes() == (out / result["cache_trace"]).read_bytes()
    assert result["cache_trace_sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
    trace = json.loads((out / "turn-1.json").read_bytes())
    (call,) = trace["calls"]
    assert (
        call["request_line"],
        call["result_line"],
        call["cache_trace_line"],
        call["cache_message_number"],
    ) == (3, 4, 2, 3)
    assert (
        call["original_evidence"] == "cache-trace.jsonl"
        and call["transcript_evidence"] == "session.jsonl"
    )
    assert call["response"] == json.loads(snapshot["messages"][2]["content"][0]["text"])
    assert trace["answer"] == "Checked."


def test_followup_reloading_guarded_reply_keeps_first_complete_source(tmp_path):
    rows, first = fixture(tmp_path)
    prompt = {"role": "user", "content": "Explain that result.", "timestamp": 2000}
    answer = {
        "role": "assistant",
        "stopReason": "stop",
        "timestamp": 2010,
        "content": [{"type": "text", "text": "Follow-up answer."}],
    }
    for identifier, message in ((4, prompt), (5, answer)):
        rows.append(
            {
                "type": "message",
                "id": str(identifier),
                "parentId": str(identifier - 1),
                "message": message,
            }
        )
    second = {
        **first,
        "runId": "followup",
        "messages": [r["message"] for r in rows[1:]],
        "messageCount": 6,
        "messageRoles": first["messageRoles"] + ["user", "assistant"],
    }
    fingerprints(second)
    _, turns = reader["read_session"](lines(rows), tmp_path, cache_data=lines([first, second]))
    assert turns[0]["calls"][0]["cache_trace_line"] == 1
    assert turns[1]["answer"] == "Follow-up answer."


@pytest.mark.parametrize(
    "bad",
    [
        "stage",
        "session",
        "project",
        "prompt",
        "answer",
        "arguments",
        "id",
        "time",
        "flag",
        "head",
        "tail",
        "count",
        "marker",
        "unseen_middle",
        "conflict",
        "incomplete",
    ],
)
def test_cache_trace_rejects_mismatched_or_incomplete_evidence(tmp_path, bad):
    native, snapshot = fixture(tmp_path)
    cached = snapshot["messages"]
    cache = [snapshot]
    preview = native[3]["message"]["content"][0]
    if bad == "stage":
        snapshot["stage"] = "stream:context"
    elif bad == "project":
        snapshot["workspaceDir"] = str(tmp_path / "elsewhere")
    elif bad == "session":
        snapshot["sessionId"] = "other-session"
    elif bad == "prompt":
        cached[0]["content"] = "Another request"
    elif bad == "answer":
        cached[-1]["content"][0]["text"] += "!"
    elif bad == "arguments":
        cached[1]["content"][0]["arguments"] = {"enabled_plugins": ["other"]}
    elif bad == "id":
        cached[2]["toolCallId"] = "other-call"
    elif bad == "time":
        cached[2]["timestamp"] += 1
    elif bad == "flag":
        cached[2]["isError"] = True
    elif bad == "head":
        preview["text"] = "x" + preview["text"][1:]
    elif bad == "tail":
        preview["text"] = preview["text"].replace("null}", "null]")
    elif bad == "marker":
        preview["text"] = preview["text"].replace("middle content", "middle")
    elif bad == "count":
        preview["text"] = preview["text"].replace(" more characters", "0 more characters")
    elif bad == "incomplete":
        cached.pop()
    elif bad == "unseen_middle":
        cached[2]["content"][0]["text"] = cached[2]["content"][0]["text"].replace(
            "a" * 9000, "a" * 8999 + "x", 1
        )
    elif bad == "conflict":
        other = copy.deepcopy(snapshot)
        other["messages"][2]["content"][0]["text"] = other["messages"][2]["content"][0][
            "text"
        ].replace("a" * 9000, "a" * 8999 + "x", 1)
        fingerprints(other)
        cache.append(other)
    with pytest.raises(ValueError):
        reader["read_session"](lines(native), tmp_path, cache_data=lines(cache))
