import json
import runpy
from pathlib import Path

import pytest

from_tool = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/host_acceptance/export_openclaw.py")
)
verify = from_tool["_verifier"]


def session(project):
    name = "autotem__validate_plan"
    plan = {"plan": {"steps": [{"operation_id": "ai.atom_columns"}]}}
    reply = {
        "schema_version": "1.0.0",
        "tool": "validate_plan",
        "request_id": "call-1",
        "ok": False,
        "result": None,
        "error": {"code": "DEFERRED_THIS_PHASE", "message": "reserved", "retryable": False},
    }
    calls = [
        {"type": "toolCall", "name": name, "id": "call-1", "arguments": plan},
        {
            "type": "toolCall",
            "name": "read",
            "id": "read-1",
            "arguments": {"path": "skills/autotem/SKILL.md"},
        },
    ]
    messages = [
        {"role": "user", "content": "Can I use the reserved analysis?"},
        {"role": "assistant", "stopReason": "toolUse", "content": calls},
        {
            "role": "toolResult",
            "toolName": "read",
            "toolCallId": "read-1",
            "content": [{"type": "text", "text": "Skill"}],
        },
        {
            "role": "toolResult",
            "toolName": name,
            "toolCallId": "call-1",
            "isError": True,
            "content": [{"type": "text", "text": json.dumps(reply)}],
        },
        {
            "role": "assistant",
            "stopReason": "stop",
            "content": [{"type": "text", "text": "This analysis is reserved."}],
        },
    ]
    records = [{"type": "session", "id": "private-test-fixture", "version": 3, "cwd": str(project)}]
    records.extend(
        {
            "type": "message",
            "id": f"e{i}",
            "parentId": f"e{i - 1}" if i else None,
            "message": message,
        }
        for i, message in enumerate(messages)
    )
    return records


def data(rows):
    return ("\r\n".join(json.dumps(row) for row in rows) + "\r\n").encode()


def test_observation_binds_raw_prompt_arguments_error_and_answer(tmp_path):
    project = tmp_path / "sample"
    project.mkdir()
    raw = tmp_path / "raw.jsonl"
    raw.write_bytes(data(session(project)))
    out = tmp_path / "operator" / "capture"
    info = from_tool["export"](raw, project, out)
    assert (out / "session.jsonl").read_bytes() == raw.read_bytes()
    trace = json.loads((out / "turn-1.json").read_bytes())
    assert (
        trace["user_line"],
        trace["calls"][0]["request_line"],
        trace["calls"][0]["result_line"],
        trace["answer_line"],
    ) == (2, 3, 5, 6)
    turn = verify["read_turn"](out, info["turns"][0], "openclaw", "observed-host-export", "autotem")
    assert turn["session"] == info["session_id"] and turn["answer"] == "This analysis is reserved."
    (call,) = turn["calls"]
    assert call["arguments"]["plan"]["steps"][0]["operation_id"] == "ai.atom_columns"
    assert call["response"]["error"]["code"] == "DEFERRED_THIS_PHASE"
    assert turn["attachments"] == [info["session_sha256"]] * 2
    with pytest.raises(FileExistsError):
        from_tool["export"](raw, project, out)
    with pytest.raises(ValueError, match="outside_project"):
        from_tool["export"](raw, project, project / "accidental")


def test_native_tool_can_resolve_with_a_typed_business_error(tmp_path):
    rows = session(tmp_path)
    rows[4]["message"]["isError"] = False
    _, turns = from_tool["read_session"](data(rows), tmp_path)
    (call,) = turns[0]["calls"]
    assert call["host_is_error"] is False and call["response"]["ok"] is False
    assert call["response"]["error"]["code"] == "DEFERRED_THIS_PHASE"


def test_native_preview_cannot_stand_in_for_the_full_return(tmp_path):
    rows = session(tmp_path)
    rows[4]["message"]["content"][0]["text"] = '{"ok":false\n[middle content omitted]\n}'
    with pytest.raises(ValueError, match="openclaw_tool_result_truncated"):
        from_tool["read_session"](data(rows), tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        "absent",
        "wrong_id",
        "wrong_tool",
        "wrong_envelope_id",
        "wrong_flag",
        "host_error_with_positive",
        "incomplete_error",
        "inherited",
        "subsequent_reuse",
        "empty_tool_use",
        "discarded_response",
        "duplicate",
        "truncated",
        "abort",
        "wrong_project",
        "only_discovery",
    ],
)
def test_unusable_openclaw_exchange_never_exports(tmp_path, change):
    rows = session(tmp_path)
    result = rows[4]["message"]
    if change == "absent":
        rows.pop(4)
        rows[-1]["parentId"] = "e2"
    elif change == "wrong_id":
        result["toolCallId"] = "different"
    elif change == "wrong_tool":
        result["toolName"] = "autotem__submit_analysis"
    elif change == "wrong_envelope_id":
        parsed = json.loads(result["content"][0]["text"])
        parsed["request_id"] = "another"
        result["content"][0]["text"] = json.dumps(parsed)
    elif change == "wrong_flag":
        result["isError"] = "false"
    elif change == "host_error_with_positive":
        positive = json.loads(result["content"][0]["text"])
        positive.update(ok=True, result={}, error=None)
        result["content"][0]["text"] = json.dumps(positive)
    elif change == "incomplete_error":
        partial = json.loads(result["content"][0]["text"])
        del partial["error"]["retryable"]
        result["content"][0]["text"] = json.dumps(partial)
        result["isError"] = False
    elif change == "inherited":
        rows[0]["parentSession"] = str(tmp_path / "other-project" / "original.jsonl")
    elif change == "subsequent_reuse":
        for i, previous in enumerate(rows[1:], 5):
            rows.append({**previous, "id": f"e{i}", "parentId": f"e{i - 1}"})
    elif change == "empty_tool_use":
        rows.append(
            {
                **rows[-1],
                "id": "dangling",
                "parentId": "e4",
                "message": {
                    "role": "assistant",
                    "stopReason": "toolUse",
                    "content": [{"type": "thinking", "thinking": "next"}],
                },
            }
        )
    elif change == "discarded_response":
        result["content"][0]["text"] = "[openclaw] missing tool result in session history"
    elif change == "duplicate":
        rows.insert(5, {**rows[4], "id": "duplicate", "parentId": "e3"})
        rows[-1]["parentId"] = "duplicate"
    elif change == "truncated":
        rows.pop()
    elif change == "abort":
        rows[-1]["message"]["stopReason"] = "aborted"
    elif change == "wrong_project":
        rows[0]["cwd"] = str(tmp_path / "other")
    elif change == "only_discovery":
        rows[2]["message"]["content"][0]["name"] = result["toolName"] = "unrelated"
    with pytest.raises((ValueError, verify["NotEvaluated"])):
        from_tool["read_session"](data(rows), tmp_path)


def test_openclaw_native_branches_and_multiple_user_turns(tmp_path):
    rows = session(tmp_path)
    rows.append(
        {
            **rows[-1],
            "id": "sibling",
            "message": {
                "role": "assistant",
                "stopReason": "stop",
                "content": [{"type": "text", "text": "Alternative wording."}],
            },
        }
    )
    rows.extend(
        [
            {**rows[1], "id": "follow", "parentId": "e4"},
            {**rows[5], "id": "end", "parentId": "follow"},
        ]
    )
    with pytest.raises(ValueError, match="select_an_openclaw_branch_leaf"):
        from_tool["read_session"](data(rows), tmp_path)
    leaf, turns = from_tool["read_session"](data(rows), tmp_path, "end")
    assert leaf == "end" and [len(t["calls"]) for t in turns] == [1, 0]
    _, alternate = from_tool["read_session"](data(rows), tmp_path, "sibling")
    assert len(alternate) == 1 and alternate[0]["answer"] == "Alternative wording."


def test_failed_export_can_retry_at_same_destination(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    raw, out = tmp_path / "test.jsonl", tmp_path / "operator" / "case"
    raw.write_bytes(data(session(project)))
    original_write = Path.write_text

    def interrupt_write(path, *args, **kwargs):
        if path.name == "turns.json":
            raise OSError("test interruption before publication")
        return original_write(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", interrupt_write)
        with pytest.raises(OSError, match="test interruption"):
            from_tool["export"](raw, project, out)
    assert not list(out.parent.iterdir())
    assert len(from_tool["export"](raw, project, out)["turns"]) == 1
