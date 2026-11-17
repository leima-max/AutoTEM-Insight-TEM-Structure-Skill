"""Synthetic controls for first-call permission, separate from actual field evidence."""

import json
from copy import deepcopy

import pytest

from tools.host_acceptance import verify


def event(method, **params):
    return {"method": method, "params": params}


def example(project):
    session = "unique-created-thread"
    turn = "unique-turn"
    context = {"threadId": session, "turnId": turn}
    name = "list_capabilities"
    item = {
        "type": "mcpToolCall",
        "id": "host-item",
        "server": "autotem",
        "tool": name,
        "arguments": {},
        "status": "inProgress",
    }
    envelope = {"schema_version": "1.0.0", "tool": name, "ok": True, "result": {}, "error": None}
    final = {
        **item,
        "status": "completed",
        "error": None,
        "result": {"structuredContent": envelope},
    }
    client = [
        dict(id=2, **event("thread/start", cwd=str(project))),
        dict(
            id=3, **event("turn/start", threadId=session, input=[{"type": "text", "text": "test"}])
        ),
        {"id": 0, "result": {"action": "accept", "content": {}}},
    ]
    server = [
        {
            "id": 2,
            "result": {"thread": {"id": session}, "cwd": str(project), "approvalsReviewer": "user"},
        },
        {"id": 3, "result": {"turn": {"id": turn}}},
        event("turn/started", threadId=session, turn={"id": turn}),
        {**event("item/started", **context, item=item), "emittedAtMs": 1767225599000},
        event(
            "thread/status/changed",
            threadId=session,
            status={"type": "active", "activeFlags": ["waitingOnApproval"]},
        ),
        {
            **event(
                "mcpServer/elicitation/request",
                **context,
                serverName="autotem",
                mode="form",
                _meta={
                    "codex_approval_kind": "mcp_tool_call",
                    "tool_name": name,
                    "tool_params": {},
                },
                requestedSchema={"type": "object", "properties": {}},
            ),
            "id": 0,
        },
        {
            **event("serverRequest/resolved", threadId=session, requestId=0),
            "emittedAtMs": 1767225601000,
        },
        event(
            "thread/status/changed", threadId=session, status={"type": "active", "activeFlags": []}
        ),
        event("item/completed", **context, item=final),
        event("turn/completed", threadId=session, turn={"id": turn, "status": "completed"}),
    ]
    launch = {
        "utc": "2025-12-31T23:59:00Z",
        "project": str(project),
        "argv": [
            "codex",
            "-c",
            f'mcp_servers.autotem.tools.{name}.approval_mode="prompt"',
            "app-server",
            "--listen",
            "stdio://",
        ],
    }
    choice = {
        "scope": "synthetic scoped test",
        "actor": "field_operator_on_user_authorization",
        "utc": "2026-01-01T00:00:00Z",
        "choice": "allow_this_call",
        "rpc_id": 0,
        "native_item_id": item["id"],
        "tool": name,
    }
    ref = {
        "format": "codex-app-server-stdio",
        "session_id": session,
        "server_trace": "server.jsonl",
        "client_trace": "client.jsonl",
        "launch": "launch.json",
        "process_end": "end.json",
        "operator_decision": "decision.json",
    }
    captured = {
        "session": session,
        "prompt": "test",
        "skill_opened": False,
        "calls": [{"tool": name, "arguments": {}, "response": deepcopy(envelope)}],
    }
    return client, server, launch, choice, ref, captured


@pytest.mark.parametrize(
    "change",
    [
        "valid",
        "wrong_thread",
        "wrong_turn",
        "wrong_project",
        "other_process_client",
        "other_launch",
        "other_end",
        "broken_process",
        "resume",
        "different_prompt",
        "preapproved",
        "auto_reviewer",
        "tool_form",
        "fake_marker",
        "no_host_wait",
        "other_arguments",
        "different_tool",
        "no_reply",
        "denied",
        "duplicate",
        "boolean_id",
        "no_resolve",
        "wrong_resolution_thread",
        "resolution_before_request",
        "wrong_item",
        "unfinished_turn",
        "different_result",
        "transport_error",
        "is_error_result",
        "missing_evidence",
        "foreign_actor",
        "wrong_choice_item",
        "late_choice",
        "prior_call",
        "ambiguous_call",
        "copied_five_lines",
        "different_host",
    ],
)
def test_only_a_bound_native_first_approval_is_credited(tmp_path, change):
    client, server, launch, choice, ref, captured = example(tmp_path)
    end = {
        "utc": "2026-01-01T00:01:00Z",
        "session": captured["session"],
        "exit_code": 0,
        "turn_status": "completed",
    }
    window = {"started_at": launch["utc"], "ended_at": end["utc"]}
    if change == "other_launch":
        launch["utc"] = "2025-12-31T22:59:00Z"
    if change == "other_end":
        end["session"] = "another-process"
    if change == "broken_process":
        end["exit_code"] = 1
    host = "codex"
    if change == "wrong_thread":
        server[5]["params"]["threadId"] = "different"
    if change == "wrong_turn":
        server[5]["params"]["turnId"] = "different"
    if change == "wrong_project":
        client[0]["params"]["cwd"] = str(tmp_path / "another")
    if change == "other_process_client":
        client[1]["params"]["threadId"] = "other-fresh-process-with-id-zero"
    if change == "resume":
        client[0]["method"] = "thread/resume"
    if change == "different_prompt":
        client[1]["params"]["input"][0]["text"] += " changed"
    if change == "preapproved":
        launch["argv"][2] = launch["argv"][2].replace('"prompt"', '"approve"')
    if change == "auto_reviewer":
        server[0]["result"]["approvalsReviewer"] = "auto_review"
    if change == "tool_form":
        server[5]["params"]["requestedSchema"]["properties"]["allow"] = {"type": "boolean"}
    if change == "fake_marker":
        del server[5]["params"]["_meta"]["codex_approval_kind"]
    if change == "no_host_wait":
        server[4]["params"]["status"]["activeFlags"] = ["waitingOnUserInput"]
    if change == "other_arguments":
        server[5]["params"]["_meta"]["tool_params"] = {"other": True}
    if change == "different_tool":
        server[5]["params"]["_meta"]["tool_name"] = "cancel_run"
    if change == "no_reply":
        client.pop()
    if change == "denied":
        client[-1]["result"]["action"] = "decline"
    if change == "duplicate":
        client.append(deepcopy(client[-1]))
    if change == "boolean_id":
        client[-1]["id"] = False
    if change == "no_resolve":
        del server[6]
    if change == "wrong_resolution_thread":
        server[6]["params"]["threadId"] = "different"
    if change == "resolution_before_request":
        server[5], server[6] = server[6], server[5]
    if change == "wrong_item":
        server[8]["params"]["item"]["id"] = "different"
    if change == "unfinished_turn":
        server.pop()
    if change == "different_result":
        captured["calls"][0]["response"]["result"] = {"unobserved": True}
    if change == "transport_error":
        server[8]["params"]["item"]["error"] = {"message": "connection lost"}
    if change == "is_error_result":
        server[8]["params"]["item"]["result"]["isError"] = True
    if change == "foreign_actor":
        choice["actor"] = "mcp_server"
    if change == "wrong_choice_item":
        choice["native_item_id"] = "different-process-item"
    if change == "late_choice":
        choice["utc"] = "2027-01-01T00:00:00Z"
    if change == "prior_call":
        server.insert(
            3,
            event(
                "item/started",
                **{**server[3]["params"], "item": {**server[3]["params"]["item"], "id": "first"}},
            ),
        )
    if change == "ambiguous_call":
        server.insert(
            5,
            event(
                "item/started",
                **{**server[3]["params"], "item": {**server[3]["params"]["item"], "id": "other"}},
            ),
        )
    if change == "copied_five_lines":
        server = [server[i] for i in (3, 5, 6, 8)]
    if change == "different_host":
        host = "workbuddy"
    for key, data in [("server_trace", server), ("client_trace", client)]:
        (tmp_path / ref[key]).write_text(
            "".join(json.dumps(r) + "\n" for r in data), encoding="utf-8"
        )
    (tmp_path / ref["launch"]).write_text(json.dumps(launch), encoding="utf-8")
    (tmp_path / ref["process_end"]).write_text(json.dumps(end), encoding="utf-8")
    if change != "missing_evidence":
        (tmp_path / ref["operator_decision"]).write_text(json.dumps(choice), encoding="utf-8")
    spec = {"host": host, "host_observations": {"authorization": ref}, **window}
    result = {"checks": {}, "evidence": {}}

    def check():
        verify.host_observations(
            spec, tmp_path, [captured], captured["calls"], result, project=tmp_path
        )

    if change == "valid":
        check()
        assert result["checks"]["first_authorization"] == "passed"
        proof = result["evidence"]["authorization"]
        assert proof["actor"] == choice["actor"] and proof["client_reply_line"] == 3
        assert proof["server_lines"]["request"] == 6 and len(proof["sha256"]) == 5
    else:
        with pytest.raises(ValueError):
            check()
        assert result["checks"]["first_authorization"] == "not_evaluated"
