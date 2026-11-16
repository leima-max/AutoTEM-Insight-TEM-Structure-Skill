"""Durable invalid-reference reply acceptance, using real Harness transactions."""

import copy
import time

import numpy as np
import pytest

from autotem.harness import Harness
from autotem.harness.cli import dispatch_request
from tools.host_acceptance import verify


@pytest.fixture(scope="module")
def rejected_journey(tmp_path_factory):
    project = tmp_path_factory.mktemp("rejected-reference")
    source = project / "lattice.npy"
    yy, xx = np.mgrid[:48, :64]
    np.save(source, np.cos(2 * np.pi * xx / 8) + np.cos(2 * np.pi * yy / 8))
    h = Harness(project / "runs")
    start = time.time()
    first = {"calls": []}

    def call(tool, **args):
        response = dispatch_request(h, {"tool": tool, "arguments": args})
        return {"tool": tool, "arguments": args, "response": response}

    plan = {
        "input_path": str(source),
        "modality": "HRTEM",
        "steps": [
            {"id": "load", "operation_id": "input.load"},
            {"id": "strain", "operation_id": "strain.gpa"},
        ],
    }
    initial = call("submit_analysis", plan=plan, asynchronous=False)
    assert initial["response"]["ok"], initial
    first["calls"].append(initial)
    state = initial["response"]["result"]
    assert state["status"] == "needs_input"
    rid = state["run_id"]
    question = state["input_requests"][0]
    evidence = question["fields"][0]["evidence"]["peak_table_artifact_id"]
    first["calls"].append(call("read_artifact", artifact_id=evidence))
    reply = copy.deepcopy(plan)
    reply["user_input"] = {
        "response": {
            "request_id": question["request_id"],
            "source": "user",
            "basis": "User selection from numbered peaks",
            "raw_reply": "I choose peak zero twice.",
            "answers": {"g_peak_indices": {"value": [0, 0]}},
            "evidence_artifact_ids": [evidence],
        }
    }
    rejected = call("validate_plan", plan=reply)
    assert rejected["response"]["error"]["code"] == "INVALID_INPUT_RESPONSE"
    validate_turns = [first, {"calls": [rejected]}]
    before = verify.run_evidence(h.root, rid, (start, time.time()), None)
    submitted = call(
        "submit_analysis",
        plan=reply,
        parent_run_id=rid,
        idempotency_key="rejected-once",
        asynchronous=False,
    )
    assert submitted["response"]["error"]["code"] == "INVALID_INPUT_RESPONSE"
    # Reopening the store is part of the regression, not an in-memory fake record.
    h = Harness(h.root)
    after = verify.run_evidence(h.root, rid, (start, time.time()), None)
    status = h.get_run_status(rid)
    assert status["input_requests"][0]["status"] == "open"
    response = status["input_responses"][0]
    assert response["validation"]["status"] == "rejected"
    assert response["child_run_id"] is None
    assert response["response"]["answers"] == {"g_peak_indices": {"value": [0, 0]}}
    assert "raw_reply" not in response["response"] and "basis" not in response["response"]
    return rid, validate_turns, before, [first, {"calls": [submitted]}], after


def check(rid, turns, run):
    calls = [c for t in turns for c in t["calls"]]
    verify.check_tool_runs(calls, {rid: run})
    return verify.check_case("J08", calls, {rid: run}, turns, {}, "invalid_reply")


def test_validation_only_rejection_cannot_prove_durable_closure(rejected_journey):
    rid, turns, before, _, _ = rejected_journey
    with pytest.raises(verify.EvidenceError, match="missing_persisted_rejected_reply"):
        check(rid, turns, before)


def test_real_rejected_submit_is_persisted_without_child_or_new_job(rejected_journey):
    rid, _, before, turns, after = rejected_journey
    assert before["jobs"] == after["jobs"]
    assert not after["_children"]
    check(rid, turns, after)


def test_unrelated_validation_error_cannot_borrow_a_persisted_rejection(rejected_journey):
    rid, _, _, turns, after = copy.deepcopy(rejected_journey)
    turns[1]["calls"][0]["arguments"]["plan"]["user_input"]["response"]["answers"][
        "g_peak_indices"
    ]["value"] = [1, 1]
    with pytest.raises(verify.EvidenceError, match="missing_persisted_rejected_reply"):
        check(rid, turns, after)


def test_rejected_reply_must_match_parent_request(rejected_journey):
    rid, _, _, turns, after = copy.deepcopy(rejected_journey)
    turns[1]["calls"][0]["arguments"]["plan"]["user_input"]["response"]["request_id"] = "f" * 32
    with pytest.raises(verify.EvidenceError, match="missing_persisted_rejected_reply"):
        check(rid, turns, after)


@pytest.mark.parametrize(
    "fault", ["closed_question", "new_job", "wrong_code", "wrong_stored_parent"]
)
def test_rejection_requires_open_question_and_no_new_job(rejected_journey, fault):
    rid, _, _, turns, after = copy.deepcopy(rejected_journey)
    request_id = turns[1]["calls"][0]["arguments"]["plan"]["user_input"]["response"]["request_id"]
    if fault == "closed_question":
        after["_questions"][request_id]["status"] = "answered"
    elif fault == "new_job":
        after["jobs"].append({"job_id": "unexpected", "status": "queued"})
    elif fault == "wrong_code":
        after["_parent_responses"][0]["validation"] = '{"status":"rejected","code":"OTHER"}'
    else:
        after["_parent_responses"][0]["parent_run_id"] = "f" * 32
    with pytest.raises(verify.EvidenceError, match="missing_persisted_rejected_reply"):
        check(rid, turns, after)


def test_unrelated_historic_rejection_does_not_invalidate_matching_reply(rejected_journey):
    rid, _, _, turns, after = copy.deepcopy(rejected_journey)
    unrelated = copy.deepcopy(after["_parent_responses"][0])
    unrelated["request_id"] = "f" * 32
    unrelated["validation"] = '{"status":"rejected","code":"OTHER"}'
    after["_parent_responses"].insert(0, unrelated)
    check(rid, turns, after)


def test_first_turn_job_id_is_sufficient_when_jobs_list_omitted(rejected_journey):
    rid, _, _, turns, after = copy.deepcopy(rejected_journey)
    for call in turns[0]["calls"]:
        data = call["response"]["result"]
        if isinstance(data, dict) and data.get("run_id") == rid:
            data.pop("jobs", None)
    check(rid, turns, after)


@pytest.mark.parametrize("user_input", [[], None, "malformed"])
def test_malformed_user_input_cannot_borrow_rejection(rejected_journey, user_input):
    rid, _, _, turns, after = copy.deepcopy(rejected_journey)
    turns[1]["calls"][0]["arguments"]["plan"]["user_input"] = user_input
    with pytest.raises(verify.EvidenceError, match="missing_persisted_rejected_reply"):
        check(rid, turns, after)


def test_empty_safe_projection_is_not_evidence_for_unsupported_value(rejected_journey):
    import json

    rid, _, _, turns, after = copy.deepcopy(rejected_journey)
    reply = turns[1]["calls"][0]["arguments"]["plan"]["user_input"]["response"]
    reply["answers"] = {"g_peak_indices": {"value": "unsupported free text"}}
    stored = json.loads(after["_parent_responses"][0]["response"])
    stored["answers"] = {}
    after["_parent_responses"][0]["response"] = json.dumps(stored)
    with pytest.raises(verify.EvidenceError, match="missing_persisted_rejected_reply"):
        check(rid, turns, after)
