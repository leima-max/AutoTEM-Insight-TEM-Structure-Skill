"""Case-contract tests. The round boundaries below are test data, never native host evidence."""

import copy
import hashlib
import json
import time

import numpy as np
import pytest

from autotem.acceptance.cases import write_fixture
from autotem.harness import Harness
from autotem.harness.cli import dispatch_request
from tools.host_acceptance import index, verify


@pytest.fixture(scope="module")
def engineering(tmp_path_factory):
    project = tmp_path_factory.mktemp("gpa-j08")
    source, _ = write_fixture("J03", project)
    store = project / "harness"
    h = Harness(store)
    window = time.time()
    turns = [
        {"prompt": p, "session": "unit-test", "trace_sha256": str(i), "calls": []}
        for i, p in enumerate(
            ("工程测试", "选择刚才展示的两个非共线峰", "我选原图中的16至47行和列")
        )
    ]

    def call(turn, tool, **args):
        if tool == "submit_analysis" and "parent_run_id" in args:
            args["idempotency_key"] = args["plan"]["user_input"]["response"]["request_id"]
        response = dispatch_request(h, {"tool": tool, "arguments": args, "request_id": "test"})
        turn["calls"].append({"tool": tool, "arguments": args, "response": response})
        assert response["ok"], response["error"]
        return response["result"]

    def answer(plan, question, values, artifact_id, prompt):
        return {
            **plan,
            "user_input": {
                "response": {
                    "request_id": question["request_id"],
                    "source": "user",
                    "basis": "现场显示的源图或编号峰表",
                    "evidence_artifact_ids": [artifact_id],
                    "raw_reply": prompt,
                    "answers": values,
                }
            },
        }

    plan = {
        "input_path": str(source),
        "modality": "HRTEM",
        "steps": [
            {"id": "load", "operation_id": "input.load"},
            {"id": "gpa", "operation_id": "strain.gpa", "depends_on": ["load"]},
        ],
    }
    parent = call(turns[0], "submit_analysis", plan=plan, asynchronous=False)
    q = parent["input_requests"][0]
    table = q["fields"][0]["evidence"]["peak_table_artifact_id"]
    rows = call(turns[0], "read_artifact", artifact_id=table)["rows"]
    g = [[float(r["gx_cycles_per_pixel"]), float(r["gy_cycles_per_pixel"])] for r in rows]
    other = next(i for i, v in enumerate(g) if abs(np.linalg.det([g[0], v])) > 0.005)
    bad_prompt = "工程控制：我用同一个峰作两束"
    bad_plan = answer(plan, q, {"g_peak_indices": {"value": [0, 0]}}, table, bad_prompt)
    bad_args = {"plan": bad_plan}
    error = dispatch_request(h, {"tool": "validate_plan", "arguments": bad_args})
    assert error["error"]["code"] == "INVALID_INPUT_RESPONSE"
    refusal = {
        **turns[1],
        "prompt": bad_prompt,
        "calls": [{"tool": "validate_plan", "arguments": bad_args, "response": error}],
    }
    p = parent["run_id"]
    # Capture the unanswered/rejected point before the engineering test submits a good reply.
    run_at_question = verify.run_evidence(store, p, (window, time.time()), None)
    middle = call(
        turns[1],
        "submit_analysis",
        parent_run_id=p,
        asynchronous=False,
        plan=answer(plan, q, {"g_peak_indices": {"value": [0, other]}}, table, turns[1]["prompt"]),
    )
    m = middle["run_id"]
    q2 = middle["input_requests"][0]
    preview = h.store.role(m, "raw_preview")
    call(turns[1], "read_artifact", artifact_id=preview)
    final = call(
        turns[2],
        "submit_analysis",
        parent_run_id=m,
        asynchronous=False,
        plan=answer(
            h.store.get_run(m)["request"],
            q2,
            {"reference_region": {"value": [16, 48, 16, 48], "unit": "pixel"}},
            preview,
            turns[2]["prompt"],
        ),
    )
    f = final["run_id"]
    call(turns[2], "export_report", run_id=f, formats=["html", "json"])
    runs = {i: verify.run_evidence(store, i, (window, time.time()), None) for i in (p, m, f)}
    assert {r["source_sha256"] for r in runs.values()} == {
        hashlib.sha256(source.read_bytes()).hexdigest()
    }
    return turns, runs, (p, m, f), refusal, run_at_question


def check(variant, turns, runs):
    calls = [c for turn in turns for c in turn["calls"]]
    verify.check_tool_runs(calls, runs)
    return verify.check_case("J08", calls, runs, turns, {}, variant)


def test_real_gpa_parent_middle_final_and_open_requests(engineering):
    turns, runs, (p, m, f), refusal, open_parent = engineering
    for variant in ("gpa_answered", "answered"):
        record = check(variant, turns, runs)
        assert record["final_run_id"] == f
        assert [
            (edge["parent_run_id"], edge["child_run_id"], edge["answer"])
            for edge in record["answered_edges"]
        ] == [(p, m, "g_peak_indices"), (m, f, "reference_region")]
    for variant, group in (
        ("gpa_unanswered", turns[:1]),
        ("gpa_invalid_reply", [turns[0], refusal]),
    ):
        record = check(variant, group, {p: open_parent})
        assert record["answered_edges"] == [] and record["parent_run_id"] == p
        with pytest.raises(verify.EvidenceError, match="reply_started_unexpected_child"):
            check(variant, group, {p: runs[p]})


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("wrong_parent", "gpa_reply_parent_or_source_mismatch"),
        ("wrong_request", "gpa_observed_reply_differs_from_store"),
        ("wrong_evidence_turn", "gpa_prompt_evidence_not_returned_in_its_turn"),
        ("swapped_user_answers", "gpa_reply_not_from_this_user_turn"),
        ("wrong_roi_reference", "gpa_reply_source_or_evidence_mismatch"),
        ("different_session", "gpa_requires_distinct_turns_in_one_session"),
        ("duplicate_turn", "gpa_requires_distinct_turns_in_one_session"),
        ("dropped_mask", "completed_gpa_fields_missing"),
        ("changed_g_pair", "gpa_selected_vectors_changed_at_roi_reply"),
        ("missing_final_report", "final_gpa_report_not_returned"),
    ],
)
def test_gpa_splices_fail_closed(engineering, fault, reason):
    turns, runs, (p, m, f), _, _ = copy.deepcopy(engineering)
    if fault == "wrong_parent":
        runs[f]["parent_run_id"] = p
        turns[2]["calls"][0]["arguments"]["parent_run_id"] = p
    elif fault == "wrong_request":
        turns[2]["calls"][0]["arguments"]["plan"]["user_input"]["response"]["request_id"] = next(
            iter(runs[p]["_questions"])
        )
    elif fault == "wrong_evidence_turn":
        turns[2]["calls"].append(turns[1]["calls"].pop())
    elif fault == "swapped_user_answers":
        turns[2]["prompt"], turns[1]["prompt"] = turns[1]["prompt"], turns[2]["prompt"]
    elif fault == "wrong_roi_reference":
        row = runs[f]["_replies"][0]
        response = json.loads(row["response"])
        response["evidence_artifact_ids"] = next(iter(runs[p]["question_evidence"].values()))
        row["response"] = json.dumps(response)
    elif fault == "different_session":
        turns[2]["session"] = "unrelated"
    elif fault == "duplicate_turn":
        turns[2]["trace_sha256"] = turns[1]["trace_sha256"]
    elif fault == "dropped_mask":
        runs[f]["artifacts"] = {
            i: a for i, a in runs[f]["artifacts"].items() if a["role"] != "gpa.quality_mask"
        }
    elif fault == "changed_g_pair":
        runs[f]["_request"]["steps"][-1]["parameters"]["g_vectors"] = [[0.1, 0], [0, 0.1]]
    elif fault == "missing_final_report":
        turns[2]["calls"].pop()
    with pytest.raises(verify.EvidenceError, match=reason):
        check("gpa_answered", turns, runs)


def test_unrelated_validation_error_does_not_count_as_gpa_rejection(engineering):
    turns, _, (p, _, _), refusal, open_parent = copy.deepcopy(engineering)
    refusal["calls"][0]["arguments"]["plan"]["user_input"]["response"]["request_id"] = "1" * 32
    with pytest.raises(verify.EvidenceError, match="missing_linked_rejected_gpa_reply"):
        check("gpa_invalid_reply", [turns[0], refusal], {p: open_parent})


def test_supplement_initialization_and_index_preserve_required_baseline(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    (release / "release-manifest.json").write_text(
        json.dumps({"release_id": "test", "profiles": {}})
    )
    raw = tmp_path / "case"
    assert (
        verify.main(
            ["init", str(raw), "--host", "codex", "--case", "J08", "--variant", "gpa_answered"]
        )
        == 0
    )
    summary = verify.verify(raw / "case.json")
    assert summary["case_verdict"] == "not_evaluated" and summary["variant"] == "gpa_answered"
    original = index.build(release)
    indexed = index.build(release, observations=[summary])
    assert indexed["required_variant_counts"] == original["required_variant_counts"]
    assert sum(indexed["required_variant_counts"].values()) == 81
    host = indexed["hosts"]["codex"]
    assert host["cases"]["J08"] == original["hosts"]["codex"]["cases"]["J08"]
    assert host["j08_gpa_observations"]["gpa_answered"]["verdict"] == "not_evaluated"
    assert set(verify.GPA_VARIANTS) == set(index.GPA_VARIANTS)
    assert verify.VARIANTS["J08"] == ["answered", "unanswered", "invalid_reply"]
