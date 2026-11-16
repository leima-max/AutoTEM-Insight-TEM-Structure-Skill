"""Persist the reference-dependent atom-strain conversation through public tools."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from autotem.harness import Harness
from autotem.harness.cli import dispatch_request
from autotem.harness.store import file_hash


def tool(harness, name, **arguments):
    return dispatch_request(
        harness, {"tool": name, "arguments": arguments, "request_id": "closure"}
    )


def result(harness, name, **arguments):
    envelope = tool(harness, name, **arguments)
    assert envelope["ok"], envelope["error"]
    return envelope["result"]


@pytest.mark.parametrize("view", ["summary", "parameters", "full"])
def test_atom_reference_gate_is_discoverable(tmp_path, view):
    (entry,) = result(
        Harness(tmp_path / "h"), "list_capabilities", view=view, operation_id="strain.atom"
    )["operations"]
    assert "needs_input" in entry["description"]
    assert "before" in entry["description"] and "reference" in entry["description"]
    if view != "summary":
        fields = entry["parameter_schema"]["properties"]
        roi = fields["reference_region"]["description"]
        assert "half-open" in roi and "row_start" in roi and "selected raw" in roi
        assert "local" in roi and "answer_schema" in roi
        for name in ("reference_region", "reference_atom_indices"):
            description = fields[name]["description"]
            assert "six" in description and "two-dimensional" in description
            assert "mutually exclusive" in description


def setup_atoms(tmp_path, *, strain=True, selection=None):
    h = Harness(tmp_path / "h")
    yy, xx = np.mgrid[:64, :80]
    points = [(y, x) for y in range(12, 53, 8) for x in range(12, 61, 8)]
    data = sum(np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 2) for y, x in points)
    source = tmp_path / "atoms.npy"
    np.save(source, data)
    plan = {
        "input_path": str(source),
        "modality": "HAADF-STEM",
        "steps": [
            {"id": "load", "operation_id": "input.load"},
            {
                "id": "cal",
                "operation_id": "calibration.resolve",
                "parameters": {"pixel_size_nm": 0.02},
            },
            {"id": "detect", "operation_id": "atomic.detect"},
            {"id": "refine", "operation_id": "atomic.refine", "parameters": {"method": "gaussian"}},
        ],
    }
    if selection is not None:
        plan["selection"] = selection
    if strain:
        plan["steps"].append({"id": "strain", "operation_id": "strain.atom"})
    state = result(h, "submit_analysis", plan=plan, asynchronous=False)
    return h, plan, state


def reply_plan(plan, request, region, raw_reply):
    reply = copy.deepcopy(plan)
    reply["user_input"] = {
        "response": {
            "request_id": request["request_id"],
            "source": "user",
            "basis": "User selected reference on this run's numbered raw-image evidence",
            "raw_reply": raw_reply,
            "evidence_artifact_ids": request["evidence_artifact_ids"],
            "answers": {"reference_region": {"value": region, "unit": "pixel"}},
        }
    }
    return reply


def test_missing_reference_persists_numbered_evidence_and_reopens(tmp_path):
    h, _, parent = setup_atoms(tmp_path)
    assert parent["status"] == "needs_input", parent
    (request,) = parent["input_requests"]
    assert request["status"] == "open" and request["request_id"]
    evidence = request["fields"][0]["evidence"]
    table = result(h, "read_artifact", artifact_id=evidence["table_artifact_id"])
    rows = table["rows"]
    assert len(rows) == 42 and [int(r["atom_index"]) for r in rows] == list(range(42))
    assert evidence["image_artifact_id"] in request["evidence_artifact_ids"]
    assert h.store.role(parent["run_id"], "refined_atoms")
    assert h.store.role(parent["run_id"], "atom_strain") is None
    reopened = result(Harness(h.root), "get_run_status", run_id=parent["run_id"])
    assert reopened["status"] == "needs_input" and reopened["input_requests"] == [request]
    assert not reopened["input_responses"]


def test_valid_reference_reply_creates_child_without_mutating_parent(tmp_path):
    h, plan, parent = setup_atoms(tmp_path)
    (request,) = parent["input_requests"]
    parent_id = parent["run_id"]
    original_plan = copy.deepcopy(h.store.get_run(parent_id)["request"])
    original_artifacts = {a["path"]: a["sha256"] for a in h.list_artifacts(parent_id)}
    reply = reply_plan(
        plan, request, [8, 36, 8, 68], "Rows 8 through 35, columns 8 through 67, inclusive"
    )
    validated = result(h, "validate_plan", plan=reply)
    assert not h.get_run_status(parent_id)["input_responses"]  # read-only preview
    child = result(
        h,
        "submit_analysis",
        plan=validated["plan"],
        parent_run_id=parent_id,
        idempotency_key="reference-answer",
        asynchronous=False,
    )
    assert child["status"] == "completed" and child["run_id"] != parent_id, child
    (accepted,) = child["input_responses"]
    assert accepted["parent_run_id"] == parent_id and accepted["child_run_id"] == child["run_id"]
    assert accepted["validation"]["status"] == "accepted"
    assert accepted["response"]["raw_reply"] == reply["user_input"]["response"]["raw_reply"]
    assert accepted["adopted_values"]["reference_region"]["parameters"]["reference_region"] == [
        8,
        36,
        8,
        68,
    ]
    assert h.store.role(child["run_id"], "atom_strain")
    assert h.store.get_run(parent_id)["request"] == original_plan
    assert all(file_hash(Path(path)) == digest for path, digest in original_artifacts.items())
    reopened = Harness(h.root).get_run_status(parent_id)
    assert reopened["status"] == "needs_input"
    assert reopened["latest_response_child_run_id"] == child["run_id"]


def test_out_of_bounds_reply_is_persisted_without_child_or_job(tmp_path, monkeypatch):
    h, plan, parent = setup_atoms(tmp_path)
    parent_id = parent["run_id"]
    (request,) = parent["input_requests"]
    reply = reply_plan(
        plan, request, [8, 1000, 8, 68], "Rows 8 through 999 and columns 8 through 67, inclusive"
    )
    preview = tool(h, "validate_plan", plan=reply)
    assert not preview["ok"] and preview["error"]["code"] == "INVALID_INPUT_RESPONSE"
    assert not h.get_run_status(parent_id)["input_responses"]
    monkeypatch.setattr(h, "_launch_scheduler", lambda *_: pytest.fail("Unexpected child worker"))
    for _ in range(2):
        rejected = tool(
            h,
            "submit_analysis",
            plan=reply,
            parent_run_id=parent_id,
            idempotency_key="invalid-reference",
            asynchronous=False,
        )
        assert not rejected["ok"] and rejected["error"]["code"] == "INVALID_INPUT_RESPONSE"
    reopened = Harness(h.root).get_run_status(parent_id)
    assert reopened["status"] == "needs_input" and reopened["input_requests"][0]["status"] == "open"
    (refusal,) = reopened["input_responses"]  # stable retries do not duplicate rejection
    assert refusal["child_run_id"] is None
    assert refusal["validation"] == {"status": "rejected", "code": "INVALID_INPUT_RESPONSE"}
    assert refusal["response"]["answers"]["reference_region"]["value"] == [8, 1000, 8, 68]
    assert "raw_reply" not in refusal["response"]  # existing safe rejection storage policy
    assert h.store.role(parent_id, "atom_strain") is None
    with h.store.connect() as db:
        assert db.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1


def test_coordinate_only_request_completes_without_reference(tmp_path):
    h, _, state = setup_atoms(tmp_path, strain=False)
    assert state["status"] == "completed", state
    assert not state["input_requests"]
    assert h.store.role(state["run_id"], "refined_atoms")
    assert h.store.role(state["run_id"], "atom_strain") is None


def test_requested_strain_can_append_to_completed_refinement(tmp_path):
    h, plan, state = setup_atoms(tmp_path, strain=False)
    package = {**plan, "steps": [{"id": "strain", "operation_id": "strain.atom"}]}
    result(h, "validate_plan", plan=package, run_id=state["run_id"])
    pending = result(
        h,
        "submit_analysis",
        plan=package,
        run_id=state["run_id"],
        idempotency_key="requested-strain",
        asynchronous=False,
    )
    assert pending["run_id"] == state["run_id"] and pending["status"] == "needs_input"
    (request,) = pending["input_requests"]
    assert request["operation_id"] == "strain.atom"
    assert request["fields"][0]["evidence"]["table_artifact_id"] in request["evidence_artifact_ids"]
    assert h.store.role(state["run_id"], "atom_strain") is None


def test_reference_reply_uses_source_coordinates_and_converts_origin_once(tmp_path):
    h, plan, parent = setup_atoms(tmp_path, selection={"roi": [8, 56, 12, 72]})
    assert parent["status"] == "needs_input", parent
    (request,) = parent["input_requests"]
    spec = request["fields"][0]["answer_schema"]["reference_region"]["coordinates"]
    assert spec["frame"] == "source_2d"
    assert spec["origin_yx"] == [8, 12]
    source_region = [16, 40, 20, 60]
    expected_local = [8, 32, 8, 48]
    reply = reply_plan(
        plan,
        request,
        source_region,
        "Original image rows 16 through 39, columns 20 through 59, inclusive",
    )
    preview = result(h, "validate_plan", plan=reply)
    assert preview["plan"]["steps"][-1]["parameters"]["reference_region"] == expected_local
    child = result(
        h,
        "submit_analysis",
        plan=preview["plan"],
        parent_run_id=parent["run_id"],
        idempotency_key="source-reference",
        asynchronous=False,
    )
    assert child["status"] == "completed", child
    (accepted,) = child["input_responses"]
    assert accepted["response"]["answers"]["reference_region"]["value"] == source_region
    assert (
        accepted["adopted_values"]["reference_region"]["parameters"]["reference_region"]
        == expected_local
    )
    assert (
        h.store.get_run(child["run_id"])["request"]["steps"][-1]["parameters"]["reference_region"]
        == expected_local
    )
    measured = h.store.load_json(h.store.role(child["run_id"], "atom_strain"))
    assert measured["provenance"]["reference_atom_count"] >= 6
