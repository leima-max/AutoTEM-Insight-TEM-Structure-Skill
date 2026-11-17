"""Reference-storage journey regression tests; engineering fixtures, not native sessions."""

import copy
import time

import numpy as np
import pytest

from autotem.harness import Harness
from autotem.harness.cli import dispatch_request
from tools.host_acceptance import verify


def journey(tmp_path, mode="reference", inspect_mode=None, later=False):
    source = tmp_path / "source.npy"
    yy, xx = np.mgrid[:32, :40]
    np.save(source, np.cos(2 * np.pi * xx / 8) + np.cos(2 * np.pi * yy / 8))
    h = Harness(tmp_path / "runs")
    start = time.time()
    turns = [{"calls": []}, {"calls": []}]

    def call(turn, tool, **args):
        response = dispatch_request(h, {"tool": tool, "arguments": args})
        assert response["ok"], response
        turns[turn]["calls"].append({"tool": tool, "arguments": args, "response": response})
        return response["result"]

    def inspection(turn):
        return call(turn, "inspect_input", path=str(source), input_mode=inspect_mode)

    if inspect_mode and not later:
        inspection(0)
    plan = {
        "input_path": str(source),
        "input_mode": mode,
        "modality": "HRTEM",
        "steps": [
            {"id": "load", "operation_id": "input.load"},
            {"id": "prep", "operation_id": "preprocessing.apply", "parameters": {"enabled": False}},
            {"id": "fft", "operation_id": "diffraction.fft"},
            {"id": "center", "operation_id": "diffraction.center"},
            {
                "id": "peaks",
                "operation_id": "diffraction.peaks",
                "parameters": {"threshold_strategy": "relative_max", "threshold_value": 0.15},
            },
        ],
    }
    parent = call(0, "submit_analysis", plan=plan, asynchronous=False)
    call(0, "list_artifacts", run_id=parent["run_id"])
    revised = copy.deepcopy(plan)
    revised["steps"][-1]["parameters"]["threshold_value"] = 0.25
    call(1, "submit_analysis", plan=revised, parent_run_id=parent["run_id"], asynchronous=False)
    if inspect_mode and later:
        inspection(1)
    return h, start, turns


def evaluate(h, start, turns, variant="reference"):
    calls = [c for t in turns for c in t["calls"]]
    ids = {
        c["response"]["result"]["run_id"]
        for c in calls
        if c["tool"] in {"inspect_input", "submit_analysis"} and c["response"]["ok"]
    }
    runs = {rid: verify.run_evidence(h.root, rid, (start, time.time()), None) for rid in ids}
    verify.check_tool_runs(calls, runs)
    verify.check_case("J09", calls, runs, turns, {}, variant)
    return runs


@pytest.mark.parametrize("later", [False, True])
def test_reference_journey_rejects_actual_snapshot_inspection(tmp_path, later):
    h, start, turns = journey(tmp_path, inspect_mode="snapshot", later=later)
    with pytest.raises(verify.EvidenceError, match="reference_journey_created_snapshot"):
        evaluate(h, start, turns)


@pytest.mark.parametrize("inspection", [None, "reference"])
def test_reference_without_source_copies_passes(tmp_path, inspection):
    h, start, turns = journey(tmp_path, inspect_mode=inspection)
    runs = evaluate(h, start, turns)
    assert all(r["_request"]["input_mode"] == "reference" for r in runs.values())
    assert all(not (r["_directory"] / "raw").exists() for r in runs.values())
    assert any(a["role"] == "raw" for r in runs.values() for a in r["artifacts"].values())


def test_snapshot_variant_keeps_snapshot_preflight_compatible(tmp_path):
    h, start, turns = journey(tmp_path, mode="snapshot", inspect_mode="snapshot")
    evaluate(h, start, turns, "snapshot")


def test_unrelated_project_snapshot_is_not_this_capture(tmp_path):
    h, start, turns = journey(tmp_path)
    source = next(iter(turns[0]["calls"]))["arguments"]["plan"]["input_path"]
    unrelated = h.inspect_input(source)
    runs = evaluate(h, start, turns)
    assert unrelated["run_id"] not in runs


def test_claimed_mode_cannot_override_actual_inspection_record(tmp_path):
    h, start, turns = journey(tmp_path, inspect_mode="snapshot")
    turns[0]["calls"][0]["arguments"]["input_mode"] = "reference"
    with pytest.raises(verify.EvidenceError, match="reference_journey_created_snapshot"):
        evaluate(h, start, turns)
