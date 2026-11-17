"""Regression checks for the installed-package acceptance driver and its evidence."""

import hashlib
import json
import queue
from pathlib import Path

import numpy as np
import pytest

from autotem.acceptance.bridge import read_reply
from autotem.acceptance.driver import run, run_case
from autotem.acceptance.evidence import audit_artifact, digest
from autotem.atomic.gaussian_refinement import refine_atom_coordinates
from autotem.harness.cli import PROTOCOL_VERSION


def test_late_reply_cannot_be_delivered_to_the_next_disk_case():
    replies = queue.Queue()
    expired = {}
    first = {"request_id": 1, "tool": "inspect_input"}
    second = {"request_id": 2, "tool": "get_run_status"}
    with pytest.raises(TimeoutError):
        read_reply(replies, first, expired, timeout=0)
    for request in (first, second):
        replies.put(json.dumps({**request, "schema_version": PROTOCOL_VERSION, "ok": True}))
    assert read_reply(replies, second, expired)["request_id"] == 2 and not expired
    replies.put(json.dumps({**first, "schema_version": PROTOCOL_VERSION, "ok": True}))
    with pytest.raises(ValueError, match="Wrong JSONL envelope"):
        read_reply(replies, second, expired)


def test_inspection_failure_reports_the_affected_run_and_error_code(tmp_path):
    def failed_inspection(tool, args):
        assert tool == "inspect_input"
        return {
            "run_id": "inspection-id",
            "status": "failed",
            "error": {"code": "WORKER_OWNERSHIP_FAILED", "message": str(tmp_path)},
        }

    with pytest.raises(ValueError) as error:
        run_case(
            failed_inspection,
            tmp_path,
            {"case_id": "J01", "modality": "HRTEM"},
            tmp_path / "x.npy",
            {},
        )
    assert str(error.value) == "Input inspection run inspection-id failed (WORKER_OWNERSHIP_FAILED)"


def test_refinement_records_each_point_including_rejections_beyond_warning_examples():
    yy, xx = np.mgrid[:24, :112]
    raw = np.zeros_like(xx, dtype=float)
    centers = [(9, x) for x in range(9, 102, 9)]
    for y, x in centers:
        raw[y, x] = 20
    coordinates = [(0, 0), (19, 9), centers[0], *((y, x + 1) for y, x in centers[1:])]
    result = refine_atom_coordinates(raw, coordinates, method="com", max_shift=0.25)
    expected = ["out_of_bounds", "fit_failed", "accepted"]
    expected.extend(["max_shift_rejected"] * (len(centers) - 1))
    assert result.provenance["point_status"] == expected
    np.testing.assert_array_equal(result.coordinates_refined, coordinates)


@pytest.mark.parametrize("field", ["run_id", "sha256", "dtype", "shape", "array_sha256", "path"])
def test_audit_artifact_rejects_mismatched_registered_evidence(tmp_path, field):
    run_id = "example"
    directory = tmp_path / "harness" / "runs" / run_id
    directory.mkdir(parents=True)
    path = directory / "raw.npy"
    pixels = np.arange(12, dtype=float).reshape(3, 4)
    np.save(path, pixels, allow_pickle=False)
    meta = {
        "dtype": str(pixels.dtype),
        "shape": [3, 4],
        "array_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
    }
    item = {
        "path": str(path),
        "run_id": run_id,
        "committed": True,
        "role": "raw",
        "step_id": "load",
        "sha256": digest(path),
        "size": path.stat().st_size,
        "kind": "array",
        "artifact_id": "item-1",
        "execution_usable": True,
        "metadata": meta,
    }
    checked = audit_artifact(tmp_path, run_id, item)
    assert checked["shape"] == [3, 4] and not Path(checked["path"]).is_absolute()
    if field == "path":
        other = tmp_path / "other.npy"
        other.write_bytes(path.read_bytes())
        item[field] = str(other)
    elif field in meta:
        meta[field] = "wrong"
    else:
        item[field] = "wrong"
    with pytest.raises(ValueError):
        audit_artifact(tmp_path, run_id, item)


def test_existing_evidence_survives_retry(tmp_path):
    report = tmp_path / "evidence.json"
    report.write_bytes(b"previous evidence")
    with pytest.raises(ValueError, match="new or empty"):
        run(tmp_path)
    assert report.read_bytes() == b"previous evidence"


def test_disk_to_report_release_evidence(tmp_path):
    evidence = run(tmp_path)
    assert evidence["case_verdict"] == "passed", (evidence.get("error"), evidence["cases"])
    assert {c["case_id"] for c in evidence["cases"]} == {"J01", "J02", "J06"}
    reply = next(c["input_response"] for c in evidence["cases"] if c["case_id"] == "J06")
    assert reply["accepted_reply"]["source"] == "user"
    assert reply["adopted"]["parameters"]["pixel_size_nm"] == 0.02
    assert evidence["host_llm_e2e"] == evidence["scientific_accuracy"] == "not_evaluated"
    assert all(c["case_verdict"] == "passed" for c in evidence["cases"])
    assert evidence["operations"]["sha256"] == digest(tmp_path / "operations.json")
    matrix = json.loads((tmp_path / "operations.json").read_text("utf-8"))["operations"]
    assert matrix["atomic.refine"]["disk_execution"] == "passed"
    assert matrix["simulation.diffraction"]["disk_execution"] == "not_evaluated"
    assert matrix["ai.denoise"]["negative_contract"]["observed_code"] == "DEFERRED_THIS_PHASE"
    assert len({r["tool"] for r in evidence["tool_trace"]}) == 9
    assert str(tmp_path) not in (tmp_path / "evidence.json").read_text("utf-8")
