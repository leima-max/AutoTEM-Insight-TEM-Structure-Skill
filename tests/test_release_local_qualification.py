"""The maintainer certificate must preserve an expected failure and reject modified evidence."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "acceptance" / "qualify_local.py"


def load_driver():
    spec = importlib.util.spec_from_file_location("test_installed_rg2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.smoke
def test_full_array_controls_and_registration_boundaries():
    q = load_driver()
    module = q.component("numerics")
    numbers = np.array([1.0, 2.0, np.nan])
    module.arrays_equal(numbers, numbers.copy())
    with pytest.raises(AssertionError):
        module.arrays_equal(numbers, np.array([1.0, 3.0, np.nan]))
    with pytest.raises(ValueError, match="shape"):
        module.arrays_equal(numbers, numbers[None])
    with pytest.raises(ValueError, match="keys"):
        module.equal({"status": "ok"}, {"availability": "ok"})
    result = q.captured(lambda: module.equal({"unit": "pixel"}, {"unit": "nm"}))
    assert result["verdict"] == "failed" and "Value differs" in result["reason"]
    assert set(q.verifier_hashes()) == {
        "qualify_local.py",
        "qualify_local_numerics.py",
        "qualify_local_faults.py",
        "qualify_local_state_probe.py",
    }


@pytest.mark.smoke
@pytest.mark.parametrize("threshold", [None, 100.0])
def test_grouped_intensity_qualification_checks_the_full_warning_list(threshold):
    from autotem.atomic.intensity_quantification import SATURATION_THRESHOLD_MISSING

    numerics = load_driver().component("numerics")
    params = {"metric": "normalized_intensity", "mode": "sum_parity"}
    grouping_warning = "No valid atom columns are available for intensity grouping."
    expected = [grouping_warning]
    if threshold is None:
        expected.append(SATURATION_THRESHOLD_MISSING)
    arrays = {
        "raw": np.zeros((32, 32)),
        "atomic_displacement.lattice_indices_ij": np.array([[0, 0]]),
        "atomic_displacement.valid_atom_mask": np.array([False]),
        "intensity_groups.labels": np.array([0]),
    }
    group = {
        "summaries": [],
        "confidence": 0.0,
        "metric": params["metric"],
        "warnings": expected,
        "provenance": {
            "source": "autotem.atomic.intensity_quantification.summarize_atomic_intensity_groups",
            "method": "lattice_sum_parity",
            "parameters": {
                "metric": params["metric"],
                "valid_count": 0,
                "group_count": 0,
                "lattice_group_mode": params["mode"],
            },
        },
    }
    records = {
        "refined_atoms": {"atoms": [{"y": 16, "x": 16, "intensity": {"value": 0}}]},
        "intensity_groups": group,
    }
    intensity_params = {"coordinate_role": "refined_atoms", "saturation_threshold": threshold}
    frame = SimpleNamespace(
        claims={"atomic.intensity": {"parameters": intensity_params}},
        value=lambda op, role: records[role],
        array=lambda op, role: arrays[role],
    )
    assert numerics.atomic_variant(frame, "atomic.intensity_groups", params)["direct_fields"] == 6
    for altered in (
        expected[:-1],
        expected[1:],
        [*expected, SATURATION_THRESHOLD_MISSING],
        [*expected, "Unexpected warning"],
        list(reversed(expected)),
    ):
        if altered != expected:
            group["warnings"] = altered
            with pytest.raises(ValueError):
                numerics.atomic_variant(frame, "atomic.intensity_groups", params)


@pytest.mark.smoke
def test_sqlite_evidence_freezes_uncheckpointed_wal_and_rejects_stale_certificate(tmp_path):
    q = load_driver()
    fault = "owned_jobs_and_isolation"
    directory = tmp_path / fault / "harness"
    directory.mkdir(parents=True)
    with sqlite3.connect(directory / "harness.sqlite3") as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE runs(run_id TEXT,status TEXT,error TEXT,run_seconds_used REAL)")
        db.execute("INSERT INTO runs VALUES ('owned', 'interrupted', NULL, 1.0)")
        db.commit()
        inventory = q.supporting_files(tmp_path, [fault])
    snapshot = tmp_path / "sqlite-snapshots" / fault / "harness" / "harness.sqlite3"
    assert inventory[snapshot.relative_to(tmp_path).as_posix()] == q.digest(snapshot)
    with sqlite3.connect(snapshot) as db:
        assert db.execute("SELECT run_id,status FROM runs").fetchall() == [("owned", "interrupted")]
    with pytest.raises(ValueError, match="Unverified"):
        q.component("faults").recheck(fault, tmp_path, {"verdict": "failed"}, inventory)
    cert = tmp_path / "local-execution.json"
    cert.write_text(
        json.dumps({"schema_version": "1.0.0", "gate": "RG2", "case_verdict": "failed"}), "utf-8"
    )
    with pytest.raises(ValueError, match="Incomplete certificate"):
        q.recheck(cert, tmp_path, tmp_path, tmp_path)


@pytest.mark.smoke
def test_private_runtime_is_required_and_failed_gate_is_saved(tmp_path):
    q = load_driver()
    result = q.qualify(tmp_path, tmp_path, tmp_path, tmp_path / "attempt")
    assert (
        result["case_verdict"] == "failed"
        and result["checks"]["qualification_initialization"]["verdict"] == "failed"
    )
    assert result == q.read(tmp_path / "attempt" / "local-execution.json")


@pytest.mark.smoke
def test_terminal_report_requires_job_settlement_for_every_terminal_state():
    module = load_driver().component("state_probe")
    settled = {"status": "failed", "jobs": [{"status": "failed"}]}
    trial = {
        "run_id": "same",
        "public_failed": settled,
        "immediate_export": {"error": "RUN_NOT_FINAL"},
        "settled_export": {"report_type": "diagnostic", "run": settled},
        "append": {"error": "RUN_NOT_READY"},
    }
    rows = {
        state: {
            "early": {"error": "RUN_NOT_FINAL"},
            "settled": {"run": state, "job": state, "type": "diagnostic"},
        }
        for state in ("failed", "needs_input", "timed_out", "cancelled", "interrupted")
    }
    trace = {"trials": [trial], "controlled_scheduler_boundaries": rows}
    assert module.assessment(trace)["verdict"] == "passed"
    rows["cancelled"]["early"] = {"job": "running"}
    assert module.assessment(trace)["gaps"] == [
        "cancelled diagnostic export accepted before job settled"
    ]
