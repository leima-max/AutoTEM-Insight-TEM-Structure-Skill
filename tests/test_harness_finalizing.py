import json
import time

import numpy as np
import pytest

from autotem.harness import Harness, HarnessError


@pytest.fixture
def queued(tmp_path, monkeypatch):
    source = tmp_path / "image.npy"
    np.save(source, np.ones((8, 12)))
    harness = Harness(tmp_path / "harness")
    monkeypatch.setattr(harness, "_launch_scheduler", lambda run: None)
    state = harness.submit_analysis(
        {"input_path": str(source), "steps": [{"id": "load", "operation_id": "input.load"}]},
    )
    return harness, state["run_id"], state["job_id"]


@pytest.mark.parametrize(
    "outcome", ["completed", "failed", "needs_input", "interrupted", "timed_out", "cancelled"]
)
def test_finalizing_until_the_matching_job_and_runtime_are_settled(queued, outcome):
    h, run, job = queued
    with h.store.connect() as db:
        db.execute("UPDATE runs SET status=? WHERE run_id=?", (outcome, run))
        db.execute("UPDATE jobs SET status='running',started=? WHERE job_id=?", (time.time(), job))
    assert h.get_run_status(run)["status"] == "finalizing"
    with pytest.raises(HarnessError) as exc:
        h.export_report(run, ["json"])
    assert exc.value.code == "RUN_NOT_FINAL"
    assert all(a["step_id"] != "_report" for a in h.list_artifacts(run))
    with h.store.connect() as db:
        db.execute("UPDATE runs SET run_seconds_used=7.5 WHERE run_id=?", (run,))
        db.execute("UPDATE jobs SET status=?,ended=? WHERE job_id=?", (outcome, time.time(), job))
    state = h._wait(run)
    assert state["status"] == state["jobs"][-1]["status"] == outcome
    assert state["execution_seconds"] == 7.5
    report = h.export_report(run, ["json"])
    manifest = h.store.load_json(report["artifacts"][0]["artifact_id"])
    assert manifest["run"]["jobs"][-1]["ended"] == state["jobs"][-1]["ended"]
    assert manifest["run"]["execution_seconds"] == 7.5


def test_orphaned_finalization_can_be_diagnosed_and_retried(queued):
    h, run, job = queued
    with h.store.connect() as db:
        db.execute(
            "UPDATE runs SET status='failed',updated=? WHERE run_id=?", (time.time() - 120, run)
        )
        db.execute("UPDATE jobs SET status='running',started=1 WHERE job_id=?", (job,))
    state = h.get_run_status(run)
    assert state["status"] == state["jobs"][-1]["status"] == "interrupted"
    assert state["error"]["code"] == "JOB_FINALIZATION_INTERRUPTED"
    assert state["error"]["recorded_run_status"] == "failed"
    assert h.export_report(run, ["json"])["report_type"] == "diagnostic"
    assert len([e for e in h.get_run_status(run)["events"] if e["kind"] == "interrupted"]) == 1


def test_report_refuses_mismatched_final_job(queued):
    h, run, job = queued
    with h.store.connect() as db:
        db.execute(
            "UPDATE runs SET status='failed',error=? WHERE run_id=?",
            (json.dumps({"code": "NUMERICAL_FAILURE"}), run),
        )
        db.execute("UPDATE jobs SET status='completed',ended=? WHERE job_id=?", (time.time(), job))
    with pytest.raises(HarnessError) as exc:
        h.export_report(run, ["json"])
    assert exc.value.code == "RUN_NOT_FINAL"
    assert all(a["step_id"] != "_report" for a in h.list_artifacts(run))
