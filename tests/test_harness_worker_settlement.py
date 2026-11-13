"""A claimed scientific job must settle once, with its original outcome and execution time."""

import os
import time
import uuid

import numpy as np
import pytest

from autotem.harness import Harness, HarnessError, worker
from autotem.harness.store import encode

pytestmark = pytest.mark.smoke


@pytest.fixture
def queued(tmp_path, monkeypatch):
    image = tmp_path / "image.npy"
    np.save(image, np.ones((8, 12)))
    h = Harness(tmp_path / "h")
    monkeypatch.setattr(h, "_launch_scheduler", lambda run: None)
    plan = {
        "input_path": str(image),
        "steps": [
            {"id": "load", "operation_id": "input.load"},
            {"id": "qa", "operation_id": "quality.assess", "depends_on": ["load"]},
            {"id": "next", "operation_id": "quality.assess", "depends_on": ["qa"]},
        ],
    }
    state = h.submit_analysis(plan)
    return h, state["run_id"], state["job_id"]


def diagnostic(h, rid, jid, status, code):
    state = h.get_run_status(rid)
    job = state["jobs"][-1]
    assert state["status"] == job["status"] == status and job["job_id"] == jid
    assert job["ended"] >= job["started"] and state["error"]["code"] == code
    raw = h.store.get_run(rid)
    assert raw["active_started"] is None and raw["step_started"] is None
    report = h.export_report(rid, ["json"])
    assert report["report_type"] == "diagnostic"
    manifest = h.store.load_json(report["artifacts"][0]["artifact_id"])
    assert manifest["run"]["execution_seconds"] == state["execution_seconds"]
    assert manifest["run"]["error"] == state["error"]
    return state


@pytest.mark.parametrize(
    "failure,code", [("spawn", "WORKER_START_FAILED"), ("tree", "WORKER_OWNERSHIP_FAILED")]
)
def test_launch_failures_close_job_and_preserve_first_diagnosis(queued, monkeypatch, failure, code):
    h, rid, jid = queued
    original = f"fixture {failure} refusal"
    tick = 10.0
    monkeypatch.setattr(worker.time, "monotonic", lambda: tick)

    def refused_launch(*args, **kwargs):
        nonlocal tick
        tick += 0.25
        raise OSError(original)

    def refused_tree(process):
        nonlocal tick
        # Model the real Windows constructor's failed attachment: it closes its own child.
        process.terminate()
        process.wait(timeout=10)
        tick += 0.25
        raise RuntimeError(original)

    if failure == "spawn":
        monkeypatch.setattr(worker.subprocess, "Popen", refused_launch)
    else:
        monkeypatch.setattr(worker, "ProcessTree", refused_tree)
    worker.schedule(str(h.root))
    state = diagnostic(h, rid, jid, "failed", code)
    assert state["error"]["message"] == original
    assert state["execution_seconds"] == 0.25
    assert all(
        s["status"] == "blocked" and s["error"]["code"] == code and s["ended"]
        for s in state["steps"]
    )
    with h.store.connect() as db:
        db.execute("UPDATE runs SET updated=? WHERE run_id=?", (time.time() - 60, rid))
    worker.schedule(str(h.root))
    late = h.get_run_status(rid)
    assert late["error"] == state["error"] and late["jobs"] == state["jobs"]
    assert late["execution_seconds"] == state["execution_seconds"]


def left_by_previous_owner(h, rid, jid, terminal):
    claim = time.time() - 5
    original = {"code": "ORIGINAL_SCIENTIFIC_FAILURE", "message": "fixture operation failed"}
    with h.store.connect() as db:
        db.execute("UPDATE jobs SET status='running',started=? WHERE job_id=?", (claim, jid))
        db.execute(
            "UPDATE runs SET status=?,error=?,worker_pid=?,worker_identity='retired-identity',"
            "active_started=?,run_seconds_used=3,current_step='qa',step_started=?,updated=? "
            "WHERE run_id=?",
            (
                "failed" if terminal else "running",
                encode(original) if terminal else None,
                os.getpid(),
                claim,
                claim + 1,
                claim + 2,
                rid,
            ),
        )
        db.execute(
            "UPDATE steps SET status='completed',started=?,ended=? "
            "WHERE run_id=? AND step_id='load'",
            (claim, claim + 1, rid),
        )
        db.execute(
            "UPDATE steps SET status='running',started=? WHERE run_id=? AND step_id='qa'",
            (claim + 1, rid),
        )
    return original


@pytest.mark.parametrize("terminal", [False, True])
def test_previous_worker_keeps_diagnosis_and_accounts_claim_once(queued, monkeypatch, terminal):
    h, rid, jid = queued
    original = left_by_previous_owner(h, rid, jid, terminal)
    if terminal:
        assert h.get_run_status(rid)["status"] == "finalizing"
        with pytest.raises(HarnessError) as blocked:
            h.export_report(rid, ["json"])
        assert blocked.value.code == "RUN_NOT_FINAL"
    # The live PID belongs to this test, with a different creation identity. Recovery
    # may inspect it but has no Popen handle or permission to signal it.
    monkeypatch.setattr(worker, "ProcessTree", lambda *_: pytest.fail("attached to inherited PID"))
    worker.schedule(str(h.root))
    status = "failed" if terminal else "interrupted"
    code = original["code"] if terminal else "WORKER_INTERRUPTED"
    state = diagnostic(h, rid, jid, status, code)
    assert [s["status"] for s in state["steps"]] == ["completed", status, "blocked"]
    assert state["steps"][1]["error"] == state["error"] and state["steps"][1]["ended"]
    assert state["steps"][2]["error"]["code"] == "UPSTREAM_NOT_COMPLETED"
    if terminal:
        assert state["error"] == original and state["execution_seconds"] == 5
    else:
        assert state["execution_seconds"] >= 8
    worker._settle_job(h.store, rid, jid, "interrupted", {"code": "LATE"})
    again = h.get_run_status(rid)
    assert (
        again["jobs"] == state["jobs"] and again["execution_seconds"] == state["execution_seconds"]
    )
    assert again["error"] == state["error"] and again["steps"] == state["steps"]


def test_late_worker_and_old_settlement_cannot_modify_an_appended_claim(queued):
    h, rid, old = queued
    new = uuid.uuid4().hex
    stamp = time.time()
    with h.store.connect() as db:
        db.execute("UPDATE jobs SET status='completed',ended=? WHERE job_id=?", (stamp, old))
        db.execute(
            "INSERT INTO jobs(job_id,run_id,created,started,status) VALUES(?,?,?,?,'running')",
            (new, rid, stamp, stamp),
        )
        db.execute(
            "UPDATE runs SET status='running',job_id=?,worker_pid=?,worker_identity=?,"
            "active_started=? WHERE run_id=?",
            (new, os.getpid(), worker.process_identity(os.getpid()), stamp, rid),
        )
    before = h.get_run_status(rid)
    worker._settle_job(h.store, rid, old, "failed", {"code": "LATE"})
    h._interrupt_run(rid, "running", {"code": "STALE_POLLER"}, unfinished_job=old)
    assert worker.execute(str(h.root), rid, old) == worker.OWNERSHIP_EXIT
    assert h.get_run_status(rid) == before and h.store.get_run(rid)["active_started"] == stamp


def test_queue_continues_with_a_real_subprocess_after_failed_launch(queued, monkeypatch):
    h, first, first_job = queued
    source = h.store.get_run(first)["request"]["input_path"]
    second = h.submit_analysis(
        {"input_path": source, "steps": [{"id": "load", "operation_id": "input.load"}]}
    )
    spawn = worker.subprocess.Popen

    def fail_first(args, **options):
        # The qualification observer maps the existing final command argument to run_id.
        if args[-1] == first:
            assert options["env"]["AUTOTEM_WORKER_JOB_ID"] == first_job
            raise OSError("First launch is unavailable")
        assert (
            args[-1] == second["run_id"]
            and options["env"]["AUTOTEM_WORKER_JOB_ID"] == second["job_id"]
        )
        return spawn(args, **options)

    monkeypatch.setattr(worker.subprocess, "Popen", fail_first)
    worker.schedule(str(h.root))
    diagnostic(h, first, first_job, "failed", "WORKER_START_FAILED")
    finished = h.get_run_status(second["run_id"])
    assert finished["status"] == finished["jobs"][-1]["status"] == "completed"
    assert finished["steps"][0]["status"] == "completed" and finished["execution_seconds"] > 0
    assert h.export_report(second["run_id"], ["json"])["report_type"] == "analysis"


def test_live_previous_worker_is_respected_even_before_legacy_job_bookkeeping(queued, monkeypatch):
    h, rid, _ = queued
    h.store.update_run(
        rid,
        status="running",
        worker_pid=os.getpid(),
        worker_identity=worker.process_identity(os.getpid()),
    )

    class ObservedWait(Exception):
        pass

    def observed(seconds):
        assert seconds == 0.1
        raise ObservedWait

    monkeypatch.setattr(worker.time, "sleep", observed)
    with pytest.raises(ObservedWait):
        worker.schedule(str(h.root))
    assert h.store.get_run(rid)["status"] == "running"


@pytest.mark.parametrize("elapsed", [0.0, 0.25], ids=["same-clock-tick", "clock-advanced"])
def test_plugin_disappearing_after_submit_fails_before_worker_ownership(
    queued, monkeypatch, elapsed
):
    h, rid, jid = queued
    tick = 10.0
    monkeypatch.setattr(worker.time, "monotonic", lambda: tick)

    def missing(*_):
        nonlocal tick
        tick += elapsed
        raise HarnessError("PLUGIN_NOT_FOUND", "vanished after submit")

    monkeypatch.setattr(worker, "default_registry", missing)
    monkeypatch.setattr(
        worker, "_start_worker", lambda *_: pytest.fail("started without the plugin")
    )
    worker.schedule(str(h.root))
    state = diagnostic(h, rid, jid, "failed", "PLUGIN_NOT_FOUND")
    assert state["error"]["message"] == "vanished after submit"
    assert state["execution_seconds"] == elapsed
    worker.schedule(str(h.root))
    assert h.get_run_status(rid)["execution_seconds"] == elapsed


def test_plugin_failure_inside_owned_child_survives_supervisor_cleanup(queued, monkeypatch):
    h, rid, jid = queued
    left_by_previous_owner(h, rid, jid, False)
    h.store.update_run(
        rid, worker_pid=os.getpid(), worker_identity=worker.process_identity(os.getpid())
    )

    def missing(*_):
        raise ImportError("changed import in the child")

    monkeypatch.setattr(worker, "default_registry", missing)
    worker.execute(str(h.root), rid, jid)
    assert h.get_run_status(rid)["status"] == "finalizing"
    worker._settle_job(h.store, rid, jid, "interrupted", {"code": "WORKER_INTERRUPTED"})
    state = diagnostic(h, rid, jid, "failed", "WORKER_PREPARATION_FAILED")
    assert state["error"]["type"] == "ImportError"


@pytest.mark.parametrize(
    "terminal,legacy", [(False, False), (False, True), (True, False), (True, True)]
)
def test_public_orphan_recovery_uses_the_same_claim_and_accounts_only_once(
    queued, terminal, legacy
):
    h, rid, jid = queued
    original = left_by_previous_owner(h, rid, jid, terminal)
    with h.store.connect() as db:
        stamp = time.time() - 80
        db.execute(
            "UPDATE runs SET updated=?,active_started=? WHERE run_id=?",
            (stamp + 2, None if legacy else stamp, rid),
        )
        if legacy:
            db.execute("UPDATE jobs SET started=NULL WHERE job_id=?", (jid,))
    state = h.get_run_status(rid)
    expected = "failed" if terminal else "interrupted"
    assert state["status"] == state["jobs"][-1]["status"] == expected
    assert state["error"] == (
        original
        if terminal
        else {
            "code": "WORKER_INTERRUPTED",
            "message": "Scientific worker exited without completion",
        }
    )
    assert state["jobs"][-1]["ended"] is not None and h.store.get_run(rid)["active_started"] is None
    if legacy:
        assert state["execution_seconds"] == 3
    elif terminal:
        assert state["execution_seconds"] == 5
    else:
        assert state["execution_seconds"] >= 83
    assert [s["status"] for s in state["steps"]] == ["completed", expected, "blocked"]
    assert h.get_run_status(rid) == state
    report = h.export_report(rid, ["json"])
    assert report["report_type"] == "diagnostic"
