"""Cancellation races at the preparing, queued and claimed job boundaries."""

import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from autotem.harness import service, store, worker
from autotem.harness.contracts import HarnessError


@pytest.fixture
def submission(tmp_path, monkeypatch):
    source = tmp_path / "source.npy"
    np.save(source, np.arange(64, dtype=np.float32).reshape(8, 8))
    harness = service.Harness(tmp_path / "runs")
    launches = []
    monkeypatch.setattr(harness, "_launch_scheduler", launches.append)
    plan = {
        "input_path": str(source),
        "modality": "HRTEM",
        "input_mode": "snapshot",
        "steps": [{"id": "load", "operation_id": "input.load"}],
    }
    return harness, plan, launches


@pytest.fixture
def live_process():
    with subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"], stdin=subprocess.PIPE
    ) as process:
        try:
            yield process
        finally:
            process.stdin.close()
            process.wait(timeout=15)


def pause_after_observed_run(harness, monkeypatch):
    observed, resume = threading.Event(), threading.Event()
    original = harness.store.get_run

    def read(run_id, connection=None):
        run = original(run_id, connection)
        if connection is None and threading.current_thread().name.startswith("paused-reader"):
            if not observed.is_set():
                observed.set()
                assert resume.wait(60), "writer did not release the observed run snapshot"
        return run

    monkeypatch.setattr(harness.store, "get_run", read)
    return observed, resume


def test_claim_committed_after_cancel_reads_queue_keeps_supervisor_ownership(
    submission, monkeypatch, live_process
):
    h, plan, _ = submission
    initial = h.submit_analysis(plan)
    rid, jid = initial["run_id"], initial["job_id"]
    observed, resume = pause_after_observed_run(h, monkeypatch)
    with ThreadPoolExecutor(1, thread_name_prefix="paused-reader") as pool:
        try:
            with h.store.connect() as writer:
                writer.execute("BEGIN IMMEDIATE")
                started = time.time() - 1.25
                writer.execute(
                    "UPDATE runs SET status='running',active_started=?,worker_pid=?,"
                    "worker_identity=? WHERE run_id=?",
                    (started, live_process.pid, worker.process_identity(live_process.pid), rid),
                )
                writer.execute(
                    "UPDATE jobs SET status='running',started=? WHERE job_id=?", (started, jid)
                )
                cancelled = pool.submit(h.cancel_run, rid)
                assert observed.wait(15), "cancellation did not see the queued WAL snapshot"
        finally:
            resume.set()
        reply = cancelled.result(15)
    assert (reply["status"], reply["jobs"][-1]["status"]) == ("running", "running")
    assert reply["error"] is None and reply["steps"][0]["status"] == "pending"
    assert h.store.get_run(rid)["cancel_requested"] == 1
    assert h.store.get_run(rid)["active_started"] == started
    assert live_process.poll() is None
    with ThreadPoolExecutor(3) as pool:
        repeats = list(pool.map(h.cancel_run, [rid] * 3))
    assert all(r["status"] == "running" for r in repeats)
    assert all([e["kind"] for e in r["events"]].count("cancel_requested") == 1 for r in repeats)
    with pytest.raises(HarnessError) as blocked:
        h.export_report(rid)
    assert blocked.value.code == "RUN_NOT_FINAL"
    error = {"code": "CANCELLED", "message": "Run cancelled by user"}
    worker._settle_job(h.store, rid, jid, "cancelled", error)
    final = h.cancel_run(rid)
    assert final["error"] == error and final["jobs"][-1]["status"] == "cancelled"
    assert final["execution_seconds"] == pytest.approx(final["jobs"][-1]["ended"] - started)
    assert final["steps"][0]["ended"] == final["jobs"][-1]["ended"]
    assert [e["kind"] for e in final["events"]].count("cancel_requested") == 1
    assert live_process.poll() is None


@pytest.mark.parametrize("late_failure", [False, True])
def test_cancelled_snapshot_cannot_publish_or_revive_on_retry(
    submission, monkeypatch, late_failure
):
    h, plan, launches = submission
    copying, resume = threading.Event(), threading.Event()
    copyfile = service.shutil.copyfile
    source = Path(plan["input_path"])
    original_bytes = source.read_bytes()

    def interrupted_copy(src, dst):
        result = copyfile(src, dst)
        copying.set()
        assert resume.wait(15), "cancellation did not finish during the staged copy"
        if late_failure:
            raise OSError("late snapshot failure")
        return result

    monkeypatch.setattr(service.shutil, "copyfile", interrupted_copy)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(h.submit_analysis, plan, idempotency_key="same-key")
        try:
            assert copying.wait(15)
            before = h.submit_analysis(plan, idempotency_key="same-key")
            assert before["status"] == "preparing"
            rid, jid = before["run_id"], before["job_id"]
            stopped = h.cancel_run(rid)
            during = h.submit_analysis(plan, idempotency_key="same-key")
            assert stopped["status"] == during["status"] == "cancelled"
        finally:
            resume.set()
        after = pending.result(15)
    retried = h.submit_analysis(plan, idempotency_key="same-key")
    for result in (stopped, during, after, retried):
        assert (result["status"], result["run_id"], result["job_id"]) == ("cancelled", rid, jid)
        assert [j["status"] for j in result["jobs"]] == ["cancelled"]
        assert result["steps"][0]["ended"] == result["jobs"][0]["ended"]
        assert result["execution_seconds"] == 0
        assert [e["kind"] for e in result["events"]] == ["cancel_requested"]
    assert h.store.get_run(rid)["input_path"] == str(source)
    assert not list((h.store.run_dir(rid) / "raw").iterdir())
    assert not list(h.store.run_dir(rid).glob(".snapshot-*.tmp"))
    assert launches == [] and source.read_bytes() == original_bytes


def test_request_evidence_cannot_be_registered_after_preparing_cancel(submission, monkeypatch):
    h, plan, launches = submission
    publishing, resume = threading.Event(), threading.Event()
    hash_file = store.file_hash
    artifacts_hashed = 0

    def hash_while_preparing(path):
        nonlocal artifacts_hashed
        digest = hash_file(path)
        if path.parent.name == "artifacts" and path.suffix == ".tmp":
            artifacts_hashed += 1
            if artifacts_hashed == 2:
                publishing.set()
                assert resume.wait(15)
        return digest

    monkeypatch.setattr(store, "file_hash", hash_while_preparing)
    with ThreadPoolExecutor(1) as pool:
        preparing = pool.submit(h.submit_analysis, plan, idempotency_key="evidence-key")
        try:
            assert publishing.wait(15)
            retry = h.submit_analysis(plan, idempotency_key="evidence-key")
            rid = retry["run_id"]
            stopped = h.cancel_run(rid)
            evidence = h.list_artifacts(rid)
            assert [item["role"] for item in evidence] == ["request"]
        finally:
            resume.set()
        assert preparing.result(15) == stopped
    assert h.list_artifacts(rid) == evidence
    assert set((h.store.run_dir(rid) / "artifacts").iterdir()) == {Path(evidence[0]["path"])}
    assert hash_file(Path(evidence[0]["path"])) == evidence[0]["sha256"]
    assert launches == []


def test_parallel_cancellation_settles_unclaimed_job_once(submission):
    h, plan, _ = submission
    initial = h.submit_analysis(plan)
    rid, jid = initial["run_id"], initial["job_id"]
    barrier = threading.Barrier(4)

    def cancel():
        barrier.wait(15)
        return h.cancel_run(rid)

    with ThreadPoolExecutor(4) as pool:
        callers = [pool.submit(cancel) for _ in range(4)]
        replies = [future.result(15) for future in callers]
    assert all(reply == replies[0] for reply in replies)
    reply = replies[0]
    assert [event["kind"] for event in reply["events"]] == ["submitted", "cancel_requested"]
    assert reply["events"][-1]["payload"] == {"job_id": jid}
    assert (
        reply["status"] == reply["jobs"][0]["status"] == reply["steps"][0]["status"] == "cancelled"
    )
    assert reply["jobs"][0]["started"] is None and reply["execution_seconds"] == 0
    assert reply["events"][-1]["created"] == reply["jobs"][0]["ended"] == reply["steps"][0]["ended"]


def test_cancel_during_real_finalization_preserves_original_outcome(submission):
    h, plan, _ = submission
    initial = h.submit_analysis(plan)
    rid, jid = initial["run_id"], initial["job_id"]
    started = time.time() - 2
    h.store.update_run(rid, status="completed", active_started=started)
    with h.store.connect() as db:
        db.execute("UPDATE jobs SET status='running',started=? WHERE job_id=?", (started, jid))
    finalizing = h.cancel_run(rid)
    assert finalizing["status"] == "finalizing" and finalizing["error"] is None
    assert h.store.get_run(rid)["cancel_requested"] == 0
    with pytest.raises(HarnessError) as blocked:
        h.export_report(rid)
    assert blocked.value.code == "RUN_NOT_FINAL"
    worker._settle_job(h.store, rid, jid, "interrupted", {"code": "WORKER_EXIT"})
    final = h.get_run_status(rid)
    assert final["status"] == final["jobs"][0]["status"] == "completed"
    assert final["execution_seconds"] > 0 and final["error"] is None


def test_cancel_observed_old_job_cannot_cancel_appended_work_package(submission, monkeypatch):
    h, plan, _ = submission
    initial = h.submit_analysis(plan)
    rid, jid = initial["run_id"], initial["job_id"]
    observed, resume = pause_after_observed_run(h, monkeypatch)
    with ThreadPoolExecutor(1, thread_name_prefix="paused-reader") as pool:
        stale = pool.submit(h.cancel_run, rid)
        try:
            assert observed.wait(15)
            worker.schedule(str(h.root))
            assert h.get_run_status(rid)["status"] == "completed"
            package = {
                **plan,
                "steps": [{"id": "qa", "operation_id": "quality.assess", "depends_on": ["load"]}],
            }
            new = h.submit_analysis(package, run_id=rid, idempotency_key="qa-package")
            assert new["job_id"] != jid and new["status"] == "queued"
        finally:
            resume.set()
        result = stale.result(15)
    assert result["job_id"] == new["job_id"] and result["status"] == "queued"
    assert [j["status"] for j in result["jobs"]] == ["completed", "queued"]
    assert [s["status"] for s in result["steps"]] == ["completed", "pending"]
    assert h.store.get_run(rid)["cancel_requested"] == 0
    assert "cancel_requested" not in [e["kind"] for e in result["events"]]


def test_late_scheduler_launch_error_cannot_interrupt_new_work_package(submission, monkeypatch):
    h, plan, _ = submission
    initial = h.submit_analysis(plan)
    rid = initial["run_id"]
    starting, release = threading.Event(), threading.Event()

    def late_failure():
        starting.set()
        assert release.wait(30)
        raise OSError("scheduler for the previous package did not start")

    monkeypatch.setattr(h, "_start_scheduler", late_failure)
    with ThreadPoolExecutor(1) as pool:
        stale_launch = pool.submit(service.Harness._launch_scheduler, h, rid)
        try:
            assert starting.wait(15)
            worker.schedule(str(h.root))
            package = {
                **plan,
                "steps": [{"id": "qa", "operation_id": "quality.assess", "depends_on": ["load"]}],
            }
            new = h.submit_analysis(package, run_id=rid, idempotency_key="new")
        finally:
            release.set()
        stale_launch.result(15)
    assert h.get_run_status(rid) == new
    assert [j["status"] for j in new["jobs"]] == ["completed", "queued"]


def test_stale_append_cannot_drop_a_newer_completed_work_package(submission, monkeypatch):
    h, plan, _ = submission
    rid = h.submit_analysis(plan)["run_id"]
    worker.schedule(str(h.root))
    qa = {
        **plan,
        "steps": [{"id": "first_qa", "operation_id": "quality.assess", "depends_on": ["load"]}],
    }
    later = {
        **plan,
        "steps": [{"id": "other_qa", "operation_id": "quality.assess", "depends_on": ["load"]}],
    }
    request = h.validate_plan(later, run_id=rid)["plan"]
    observed, release = pause_after_observed_run(h, monkeypatch)
    with ThreadPoolExecutor(1, thread_name_prefix="paused-reader") as pool:
        stale = pool.submit(h._append_work_package, rid, request, "other", "", True)
        try:
            assert observed.wait(15)
            h.submit_analysis(qa, run_id=rid, idempotency_key="first")
            worker.schedule(str(h.root))
            complete = h.get_run_status(rid)
            assert complete["status"] == "completed"
        finally:
            release.set()
        with pytest.raises(HarnessError) as denied:
            stale.result(15)
    assert denied.value.code == "RUN_NOT_READY"
    assert h.get_run_status(rid) == complete
    after_retry = h.submit_analysis(later, run_id=rid, idempotency_key="other")
    assert [s["step_id"] for s in after_retry["steps"]] == ["load", "first_qa", "other_qa"]
    assert [s["id"] for s in h.store.get_run(rid)["request"]["steps"]] == [
        "load",
        "first_qa",
        "other_qa",
    ]


def test_preparation_failure_owner_preserves_first_diagnosis(submission, monkeypatch):
    h, plan, launches = submission

    def partial_copy(src, dst):
        Path(dst).write_bytes(b"partial")
        raise OSError("snapshot disk unavailable")

    monkeypatch.setattr(service.shutil, "copyfile", partial_copy)
    with pytest.raises(OSError, match="snapshot disk unavailable"):
        h.submit_analysis(plan, idempotency_key="failed-key")
    result = h.submit_analysis(plan, idempotency_key="failed-key")
    assert result["status"] == result["jobs"][0]["status"] == "failed"
    assert result["error"] == {"code": "PREPARATION_FAILED", "message": "snapshot disk unavailable"}
    assert result["steps"][0]["error"] == result["error"]
    assert result["steps"][0]["ended"] == result["jobs"][0]["ended"]
    assert not list((h.store.run_dir(result["run_id"]) / "raw").iterdir())
    assert launches == [] and result["execution_seconds"] == 0
