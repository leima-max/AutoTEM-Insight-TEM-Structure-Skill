"""A single bounded scientific subprocess, supervised across CLI invocations."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from autotem._paths import io_path

from .compiler import required_roles
from .contracts import HarnessError, NeedsInput, OperationContext, ServiceResult
from .process_tree import ProcessTree
from .registry import default_registry
from .runtime_binding import selected_runtime
from .store import Store, encode, file_hash
from .workspace import job_workspace, workspace_file

# A child that never received ownership can signal only its supervisor, never change a run.
OWNERSHIP_EXIT = 73


def process_identity(pid: int | None) -> str | None:
    if not pid:
        return None
    if os.name == "nt":
        import ctypes

        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        times = [ctypes.c_ulonglong() for _ in range(4)]
        try:
            result = kernel.GetProcessTimes(
                ctypes.c_void_p(handle), *(ctypes.byref(value) for value in times)
            )
            return str(times[0].value) if result else None
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def process_alive(pid: int | None, identity: str | None = None) -> bool:
    if not pid:
        return False
    if identity is not None and process_identity(pid) != identity:
        return False
    if os.name == "nt":
        import ctypes

        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000 | 0x00100000, False, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(ctypes.c_void_p(handle), 0) == 258
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def cache_path(path: Path) -> str:
    """Numba appends long function/cache names; support run-local Windows paths."""
    value = str(path.resolve())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        return "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return value


def fail_remaining(store, run_id, current, error, status="failed"):
    with store.connect() as db:
        _fail_steps(db, run_id, current, error, status, time.time())


def _fail_steps(db, run_id, current, error, status, ended):
    if current:
        db.execute(
            "UPDATE steps SET status=?,error=?,ended=? WHERE run_id=? AND step_id=? "
            "AND status IN ('pending','running')",
            (status, encode(error), ended, run_id, current),
        )
    db.execute(
        "UPDATE steps SET status='blocked',error=?,ended=? WHERE run_id=? AND status='pending'",
        (encode({"code": "UPSTREAM_NOT_COMPLETED"} if current else error), ended, run_id),
    )


def execute(root: str, run_id: str, job_id: str | None = None):
    store = Store(root)
    # Keep WAL open during bulk publication. Closing the last connection after every
    # small result repeatedly checkpoints and recreates SQLite sidecars on Windows.
    with store.connect():
        return _execute(store, run_id, job_id)


def _preparation_error(exc):
    if isinstance(exc, HarnessError):
        return exc.as_dict()
    return {"code": "WORKER_PREPARATION_FAILED", "message": str(exc), "type": type(exc).__name__}


def _execute(store, run_id, job_id=None):
    pid = os.getpid()
    identity = process_identity(pid)
    ownership_deadline = time.monotonic() + 5
    while time.monotonic() < ownership_deadline:
        ownership = store.get_run(run_id)
        if job_id is not None and (
            ownership["job_id"] != job_id or ownership["status"] != "running"
        ):
            return OWNERSHIP_EXIT
        if (
            identity
            and ownership["worker_pid"] == pid
            and ownership["worker_identity"] == identity
            and ownership["status"] == "running"
        ):
            break
        time.sleep(0.02)
    else:
        return OWNERSHIP_EXIT
    run = store.get_run(run_id)
    request = run["request"]
    workspace = job_workspace(store, run_id, run["job_id"])
    try:
        registry = default_registry(request["enabled_plugins"])
    except Exception as exc:
        error = _preparation_error(exc)
        fail_remaining(store, run_id, None, error)
        store.update_run(run_id, status="failed", error=error)
        return
    store.event(run_id, "worker_started", {"pid": pid, "process_identity": identity})
    status_by_id = {}
    with store.connect() as db:
        existing = {
            row["step_id"]: row["status"]
            for row in db.execute(
                "SELECT step_id,status FROM steps WHERE run_id=?", (run_id,)
            ).fetchall()
        }
    for step in request["steps"]:
        step_id = step["id"]
        if existing.get(step_id) == "completed":
            status_by_id[step_id] = "completed"
            continue
        operation = registry.get(step["operation_id"])
        if step.get("operation_version") != operation.version:
            error = {"code": "OPERATION_VERSION_MISMATCH", "operation_id": operation.operation_id}
            fail_remaining(store, run_id, step_id, error)
            store.update_run(run_id, status="failed", error=error)
            return
        if store.get_run(run_id)["cancel_requested"]:
            store.update_run(run_id, status="cancelled", error={"code": "CANCELLED"})
            fail_remaining(store, run_id, step_id, {"code": "CANCELLED"}, "cancelled")
            return
        if any(status_by_id.get(dep) != "completed" for dep in step["depends_on"]):
            error = {"code": "UPSTREAM_NOT_COMPLETED", "message": step_id}
            if step["optional"]:
                with store.connect() as db:
                    db.execute(
                        "UPDATE steps SET status='skipped',error=?,ended=? "
                        "WHERE run_id=? AND step_id=?",
                        (encode(error), time.time(), run_id, step_id),
                    )
                status_by_id[step_id] = "skipped"
                continue
            store.update_run(run_id, status="failed", error=error)
            fail_remaining(store, run_id, step_id, error)
            return
        try:
            source = workspace_file(Path(store.get_run(run_id)["input_path"]), workspace)
            input_file = io_path(source)
            if not input_file.is_file():
                raise HarnessError("INPUT_UNAVAILABLE", "Pinned source file or snapshot is missing")
            if file_hash(input_file) != run["source_hash"]:
                raise HarnessError("INPUT_CHANGED", "Input changed before calculation")
            availability = registry.availability(operation)
            if availability["state"] != "available":
                raise HarnessError(
                    availability["state"].upper(), operation.operation_id, availability
                )
            context = OperationContext(store, run_id, step_id, request, step["inputs"],
                                       workspace=workspace)
            effective_roles = required_roles(operation, step["parameters"])
            for role in effective_roles:
                context._artifact(role)
            started = time.time()
            details = {
                "step_id": step_id,
                "operation_id": operation.operation_id,
                "operation_version": operation.version,
                "parameters": step["parameters"],
                "profile_id": operation.profile_id,
                "input_artifacts": {
                    role: context._artifact(role)["artifact_id"]
                    for role in effective_roles | set(step["inputs"])
                },
            }
            with store.connect() as db:
                db.execute(
                    "UPDATE runs SET current_step=?,step_started=?,updated=? WHERE run_id=?",
                    (step_id, started, started, run_id),
                )
                db.execute(
                    "UPDATE steps SET status='running',started=? WHERE run_id=? AND step_id=?",
                    (started, run_id, step_id),
                )
                db.execute(
                    "INSERT INTO events(run_id,kind,payload,created) VALUES(?,?,?,?)",
                    (run_id, "operation_started", encode(details), started),
                )
            result = operation.handler(context, step["parameters"])
            if not isinstance(result, ServiceResult):
                raise HarnessError("INVALID_RESULT_CONTRACT", operation.operation_id)
            for role, value in result.outputs.items():
                from autotem.core.models import ArrayReference

                if isinstance(value, ArrayReference):
                    matches = [
                        a
                        for a in store.artifacts(run_id, step_id=step_id)
                        if a["role"] == role and a["path"] == value.uri and a["kind"] == "array"
                    ]
                    if len(matches) != 1:
                        raise HarnessError("UNREGISTERED_ARRAY", role)
                else:
                    context.emit_json(role, value, {"scientific_level": result.scientific_level})
            pending = store.artifacts(run_id, step_id=step_id)
            missing = set(operation.output_roles) - {a["role"] for a in pending}
            if missing:
                raise HarnessError(
                    "MISSING_RESULT", "Handler omitted declared outputs", {"roles": sorted(missing)}
                )
            # Nothing is usable as a downstream role until the step commits successfully.
            store.commit_step(
                run_id,
                step_id,
                {
                    "warnings": result.warnings,
                    "scientific_level": result.scientific_level,
                    "scientific_accuracy": "not_evaluated",
                    "artifact_ids": [a["artifact_id"] for a in pending],
                    "consumed_artifacts": context.consumed_artifacts,
                },
            )
            status_by_id[step_id] = "completed"
        except Exception as exc:
            error = (
                exc.as_dict()
                if isinstance(exc, HarnessError)
                else {"code": "NUMERICAL_FAILURE", "message": str(exc), "type": type(exc).__name__}
            )
            if isinstance(exc, NeedsInput):
                store.record_input_request(
                    run, step, error, context.consumed_artifacts.values(), registry=registry
                )
                return
            state = "failed"
            store.event(run_id, "operation_failed", {"step_id": step_id, "error": error})
            if step["optional"]:
                with store.connect() as db:
                    db.execute(
                        "UPDATE steps SET status='skipped',error=?,ended=? "
                        "WHERE run_id=? AND step_id=?",
                        (encode(error), time.time(), run_id, step_id),
                    )
                status_by_id[step_id] = "skipped"
                continue
            fail_remaining(store, run_id, step_id, error, state)
            store.update_run(run_id, status=state, error=error)
            return
    store.update_run(run_id, status="completed", current_step=None, step_started=None)
    store.event(run_id, "run_completed", {"scientific_accuracy": "not_evaluated"})


def _start_worker(store, run_id, job_id):
    directory = store.run_dir(run_id)
    environment = os.environ.copy()
    temporary = directory / "tmp"
    temporary.mkdir(exist_ok=True)
    environment.update(
        {
            "TEMP": str(temporary),
            "TMP": str(temporary),
            "AUTOTEM_TMP_DIR": str(temporary),
            "MPLCONFIGDIR": str(temporary / "matplotlib"),
            "MPLBACKEND": "Agg",
            "NUMBA_CACHE_DIR": cache_path(temporary / "numba-cache"),
            "PYTHONUNBUFFERED": "1",
            "AUTOTEM_WORKER_JOB_ID": job_id,
        }
    )
    with (directory / "worker.log").open("ab") as log:
        options = ["-I"] if sys.flags.isolated else []
        # CPython's Windows multiprocessing launcher (bpo-35797) starts the base
        # interpreter directly and preserves the venv through this startup marker.
        # The Popen handle and the Job Object must refer to the actual worker PID.
        executable = sys.executable
        if os.name == "nt" and os.path.normcase(sys._base_executable) != os.path.normcase(
            executable
        ):
            environment["__PYVENV_LAUNCHER__"] = executable
            executable = sys._base_executable
        return subprocess.Popen(
            [
                sys.executable,
                *options,
                "-m",
                "autotem.harness.worker",
                "execute",
                str(store.root),
                run_id,
            ],
            executable=executable,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )


def _settle_job(store, run_id, job_id, status, error, *, started=None):
    """Settle only the claimed job; another caller may already have recovered or appended work."""
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        settle_claimed_job(db, run_id, job_id, status, error, started=started)


def settle_claimed_job(db, run_id, job_id, status, error, *, started=None, expected_status=None):
    """Finish a matching active claim under the caller's immediate SQLite transaction."""
    run = db.execute(
        "SELECT r.status,r.error,r.current_step,r.active_started,r.updated FROM runs r "
        "JOIN jobs j ON j.job_id=r.job_id AND j.run_id=r.run_id "
        "WHERE r.run_id=? AND j.job_id=? AND j.status='running'",
        (run_id, job_id),
    ).fetchone()
    if not run or run["status"] in {"queued", "preparing"}:
        return False
    if expected_status is not None and (
        expected_status != run["status"]
        or (expected_status != "running" and run["active_started"] is None and not run["error"])
    ):
        # The emergency reader cannot certify a terminal record without a timed claim
        # or a concrete outcome. Its existing orphan diagnosis handles that case.
        return False
    ended = time.time()
    # A successor has only the persisted wall-clock claim. If the worker already
    # published its outcome, its last update bounds that interval before recovery.
    if started is None:
        last = ended if run["status"] == "running" else min(ended, run["updated"])
        elapsed = max(0, last - (run["active_started"] or last))
    else:
        elapsed = max(0, time.monotonic() - started)
    if run["status"] == "running":
        _fail_steps(db, run_id, run["current_step"], error, status, ended)
        db.execute(
            "UPDATE runs SET status=?,error=? WHERE run_id=?", (status, encode(error), run_id)
        )
        db.execute(
            "INSERT INTO events(run_id,kind,payload,created) VALUES(?,?,?,?)",
            (run_id, status, encode(error), ended),
        )
    else:
        # A worker outcome takes precedence over an observed process exit.
        status = run["status"]
        if run["error"] and status in {"failed", "cancelled", "timed_out", "interrupted"}:
            _fail_steps(db, run_id, run["current_step"], json.loads(run["error"]), status, ended)
    db.execute(
        "UPDATE runs SET run_seconds_used=coalesce(run_seconds_used,0)+?,active_started=NULL,"
        "current_step=NULL,step_started=NULL,updated=? WHERE run_id=?",
        (elapsed, ended, run_id),
    )
    db.execute("UPDATE jobs SET status=?,ended=? WHERE job_id=?", (status, ended, job_id))
    return True


def schedule(root: str):
    store = Store(root)
    pid = os.getpid()
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        try:
            selected_runtime(store.root)
        except HarnessError:
            return
        previous = db.execute("SELECT pid,identity FROM scheduler WHERE singleton=1").fetchone()
        if previous["identity"] and process_alive(previous["pid"], previous["identity"]):
            return
        db.execute(
            "UPDATE scheduler SET pid=?,identity=?,heartbeat=? WHERE singleton=1",
            (pid, process_identity(pid), time.time()),
        )
    next_heartbeat = time.monotonic() + 2.0

    def heartbeat_if_due():
        nonlocal next_heartbeat
        now = time.monotonic()
        if now >= next_heartbeat:
            with store.connect() as db:
                db.execute(
                    "UPDATE scheduler SET heartbeat=? WHERE singleton=1 AND pid=?",
                    (time.time(), pid),
                )
            next_heartbeat = now + 2.0

    try:
        while True:
            heartbeat_if_due()
            with store.connect() as db:
                running = db.execute(
                    "SELECT r.run_id,r.job_id,r.worker_pid,r.worker_identity FROM runs r "
                    "LEFT JOIN jobs j ON j.job_id=r.job_id AND j.run_id=r.run_id "
                    "WHERE r.status='running' OR (j.status='running' "
                    "AND r.active_started IS NOT NULL AND r.status NOT IN ('queued','preparing'))"
                ).fetchall()
            still_running = False
            for row in running:
                if process_alive(row["worker_pid"], row["worker_identity"]):
                    still_running = True
                else:
                    error = {"code": "WORKER_INTERRUPTED", "message": "Previous worker stopped"}
                    _settle_job(store, row["run_id"], row["job_id"], "interrupted", error)
            if still_running:
                time.sleep(0.1)
                continue
            with store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                if db.execute("SELECT pid FROM scheduler WHERE singleton=1").fetchone()[0] != pid:
                    return
                try:
                    selected_runtime(store.root)
                except HarnessError:
                    db.execute("UPDATE scheduler SET pid=NULL WHERE singleton=1 AND pid=?", (pid,))
                    return
                row = db.execute(
                    "SELECT run_id,job_id FROM runs WHERE status='queued' ORDER BY updated LIMIT 1"
                ).fetchone()
                if row is None:
                    # Release under the same lock as the queue check: no lost concurrent submit.
                    db.execute("UPDATE scheduler SET pid=NULL WHERE singleton=1 AND pid=?", (pid,))
                    return
                run_id, job_id = row["run_id"], row["job_id"]
                claimed, started = time.time(), time.monotonic()
                db.execute(
                    "UPDATE runs SET status='running',updated=?,active_started=?, "
                    "worker_pid=NULL,worker_identity=NULL "
                    "WHERE run_id=? AND status='queued'",
                    (claimed, claimed, run_id),
                )
                db.execute(
                    "UPDATE jobs SET status='running',started=? WHERE job_id=?",
                    (claimed, job_id),
                )
            activation = store.get_run(run_id)
            request = activation["request"]
            elapsed_before = activation["run_seconds_used"] or 0
            try:
                registry = default_registry(request["enabled_plugins"])
            except Exception as exc:
                # An enabled plugin can disappear or fail to import after submission.
                error = _preparation_error(exc)
                _settle_job(store, run_id, job_id, "failed", error, started=started)
                continue
            try:
                process = _start_worker(store, run_id, job_id)
            except OSError as exc:
                error = {"code": "WORKER_START_FAILED", "message": str(exc)}
                _settle_job(store, run_id, job_id, "failed", error, started=started)
                continue
            try:
                tree = ProcessTree(process)
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                error = {"code": "WORKER_OWNERSHIP_FAILED", "message": str(exc)}
                _settle_job(store, run_id, job_id, "failed", error, started=started)
                continue
            store.update_run(
                run_id, worker_pid=process.pid, worker_identity=process_identity(process.pid)
            )
            reason = None
            terminal = "interrupted"
            try:
                while process.poll() is None:
                    run = store.get_run(run_id)
                    if run["cancel_requested"]:
                        reason = {"code": "CANCELLED", "message": "User cancelled this worker"}
                    if (
                        elapsed_before + time.monotonic() - started
                        > request["budgets"]["run_seconds"]
                    ):
                        reason = {"code": "TIMEOUT", "message": "Run budget exceeded"}
                    if run["current_step"] and run["step_started"]:
                        step = next(s for s in request["steps"] if s["id"] == run["current_step"])
                        operation = registry.get(step["operation_id"])
                        budget = min(
                            operation.timeout_seconds,
                            request["budgets"][
                                "simulation_seconds"
                                if step["operation_id"].startswith("simulation.")
                                else "operation_seconds"
                            ],
                        )
                        if time.time() - run["step_started"] > budget:
                            reason = {
                                "code": "TIMEOUT",
                                "message": "Operation budget exceeded",
                                "step_id": step["id"],
                                "seconds": budget,
                            }
                    if reason:
                        # Only the Popen object created for this run is terminated.
                        tree.stop()
                        terminal = "cancelled" if reason["code"] == "CANCELLED" else "timed_out"
                        break
                    # Cancellation and budgets are checked every poll. The diagnostic
                    # heartbeat has a separate cadence to avoid competing WAL flushes.
                    heartbeat_if_due()
                    time.sleep(0.05)
            finally:
                tree.close()
            if reason is None:
                if process.returncode == OWNERSHIP_EXIT:
                    terminal = "failed"
                    reason = {
                        "code": "WORKER_OWNERSHIP_FAILED",
                        "message": "Owner handshake not received",
                    }
                else:
                    reason = {"code": "WORKER_INTERRUPTED", "exit_code": process.returncode}
            _settle_job(store, run_id, job_id, terminal, reason, started=started)
    finally:
        with store.connect() as db:
            db.execute("UPDATE scheduler SET pid=NULL WHERE singleton=1 AND pid=?", (pid,))


def main():
    if sys.argv[1] == "schedule":
        schedule(sys.argv[2])
    elif sys.argv[1] == "execute":
        raise SystemExit(execute(sys.argv[2], sys.argv[3], os.environ.get("AUTOTEM_WORKER_JOB_ID")))
    else:
        raise SystemExit("Unknown worker command")


if __name__ == "__main__":
    main()
