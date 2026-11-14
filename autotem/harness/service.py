"""The nine coarse host tools share this local, headless execution service."""

from __future__ import annotations

import csv
import hashlib
import html
import importlib.metadata
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from autotem._paths import io_path

from .compiler import compile_plan, validate_graph
from .contracts import HarnessError
from .gpa_evidence import GPA_MASK_INTERPRETATION, gpa_result_mask
from .input_requests import apply_response, parse_response, reject_credentials, rejected_response
from .registry import default_registry
from .runtime_binding import record_job_runtime
from .store import Store, encode, file_hash, owns_preparation
from .workspace import record_workspace, workspace_file

TERMINAL = {"completed", "failed", "needs_input", "cancelled", "timed_out", "interrupted"}


def _finish_unclaimed(db, run_id, job_id, status, step_status, error, ended):
    """Settle a verified unclaimed package under its caller's write transaction."""
    encoded = encode(error)
    db.execute(
        "UPDATE runs SET status=?,error=?,updated=?,worker_pid=NULL,worker_identity=NULL "
        "WHERE run_id=?",
        (status, encoded, ended, run_id),
    )
    db.execute(
        "UPDATE jobs SET status=?,ended=? WHERE run_id=? AND job_id=? AND status='queued'",
        (status, ended, run_id, job_id),
    )
    db.execute(
        "UPDATE steps SET status=?,error=?,ended=? WHERE run_id=? AND status='pending'",
        (step_status, encoded, ended, run_id),
    )


class Harness:
    def __init__(
        self, root: str | Path = "outputs/harness", *, queue_limit: int = 32,
        workspace: Path | None = None,
    ):
        self.store = Store(root)
        self.root = self.store.root
        self.queue_limit = queue_limit
        self.workspace = workspace.resolve(strict=True) if workspace is not None else None

    def list_capabilities(self, enabled_plugins: list[str] | None = None) -> dict:
        self._check_workspace_plugins(enabled_plugins)
        return {
            "envelope_version": "1.0.0",
            "operations": default_registry(enabled_plugins).capabilities(),
            "installed_plugins": [
                {
                    "name": entry.name,
                    "entry_point": entry.value,
                    "version": entry.dist.version if entry.dist else None,
                    "enabled": entry.name in (enabled_plugins or []),
                }
                for entry in importlib.metadata.entry_points(group="autotem.operations")
            ],
            "execution": "headless",
            "scientific_accuracy": "not_evaluated",
        }

    def validate_plan(self, plan, run_id: str | None = None) -> dict:
        compiled = compile_plan(plan)
        self._check_workspace_plugins(compiled.enabled_plugins)
        reply = parse_response(compiled.user_input)
        response_validation = None
        if reply:
            compiled = compiled.model_copy(deep=True)
            if run_id:
                raise HarnessError("CHILD_RUN_REQUIRED", "A scientific answer creates a child run")
            record = self.store.input_request(reply.request_id)
            parent = self.store.get_run(record["run_id"])
            if parent["status"] != "needs_input":
                raise HarnessError(
                    "INPUT_REQUEST_NOT_ACTIVE", "The parent is not awaiting scientific input"
                )
            adopted = apply_response(compiled, reply, record, self.store, parent)
            compiled.user_input = {"response": reply.model_dump(mode="json", exclude_none=True)}
            response_validation = {
                "request_id": reply.request_id,
                "parent_run_id": record["run_id"],
                "adopted_values": adopted,
            }
        for step in compiled.steps:
            if step.operation_id == "input.load":
                step.parameters = {**compiled.selection, **step.parameters}
                step.parameters.setdefault("modality", compiled.modality)
        registry = default_registry(compiled.enabled_plugins)
        for step in compiled.steps:
            for artifact_id in step.inputs.values():
                if self.store.artifact(artifact_id)["run_id"] != run_id:
                    raise HarnessError(
                        "CROSS_RUN_ARTIFACT", "Bind artifacts from the shared run only"
                    )
        seed_roles, seed_steps = set(), set()
        if run_id:
            state = self.get_run_status(run_id)
            seed_steps = {s["step_id"] for s in state["steps"] if s["status"] == "completed"}
            with self.store.connect() as db:
                seed_roles = {
                    row[0]
                    for row in db.execute(
                        "SELECT role FROM roles WHERE run_id=?", (run_id,)
                    ).fetchall()
                }
        warnings = validate_graph(compiled, registry, self.store, seed_roles, seed_steps)
        if response_validation:
            from .input_geometry import check_geometry

            check_geometry(compiled, record, adopted, parent, self.store)
        return {
            "valid": True,
            "plan": compiled.model_dump(mode="json"),
            "warnings": warnings,
            "scientific_accuracy": "not_evaluated",
            **({"input_response_validation": response_validation} if reply else {}),
        }

    def _check_workspace_plugins(self, enabled_plugins):
        if self.workspace is not None and enabled_plugins:
            raise HarnessError(
                "WORKSPACE_INPUT_RESTRICTED", "Additional operation plugins require a reviewed service."
            )

    def inspect_input(self, path: str, **selection) -> dict:
        modality = selection.pop("modality", "unknown")
        result = self.submit_analysis(
            {
                "input_path": path,
                "modality": modality,
                "selection": selection,
                "steps": [{"id": "inspect", "operation_id": "input.load"}],
            },
            asynchronous=False,
            execution_mode="inspection",
        )
        result["artifacts"] = self.list_artifacts(result["run_id"])
        dataset_id = self.store.role(result["run_id"], "dataset")
        result["dataset"] = self.store.load_json(dataset_id) if dataset_id else None
        return result

    def submit_analysis(
        self,
        plan,
        idempotency_key: str | None = None,
        parent_run_id: str | None = None,
        execution_mode: str = "sequential",
        user_request: str | None = None,
        asynchronous: bool = True,
        run_id: str | None = None,
    ) -> dict:
        from .worker import process_identity

        reject_credentials(user_request)
        reject_credentials(idempotency_key)
        if idempotency_key is not None and (not idempotency_key or len(idempotency_key) > 200):
            raise HarnessError("INVALID_IDEMPOTENCY_KEY", "Key length must be 1..200")
        if run_id and parent_run_id:
            raise HarnessError("INVALID_SUBMISSION", "Choose shared run or child run")

        def digest(value):
            return hashlib.sha256(
                encode(
                    {
                        "plan": value,
                        "parent": parent_run_id,
                        "mode": execution_mode,
                        "user_request": user_request,
                    }
                ).encode()
            ).hexdigest()

        previous = None
        submitted_hash = None
        if idempotency_key and not run_id:
            with self.store.connect() as db:
                previous = db.execute(
                    "SELECT run_id,submitted_hash,request_hash FROM runs WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                if db.execute(
                    "SELECT 1 FROM submissions WHERE idempotency_key=?", (idempotency_key,)
                ).fetchone():
                    raise HarnessError(
                        "IDEMPOTENCY_CONFLICT", "Key was used for another work package"
                    )
        # A work-package retry must be checked before duplicate immutable step validation.
        if run_id and idempotency_key:
            with self.store.connect() as db:
                prior = db.execute(
                    "SELECT * FROM submissions WHERE idempotency_key=?", (idempotency_key,)
                ).fetchone()
            if prior:
                compiled = compile_plan(plan)
                self._check_workspace_plugins(compiled.enabled_plugins)
                registry = default_registry(compiled.enabled_plugins)
                for step in compiled.steps:
                    step.operation_version = registry.get(step.operation_id).version
                    step.parameters = registry.validate_parameters(
                        registry.get(step.operation_id), step.parameters
                    )
                return self._append_work_package(
                    run_id,
                    compiled.model_dump(mode="json"),
                    idempotency_key,
                    user_request,
                    asynchronous,
                )
        parent = self.store.get_run(parent_run_id) if parent_run_id else None
        records = (
            self.store.input_records(parent)[0]
            if parent and parent["status"] == "needs_input"
            else []
        )
        try:
            if idempotency_key and not run_id:
                # Read a file-backed plan once. Successful retries can refer to their
                # existing child even after a reference file or parent evidence changes.
                plan = compile_plan(plan).model_dump(mode="json")
                reject_credentials(plan["user_input"])
                submitted_hash = digest(plan)
                if previous and submitted_hash in (
                    previous["submitted_hash"],
                    previous["request_hash"],
                ):
                    return self._existing_submission(previous["run_id"], asynchronous)
            validated = self.validate_plan(plan, run_id=run_id)
        except HarnessError as exc:
            if previous:
                raise HarnessError(
                    "IDEMPOTENCY_CONFLICT", "Key was used for another request"
                ) from exc
            # A rejected reply is auditable without allocating a child or starting a worker.
            submitted = plan.get("user_input", {}) if isinstance(plan, dict) else {}
            response = submitted.get("response") if isinstance(submitted, dict) else None
            if (
                records
                and isinstance(response, dict)
                and response.get("request_id") == records[0]["request_id"]
            ):
                with self.store.connect() as db:
                    self.store.record_input_response(
                        db,
                        records[0],
                        rejected_response(submitted, records[0], exc),
                        {},
                        None,
                        {"status": "rejected", "code": exc.code},
                        retry_key=idempotency_key,
                    )
            raise
        request = validated["plan"]
        response_validation = validated.get("input_response_validation")
        if response_validation:
            if not parent or response_validation["parent_run_id"] != parent_run_id:
                raise HarnessError(
                    "INPUT_REQUEST_PARENT_MISMATCH", "Use the run that requested this input"
                )
            if not idempotency_key:
                raise HarnessError("IDEMPOTENCY_KEY_REQUIRED", "Use a stable key for this reply")
        elif records:
            raise HarnessError(
                "INPUT_RESPONSE_REQUIRED",
                "Supply a sourced plan.user_input.response",
                {"request_id": records[0]["request_id"], "fields": records[0]["fields"]},
            )
        if run_id:
            return self._append_work_package(
                run_id, request, idempotency_key, user_request, asynchronous
            )
        request_hash = digest(request)
        if previous:
            if previous["request_hash"] != request_hash:
                raise HarnessError("IDEMPOTENCY_CONFLICT", "Key was used for another request")
            return self._existing_submission(previous["run_id"], asynchronous)
        if parent and parent["request"]["input_path"] != request["input_path"]:
            raise HarnessError("CHILD_INPUT_MISMATCH", "A child run must use the same source")
        source = Path(request["input_path"]).expanduser().resolve()
        if parent and parent["request"]["input_mode"] == request["input_mode"] == "snapshot":
            source = Path(parent["input_path"])
        elif parent and parent["request"]["input_mode"] == "reference":
            source = Path(parent["input_path"])
        source_io = io_path(workspace_file(source, self.workspace))
        if not source_io.is_file():
            raise HarnessError("INPUT_NOT_FOUND", str(source))
        source_digest = file_hash(source_io)
        if parent and source_digest != parent["source_hash"]:
            raise HarnessError("INPUT_CHANGED", "Source changed since parent run")
        now = time.time()
        run_id = uuid.uuid4().hex
        job_id = uuid.uuid4().hex
        # Before queue publication, the submitting process owns preparation. The scheduler
        # and scientific worker claim their own identities only after this handoff.
        preparer_pid = os.getpid()
        preparer_identity = process_identity(preparer_pid)
        parameter_diff = None
        if parent:
            parameter_diff = {
                key: {"before": parent["request"].get(key), "after": value}
                for key, value in request.items()
                if parent["request"].get(key) != value
            }
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if idempotency_key:
                if db.execute(
                    "SELECT 1 FROM submissions WHERE idempotency_key=?", (idempotency_key,)
                ).fetchone():
                    raise HarnessError(
                        "IDEMPOTENCY_CONFLICT", "Key was used for another work package"
                    )
                previous = db.execute(
                    "SELECT run_id,request_hash FROM runs WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                if previous:
                    if previous["request_hash"] != request_hash:
                        raise HarnessError(
                            "IDEMPOTENCY_CONFLICT", "Key was used for another request"
                        )
                    run_id = previous["run_id"]
                    db.commit()
                    return self._existing_submission(run_id, asynchronous)
            count = db.execute(
                "SELECT COUNT(*) FROM runs WHERE status IN ('queued','preparing')"
            ).fetchone()[0]
            if count >= self.queue_limit:
                raise HarnessError("QUEUE_FULL", "The bounded scientific queue is full")
            db.execute(
                "INSERT INTO runs(run_id,parent_run_id,idempotency_key,request_hash,request,"
                "status,created,updated,input_path,source_hash,execution_mode,user_request,"
                "parameter_diff,submitted_hash,worker_pid,worker_identity) "
                "VALUES(?,?,?,?,?,'preparing',?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    parent_run_id,
                    idempotency_key,
                    request_hash,
                    encode(request),
                    now,
                    now,
                    str(source),
                    source_digest,
                    execution_mode,
                    user_request,
                    encode(parameter_diff) if parameter_diff else None,
                    submitted_hash,
                    preparer_pid,
                    preparer_identity,
                ),
            )
            for step in request["steps"]:
                db.execute(
                    "INSERT INTO steps(run_id,step_id,operation_id,status) VALUES(?,?,?,'pending')",
                    (run_id, step["id"], step["operation_id"]),
                )
            db.execute("UPDATE runs SET job_id=? WHERE run_id=?", (job_id, run_id))
            db.execute(
                "INSERT INTO jobs(job_id,run_id,created,status) VALUES(?,?,?,'queued')",
                (job_id, run_id, now),
            )
            record_job_runtime(db, self.root, run_id, job_id, encode(request), request)
            record_workspace(db, run_id, job_id, self.workspace)
            if response_validation:
                self.store.record_input_response(
                    db,
                    records[0],
                    request["user_input"]["response"],
                    response_validation["adopted_values"],
                    run_id,
                    {"status": "accepted"},
                )
        staged_snapshot = None
        input_path = source
        queued = False
        try:
            directory = self.store.run_dir(run_id)
            directory.mkdir(parents=True, exist_ok=False)
            self.store.write_json(run_id, "_request", "request", request, preparing_job=job_id)
            environment = {"python": sys.version, "packages": {}, "operations": []}
            for name in ("autotem-insight", "numpy", "scipy", "scikit-image", "pydantic", "Pillow"):
                try:
                    environment["packages"][name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    environment["packages"][name] = "source_checkout"
            registry = default_registry(request["enabled_plugins"])
            for step in request["steps"]:
                definition = registry.get(step["operation_id"])
                source_file = Path(inspect.getfile(definition.handler))
                environment["operations"].append(
                    {
                        "operation_id": definition.operation_id,
                        "version": definition.version,
                        "source": str(source_file),
                        "source_sha256": file_hash(source_file),
                        "plugin": registry.plugin_sources.get(definition.operation_id),
                    }
                )
            self.store.write_json(
                run_id, "_request", "environment", environment, preparing_job=job_id
            )
            code_root = Path(__file__).resolve().parents[1]
            self.store.write_json(
                run_id,
                "_request",
                "implementation_fingerprints",
                {
                    path.relative_to(code_root).as_posix(): file_hash(path)
                    for path in sorted(code_root.rglob("*.py"))
                },
                preparing_job=job_id,
            )
            if request["input_mode"] == "snapshot":
                if source_io.stat().st_size > request["budgets"]["max_snapshot_bytes"]:
                    raise HarnessError(
                        "SNAPSHOT_QUOTA", "Snapshot quota exceeded; select reference mode"
                    )
                raw_directory = directory / "raw"
                raw_directory.mkdir()
                input_path = raw_directory / source.name
                staged_snapshot = directory / f".snapshot-{job_id}.tmp"
                shutil.copyfile(source_io, staged_snapshot)
                if (
                    file_hash(staged_snapshot) != source_digest
                    or file_hash(source_io) != source_digest
                ):
                    raise HarnessError("INPUT_CHANGED", "Input changed during snapshot")
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                preparing = owns_preparation(db, run_id, job_id)
                if preparing:
                    stamp = time.time()
                    db.execute(
                        "UPDATE runs SET input_path=?,status='queued',updated=?,"
                        "worker_pid=NULL,worker_identity=NULL WHERE run_id=?",
                        (str(input_path), stamp, run_id),
                    )
                    submitted = {
                        "request_hash": request_hash,
                        "source_hash": source_digest,
                        "execution_mode": execution_mode,
                        "parent_run_id": parent_run_id,
                        "parameter_diff": parameter_diff,
                    }
                    db.execute(
                        "INSERT INTO events(run_id,kind,payload,created) VALUES(?,'submitted',?,?)",
                        (run_id, encode(submitted), stamp),
                    )
                    if staged_snapshot is not None:
                        os.replace(staged_snapshot, io_path(input_path))
            queued = preparing
        except Exception as exc:
            error = (
                exc.as_dict()
                if isinstance(exc, HarnessError)
                else {"code": "PREPARATION_FAILED", "message": str(exc)}
            )
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                failed = owns_preparation(db, run_id, job_id)
                if failed:
                    _finish_unclaimed(db, run_id, job_id, "failed", "blocked", error, time.time())
            if failed:
                raise
        finally:
            if staged_snapshot is not None:
                staged_snapshot.unlink(missing_ok=True)
        if queued:
            self._launch_scheduler(run_id)
        return self.get_run_status(run_id) if asynchronous else self._wait(run_id)

    def _existing_submission(self, run_id, asynchronous):
        run = self.store.get_run(run_id)
        if run["status"] == "queued":
            self._launch_scheduler(run_id)
        elif run["status"] == "preparing" and run["worker_pid"] is None:
            # Pre-upgrade clients left no owner identity. Return the recorded status to
            # a waiting retry so the user can inspect or cancel it without spawning work.
            return self.get_run_status(run_id)
        return self.get_run_status(run_id) if asynchronous else self._wait(run_id)

    def _append_work_package(self, run_id, request, key, user_request, asynchronous):
        original = self.store.get_run(run_id)
        digest = hashlib.sha256(
            encode({"request": request, "run_id": run_id, "user_request": user_request}).encode()
        ).hexdigest()
        for name in (
            "input_path",
            "modality",
            "selection",
            "input_mode",
            "enabled_plugins",
            "network_enabled",
            "budgets",
        ):
            if request[name] != original["request"][name]:
                raise HarnessError("CHILD_RUN_REQUIRED", f"Change to {name} needs a child run")
        combined = {**original["request"], "steps": original["request"]["steps"] + request["steps"]}
        job_id = uuid.uuid4().hex
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if key:
                previous = db.execute(
                    "SELECT * FROM submissions WHERE idempotency_key=?", (key,)
                ).fetchone()
                if previous:
                    if previous["request_hash"] != digest or previous["run_id"] != run_id:
                        raise HarnessError(
                            "IDEMPOTENCY_CONFLICT", "Key was used for another work package"
                        )
                    db.commit()
                    return self.get_run_status(run_id) if asynchronous else self._wait(run_id)
                if db.execute("SELECT 1 FROM runs WHERE idempotency_key=?", (key,)).fetchone():
                    raise HarnessError(
                        "IDEMPOTENCY_CONFLICT", "Key was used for initial submission"
                    )
            state = db.execute(
                "SELECT r.status,r.job_id,j.status AS job_status FROM runs r LEFT JOIN jobs j "
                "ON j.job_id=r.job_id AND j.run_id=r.run_id WHERE r.run_id=?",
                (run_id,),
            ).fetchone()
            if state["job_id"] != original["job_id"]:
                raise HarnessError(
                    "RUN_NOT_READY",
                    "Work package changed during submission. Retry against the current run.",
                )
            if state["job_status"] in {"queued", "running"}:
                raise HarnessError("RUN_NOT_READY", "Wait for the current worker to finalize")
            if state["status"] != "completed":
                raise HarnessError(
                    "RUN_NOT_READY",
                    "Append only after this run completes; failures need a child run",
                )
            if len(combined["steps"]) > 100:
                raise HarnessError("RUN_STEP_LIMIT", "A run supports at most 100 immutable steps")
            if original["run_seconds_used"] >= request["budgets"]["run_seconds"]:
                raise HarnessError(
                    "RUN_BUDGET_EXHAUSTED", "Create a child run after reviewing the budget"
                )
            queued = db.execute(
                "SELECT COUNT(*) FROM runs WHERE status IN ('queued','preparing')"
            ).fetchone()[0]
            if queued >= self.queue_limit:
                raise HarnessError("QUEUE_FULL", "The bounded scientific queue is full")
            for step in request["steps"]:
                db.execute(
                    "INSERT INTO steps(run_id,step_id,operation_id,status) VALUES(?,?,?,'pending')",
                    (run_id, step["id"], step["operation_id"]),
                )
            if key:
                db.execute("INSERT INTO submissions VALUES(?,?,?)", (key, digest, run_id))
            db.execute(
                "UPDATE runs SET request=?,status='queued',updated=?,current_step=NULL,"
                "step_started=NULL,worker_pid=NULL,error=NULL,cancel_requested=0 WHERE run_id=?",
                (encode(combined), time.time(), run_id),
            )
            db.execute("UPDATE runs SET job_id=? WHERE run_id=?", (job_id, run_id))
            db.execute(
                "INSERT INTO jobs(job_id,run_id,created,status) VALUES(?,?,?,'queued')",
                (job_id, run_id, time.time()),
            )
            record_job_runtime(db, self.root, run_id, job_id, encode(combined), combined)
            record_workspace(db, run_id, job_id, self.workspace)
            db.execute(
                "INSERT INTO events(run_id,kind,payload,created) "
                "VALUES(?,'work_package_submitted',?,?)",
                (
                    run_id,
                    encode(
                        {"request": request, "user_request": user_request, "request_hash": digest}
                    ),
                    time.time(),
                ),
            )
        self._launch_scheduler(run_id)
        return self.get_run_status(run_id) if asynchronous else self._wait(run_id)

    def _interrupt_run(self, run_id, expected_status, error, *, unfinished_job=None):
        from .worker import settle_claimed_job

        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if (
                unfinished_job
                and expected_status != "queued"
                and settle_claimed_job(
                    db,
                    run_id,
                    unfinished_job,
                    "interrupted",
                    error,
                    expected_status=expected_status,
                )
            ):
                return
            if unfinished_job and not db.execute(
                "SELECT 1 FROM jobs JOIN runs ON runs.job_id=jobs.job_id "
                "WHERE runs.run_id=? AND jobs.job_id=? AND jobs.status IN ('queued','running')",
                (run_id, unfinished_job),
            ).fetchone():
                return
            changed = db.execute(
                "UPDATE runs SET status='interrupted',error=?,updated=? "
                "WHERE run_id=? AND status=?",
                (encode(error), time.time(), run_id, expected_status),
            ).rowcount
            if changed:
                db.execute(
                    "UPDATE steps SET status='interrupted',error=?,ended=? WHERE run_id=? "
                    "AND status IN ('pending','running')",
                    (encode(error), time.time(), run_id),
                )
                db.execute(
                    "UPDATE jobs SET status='interrupted',ended=? WHERE job_id="
                    "(SELECT job_id FROM runs WHERE run_id=?)",
                    (time.time(), run_id),
                )
        if changed:
            self.store.event(run_id, "interrupted", error)

    def _launch_scheduler(self, run_id):
        queued = self.store.get_run(run_id)
        if queued["status"] != "queued":
            return
        try:
            self._start_scheduler()
        except OSError as exc:
            from .worker import process_alive

            with self.store.connect() as db:
                owner = db.execute("SELECT * FROM scheduler WHERE singleton=1").fetchone()
            if not (owner["identity"] and process_alive(owner["pid"], owner["identity"])):
                self._interrupt_run(
                    run_id,
                    "queued",
                    {"code": "SCHEDULER_UNAVAILABLE", "message": str(exc)},
                    unfinished_job=queued["job_id"],
                )

    def _start_scheduler(self):
        log_dir = self.root / "logs"
        log_dir.mkdir(exist_ok=True)
        environment = os.environ.copy()
        if sys.flags.isolated:
            environment.pop("PYTHONPATH", None)
        else:
            package_root = str(Path(__file__).resolve().parents[2])
            environment["PYTHONPATH"] = (
                package_root + os.pathsep + environment.get("PYTHONPATH", "")
            )
        options = ["-I"] if sys.flags.isolated else []
        with (log_dir / "scheduler.log").open("ab") as log:
            subprocess.Popen(
                [
                    sys.executable,
                    *options,
                    "-m",
                    "autotem.harness.worker",
                    "schedule",
                    str(self.root),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                env=environment,
                cwd=self.root,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                start_new_session=os.name != "nt",
            )

    def _wait(self, run_id):
        while True:
            state = self.get_run_status(run_id)
            if state["status"] in TERMINAL:
                return state
            time.sleep(0.1)

    def get_run_status(self, run_id: str) -> dict:
        from .worker import process_alive

        run = self.store.get_run(run_id)
        if (
            run["status"] == "preparing"
            and run["worker_pid"]
            and not process_alive(run["worker_pid"], run["worker_identity"])
        ):
            job_id = run["job_id"]
            error = {
                "code": "PREPARATION_INTERRUPTED",
                "message": "The submitting process exited before the job was queued. "
                "Use a new idempotency key to submit a replacement.",
            }
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                if owns_preparation(db, run_id, job_id):
                    stamp = time.time()
                    _finish_unclaimed(
                        db, run_id, job_id, "interrupted", "interrupted", error, stamp
                    )
                    db.execute(
                        "INSERT INTO events(run_id,kind,payload,created) "
                        "VALUES(?,'interrupted',?,?)",
                        (run_id, encode(error), stamp),
                    )
            run = self.store.get_run(run_id)
        if (
            run["status"] == "interrupted"
            and (run["error"] or {}).get("code") == "PREPARATION_INTERRUPTED"
        ):
            # No producer can publish this staging file. A locked file can be retried
            # on the next query without undoing the already committed job diagnosis.
            directory = self.store.run_dir(run_id)
            if directory.resolve() == directory and re.fullmatch(
                r"[0-9a-f]{32}", run["job_id"] or ""
            ):
                try:
                    (directory / f".snapshot-{run['job_id']}.tmp").unlink(missing_ok=True)
                except PermissionError:
                    pass
        with self.store.connect() as db:
            scheduler = db.execute("SELECT * FROM scheduler WHERE singleton=1").fetchone()
        scheduler_healthy = bool(
            scheduler["identity"] and process_alive(scheduler["pid"], scheduler["identity"])
        )
        if (
            run["status"] == "queued"
            and time.time() - run["updated"] > 30
            and not scheduler_healthy
        ):
            self._interrupt_run(
                run_id,
                "queued",
                {
                    "code": "SCHEDULER_UNAVAILABLE",
                    "message": "Queue owner failed to start or exited",
                },
                unfinished_job=run["job_id"],
            )
            run = self.store.get_run(run_id)
        if (
            run["status"] == "running"
            and not process_alive(run["worker_pid"], run["worker_identity"])
            and not scheduler_healthy
        ):
            # The scheduler normally records this. A stopped scheduler must not leave false running.
            self._interrupt_run(
                run_id,
                "running",
                {
                    "code": "WORKER_INTERRUPTED",
                    "message": "Scientific worker exited without completion",
                },
                unfinished_job=run["job_id"],
            )
            run = self.store.get_run(run_id)
        if (run["status"] in TERMINAL and run["job_id"] and not scheduler_healthy
                and time.time() - run["updated"] > 30):
            with self.store.connect() as db:
                pending = db.execute(
                    "SELECT 1 FROM jobs WHERE job_id=? AND status IN ('queued','running')",
                    (run["job_id"],),
                ).fetchone()
            if pending and not process_alive(run["worker_pid"], run["worker_identity"]):
                self._interrupt_run(
                    run_id, run["status"],
                    {"code": "JOB_FINALIZATION_INTERRUPTED", "recorded_run_status": run["status"]},
                    unfinished_job=run["job_id"],
                )
        with self.store.connect() as db:
            # Select run, steps, events and jobs from one WAL snapshot. Otherwise an old
            # execution_seconds and a newly settled job can appear together in a report.
            db.execute("BEGIN")
            run = self.store.get_run(run_id, db)
            steps = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM steps WHERE run_id=? ORDER BY rowid", (run_id,)
                ).fetchall()
            ]
            events = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM events WHERE run_id=? ORDER BY event_id", (run_id,)
                ).fetchall()
            ]
            jobs = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM jobs WHERE run_id=? ORDER BY created", (run_id,)
                ).fetchall()
            ]
        for step in steps:
            for key in ("result", "error"):
                step[key] = json.loads(step[key]) if step[key] else None
        for event in events:
            event["payload"] = json.loads(event["payload"])
        public_status = run["status"]
        current_job = next((job for job in jobs if job["job_id"] == run["job_id"]), None)
        if (public_status in TERMINAL and current_job
                and current_job["status"] in {"queued", "running"}):
            public_status = "finalizing"
        acceptance = {}
        for role in ("validation", "benchmark"):
            artifact_id = self.store.role(run_id, role)
            if artifact_id:
                acceptance[role] = self.store.load_json(artifact_id).get("acceptance", {})
        gate_failed = any(value.get("all_passed") is False for value in acceptance.values())
        input_requests, input_responses = self.store.input_records(run)
        latest_child = next(
            (
                r["child_run_id"]
                for r in reversed(input_responses)
                if r["parent_run_id"] == run_id and r["child_run_id"]
            ),
            None,
        )
        next_action = None
        if public_status == "completed":
            next_action = "export_report"
        elif public_status == "preparing" and run["worker_pid"] is None:
            next_action = "check_submitting_client_or_cancel_run"
        elif public_status == "needs_input":
            next_action = "provide_required_input_in_child_run"
            if latest_child:
                next_action = "get_child_run_status"
            elif any(f["name"] == "input_path" for item in input_requests for f in item["fields"]):
                next_action = "inspect_replacement_input"
        return {
            "envelope_version": "1.0.0",
            "run_id": run_id,
            "job_id": run["job_id"],
            "jobs": jobs,
            "execution_seconds": run["run_seconds_used"] or 0,
            "parent_run_id": run["parent_run_id"],
            "status": public_status,
            "error": run["error"],
            "steps": steps,
            "events": events,
            "execution_mode": run["execution_mode"],
            "acceptance": acceptance,
            "acceptance_status": "blocked" if gate_failed else "not_evaluated",
            "parameter_diff": run["parameter_diff"],
            "input_requests": input_requests,
            "input_responses": input_responses,
            "latest_response_child_run_id": latest_child,
            "next_action": next_action,
            "scientific_accuracy": "not_evaluated",
        }

    def cancel_run(self, run_id: str) -> dict:
        observed = self.store.get_run(run_id)
        if observed["status"] not in TERMINAL:
            # A run can gain another work package while this caller waits for the writer.
            # Bind to the observed job, then decide ownership from the committed claim.
            job_id = observed["job_id"]
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                current = db.execute(
                    "SELECT r.status,r.cancel_requested,j.status AS job_status "
                    "FROM runs r LEFT JOIN jobs j "
                    "ON j.job_id=r.job_id AND j.run_id=r.run_id "
                    "WHERE r.run_id=? AND r.job_id IS ?",
                    (run_id, job_id),
                ).fetchone()
                if current and current["status"] not in TERMINAL:
                    stamp = time.time()
                    if not current["cancel_requested"]:
                        db.execute(
                            "UPDATE runs SET cancel_requested=1,updated=? WHERE run_id=?",
                            (stamp, run_id),
                        )
                        db.execute(
                            "INSERT INTO events(run_id,kind,payload,created) "
                            "VALUES(?,'cancel_requested',?,?)",
                            (run_id, encode({"job_id": job_id}), stamp),
                        )
                    if current["status"] in {"queued", "preparing"} and (
                        current["job_status"] == "queued" or job_id is None
                    ):
                        _finish_unclaimed(
                            db,
                            run_id,
                            job_id,
                            "cancelled",
                            "cancelled",
                            {"code": "CANCELLED"},
                            stamp,
                        )
        return self.get_run_status(run_id)

    def list_artifacts(self, run_id: str) -> list[dict]:
        return self.store.artifacts(run_id)

    def read_artifact(self, artifact_id: str, offset: int = 0, limit: int = 100) -> dict:
        if (
            isinstance(offset, bool)
            or isinstance(limit, bool)
            or offset < 0
            or not 1 <= limit <= 1000
        ):
            raise HarnessError("INVALID_READ_RANGE", "offset >= 0, limit 1..1000 required")
        artifact = self.store._check(artifact_id)
        result = {"artifact": artifact, "preview_limit": limit}
        if artifact["kind"] == "array":
            array = self.store.load_array(artifact_id)
            flat = array.ravel()[offset : offset + limit]
            result["values"] = [
                None if isinstance(v, float) and not np.isfinite(v) else v for v in flat.tolist()
            ]
            result["is_preview"] = True
            if np.iscomplexobj(array):
                result["values"] = [{"real": float(v.real), "imag": float(v.imag)} for v in flat]
            if (
                artifact["metadata"].get("validity_role") == "gpa.quality_mask"
                and artifact["role"] != "gpa.quality_mask"
            ):
                masks = [
                    a
                    for a in self.store.artifacts(artifact["run_id"], step_id=artifact["step_id"])
                    if a["role"] == "gpa.quality_mask"
                    and a["kind"] == "array"
                    and a["metadata"].get("array_sha256")
                    == artifact["metadata"].get("validity_sha256")
                ]
                if not masks:
                    raise HarnessError(
                        "MISSING_ARTIFACT",
                        "The GPA field has no matching quality mask in this step",
                    )
                mask_id = masks[0]["artifact_id"]
                valid = self.store.load_array(mask_id)
                if valid.dtype != np.bool_ or valid.shape != array.shape:
                    raise HarnessError(
                        "ARRAY_VALIDATION_FAILED", "GPA field and quality mask do not match"
                    )
                result["validity_mask_artifact_id"] = mask_id
                result["validity_interpretation"] = GPA_MASK_INTERPRETATION
                result["values"] = [
                    value if ok else None
                    for value, ok in zip(
                        result["values"], valid.ravel()[offset : offset + limit], strict=True
                    )
                ]
        elif artifact["kind"] == "json":
            if artifact["size"] <= 256_000:
                result["value"] = self.store.load_json(artifact_id)
            else:
                result["preview_limitation"] = (
                    "JSON exceeds 256 kB; use the verified local reference"
                )
            if artifact["role"] == "gpa":
                reference = result.get("value", {}).get("provenance", {}).get("quality_mask")
                step_artifacts = self.store.artifacts(
                    artifact["run_id"], step_id=artifact["step_id"]
                )
                mask = gpa_result_mask(reference, step_artifacts)
                result["validity_mask_artifact_id"] = mask["artifact_id"]
                result["validity_interpretation"] = mask["interpretation"]
        elif artifact["kind"] == "table":
            with Path(artifact["path"]).open(encoding="utf-8-sig", newline="") as stream:
                import itertools

                result["rows"] = list(
                    itertools.islice(csv.DictReader(stream), offset, offset + limit)
                )
            result["is_preview"] = True
        else:
            result["preview_limitation"] = "Open the static file using its verified local path"
        return result

    def export_report(self, run_id: str, formats: list[str] | None = None) -> dict:
        from autotem.reporting.harness_evidence import claim_html, delivery_index

        formats = ["json", "html"] if formats is None else formats
        if not set(formats) <= {"json", "html", "pdf"} or not formats:
            raise HarnessError("UNSUPPORTED_REPORT_FORMAT", "Supported: json/html/pdf")
        state = self.get_run_status(run_id)
        if state["status"] not in TERMINAL:
            raise HarnessError("RUN_NOT_FINAL", "Wait for execution to finish before exporting")
        if state["job_id"] and not any(
            job["job_id"] == state["job_id"] and job["status"] == state["status"]
            and job["ended"] is not None for job in state["jobs"]
        ):
            raise HarnessError("RUN_NOT_FINAL", "The current job has no matching final outcome")
        previous_reports = {"report_manifest", "report_html", "report_pdf"}
        evidence = [
            a for a in self.list_artifacts(run_id)
            if not (a["step_id"] == "_report" and a["role"] in previous_reports)
        ]
        for artifact in evidence:
            self.store._check(artifact["artifact_id"])
        run = self.store.get_run(run_id)
        if run["job_id"] != state["job_id"] or run["status"] != state["status"]:
            raise HarnessError("RUN_NOT_FINAL", "Wait for the newly submitted work package")
        delivery = delivery_index(run, state, evidence, self.store.load_json)
        latest = {
            role: values[-1]["artifact_id"] for role, values in delivery["results_by_role"].items()
        }
        counts = delivery["completeness"]
        manifest = {
            "envelope_version": "1.0.0",
            "run": state,
            "report_type": "analysis"
            if (state["status"] == "completed" and state["acceptance_status"] != "blocked"
                and counts["required_complete"] and counts["delivered_steps"])
            else "diagnostic",
            "request": run["request"],
            "input_response_history": self.store.input_response_history(run_id),
            "artifacts": evidence,
            "scientific_accuracy": "not_evaluated",
            "result_index": latest,
            "primary_results": {
                role: latest.get(role)
                for role in ("diffraction", "refined_atoms", "gpa", "atom_strain")
            },
            **delivery,
        }
        generated = []
        if "json" in formats:
            generated.append(self.store.write_json(run_id, "_report", "report_manifest", manifest))
        if "html" in formats:
            figures = "".join(
                f'<figure><img width="600" style="max-width:100%" '
                f'src="{html.escape(Path(a["path"]).name)}" alt="{html.escape(a["role"])}">'
                f"<figcaption>{html.escape(a['role'])}</figcaption></figure>"
                for a in evidence
                if a["kind"] == "image"
            )
            summaries = ""
            for role in ("atom_distances", "diffraction", "atomic_intensity", "gpa", "atom_strain"):
                artifact_id = latest.get(role)
                if artifact_id:
                    value = self.store.load_json(artifact_id)
                    summaries += (
                        f"<h2>{html.escape(role)}</h2><pre>"
                        + html.escape(json.dumps(value, ensure_ascii=False, indent=2)[:50000])
                        + "</pre><p>Bounded summary; complete JSON is linked "
                        "in the evidence table.</p>"
                    )
            rows = "".join(
                f"<tr><td>{html.escape(a['role'])}</td><td>{a['kind']}</td>"
                f"<td>{html.escape(a['operation_status'])}</td>"
                f'<td><a href="{html.escape(Path(a["path"]).name)}">'
                f"{a['artifact_id']}</a></td><td>{a['sha256']}</td></tr>"
                for a in evidence
            )
            page = (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>AutoTEM evidence report</title>"
                "</head><body><h1>AutoTEM "
                + manifest["report_type"]
                + " report</h1><p>"
                + html.escape(state["status"])
                + " · scientific accuracy: not_evaluated</p>"
                + claim_html(manifest, evidence)
                + "<pre>"
                + html.escape(json.dumps(state, ensure_ascii=False, indent=2))
                + "</pre><table><tr><th>Role</th><th>Type</th><th>Step status</th>"
                + "<th>Evidence</th><th>SHA256</th></tr>"
                + rows
                + "</table>"
                + summaries
                + figures
                + "</body></html>"
            )
            # Reports live in artifacts; use ../artifacts links from that directory.
            page = page.replace('href="artifacts/', 'href="')
            generated.append(
                self.store._publish(
                    run_id,
                    "_report",
                    "report_html",
                    "html",
                    ".html",
                    page.encode("utf-8"),
                    {"report_type": manifest["report_type"]},
                )
            )
        if "pdf" in formats:
            # Existing matplotlib dependency suffices; Unicode text is retained in JSON/HTML.
            import io

            from matplotlib.backends.backend_pdf import PdfPages
            from matplotlib.figure import Figure

            stream = io.BytesIO()
            with PdfPages(stream) as pdf:
                chunks = [evidence[i : i + 20] for i in range(0, len(evidence), 20)] or [[]]
                for chunk in chunks:
                    figure = Figure(figsize=(8.27, 11.69))
                    text = (
                        f"AutoTEM {manifest['report_type']} report\nRun: {run_id}\n"
                        f"Status: {state['status']}\n"
                    )
                    text += "Scientific accuracy: not_evaluated\n\n"
                    text += (
                        f"Required results delivered: {counts['delivered_required_steps']}"
                        f"/{counts['required_steps']}\n"
                    )
                    primary = manifest["strain_results"]["primary"]
                    if primary:
                        text += f"Primary strain: {primary['role']} ({primary['step_id']})\n"
                    text += "\n".join(
                        f"{a['role']} [{a['kind']}]\n{a['artifact_id']}" for a in chunk
                    )
                    figure.text(0.08, 0.93, text, va="top", fontsize=9)
                    pdf.savefig(figure)
            generated.append(
                self.store._publish(
                    run_id,
                    "_report",
                    "report_pdf",
                    "other",
                    ".pdf",
                    stream.getvalue(),
                    {"report_type": manifest["report_type"]},
                )
            )
        self.store.event(
            run_id,
            "report_exported",
            {"job_id": state["job_id"], "formats": formats,
             "artifact_ids": [a["artifact_id"] for a in generated]},
        )
        return {
            "run_id": run_id, "job_id": state["job_id"], "report_type": manifest["report_type"],
            "artifacts": generated,
            "completeness": manifest["completeness"],
            "primary_strain": manifest["strain_results"]["primary"],
        }
