"""SQLite metadata and immutable, hash-checked run artifacts."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import closing, contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from .contracts import HarnessError

STORE_SCHEMA_VERSION = 1
ARTIFACT_ROWS = (
    "SELECT a.*,s.status AS operation_status FROM artifacts a LEFT JOIN steps s "
    "ON a.run_id=s.run_id AND a.step_id=s.step_id WHERE a.committed=1 "
)


def json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return json_value(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        raise HarnessError("ARRAY_REQUIRES_ARTIFACT", "Store arrays separately with emit_array")
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def encode(value: Any) -> str:
    return json.dumps(json_value(value), ensure_ascii=False, allow_nan=False, sort_keys=True)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(array: np.ndarray) -> str:
    """Hash the original dtype's C-order bytes without materializing the entire array."""
    digest = hashlib.sha256()
    if array.nbytes:
        if array.flags.c_contiguous:
            digest.update(array.reshape(-1).view(np.uint8))
        else:
            # Structured element copies can omit their padding. Traverse complete records
            # as fixed-width bytes so even strided or Fortran inputs preserve those bits.
            if array.dtype.fields is not None:
                array = array.view(np.dtype((np.void, array.dtype.itemsize)))
            elements = max(1, (1024 * 1024) // array.dtype.itemsize)
            for chunk in np.nditer(
                array, flags=["buffered", "external_loop"], order="C", buffersize=elements
            ):
                digest.update(chunk.tobytes())
    return digest.hexdigest()


def _aligned_records(dtype):
    """Record aligned struct nodes; the NPY header already stores every offset and size."""
    if dtype.subdtype is not None:
        return _aligned_records(dtype.subdtype[0])
    paths = [()] if dtype.isalignedstruct else []
    for name in dtype.names or ():
        paths.extend((name, *p) for p in _aligned_records(dtype.fields[name][0]))
    return paths


def _record_dtype(dtype, description, alignments, path=()):
    """Restore alignment flags using the exact layout read from the NPY header."""
    if dtype.subdtype is not None:
        base, shape = dtype.subdtype
        if isinstance(description, tuple):  # subarray format in an explicit-offset dtype
            description = description[0]
        return np.dtype((_record_dtype(base, description, alignments, path), shape))
    if dtype.names is None:
        return dtype
    descriptions = (
        description["formats"] if isinstance(description, dict) else [f[1] for f in description]
    )
    fields = [dtype.fields[n] for n in dtype.names]
    formats = [
        _record_dtype(f[0], d, alignments, (*path, n))
        for n, f, d in zip(dtype.names, fields, descriptions, strict=True)
    ]
    offsets = [f[1] for f in fields]
    if alignments is not None:
        aligned = path in alignments
    elif isinstance(description, dict):
        aligned = description.get("aligned", False)
    else:
        # A legacy nested record could print as a list with implicit alignment. Its
        # header disambiguates this from a packed record whenever the offsets differ.
        packed = np.dtype(list(zip(dtype.names, formats, strict=True)))
        packed_offsets = [packed.fields[n][1] for n in dtype.names]
        aligned = packed.itemsize != dtype.itemsize or packed_offsets != offsets
    return np.dtype({
        "names": dtype.names, "formats": formats, "offsets": offsets, "itemsize": dtype.itemsize,
        "titles": [f[2] if len(f) > 2 else None for f in fields], "aligned": aligned,
    })


def _verified_npy(path, array, expected, role):
    """Verify the temp file's header and exact values, returning its complete file hash."""
    stored = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        fortran = array.flags.f_contiguous and not array.flags.c_contiguous
        if (
            stored.shape != array.shape
            or stored.dtype != array.dtype
            or (not stored.flags.c_contiguous) != fortran
            or stored.offset + stored.nbytes != path.stat().st_size
        ):
            raise HarnessError("ARRAY_VALIDATION_FAILED", role)
        if fortran:
            observed = _array_hash(stored)
            digest = file_hash(path)
        else:
            # One pass over the actual payload feeds both hashes for ordinary C-order files.
            with path.open("rb") as stream:
                whole = hashlib.sha256(stream.read(stored.offset))
                payload = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    payload.update(block)
                    whole.update(block)
            observed, digest = payload.hexdigest(), whole.hexdigest()
        if observed != expected:
            raise HarnessError("ARRAY_VALIDATION_FAILED", role)
        return digest
    finally:
        # Release the read-only mapping before Windows replaces or removes the file.
        del stored


def owns_preparation(db, run_id, job_id):
    """Check the initial submission's claim under its caller's write transaction."""
    return (
        db.execute(
            "SELECT 1 FROM runs r JOIN jobs j ON j.job_id=r.job_id AND j.run_id=r.run_id "
            "WHERE r.run_id=? AND r.job_id=? AND r.status='preparing' "
            "AND coalesce(r.cancel_requested,0)=0 AND j.status='queued'",
            (run_id, job_id),
        ).fetchone()
        is not None
    )


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "harness.sqlite3"
        with self.connect(initializing=True) as db:
            if db.execute("PRAGMA user_version").fetchone()[0] == STORE_SCHEMA_VERSION:
                return
            # Serialize first open / upgrade across processes. executescript() implicitly
            # commits an open transaction, so execute each fixed schema statement separately.
            db.execute("BEGIN IMMEDIATE")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version == STORE_SCHEMA_VERSION:
                return
            if version != 0:
                raise HarnessError("STORE_SCHEMA_UNSUPPORTED", "Use a compatible Harness runtime")
            tables = {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if tables:
                if not {"runs", "steps", "artifacts", "scheduler"} <= tables:
                    raise HarnessError(
                        "STORE_SCHEMA_UNKNOWN", "Database is not a recognized legacy Harness store"
                    )
                active = db.execute(
                    "SELECT 1 FROM runs WHERE status IN ('preparing','queued','running') LIMIT 1"
                ).fetchone()
                if not active and "jobs" in tables:
                    active = db.execute(
                        "SELECT 1 FROM jobs WHERE status IN ('queued','running') LIMIT 1"
                    ).fetchone()
                if active:
                    raise HarnessError(
                        "STORE_MIGRATION_BUSY", "Finish or recover legacy jobs before upgrading"
                    )
                self._backup_legacy_schema()
            schema = """
                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY, parent_run_id TEXT, idempotency_key TEXT UNIQUE,
                  request_hash TEXT NOT NULL, request TEXT NOT NULL, status TEXT NOT NULL,
                  created REAL NOT NULL, updated REAL NOT NULL, input_path TEXT,
                  source_hash TEXT, error TEXT, cancel_requested INTEGER DEFAULT 0,
                  execution_mode TEXT, worker_pid INTEGER, current_step TEXT,
                  step_started REAL, user_request TEXT, parameter_diff TEXT);
                CREATE TABLE IF NOT EXISTS steps (
                  run_id TEXT NOT NULL, step_id TEXT NOT NULL, operation_id TEXT NOT NULL,
                  status TEXT NOT NULL, result TEXT, error TEXT, started REAL, ended REAL,
                  PRIMARY KEY(run_id, step_id));
                CREATE TABLE IF NOT EXISTS artifacts (
                  artifact_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, step_id TEXT NOT NULL,
                  role TEXT NOT NULL, kind TEXT NOT NULL, path TEXT NOT NULL,
                  sha256 TEXT NOT NULL, size INTEGER NOT NULL, metadata TEXT NOT NULL,
                  created REAL NOT NULL, committed INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS roles (
                  run_id TEXT NOT NULL, role TEXT NOT NULL, artifact_id TEXT NOT NULL,
                  PRIMARY KEY(run_id, role));
                CREATE TABLE IF NOT EXISTS events (
                  event_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                  kind TEXT NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS scheduler (
                  singleton INTEGER PRIMARY KEY CHECK(singleton=1), pid INTEGER, heartbeat REAL);
                CREATE TABLE IF NOT EXISTS submissions (
                  idempotency_key TEXT PRIMARY KEY, request_hash TEXT NOT NULL,
                  run_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                  job_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, created REAL NOT NULL,
                  started REAL, ended REAL, status TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS input_requests (
                  request_id TEXT PRIMARY KEY, schema_version TEXT NOT NULL, run_id TEXT UNIQUE,
                  step_id TEXT NOT NULL, operation_id TEXT NOT NULL, operation_version TEXT,
                  fields TEXT NOT NULL, reason TEXT NOT NULL, evidence_artifact_ids TEXT NOT NULL,
                  created REAL NOT NULL, status TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS input_responses (
                  response_id TEXT PRIMARY KEY, schema_version TEXT NOT NULL,
                  request_id TEXT NOT NULL,
                  parent_run_id TEXT NOT NULL, child_run_id TEXT UNIQUE, response TEXT,
                  validation TEXT NOT NULL, adopted_values TEXT NOT NULL, created REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS input_response_parent ON input_responses(parent_run_id);
                INSERT OR IGNORE INTO scheduler(singleton,pid,heartbeat) VALUES(1,NULL,0);
            """
            for statement in schema.split(";"):
                if statement.strip():
                    db.execute(statement)
            columns = {row[1] for row in db.execute("PRAGMA table_info(runs)")}
            for name, sql_type in (
                ("worker_identity", "TEXT"),
                ("job_id", "TEXT"),
                ("run_seconds_used", "REAL DEFAULT 0"),
                ("active_started", "REAL"),
                ("submitted_hash", "TEXT"),
            ):
                if name not in columns:
                    db.execute(f"ALTER TABLE runs ADD COLUMN {name} {sql_type}")
            columns = {row[1] for row in db.execute("PRAGMA table_info(scheduler)")}
            if "identity" not in columns:
                db.execute("ALTER TABLE scheduler ADD COLUMN identity TEXT")
            db.execute(f"PRAGMA user_version={STORE_SCHEMA_VERSION}")
            db.commit()
            if db.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
                db.execute("PRAGMA journal_mode=WAL")

    def _backup_legacy_schema(self):
        """Capture committed WAL pages while the migration connection excludes other writers."""
        directory = self.root / "schema-backups"
        if directory.resolve() != directory:
            raise HarnessError("STORE_BACKUP_PATH_INVALID", "Store backups must stay in this root")
        directory.mkdir(exist_ok=True)
        temp = directory / f".{uuid.uuid4().hex}.tmp"
        temp.touch(exist_ok=False)
        try:
            uri = self.db_path.as_uri() + "?mode=ro"
            with (
                closing(sqlite3.connect(uri, uri=True)) as source,
                closing(sqlite3.connect(temp)) as target,
            ):
                source.backup(target)
                target.execute("PRAGMA journal_mode=DELETE")
                if target.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise HarnessError(
                        "STORE_BACKUP_INVALID", "Legacy store backup failed integrity check"
                    )
            checksum = file_hash(temp)
            backup = directory / f"harness-v0-{checksum}.sqlite3"
            if backup.exists():
                if backup.resolve() != backup or file_hash(backup) != checksum:
                    raise HarnessError("STORE_BACKUP_INVALID", "Existing backup content changed")
            else:
                os.replace(temp, backup)
        finally:
            temp.unlink(missing_ok=True)

    @contextmanager
    def connect(self, *, initializing=False):
        db = sqlite3.connect(self.db_path, timeout=30)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA busy_timeout=30000")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version != STORE_SCHEMA_VERSION and not (initializing and version == 0):
                raise HarnessError(
                    "STORE_SCHEMA_UNSUPPORTED",
                    "Use a compatible Harness runtime",
                    {"store_version": version, "supported_version": STORE_SCHEMA_VERSION},
                )
            with db:
                yield db
        finally:
            db.close()

    def run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise HarnessError("INVALID_RUN_ID", "Invalid run ID")
        return self.root / "runs" / run_id

    def get_run(self, run_id: str, connection=None) -> dict:
        self.run_dir(run_id)
        with self.connect() if connection is None else nullcontext(connection) as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise HarnessError("RUN_NOT_FOUND", run_id)
        result = dict(row)
        for key in ("request", "error", "parameter_diff"):
            if result[key] is not None:
                result[key] = json.loads(result[key])
        return result

    def update_run(self, run_id: str, **fields):
        allowed = {
            "status",
            "input_path",
            "source_hash",
            "error",
            "cancel_requested",
            "worker_pid",
            "current_step",
            "step_started",
            "worker_identity",
            "job_id",
            "run_seconds_used",
            "active_started",
        }
        if not fields or not set(fields) <= allowed:
            raise ValueError("Invalid run update")
        if "error" in fields and fields["error"] is not None:
            fields["error"] = encode(fields["error"])
        fields["updated"] = time.time()
        with self.connect() as db:
            db.execute(
                "UPDATE runs SET " + ",".join(f"{key}=?" for key in fields) + " WHERE run_id=?",
                (*fields.values(), run_id),
            )

    def event(self, run_id: str, kind: str, payload: Any):
        with self.connect() as db:
            db.execute(
                "INSERT INTO events(run_id,kind,payload,created) VALUES(?,?,?,?)",
                (run_id, kind, encode(payload), time.time()),
            )

    def record_input_request(
        self, run, step, error, consumed=(), *, registry=None, historical=False
    ):
        from .input_requests import RECORD_VERSION, describe_fields
        from .registry import default_registry

        run_id = run["run_id"]
        registry = registry or default_registry()
        raw_fields = error["details"]["fields"]
        modality = run["request"]["modality"]
        if any(f["name"] == "physical_calibration" for f in raw_fields):
            if dataset_id := self.role(run_id, "dataset"):
                modality = self.load_json(dataset_id)["modality"]
        if step["operation_id"] in registry.operations:
            fields = describe_fields(raw_fields, step["operation_id"], modality, registry)
        else:
            # Reading an old run must not load missing third-party code. Keep its original
            # question available, and require reinspection before accepting an untyped reply.
            fields = [{**field, "answer_options": [], "answer_schema": {}} for field in raw_fields]
        request_id, now = uuid.uuid4().hex, time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT request_id FROM input_requests WHERE run_id=?", (run_id,)
            ).fetchone()
            if existing:
                return existing[0]
            rows = db.execute(
                "SELECT artifact_id,role,step_id,metadata FROM artifacts WHERE run_id=? "
                "AND committed=1 ORDER BY created",
                (run_id,),
            ).fetchall()
            raw = db.execute(
                "SELECT a.metadata FROM roles r JOIN artifacts a ON a.artifact_id=r.artifact_id "
                "WHERE r.run_id=? AND r.role='raw'",
                (run_id,),
            ).fetchone()
            evidence = set(consumed)
            for row in rows:
                if row["step_id"] == step["id"] or row["role"] in {
                    "raw",
                    "raw_preview",
                    "fft_preview",
                    "peak_table",
                    "g_vector_table",
                }:
                    evidence.add(row["artifact_id"])
            for field in fields:
                evidence.update(field.get("evidence", {}).values())
                if field["name"] == "reference_region":
                    mask = next(
                        (
                            row["artifact_id"]
                            for row in rows
                            if row["step_id"] == step["id"]
                            and row["role"] == "gpa_reference_quality"
                        ),
                        None,
                    )
                    if mask:
                        field.setdefault("evidence", {})["quality_mask_artifact_id"] = mask
                for name in {"roi", "reference_region", "reference_regions"} & field[
                    "answer_schema"
                ].keys():
                    if raw:
                        meta = json.loads(raw[0])
                        field["answer_schema"][name]["coordinates"] = {
                            "frame": "source_2d",
                            "order": "[y0,y1,x0,x1], exclusive upper bounds",
                            "origin_yx": meta.get("selection", {}).get("roi_origin_yx", [0, 0]),
                            "shape_yx": meta["shape"],
                        }
            ids = [row["artifact_id"] for row in rows if row["artifact_id"] in evidence]
            db.execute(
                "INSERT INTO input_requests VALUES(?,?,?,?,?,?,?,?,?,?,'open')",
                (
                    request_id,
                    RECORD_VERSION,
                    run_id,
                    step["id"],
                    step["operation_id"],
                    step.get("operation_version"),
                    encode(fields),
                    error["message"],
                    encode(ids),
                    run["updated"] if historical else now,
                ),
            )
            if not historical:
                error["details"]["request_id"] = request_id
                db.execute(
                    "UPDATE steps SET status='needs_input',error=?,ended=? "
                    "WHERE run_id=? AND step_id=?",
                    (encode(error), now, run_id, step["id"]),
                )
                db.execute(
                    "UPDATE steps SET status='blocked',error=? WHERE run_id=? AND status='pending'",
                    (encode({"code": "UPSTREAM_NOT_COMPLETED"}), run_id),
                )
                db.execute(
                    "UPDATE runs SET status='needs_input',error=?,updated=? WHERE run_id=?",
                    (encode(error), now, run_id),
                )
                db.execute(
                    "INSERT INTO events(run_id,kind,payload,created) VALUES(?,?,?,?)",
                    (
                        run_id,
                        "operation_failed",
                        encode({"step_id": step["id"], "error": error}),
                        now,
                    ),
                )
        return request_id

    def input_request(self, request_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM input_requests WHERE request_id=?", (request_id,)
            ).fetchone()
        if row is None:
            raise HarnessError("INPUT_REQUEST_NOT_FOUND", "Unknown input request")
        return {
            **dict(row),
            "fields": json.loads(row["fields"]),
            "evidence_artifact_ids": json.loads(row["evidence_artifact_ids"]),
        }

    def input_records(self, run):
        with self.connect() as db:
            request = db.execute(
                "SELECT request_id FROM input_requests WHERE run_id=?", (run["run_id"],)
            ).fetchone()
            # Upgrade a previous-format needs_input record from its persisted scientific error.
            # Do not rewrite its plan, events, artifacts or terminal state.
            missing = not request and run["status"] == "needs_input"
            legacy = (
                db.execute(
                    "SELECT step_id,error FROM steps WHERE run_id=? AND status='needs_input'",
                    (run["run_id"],),
                ).fetchone()
                if missing
                else None
            )
            responses = db.execute(
                "SELECT * FROM input_responses WHERE parent_run_id=? "
                "OR child_run_id=? ORDER BY created,response_id",
                (run["run_id"], run["run_id"]),
            ).fetchall()
        request_id = request[0] if request else None
        if legacy:
            step = next(s for s in run["request"]["steps"] if s["id"] == legacy["step_id"])
            request_id = self.record_input_request(
                run, step, json.loads(legacy["error"]), historical=True
            )
        requests = [self.input_request(request_id)] if request_id else []
        decoded = []
        for row in responses:
            values = dict(row)
            for key in ("response", "validation", "adopted_values"):
                values[key] = json.loads(row[key]) if row[key] is not None else None
            decoded.append(values)
        return requests, decoded

    def input_response_history(self, run_id):
        """Return accepted answers on this run's parent chain, in the order they were given."""
        history = []
        visited = set()
        with self.connect() as db:
            while run_id:
                if run_id in visited:
                    raise HarnessError("INVALID_RUN_HISTORY", "Cycle in report input history")
                visited.add(run_id)
                row = db.execute(
                    "SELECT r.parent_run_id, a.response_id, a.request_id, a.response, "
                    "a.adopted_values, q.operation_id, q.step_id, q.fields, "
                    "q.evidence_artifact_ids FROM runs r "
                    "LEFT JOIN input_responses a ON a.child_run_id=r.run_id "
                    "AND a.parent_run_id=r.parent_run_id "
                    "LEFT JOIN input_requests q ON q.request_id=a.request_id "
                    "AND q.run_id=r.parent_run_id "
                    "WHERE r.run_id=?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    break  # Older runs can reference a parent whose records are unavailable.
                if row["response_id"] and row["operation_id"]:
                    reply = json.loads(row["response"])
                    history.append(
                        {
                            "parent_run_id": row["parent_run_id"],
                            "child_run_id": run_id,
                            "request_id": row["request_id"],
                            "response_id": row["response_id"],
                            "operation_id": row["operation_id"],
                            "step_id": row["step_id"],
                            "requested_fields": json.loads(row["fields"]),
                            "request_evidence_artifact_ids": json.loads(
                                row["evidence_artifact_ids"]
                            ),
                            "source": reply["source"],
                            "basis": reply["basis"],
                            "answers": reply["answers"],
                            "cited_artifact_ids": reply.get("evidence_artifact_ids", []),
                            "adopted_values": json.loads(row["adopted_values"]),
                        }
                    )
                run_id = row["parent_run_id"]
        return history[::-1]

    @staticmethod
    def record_input_response(
        db, record, response, adopted, child_run_id, validation, *, retry_key=None
    ):
        from .input_requests import RECORD_VERSION

        response_id = uuid.uuid4().hex
        if child_run_id is None and retry_key:
            response_id = uuid.uuid5(
                uuid.NAMESPACE_URL, encode([record["request_id"], retry_key, response, validation])
            ).hex
        db.execute(
            "INSERT INTO input_responses VALUES(?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(response_id) DO NOTHING",
            (
                response_id,
                RECORD_VERSION,
                record["request_id"],
                record["run_id"],
                child_run_id,
                encode(response) if response is not None else None,
                encode(validation),
                encode(adopted),
                time.time(),
            ),
        )
        if child_run_id:
            db.execute(
                "UPDATE input_requests SET status='answered' WHERE request_id=?",
                (record["request_id"],),
            )

    def role(self, run_id: str, role: str) -> str | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT artifact_id FROM roles WHERE run_id=? AND role=?", (run_id, role)
            ).fetchone()
        return row[0] if row else None

    def artifact(self, artifact_id: str) -> dict:
        with self.connect() as db:
            row = db.execute(
                ARTIFACT_ROWS + "AND a.artifact_id=?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            raise HarnessError("ARTIFACT_NOT_FOUND", artifact_id)
        return self._artifact_row(row)

    @staticmethod
    def _artifact_row(row):
        result = dict(row)
        result["metadata"] = json.loads(result["metadata"])
        result["operation_status"] = result["operation_status"] or "system_artifact"
        result["execution_usable"] = result["operation_status"] == "completed"
        return result

    def artifacts(self, run_id: str, *, step_id: str | None = None) -> list[dict]:
        self.get_run(run_id)
        with self.connect() as db:
            query = ARTIFACT_ROWS + "AND a.run_id=?"
            args = [run_id]
            if step_id is not None:
                query += " AND a.step_id=?"
                args.append(step_id)
            rows = db.execute(query + " ORDER BY a.created,a.artifact_id", args).fetchall()
        return [self._artifact_row(row) for row in rows]

    def _publish(
        self, run_id, step_id, role, kind, suffix, content: bytes, metadata, *, preparing_job=None
    ):
        with self._stage_artifact(
            run_id, step_id, role, kind, suffix, metadata, preparing_job=preparing_job
        ) as (stream, artifact_id):
            stream.write(content)
        return self.artifact(artifact_id)

    @contextmanager
    def _stage_artifact(
        self, run_id, step_id, role, kind, suffix, metadata, *,
        preparing_job=None, checksum=None,
    ):
        self.get_run(run_id)
        if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,120}", role):
            raise HarnessError("INVALID_ARTIFACT_ROLE", role)
        artifact_id = uuid.uuid4().hex
        directory = self.run_dir(run_id) / "artifacts"
        directory.mkdir(parents=True, exist_ok=True)
        # Roles belong in metadata. Short opaque names keep nested Windows runs usable.
        target = directory / f"{artifact_id}{suffix}"
        temporary = target.with_suffix(target.suffix + ".tmp")
        published = False
        stream = temporary.open("xb")
        try:
            with stream:
                yield stream, artifact_id
                stream.flush()
                os.fsync(stream.fileno())
            size = temporary.stat().st_size
            digest = file_hash(temporary) if checksum is None else checksum(temporary)
            with self.connect() as db:
                if preparing_job is not None:
                    db.execute("BEGIN IMMEDIATE")
                    if not owns_preparation(db, run_id, preparing_job):
                        raise HarnessError(
                            "PREPARATION_CLOSED", "Initial submission is no longer active"
                        )
                os.replace(temporary, target)
                published = True
                db.execute(
                    "INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?,1)",
                    (
                        artifact_id,
                        run_id,
                        step_id,
                        role,
                        kind,
                        str(target),
                        digest,
                        size,
                        encode(metadata or {}),
                        time.time(),
                    ),
                )
        except BaseException:
            # Unregistered or incomplete files never become evidence.
            temporary.unlink(missing_ok=True)
            if published:
                target.unlink(missing_ok=True)
            raise

    def commit_step(self, run_id: str, step_id: str, result: dict):
        with self.connect() as db:
            rows = db.execute(
                "SELECT role,artifact_id FROM artifacts WHERE run_id=? "
                "AND step_id=? AND committed=1 ORDER BY created",
                (run_id, step_id),
            ).fetchall()
            for row in rows:
                db.execute(
                    "INSERT INTO roles VALUES(?,?,?) ON CONFLICT(run_id,role) "
                    "DO UPDATE SET artifact_id=excluded.artifact_id",
                    (run_id, row["role"], row["artifact_id"]),
                )
            db.execute(
                "UPDATE steps SET status='completed',result=?,ended=? WHERE run_id=? AND step_id=?",
                (encode(result), time.time(), run_id, step_id),
            )
            db.execute(
                "INSERT INTO events(run_id,kind,payload,created) VALUES(?,?,?,?)",
                (
                    run_id,
                    "operation_completed",
                    encode({"step_id": step_id, **result}),
                    time.time(),
                ),
            )

    def _check(self, artifact_id):
        artifact = self.artifact(artifact_id)
        path = Path(artifact["path"])
        if not path.is_relative_to(self.run_dir(artifact["run_id"])):
            raise HarnessError("INVALID_ARTIFACT_PATH", "Artifact escaped run directory")
        if not path.is_file() or file_hash(path) != artifact["sha256"]:
            raise HarnessError("ARTIFACT_CHANGED", artifact_id)
        return artifact

    def write_array(self, run_id, step_id, role, value, metadata=None):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise HarnessError("UNSAFE_ARRAY", "Object arrays are not supported")
        expected = _array_hash(array)
        meta = {
            **(metadata or {}),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "array_sha256": expected,
            "preview_is_full_array": False,
        }
        if array.dtype.names is not None:
            meta["npy_aligned_records"] = _aligned_records(array.dtype)
        with self._stage_artifact(
            run_id, step_id, role, "array", ".npy", meta,
            checksum=lambda path: _verified_npy(path, array, expected, role),
        ) as (stream, aid):
            np.save(stream, array, allow_pickle=False)
        return self.artifact(aid)

    def write_json(self, run_id, step_id, role, value, metadata=None, *, preparing_job=None):
        content = encode(value).encode("utf-8")
        json.loads(content)
        return self._publish(
            run_id, step_id, role, "json", ".json", content, metadata, preparing_job=preparing_job
        )

    def write_table(self, run_id, step_id, role, rows, metadata=None):
        rows = json_value(rows)
        columns = list(dict.fromkeys(key for row in rows for key in row))
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        return self._publish(
            run_id,
            step_id,
            role,
            "table",
            ".csv",
            stream.getvalue().encode("utf-8-sig"),
            {**(metadata or {}), "rows": len(rows), "columns": columns},
        )

    def write_image(self, run_id, step_id, role, value, metadata=None):
        from PIL import Image

        if isinstance(value, (str, Path)):
            with Image.open(value) as image:
                image.load()
                image = image.copy()
        else:
            array = np.asarray(value)
            if array.dtype != np.uint8:
                finite = np.isfinite(array)
                lower, upper = (
                    (float(array[finite].min()), float(array[finite].max()))
                    if finite.any()
                    else (0, 0)
                )
                array = np.nan_to_num((array - lower) / (upper - lower or 1) * 255)
                array = np.clip(array, 0, 255).astype(np.uint8)
            image = Image.fromarray(array)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return self._publish(run_id, step_id, role, "image", ".png", stream.getvalue(), metadata)

    def load_array(self, artifact_id) -> np.ndarray:
        artifact = self._check(artifact_id)
        if artifact["kind"] != "array":
            raise HarnessError("ARTIFACT_TYPE_MISMATCH", "Expected an array")
        array = np.load(artifact["path"], allow_pickle=False)
        metadata = artifact["metadata"]
        stored_dtype = metadata["dtype"]
        dtype_matches = str(array.dtype) == stored_dtype
        flags = metadata.get("npy_aligned_records")
        if array.dtype.names is not None and (not dtype_matches or flags):
            try:
                paths = None if flags is None else {tuple(p) for p in flags}
                original = _record_dtype(array.dtype, ast.literal_eval(stored_dtype), paths)
                dtype_matches = original == array.dtype and str(original) == stored_dtype
                if paths is not None:
                    dtype_matches = dtype_matches and set(_aligned_records(original)) == paths
                if dtype_matches:
                    array = array.view(original)
            except (SyntaxError, ValueError, TypeError, KeyError, IndexError):
                dtype_matches = False
        if list(array.shape) != metadata["shape"] or not dtype_matches:
            raise HarnessError("ARRAY_VALIDATION_FAILED", artifact_id)
        return array

    def load_json(self, artifact_id):
        artifact = self._check(artifact_id)
        if artifact["kind"] != "json":
            raise HarnessError("ARTIFACT_TYPE_MISMATCH", "Expected JSON")
        return json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
