"""Qualify two native Windows release bundles in one fresh, project-local installation."""

import argparse
import hashlib
import json
import math
import os
import queue
import re
import runpy
import sqlite3
import subprocess
import sys
import threading
import time
import zipfile
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

FIRST_VERIFIER_SHA256 = "3a6aad4419cc1dd2c1192a77f0dd3aaf37f7d39d3fda11327fe2b80c28d2d86c"
PRE_COMPARISON_VERIFIER_SHA256 = "3dfd55bd1e6e64758647c6ac07863c59a41569cc4b06e8f6e3ad7ce101e06c22"
PRE_RECONNECT_VERIFIER_SHA256 = "35d8309cae9832dca057bc448af8f143766e7e4c3b04311969f015cd318a274c"
EARLIER_DRIVERS = {
    FIRST_VERIFIER_SHA256,
    PRE_COMPARISON_VERIFIER_SHA256,
    PRE_RECONNECT_VERIFIER_SHA256,
}
ENTRY_ERROR = "AutoTEM host setup required: The active runtime requires repair."
TOOL_NAMES = {
    "list_capabilities",
    "inspect_input",
    "validate_plan",
    "submit_analysis",
    "get_run_status",
    "cancel_run",
    "list_artifacts",
    "read_artifact",
    "export_report",
}

USER_CONFIG = (
    b"# User-owned preference for this synthetic project.\r\nproject_doc_max_bytes = 16400"
)
PHASES = ("before_upgrade", "after_upgrade", "after_rollback")
MILESTONES = (
    "previous_release_installed",
    "previous_bound_science_and_reports",
    "competing_installer_rejected_without_changes",
    "upgrade_and_old_history_verified",
    "stale_session_refused_without_job",
    "upgraded_bound_science_and_reports",
    "user_modified_host_asset_preserved",
    "previous_activation_and_both_histories_restored",
    "three_distinct_project_bound_jobs_and_immutable_reports",
)
STAGE_FILES = {
    "pointer_sha256": ".autotem/active-runtime.json",
    "stable_entry_sha256": ".autotem/host-entry.py",
    "host_config_sha256": ".codex/config.toml",
    "adapter_sha256": "autotem-host/codex/adapter.json",
}
MAINTENANCE = {
    "preview_previous": ("planned", None),
    "install_previous": ("host_setup_required", None),
    "concurrent_setup": ("blocked", "INSTALLATION_BUSY"),
    "upgrade_current": ("host_setup_required", None),
    "rollback_conflict": ("blocked", "HOST_FILE_CONFLICT"),
    "rollback_previous": ("rolled_back", None),
}
J01 = {
    "input.load",
    "quality.assess",
    "calibration.resolve",
    "preprocessing.apply",
    "diffraction.fft",
    "diffraction.center",
    "diffraction.peaks",
    "diffraction.metrics",
    "diffraction.index",
}


class GateError(ValueError):
    """A stable, path-free failure code safe to publish."""


def check(condition, code):
    if not condition:
        raise GateError(code)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8") as writer:
        json.dump(doc, writer, indent=2, ensure_ascii=True, allow_nan=False)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(pending, path)


def release(bundle):
    path = bundle / "release-manifest.json"
    meta = read(path)
    launcher = bundle / "AutoTEM-Setup.exe"
    records = [row for row in meta["artifacts"] if row["path"] == launcher.name]
    check(len(records) == 1 and sha(launcher) == records[0]["sha256"], "launcher_digest")
    archive = meta["source"].get("archive")
    if archive:
        check(
            archive == "source/autotem-source.zip"
            and sha(bundle / archive) == meta["source"]["archive_sha256"],
            "source_archive_digest",
        )
    digest = sha(path)
    return {
        "release_id": meta["release_id"],
        "manifest_sha256": digest,
        "runtime_id": "rt-" + digest[:20],
        "source_sha256": meta["source"]["archive_sha256"],
        "launcher_sha256": records[0]["sha256"],
        "package_version": meta["package_version"],
    }


def owned_file(project, path):
    path = Path(path)
    check(
        path.is_absolute() and path.is_file() and path.resolve().is_relative_to(project.resolve()),
        "project_file_outside_or_missing",
    )
    return path


def conflict_copy(project, response, owner, expected, edited):
    details = response["conflict_resolution"]
    name = "autotem-host/codex/adapter.json"
    matches = [row for row in details["files"] if row["path"] == name]
    check(len(matches) == 1 and re.fullmatch(r"install-[a-f0-9]{32}", owner), "conflict_file_owner")
    row = matches[0]
    directory = project / ".autotem/host-installations" / owner
    comparison = owned_file(project, project / row["comparison_path"])
    original_receipt = owned_file(project, project / row["receipt_path"])
    files = read(original_receipt)["files"]
    recorded = [entry for entry in files if entry["path"] == name]
    check(
        original_receipt == directory / "manifest.json"
        and comparison.resolve().is_relative_to(directory.resolve())
        and len(recorded) == 1
        and recorded[0]["after_sha256"] == expected
        and row["current_sha256"] == edited
        and row["current_state"] == "file"
        and row["condition"] == "recorded_file"
        and row["expected_sha256"] == row["comparison_sha256"] == sha(comparison) == expected,
        "conflict_comparison_not_bound_to_installed_file",
    )
    return comparison


def activation(project, expected):
    control = project / ".autotem"
    pointer_file = control / "active-runtime.json"
    pointer = read(pointer_file)
    check(
        pointer["manifest_sha256"] == expected["manifest_sha256"]
        and pointer["runtime_id"] == expected["runtime_id"],
        "active_pointer",
    )
    adapter = project / "autotem-host/codex/adapter.json"
    note = read(adapter)
    command = note["jsonl_command"]
    entry = control / "host-entry.py"
    check(len(command) == 4 and command[1:] == ["-I", str(entry), "stdio"], "stable_entry_route")
    check(owned_file(project, command[0]).parent.parent == control / "python", "private_base")
    rt = control / "runtimes" / expected["runtime_id"]
    check(
        owned_file(project, note["python_executable"]) == rt / "Scripts/python.exe",
        "adapter_runtime",
    )
    marker = read(rt / ".autotem-runtime.json")
    check(
        marker["state"] == "ready"
        and marker["manifest_sha256"] == expected["manifest_sha256"]
        and Path(marker["project"]) == project
        and Path(marker["runtime"]) == rt,
        "runtime_marker",
    )
    cfg = project / ".codex/config.toml"
    check(cfg.read_bytes().startswith(USER_CONFIG), "original_user_config")
    return command, {
        "runtime_id": pointer["runtime_id"],
        "manifest_sha256": pointer["manifest_sha256"],
        "pointer_sha256": sha(pointer_file),
        "stable_entry_sha256": sha(entry),
        "host_config_sha256": sha(cfg),
        "adapter_sha256": sha(adapter),
        "store_schema_version": pointer["store_schema_version"],
    }


def mcp_command(adapter):
    server = adapter["mcp"]["mcpServers"]["autotem"]
    command = [server["command"], *server["args"]]
    check(command == [*adapter["jsonl_command"][:-1], "mcp"], "installed_mcp_entry")
    return command


def observe_reconnect_failure(project, current):
    """In this synthetic project, make only the upgraded runtime fail the stable entry check."""
    local = project / ".upgrade-verifier/reconnect"
    local.mkdir()
    control = project / ".autotem"
    active = read(control / "active-runtime.json")
    marker = owned_file(
        project, control / "runtimes" / current["runtime_id"] / ".autotem-runtime.json"
    )
    original = marker.read_bytes()
    (local / "ready-marker.json").write_bytes(original)
    command = mcp_command(read(project / "autotem-host/codex/adapter.json"))
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "autotem-upgrade-qualification", "version": "1.0.0"},
        },
    }
    try:
        save(marker, {**read(marker), "state": "failed"})
        (local / "fault-marker.json").write_bytes(marker.read_bytes())
        proc = subprocess.run(
            command,
            input=(json.dumps(request) + "\n").encode(),
            cwd=project,
            capture_output=True,
            timeout=120,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    finally:
        with marker.with_suffix(".restore").open("xb") as backup:
            backup.write(original)
        os.replace(marker.with_suffix(".restore"), marker)
    (local / "entry.stdout").write_bytes(proc.stdout)
    (local / "entry.stderr").write_bytes(proc.stderr)
    save(
        local / "attempt.json", {"argv": command, "request": request, "exit_code": proc.returncode}
    )
    message = proc.stderr.decode("ascii").strip()
    check(
        proc.returncode == 2 and not proc.stdout and message == ENTRY_ERROR,
        "new_mcp_entry_did_not_fail",
    )
    report = local / "operator-report.json"
    save(
        report,
        {
            "schema_version": 1,
            "installation_id": active["installation_id"],
            "release_id": current["release_id"],
            "host": "codex",
            "stage": "connect",
            "error_code": "MCP_HANDSHAKE_FAILED",
            "message": message,
        },
    )
    return report


def reconnect_evidence(project, current):
    local = project / ".upgrade-verifier"
    folder = local / "reconnect"
    upgraded = read(local / "after_upgrade/pointer_sha256")
    previous = read(local / "before_upgrade/pointer_sha256")
    attempt = read(folder / "attempt.json")
    request = attempt["request"]
    check(
        attempt["argv"] == mcp_command(read(local / "after_upgrade/adapter_sha256"))
        and request["jsonrpc"] == "2.0"
        and request["method"] == "initialize"
        and request["id"] == 1
        and attempt["exit_code"] == 2
        and not (folder / "entry.stdout").read_bytes()
        and (folder / "entry.stderr").read_bytes().decode("ascii").strip() == ENTRY_ERROR,
        "new_mcp_process_failure_evidence",
    )
    runtime_marker = owned_file(
        project, project / ".autotem/runtimes" / current["runtime_id"] / ".autotem-runtime.json"
    )
    ready = read(folder / "ready-marker.json")
    check(
        ready["state"] == "ready"
        and ready["manifest_sha256"] == current["manifest_sha256"]
        and sha(runtime_marker) == sha(folder / "ready-marker.json")
        and read(folder / "fault-marker.json") == {**ready, "state": "failed"},
        "scoped_failure_control",
    )
    item = read(folder / "operator-report.json")
    installation = upgraded["installation_id"]
    check(
        item
        == {
            "schema_version": 1,
            "installation_id": installation,
            "release_id": current["release_id"],
            "host": "codex",
            "stage": "connect",
            "error_code": "MCP_HANDSHAKE_FAILED",
            "message": ENTRY_ERROR,
        },
        "original_connection_failure_report",
    )
    digest = hashlib.sha256(
        json.dumps(item, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    journal = project / ".autotem/installations" / installation / "transaction.json"
    base = {"from_installation_id": installation, "to_installation_id": previous["installation_id"]}
    success = {**base, "status": "rolled_back"}
    record = read(owned_file(project, journal))
    receipt = record["host_reconnect_recovery"]
    conflict = read(local / "maintenance/rollback_conflict.json")["error"]
    check(
        record["state"] == "rolled_back"
        and record["previous"] == previous
        and receipt["report"] == item
        and receipt["report_sha256"] == digest
        and receipt["rollback"] == success
        and receipt["last_blocked_error"] == conflict
        and conflict["code"] == "HOST_FILE_CONFLICT",
        "joint_recovery_journal",
    )
    for name in ("rollback_conflict", "rollback_previous", "reconnect_replay"):
        reply = read(local / "maintenance" / (name + ".json"))
        blocked = name == "rollback_conflict"
        outcome = reply["rollback"]
        check(
            reply["host_reconnect_failure"] == item
            and reply["report_sha256"] == digest
            and reply["journal_path"] == journal.relative_to(project).as_posix()
            and reply["host_connection"] == "reported_failed"
            and reply["restored_host_connection"] == "not_evaluated"
            and reply["active_runtime"] == (upgraded if blocked else previous)
            and all(outcome.get(k) == v for k, v in base.items())
            and (
                outcome["status"] == "blocked" and outcome["error"] == conflict
                if blocked
                else outcome == success
            ),
            "bound_reconnect_outcome_" + name,
        )
    restored = read(folder / "restored-mcp.json")
    checks = restored["observation"]
    envelope = checks["capabilities"]
    check(
        restored["argv"] == mcp_command(read(local / "after_rollback/adapter_sha256"))
        and checks["server"] == "autotem-harness"
        and set(checks["tools"]) == TOOL_NAMES
        and envelope["ok"] is True
        and envelope["tool"] == "list_capabilities"
        and bool(envelope["result"]),
        "restored_adapter_real_mcp_call",
    )
    return {
        "verdict": "passed",
        "scope": "controlled new local MCP connection failure and native recovery",
        "host_llm_journey": "not_evaluated",
        "from_installation_id": installation,
        "to_installation_id": previous["installation_id"],
        "failure_code": item["error_code"],
        "rollback_blocker": conflict["code"],
        "restored_mcp_tools": len(checks["tools"]),
        "journal_sha256": sha(journal),
        "report_sha256": digest,
        "process_evidence_sha256": {
            name: sha(owned_file(project, folder / name))
            for name in (
                "attempt.json",
                "entry.stdout",
                "entry.stderr",
                "ready-marker.json",
                "fault-marker.json",
                "restored-mcp.json",
            )
        },
    }


def capture(project, stage):
    directory = project / ".upgrade-verifier" / stage
    directory.mkdir(parents=True)
    for key, relative in STAGE_FILES.items():
        (directory / key).write_bytes((project / relative).read_bytes())


def setup(bundle, project, report, action, flags=(), expected="host_setup_required", error=None):
    start = time.monotonic()
    proc = subprocess.run(
        [str(bundle / "AutoTEM-Setup.exe"), "--project", str(project), "--host", "codex", *flags],
        cwd=project,
        capture_output=True,
        timeout=2100,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    row = json.loads(proc.stdout)
    if action == "preview_previous":
        check(not list(project.iterdir()), "read_only_preview")
    record = project / ".upgrade-verifier/maintenance" / (action + ".json")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_bytes(proc.stdout)
    observed_error = (row.get("error") or {}).get("code")
    # Only predefined public codes are retained, never command lines, paths or error text.
    safe_code = observed_error if re.fullmatch(r"[A-Z0-9_]{1,80}", observed_error or "") else None
    report["maintenance"].append(
        {
            "action": action,
            "record_sha256": sha(record),
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "exit_code": proc.returncode,
            "state": row["state"],
            "error_code": safe_code,
        }
    )
    check(
        row.get("ok") is (error is None) and proc.returncode == (2 if error else 0),
        "native_setup_exit_" + action,
    )
    if error:
        check(observed_error == error, "native_setup_error_" + action)
    else:
        check(row["state"] == expected, "native_setup_state_" + action)
        if expected == "host_setup_required":
            check(
                all(
                    row.get(key) == "passed"
                    for key in ("runtime", "local_transport", "host_assets")
                ),
                "native_setup_readiness_" + action,
            )
    return row


@contextmanager
def bridge(command, project, stage):
    # Keep one already-open project host connection across the actual upgrade.
    proc = subprocess.Popen(
        command,
        cwd=project,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="ascii",
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    replies = queue.Queue()

    def listen():
        try:
            for line in proc.stdout:
                replies.put(line)
        finally:
            replies.put(None)

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()
    sequence = 0

    def call(tool, arguments=None, *, error=None):
        nonlocal sequence
        sequence += 1
        req = {"tool": tool, "arguments": arguments or {}, "request_id": sequence}
        proc.stdin.write(json.dumps(req, ensure_ascii=True) + "\n")
        proc.stdin.flush()
        line = replies.get(timeout=650)
        check(bool(line), "stable_bridge_eof")
        response = json.loads(line)
        with (project / ".upgrade-verifier" / (stage + ".jsonl")).open(
            "a", encoding="utf-8"
        ) as trace:
            trace.write(
                json.dumps({"request": req, "response": response}, ensure_ascii=True) + "\n"
            )
        check(
            response.get("request_id") == sequence and response.get("tool") == tool,
            "stable_bridge_response",
        )
        if error:
            check(not response["ok"] and response["error"]["code"] == error, "stale_bridge_error")
        else:
            check(response["ok"], "stable_bridge_" + tool)
        return response.get("result")

    try:
        yield call
    finally:
        proc.stdin.close()
        proc.wait(timeout=40)
        thread.join(timeout=3)
        proc.stdout.close()


@contextmanager
def competing_installer(control):
    import msvcrt

    with (control / "install.lock").open("r+b") as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def fixture(project, executable):
    directory = project / "fixture"
    directory.mkdir()
    code = (
        "import json,hashlib,sys; from pathlib import Path; "
        "from autotem.acceptance.cases import plan_for,write_fixture; "
        "p,meta=write_fixture('J01',Path(sys.argv[1])); "
        "print(json.dumps({'plan':plan_for('J01',p),'input':str(p),"
        "'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))"
    )
    reply = subprocess.run(
        [executable, "-I", "-c", code, str(directory)],
        cwd=project,
        capture_output=True,
        timeout=90,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    check(reply.returncode == 0, "installed_synthetic_fixture")
    row = json.loads(reply.stdout)
    check(sha(owned_file(project, row["input"])) == row["sha256"], "input_fixture_hash")
    return row


def evidence(project, run_id, expected, source, planned):
    """Reopen the store, reconcile artifacts and measure the synthetic lattice."""
    dbfile = project / ".autotem/runs/harness.sqlite3"
    with closing(sqlite3.connect(dbfile.as_uri() + "?mode=ro", uri=True)) as db:
        run = db.execute(
            "SELECT status,source_hash,input_path,request,job_id FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        check(
            run
            and run[0] == "completed"
            and run[1] == source["sha256"]
            and sha(owned_file(project, run[2])) == run[1],
            "stored_run_snapshot",
        )
        job = db.execute("SELECT run_id,status FROM jobs WHERE job_id=?", (run[4],)).fetchone()
        check(job == (run_id, "completed"), "stored_job")
        events = db.execute(
            "SELECT payload FROM events WHERE run_id=? AND kind='runtime_bound'", (run_id,)
        ).fetchall()
        check(len(events) == 1, "unique_runtime_binding")
        bound = json.loads(events[0][0])
        check(
            bound["runtime_id"] == expected["runtime_id"]
            and bound["job_id"] == run[4]
            and bound["package_version"] == expected["package_version"]
            and bound["plan_sha256"] == hashlib.sha256(run[3].encode()).hexdigest(),
            "bound_runtime_plan_job",
        )
        steps = db.execute(
            "SELECT step_id,operation_id,status FROM steps WHERE run_id=? ORDER BY step_id",
            (run_id,),
        ).fetchall()
        check(
            {s[1] for s in steps} == planned and all(s[2] == "completed" for s in steps),
            "completed_plan_operations",
        )
        rows = db.execute(
            "SELECT artifact_id,role,step_id,path,sha256,size FROM artifacts "
            "WHERE run_id=? AND committed=1 ORDER BY artifact_id",
            (run_id,),
        ).fetchall()
    roles = {r[1] for r in rows}
    check(
        {"raw", "fft_complex", "g_vector_table", "diffraction", "report_manifest", "report_html"}
        <= roles,
        "science_artifact_roles",
    )
    artifacts = []
    spacing = []
    for key, role, step, filename, fingerprint, size in rows:
        file = owned_file(project, filename)
        check(file.stat().st_size == size and sha(file) == fingerprint, "artifact_bytes")
        artifacts.append(
            {
                "artifact_id": key,
                "role": role,
                "step_id": step,
                "sha256": fingerprint,
                "size_bytes": size,
            }
        )
        if (
            role == "diffraction"
            and dict((s[0], s[1]) for s in steps).get(step) == "diffraction.metrics"
        ):
            values = read(file)["d_spacings"]
            check(all(v["unit"] == "nm" for v in values), "d_spacing_unit")
            spacing = sorted(v["value"] for v in values)
        if role == "report_manifest":
            doc = read(file)
            check(
                doc["report_type"] == "analysis"
                and doc["completeness"]["required_complete"]
                and doc["run"]["run_id"] == run_id,
                "required_report_results",
            )
    check(
        len(spacing) == 4 and all(math.isclose(v, 0.16, rel_tol=1e-5) for v in spacing),
        "synthetic_lattice_spacing",
    )
    return {
        "run_id": run_id,
        "job_id": run[4],
        "release_manifest_sha256": expected["manifest_sha256"],
        "runtime_id": bound["runtime_id"],
        "package_version": bound["package_version"],
        "plugins": bound["plugins"],
        "plan_sha256": bound["plan_sha256"],
        "snapshot_sha256": run[1],
        "d_spacings_nm": spacing,
        "steps": [{"step_id": k, "operation_id": op, "status": status} for k, op, status in steps],
        "artifacts": artifacts,
        "reports": {
            r["role"]: r["sha256"]
            for r in artifacts
            if r["role"] in {"report_manifest", "report_html"}
        },
    }


def science(call, project, source, expected, phase):
    plan = source["plan"]
    response = call(
        "submit_analysis",
        {
            "plan": plan,
            "asynchronous": False,
            "idempotency_key": "upgrade-" + phase,
            "user_request": "Measure lattice spacings in nm and index against synthetic-cubic.",
        },
    )
    rid = response["run_id"]
    check(response["status"] == "completed", "public_analysis_" + phase)
    exported = call("export_report", {"run_id": rid, "formats": ["json", "html"]})
    check(
        exported["report_type"] == "analysis" and exported["completeness"]["required_complete"],
        "public_report_" + phase,
    )
    return evidence(
        project, rid, expected, source, {step["operation_id"] for step in plan["steps"]}
    )


def recovery_evidence(project, previous):
    local = project / ".upgrade-verifier"
    original = owned_file(project, local / "maintenance/install_previous-mcp-timeout.json")
    failure = read(original)
    initial = read(owned_file(project, local / "initial-qualified-attempt.json"))
    check(
        initial["case_verdict"] == "failed"
        and initial["failure"]["phase"] == "install_previous"
        and initial["previous_manifest_sha256"] == previous["manifest_sha256"]
        and initial["maintenance"][-1]["record_sha256"] == sha(original)
        and failure["manifest_sha256"] == previous["manifest_sha256"]
        and failure["error"]["code"] == "MCP_HANDSHAKE_FAILED"
        and failure["runtime_diagnostic"]["local_mcp"]["code"] == "MCP_PROBE_TIMEOUT"
        and {"pip_check_passed", "installed_profile_versions_and_origins_passed"}
        <= set(failure["checks"]),
        "first_native_probe_failure",
    )
    failed_tx = project / ".autotem/installations" / failure["installation_id"] / "transaction.json"
    tx = read(owned_file(project, failed_tx))
    check(
        tx["state"] == "failed"
        and tx["manifest_sha256"] == previous["manifest_sha256"]
        and tx["error"]["code"] == "MCP_HANDSHAKE_FAILED",
        "failed_native_transaction",
    )
    attempts = []
    for path in sorted((local / "recovery").glob("attempt-*.json")):
        info = read(owned_file(project, path))
        native = owned_file(project, path.with_suffix(".native"))
        marker = owned_file(project, path.with_suffix(".marker"))
        row, state = read(native), read(marker)
        check(
            info["manifest_sha256"]
            == row["manifest_sha256"]
            == state["manifest_sha256"]
            == previous["manifest_sha256"]
            and info["runtime_id"] == row["runtime_id"] == previous["runtime_id"]
            and info["verifier_sha256"] == sha(Path(__file__).with_name("verify_setup_retry.py"))
            and info["launcher_sha256"] == previous["launcher_sha256"]
            and info["native_sha256"] == sha(native)
            and info["before_marker_sha256"] == sha(marker)
            and info["initial_failure_sha256"] == sha(original)
            and info["before_pointer"] == "absent"
            and state["state"] == "failed"
            and state["error"]["code"] == "MCP_HANDSHAKE_FAILED"
            and Path(state["project"]) == project
            and info["user_config_before_sha256"] == hashlib.sha256(USER_CONFIG).hexdigest()
            and info["ok"] is row["ok"]
            and info["state"] == row["state"]
            and info["error_code"] == (row.get("error") or {}).get("code")
            and info["exit_code"] == (0 if row["ok"] else 2)
            and info["probe_timeout_seconds"] == 60
            and info["elapsed_seconds"] >= 0,
            "native_recovery_evidence",
        )
        if row["ok"]:
            check(
                not any(a["ok"] for a in attempts)
                and info["installed_files_reused"]
                and row["local_transport"] == "passed"
                and row["state"] == "host_setup_required",
                "recovered_local_mcp",
            )
            root = project / ".autotem/runtimes" / previous["runtime_id"]
            check(
                sha(owned_file(project, root / "Scripts/python.exe")) == info["python_sha256"]
                and sha(owned_file(project, root / "Lib/site-packages/autotem/harness/worker.py"))
                == info["worker_sha256"],
                "reused_binary_or_worker",
            )
        else:
            check(
                info["error_code"] == "MCP_HANDSHAKE_FAILED"
                and row["runtime_diagnostic"]["local_mcp"]["code"] == "MCP_PROBE_TIMEOUT",
                "unrecognized_retry_failure",
            )
        attempts.append(info)
    return {
        "initial_code": "MCP_HANDSHAKE_FAILED",
        "initial_transport_code": "MCP_PROBE_TIMEOUT",
        "initial_native_sha256": sha(original),
        "initial_verifier_sha256": initial["verifier_sha256"],
        "initial_elapsed_seconds": initial["maintenance"][-1]["elapsed_seconds"],
        "attempts": attempts,
    }


def resume_install(project, output, previous_bundle, report):
    local = project / ".upgrade-verifier"
    record = local / "initial-qualified-attempt.json"
    if not record.exists():
        record.write_bytes(output.read_bytes())
    first = read(record)
    check(
        first["previous_manifest_sha256"] == report["previous_manifest_sha256"]
        and first["release_manifest_sha256"] == report["release_manifest_sha256"]
        and first["case_verdict"] == "failed"
        and first["failure"]["phase"] == "install_previous"
        and [r["action"] for r in first["maintenance"]] == ["preview_previous", "install_previous"]
        and first["maintenance"][0]["record_sha256"]
        == sha(local / "maintenance/preview_previous.json")
        and not (local / "before_upgrade").exists()
        and not (project / "fixture").exists(),
        "resume_requires_original_failed_cold_attempt",
    )
    attempts = sorted((local / "recovery").glob("attempt-*.json"))
    if not attempts or not read(attempts[-1])["ok"]:
        if __package__:
            from .verify_setup_retry import retry
        else:
            from verify_setup_retry import retry
        check(retry(previous_bundle, project), "native_retry_still_blocked")
    info = recovery_evidence(project, report["releases"]["previous"])
    check(info["attempts"] and info["attempts"][-1]["ok"], "missing_successful_native_retry")
    last = info["attempts"][-1]
    latest = sorted((local / "recovery").glob("attempt-*.native"))[-1]
    native = local / "maintenance/install_previous.json"
    native.write_bytes(latest.read_bytes())
    report["started_utc"] = first["started_utc"]
    report["maintenance"] = [
        first["maintenance"][0],
        {
            "action": "install_previous",
            "record_sha256": last["native_sha256"],
            "elapsed_seconds": last["elapsed_seconds"],
            "exit_code": last["exit_code"],
            "state": last["state"],
            "error_code": last["error_code"],
        },
    ]
    report["install_recovery"] = info
    return read(native)


def main_run(previous_bundle, bundle, project, output, *, resume_failed_install=False):
    project = project.resolve()
    previous, current = release(previous_bundle), release(bundle)
    report = {
        "schema_version": "1.0.0",
        "case_verdict": "failed",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "previous_manifest_sha256": previous["manifest_sha256"],
        "release_manifest_sha256": current["manifest_sha256"],
        "verifier_sha256": sha(Path(__file__)),
        "releases": {"previous": previous, "current": current},
        "scope": "Native Windows Setup, project-local Codex JSONL entry and synthetic J01 HRTEM",
        "scientific_accuracy": "not_evaluated",
        "host_llm_journey": "not_evaluated",
        "release_ready": False,
        "checks": [],
        "maintenance": [],
        "activations": {},
        "runs": {},
        "history_rechecks": {},
        "original_user_config_sha256": hashlib.sha256(USER_CONFIG).hexdigest(),
    }
    phase = "preflight"

    def milestone(code):
        report["checks"].append(code)
        print(code, flush=True)

    def recheck(name):
        seen = []
        for stage, original in report["runs"].items():
            rel = current if stage == "after_upgrade" else previous
            checked = evidence(project, original["run_id"], rel, source, planned)
            check(checked == original, "earlier_scientific_evidence_changed")
            seen.append(original["run_id"])
        check(sha(Path(source["input"])) == source["sha256"], "original_input_changed")
        report["history_rechecks"][name] = seen

    try:
        check(
            os.name == "nt" and previous["manifest_sha256"] != current["manifest_sha256"],
            "two_distinct_windows_releases",
        )
        check(
            not output.is_relative_to(project)
            and all(
                not project.is_relative_to(b) and not b.is_relative_to(project)
                for b in (previous_bundle, bundle)
            ),
            "isolated_project",
        )
        cfg = project / ".codex/config.toml"
        phase = "install_previous"
        if resume_failed_install:
            installed = resume_install(project, output, previous_bundle, report)
        else:
            check(
                not project.exists() or project.is_dir() and not list(project.iterdir()),
                "fresh_project",
            )
            project.mkdir(parents=True, exist_ok=True)
            setup(previous_bundle, project, report, "preview_previous", expected="planned")
            cfg.parent.mkdir()
            cfg.write_bytes(USER_CONFIG)
            installed = setup(previous_bundle, project, report, phase, ("--apply",))
        (
            project / ".upgrade-verifier" / ("driver-" + report["verifier_sha256"] + ".py")
        ).write_bytes(Path(__file__).read_bytes())
        command, first = activation(project, previous)
        report["activations"]["before_upgrade"] = first
        capture(project, "before_upgrade")
        source = fixture(project, installed["python_executable"])
        save(project / ".upgrade-verifier/fixture.json", source)
        planned = {step["operation_id"] for step in source["plan"]["steps"]}
        check(planned == J01, "canonical_j01_operations")
        milestone("previous_release_installed")
        phase = "previous_science"
        with bridge(command, project, "before_upgrade") as old:
            report["runs"]["before_upgrade"] = science(
                old, project, source, previous, "before_upgrade"
            )
            milestone("previous_bound_science_and_reports")
            phase = "concurrent_setup"
            with competing_installer(project / ".autotem"):
                setup(bundle, project, report, phase, ("--apply",), error="INSTALLATION_BUSY")
            check(activation(project, previous)[1] == first, "busy_activation_changed")
            recheck("after_competing_installer")
            milestone("competing_installer_rejected_without_changes")
            phase = "upgrade_current"
            setup(bundle, project, report, phase, ("--apply",))
            new_command, second = activation(project, current)
            check(
                new_command == command
                and second["host_config_sha256"] == first["host_config_sha256"]
                and second["store_schema_version"] == first["store_schema_version"],
                "stable_entry_user_config_or_schema_changed",
            )
            report["activations"]["after_upgrade"] = second
            capture(project, "after_upgrade")
            recheck("after_upgrade")
            milestone("upgrade_and_old_history_verified")
            phase = "stale_connector"
            old(
                "submit_analysis",
                {
                    "plan": source["plan"],
                    "asynchronous": True,
                    "idempotency_key": "upgrade-stale-session",
                },
                error="HOST_RECONNECT_REQUIRED",
            )
            dbfile = project / ".autotem/runs/harness.sqlite3"
            with closing(sqlite3.connect(dbfile.as_uri() + "?mode=ro", uri=True)) as db:
                check(
                    db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1,
                    "stale_session_created_job",
                )
            milestone("stale_session_refused_without_job")
        phase = "upgraded_science"
        with bridge(new_command, project, "after_upgrade") as new:
            report["runs"]["after_upgrade"] = science(
                new, project, source, current, "after_upgrade"
            )
        milestone("upgraded_bound_science_and_reports")
        phase = "new_mcp_connection_failure"
        reconnect_report = observe_reconnect_failure(project, current)
        recover_flags = ("--host-reconnect-failure", str(reconnect_report))
        phase = "rollback_conflict"
        adapter = project / "autotem-host/codex/adapter.json"
        expected = sha(adapter)
        modified = adapter.read_bytes() + b"\n  \n"
        adapter.write_bytes(modified)
        blocked = setup(bundle, project, report, phase, recover_flags, error="HOST_FILE_CONFLICT")
        pointer = project / ".autotem/active-runtime.json"
        check(
            adapter.read_bytes() == modified
            and sha(pointer) == second["pointer_sha256"]
            and sha(cfg) == first["host_config_sha256"],
            "modified_host_asset_overwritten",
        )
        recheck("after_rollback_conflict")
        changed_record = project / ".upgrade-verifier/conflict-user-asset.json"
        changed_record.write_bytes(adapter.read_bytes())
        edited = sha(changed_record)
        comparison = conflict_copy(
            project, blocked, read(pointer)["host_installations"]["codex"], expected, edited
        )
        report["host_conflict"] = {
            "original_sha256": expected,
            "preserved_sha256": edited,
            "verified_comparison_sha256": sha(comparison),
        }
        adapter.write_bytes(comparison.read_bytes())
        milestone("user_modified_host_asset_preserved")
        phase = "rollback_previous"
        setup(bundle, project, report, phase, recover_flags, expected="rolled_back")
        setup(bundle, project, report, "reconnect_replay", recover_flags, expected="rolled_back")
        restored_command, restored = activation(project, previous)
        check(restored == first and restored_command == command, "rollback_pointer_host_bytes")
        report["activations"]["after_rollback"] = restored
        capture(project, "after_rollback")
        recheck("after_rollback")
        milestone("previous_activation_and_both_histories_restored")
        phase = "restored_mcp"
        adapter = read(project / "autotem-host/codex/adapter.json")
        command = mcp_command(adapter)
        probe = Path(__file__).resolve().parents[1] / "release/contract_probe.py"
        local = project / ".upgrade-verifier/reconnect"
        observation = runpy.run_path(str(probe))["mcp_probe"](local, command=command)
        save(local / "restored-mcp.json", {"argv": command, "observation": observation})
        phase = "restored_science"
        with bridge(restored_command, project, "after_rollback") as restored_host:
            report["runs"]["after_rollback"] = science(
                restored_host, project, source, previous, "after_rollback"
            )
        recheck("final")
        check(
            len({r["job_id"] for r in report["runs"].values()}) == 3
            and len({r["run_id"] for r in report["runs"].values()}) == 3,
            "non_unique_scientific_jobs",
        )
        milestone("three_distinct_project_bound_jobs_and_immutable_reports")
        report["bridge_traces"] = {
            stage: sha(project / ".upgrade-verifier" / (stage + ".jsonl")) for stage in PHASES
        }
        report["host_reconnect_recovery"] = reconnect_evidence(project, current)
        report["case_verdict"] = "passed"
        _verify(report, bundle, previous_bundle, project)
    except Exception as exc:
        report["case_verdict"] = "failed"
        report["failure"] = {
            "phase": phase,
            "code": str(exc) if isinstance(exc, GateError) else type(exc).__name__,
        }
        print("failed", report["failure"], file=sys.stderr)
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(output, report)
    return report["case_verdict"] == "passed"


def _verify(report, bundle, previous_bundle, project):
    previous, current = release(previous_bundle), release(bundle)
    check(
        report["schema_version"] == "1.0.0"
        and report["case_verdict"] == "passed"
        and report["checks"] == list(MILESTONES)
        and report["release_ready"] is False
        and report["host_llm_journey"] == "not_evaluated"
        and report["scientific_accuracy"] == "not_evaluated",
        "upgrade_evidence_schema",
    )
    local = project / ".upgrade-verifier"
    driver_hash = report["verifier_sha256"]
    accepted_driver = driver_hash == sha(Path(__file__))
    if driver_hash == FIRST_VERIFIER_SHA256:
        accepted_driver = (
            sha(owned_file(project, local / "driver-source.py")) == FIRST_VERIFIER_SHA256
        )
    if driver_hash in {PRE_COMPARISON_VERIFIER_SHA256, PRE_RECONNECT_VERIFIER_SHA256}:
        meta = read(bundle / "release-manifest.json")["source"]
        frozen = owned_file(bundle, bundle / meta["archive"])
        with zipfile.ZipFile(frozen) as archive:
            accepted_driver = (
                sha(frozen) == meta["archive_sha256"]
                and hashlib.sha256(archive.read("tools/setup/verify_upgrade.py")).hexdigest()
                == driver_hash
                and sha(owned_file(project, local / ("driver-" + driver_hash + ".py")))
                == driver_hash
            )
    check(
        accepted_driver
        and previous != current
        and report["previous_manifest_sha256"] == previous["manifest_sha256"]
        and report["release_manifest_sha256"] == current["manifest_sha256"]
        and report["releases"] == {"previous": previous, "current": current},
        "upgrade_evidence_identity",
    )
    actions = dict(MAINTENANCE)
    if driver_hash not in EARLIER_DRIVERS:
        actions["reconnect_replay"] = ("rolled_back", None)
    check([m["action"] for m in report["maintenance"]] == list(actions), "maintenance_sequence")
    if (local / "initial-qualified-attempt.json").exists() or "install_recovery" in report:
        extra = recovery_evidence(project, previous)
        check(
            report.get("install_recovery") == extra
            and extra["attempts"][-1]["ok"]
            and extra["attempts"][-1]["native_sha256"] == report["maintenance"][1]["record_sha256"],
            "published_recovery_proof",
        )
    for item in report["maintenance"]:
        action = item["action"]
        path = owned_file(project, local / "maintenance" / (action + ".json"))
        raw = read(path)
        state, error = actions[action]
        rel = previous if action in {"preview_previous", "install_previous"} else current
        check(
            sha(path) == item["record_sha256"]
            and item["state"] == raw["state"] == state
            and item["error_code"] == (raw.get("error") or {}).get("code") == error
            and raw["ok"] is (error is None)
            and item["exit_code"] == (2 if error else 0)
            and (action == "preview_previous" or raw["manifest_sha256"] == rel["manifest_sha256"]),
            "native_record_" + action,
        )
    source = read(owned_file(project, local / "fixture.json"))
    check(
        sha(owned_file(project, source["input"])) == source["sha256"]
        and {step["operation_id"] for step in source["plan"]["steps"]} == J01,
        "recorded_fixture",
    )
    paths = []
    ids = []
    jobs = []
    total = 0
    for phase in PHASES:
        rel = current if phase == "after_upgrade" else previous
        release_path = bundle if phase == "after_upgrade" else previous_bundle
        stage = report["activations"][phase]
        directory = local / phase
        for key in STAGE_FILES:
            check(
                sha(owned_file(project, directory / key)) == stage[key],
                "activation_snapshot_" + phase,
            )
        pointer = read(directory / "pointer_sha256")
        pointer_id = pointer["installation_id"]
        check(
            re.fullmatch("[a-f0-9]{32}", pointer_id)
            and pointer["runtime_id"] == stage["runtime_id"] == rel["runtime_id"]
            and pointer["manifest_sha256"] == stage["manifest_sha256"] == rel["manifest_sha256"]
            and pointer["store_schema_version"] == stage["store_schema_version"],
            "stage_pointer",
        )
        journal = read(
            owned_file(
                project, project / ".autotem/installations" / pointer_id / "transaction.json"
            )
        )
        check(
            journal["manifest_sha256"] == rel["manifest_sha256"]
            and journal["installation_id"] == pointer_id,
            "activation_transaction",
        )
        check(
            (directory / "host_config_sha256").read_bytes().startswith(USER_CONFIG)
            and report["original_user_config_sha256"] == hashlib.sha256(USER_CONFIG).hexdigest(),
            "user_configuration_snapshot",
        )
        with zipfile.ZipFile(release_path / "source/autotem-source.zip") as archive:
            entry = archive.read("autotem/installation/host_entry.py")
            worker_digest = hashlib.sha256(archive.read("autotem/harness/worker.py")).hexdigest()
        runtime = project / ".autotem/runtimes" / rel["runtime_id"]
        marker = read(owned_file(project, runtime / ".autotem-runtime.json"))
        check(
            marker["state"] == "ready"
            and marker["manifest_sha256"] == rel["manifest_sha256"]
            and Path(marker["runtime"]) == runtime
            and Path(marker["project"]) == project
            and sha(owned_file(project, runtime / "Lib/site-packages/autotem/harness/worker.py"))
            == worker_digest,
            "retained_release_worker",
        )
        check(
            hashlib.sha256(entry).hexdigest() == stage["stable_entry_sha256"], "stable_entry_source"
        )
        adapter = read(directory / "adapter_sha256")
        check(
            Path(adapter["python_executable"])
            == project / ".autotem/runtimes" / rel["runtime_id"] / "Scripts/python.exe"
            and adapter["jsonl_command"][1:]
            == ["-I", str(project / ".autotem/host-entry.py"), "stdio"],
            "stage_adapter_route",
        )
        paths.append(adapter["jsonl_command"])
        row = report["runs"][phase]
        check(
            evidence(project, row["run_id"], rel, source, J01) == row, "persisted_science_" + phase
        )
        ids.append(row["run_id"])
        jobs.append(row["job_id"])
        total += len(row["artifacts"])
        trace = owned_file(project, local / (phase + ".jsonl"))
        check(sha(trace) == report["bridge_traces"][phase], "bridge_trace_hash")
        calls = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        tools = ["submit_analysis", "export_report"] + (
            ["submit_analysis"] if phase == "before_upgrade" else []
        )
        check([pair["request"]["tool"] for pair in calls] == tools, "stage_tool_calls")
        for i, pair in enumerate(calls):
            req, resp = pair["request"], pair["response"]
            check(
                req["request_id"] == resp["request_id"] == i + 1 and resp["tool"] == req["tool"],
                "trace_correlation",
            )
            if i == 2:
                check(
                    not resp["ok"] and resp["error"]["code"] == "HOST_RECONNECT_REQUIRED",
                    "old_live_session_response",
                )
            else:
                check(resp["ok"] and resp["result"]["run_id"] == row["run_id"], "trace_run_binding")
        check(
            calls[0]["request"]["arguments"]["plan"] == source["plan"]
            and calls[0]["response"]["result"]["status"] == "completed"
            and calls[1]["response"]["result"]["completeness"]["required_complete"],
            "trace_science",
        )
    first, up, restored = (report["activations"][k] for k in PHASES)
    check(
        first == restored
        and up["runtime_id"] != first["runtime_id"]
        and up["host_config_sha256"] == first["host_config_sha256"]
        and paths[0] == paths[1] == paths[2]
        and activation(project, previous)[1] == first,
        "restored_project_state",
    )
    history = {
        "after_competing_installer": ids[:1],
        "after_upgrade": ids[:1],
        "after_rollback_conflict": ids[:2],
        "after_rollback": ids[:2],
        "final": ids,
    }
    check(
        len(set(ids)) == 3 and len(set(jobs)) == 3 and report["history_rechecks"] == history,
        "history_recheck_ids",
    )
    asset = (local / "after_upgrade/adapter_sha256").read_bytes()
    changed = owned_file(project, local / "conflict-user-asset.json")
    original = hashlib.sha256(asset).hexdigest()
    conflict = {"original_sha256": original, "preserved_sha256": sha(changed)}
    if driver_hash not in {FIRST_VERIFIER_SHA256, PRE_COMPARISON_VERIFIER_SHA256}:
        raw_conflict = read(owned_file(project, local / "maintenance/rollback_conflict.json"))
        pointer = read(owned_file(project, local / "after_upgrade/pointer_sha256"))
        comparison = conflict_copy(
            project, raw_conflict, pointer["host_installations"]["codex"], original, sha(changed)
        )
        conflict["verified_comparison_sha256"] = sha(comparison)
    check(
        changed.read_bytes() == asset + b"\n  \n" and report["host_conflict"] == conflict,
        "modified_asset_evidence",
    )
    reconnect = "not_evaluated"
    if driver_hash not in EARLIER_DRIVERS:
        observed = reconnect_evidence(project, current)
        check(report.get("host_reconnect_recovery") == observed, "post_switch_recovery_evidence")
        reconnect = observed["verdict"]
    else:
        check("host_reconnect_recovery" not in report, "historical_reconnect_evidence")
    return {
        "case_verdict": "passed",
        "release_manifest_sha256": current["manifest_sha256"],
        "previous_manifest_sha256": previous["manifest_sha256"],
        "run_ids": ids,
        "verified_artifact_count": total,
        "post_switch_host_failure_recovery": reconnect,
    }


def recheck(report_path, bundle, previous_bundle, project):
    """Verify the public summary against both bundles and the retained local project, offline."""
    try:
        return _verify(
            read(Path(report_path)),
            Path(bundle).resolve(),
            Path(previous_bundle).resolve(),
            Path(project).resolve(),
        )
    except GateError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        sqlite3.Error,
        zipfile.BadZipFile,
    ) as exc:
        raise GateError("unreadable_or_incomplete_upgrade_evidence") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-bundle", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--resume-failed-install",
        action="store_true",
        help="Resume only this pair's documented native first-install MCP timeout.",
    )
    args = parser.parse_args()
    raise SystemExit(
        0
        if main_run(
            args.previous_bundle.resolve(),
            args.bundle.resolve(),
            args.project.resolve(),
            args.output.resolve(),
            resume_failed_install=args.resume_failed_install,
        )
        else 2
    )
