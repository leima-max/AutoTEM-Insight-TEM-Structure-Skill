"""Interrupt a native Windows activation with a project-owned SQLite writer and retry Setup."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import re
import sqlite3
import stat
import subprocess
import sys
import time
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

# Also works under the target PSF interpreter's -I, outside the source checkout.
HERE = Path(__file__).resolve()
HELPER = HERE.with_name("verify_upgrade.py")
_spec = importlib.util.spec_from_file_location("autotem_upgrade_evidence", HELPER)
up = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(up)
check, read, sha = up.check, up.read, up.sha
PROFILE = "traditional-core-mcp"
ACTIONS = {
    "historical_install": ("--apply",),
    "rollback_write_fault": ("--rollback", "--apply"),
    "historical_rollback": ("--rollback", "--apply"),
    "protected_uninstall": ("--uninstall", "--apply"),
    "blocked_activation": ("--apply",),
    "retry": ("--apply",),
}
READY_CHECKS = {
    "pip_check_passed",
    "installed_autotem_wheel_bytes_passed",
    "installed_profile_versions_and_origins_passed",
    "local_mcp_handshake_passed",
}
INVENTORY = "runtime-inventory.json"
LEASE_FAULT = "AUTOTEM_QUALIFICATION_REJECT_LEASE_WRITE"
LEASE_IDENTITY = "autotem-qualification-retired-lease"


def write(path, data):
    """Refuse to replace original evidence, including on a failed invocation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def record(path, value):
    write(path, (json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode())


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def new_paths(bundle, project, out):
    for path in (project, out):
        check(
            path.is_absolute() and ".." not in path.parts, "noncanonical_project_or_evidence_path"
        )
        check(not os.path.lexists(path), "use_a_new_project_and_evidence_directory")
        for parent in path.parents:
            if parent.exists():
                mode = parent.lstat()
                check(
                    not stat.S_ISLNK(mode.st_mode)
                    and not getattr(mode, "st_file_attributes", 0) & 0x400,
                    "linked_parent",
                )
    for a, b in ((project, bundle), (out, bundle), (project, out)):
        check(not a.is_relative_to(b) and not b.is_relative_to(a), "overlapping_paths")


def identity(bundle, development):
    release = up.release(bundle)
    meta = read(bundle / "release-manifest.json")
    check(set(meta["profiles"]) == {PROFILE}, "expected_core_mcp_bundle")
    with zipfile.ZipFile(bundle / meta["source"]["archive"]) as source:
        helper_bytes = source.read("tools/setup/verify_upgrade.py")
        check(helper_bytes == HELPER.read_bytes(), "wrong_supporting_verifier")
        name = "tools/setup/verify_recovery.py"
        if name in source.namelist():
            check(source.read(name) == HERE.read_bytes(), "recovery_verifier_changed_after_freeze")
            origin = "release_source_archive"
        else:
            check(development, "recovery_verifier_missing_from_release")
            origin = "development_after_release_freeze"
    return release, origin


def native(bundle, project, out, action):
    exe = bundle / "AutoTEM-Setup.exe"
    flags = ACTIONS[action]
    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper() not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"}
    }
    env.update(PATH=str(Path(os.environ["SystemRoot"]) / "System32"), PIP_NO_INDEX="1")
    started, tick = now(), time.monotonic()
    proc = subprocess.run(
        [str(exe), "--project", str(project), "--host", "codex", *flags],
        cwd=project,
        env=env,
        capture_output=True,
        timeout=2100,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    prefix = out / "native" / action
    write(prefix.with_suffix(".stdout.json"), proc.stdout)
    write(prefix.with_suffix(".stderr"), proc.stderr)
    record(
        prefix.with_suffix(".process.json"),
        {
            "exit_code": proc.returncode,
            "started_utc": started,
            "ended_utc": now(),
            "elapsed_seconds": round(time.monotonic() - tick, 3),
            "launcher_sha256": sha(exe),
            "flags": list(flags),
            "project": str(project),
        },
    )
    return json.loads(proc.stdout.decode("utf-8"))


def native_evidence(out, project, rel):
    rows = {}
    states = {
        "historical_install": "host_setup_required",
        "rollback_write_fault": "blocked",
        "historical_rollback": "rolled_back",
        "protected_uninstall": "uninstalled",
        "blocked_activation": "blocked",
        "retry": "host_setup_required",
    }
    for name, state in states.items():
        prefix = out / "native" / name
        data = read(prefix.with_suffix(".stdout.json"))
        event = read(prefix.with_suffix(".process.json"))
        check(prefix.with_suffix(".stderr").is_file(), "missing_native_stderr_" + name)
        success = name not in {"blocked_activation", "rollback_write_fault"}
        check(
            event["exit_code"] == (0 if success else 2)
            and event["launcher_sha256"] == rel["launcher_sha256"]
            and event["project"] == str(project)
            and event["flags"] == list(ACTIONS[name])
            and data["ok"] is success
            and data["state"] == state
            and data["manifest_sha256"] == rel["manifest_sha256"]
            and data["runtime_id"] == rel["runtime_id"]
            and data["profile"] == PROFILE
            and data["release_ready"] is False,
            "native_result_" + name,
        )
        if name in {"historical_install", "retry", "blocked_activation"}:
            check(
                READY_CHECKS <= set(data["checks"])
                and data["runtime"] == data["local_transport"] == data["local_mcp"] == "passed"
                and data["host_connection"] == data["host_journey"] == "not_evaluated",
                "staged_runtime_" + name,
            )
        if name == "blocked_activation":
            check(
                data["error"]["code"] == "PROJECT_DATABASE_BUSY"
                and "database is locked" in data["error"]["message"].lower()
                and "activation" not in data
                and data["host_assets"] == "not_evaluated",
                "expected_precommit_write_lock_failure",
            )
            action = data.get("next_action", "")
            check(
                all(
                    term in action for term in ("保留项目文件", "等待", "写入", "同一项目", "重试")
                ),
                "actionable_same_project_retry",
            )
        elif name == "rollback_write_fault":
            check(LEASE_FAULT in data["error"]["message"], "expected_lease_write_failure")
        elif name in {"historical_install", "retry"}:
            check(
                data["host_assets"] == "passed" and data["activation"] == state,
                "activation_receipt_" + name,
            )
        rows[name] = data
    return rows


def competing_writer(database):
    """Probe exactly this file from a second connection without committing any data."""
    with closing(sqlite3.connect(database.as_uri() + "?mode=rw", uri=True, timeout=0)) as other:
        try:
            other.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc).lower():
                raise
            return {
                "result": "busy",
                "error": str(exc),
                "sqlite_errorcode": getattr(exc, "sqlite_errorcode", None),
            }
        other.rollback()
        return {"result": "acquired_and_rolled_back"}


def journal(project, transaction):
    check(
        isinstance(transaction, str) and re.fullmatch("[0-9a-f]{32}", transaction),
        "invalid_transaction_id",
    )
    return up.owned_file(
        project, project / ".autotem/installations" / transaction / "transaction.json"
    )


def status(project, rel):
    control = project / ".autotem"
    marker = control / "runtimes" / rel["runtime_id"] / ".autotem-runtime.json"
    pointer = control / "active-runtime.json"
    adapter = project / "autotem-host/codex/adapter.json"
    with closing(
        sqlite3.connect((control / "runs/harness.sqlite3").as_uri() + "?mode=ro", uri=True)
    ) as db:
        runs = db.execute("SELECT run_id,job_id,status FROM runs ORDER BY run_id").fetchall()
        jobs = db.execute("SELECT run_id,job_id,status FROM jobs ORDER BY run_id").fetchall()
    check(runs == jobs and all(r[2] == "completed" for r in runs), "project_history_or_active_jobs")
    return {
        "active_pointer": read(pointer) if pointer.exists() else None,
        "adapter_sha256": sha(adapter) if adapter.exists() else None,
        "user_config_sha256": sha(project / ".codex/config.toml"),
        "runtime_marker": read(marker) if marker.exists() else None,
        "completed": [r[0] for r in runs],
    }


def analyse(project, out, phase, source, rel):
    command, _ = up.activation(project, rel)
    index = 0

    def call(tool, args):
        nonlocal index
        index += 1
        req = {"tool": tool, "arguments": args, "request_id": index}
        data = (json.dumps(req, ensure_ascii=True) + "\n").encode("ascii")
        p = subprocess.run(
            command,
            input=data,
            cwd=project,
            capture_output=True,
            timeout=900,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        base = out / "science" / f"{phase}-{index}"
        write(base.with_suffix(".request.json"), data)
        write(base.with_suffix(".response.json"), p.stdout)
        response = json.loads(p.stdout.decode("utf-8"))
        check(
            p.returncode == 0
            and response["ok"]
            and response["request_id"] == index
            and response["tool"] == tool,
            "stable_entry_call",
        )
        return response["result"]

    return up.science(call, project, source, rel, phase)


def capture(out, project, rel, source, before, stage):
    old = up.evidence(project, before["run_id"], rel, source, up.J01)
    check(old == before, "historical_artifacts_changed_" + stage)
    record(
        out / "observed" / (stage + ".json"),
        {"state": status(project, rel), "historical_run_sha256": fingerprint(old)},
    )


def rejected_rollback(bundle, project, out, rel, source, before):
    """Reject a real lease write in this disposable project before a native rollback."""
    database = up.owned_file(project, project / ".autotem/runs/harness.sqlite3")
    uri = database.as_uri() + "?mode=rw"
    lease_sql = "SELECT pid,identity,heartbeat FROM scheduler WHERE singleton=1"
    trigger = (
        "CREATE TRIGGER autotem_qualification_lease BEFORE UPDATE OF pid ON scheduler "
        f"WHEN OLD.identity='{LEASE_IDENTITY}' AND NEW.pid IS NULL "
        f"BEGIN SELECT RAISE(ABORT,'{LEASE_FAULT}'); END"
    )
    deadline = time.monotonic() + 35
    with closing(sqlite3.connect(uri, uri=True)) as db:
        while True:
            db.execute("BEGIN IMMEDIATE")
            saved = db.execute(lease_sql).fetchone()
            if saved[0] is None:
                break
            db.rollback()
            check(time.monotonic() < deadline, "historical_scheduler_still_owns_lease")
            time.sleep(0.15)
        db.execute(
            "UPDATE scheduler SET pid=?,identity=? WHERE singleton=1", (os.getpid(), LEASE_IDENTITY)
        )
        db.execute(trigger)
        db.commit()
        installed_trigger = db.execute(
            "SELECT sql FROM sqlite_master WHERE name='autotem_qualification_lease'"
        ).fetchone()[0]
    try:
        reply = native(bundle, project, out, "rollback_write_fault")
        capture(out, project, rel, source, before, "rejected_rollback")
        check(
            not reply["ok"] and LEASE_FAULT in reply["error"]["message"], "lease_fault_not_observed"
        )
    finally:
        with closing(sqlite3.connect(uri, uri=True)) as db:
            db.execute("BEGIN IMMEDIATE")
            observed = db.execute(lease_sql).fetchone()
            check(observed[:2] == (os.getpid(), LEASE_IDENTITY), "controlled_lease_changed")
            db.execute("DROP TRIGGER autotem_qualification_lease")
            db.execute("UPDATE scheduler SET pid=?,identity=?,heartbeat=? WHERE singleton=1", saved)
            db.commit()
            restored = db.execute(lease_sql).fetchone()
        record(
            out / "observed/lease-fault.json",
            {
                "trigger_sql": installed_trigger,
                "previous": saved,
                "on_failure": observed,
                "restored": restored,
                "original_science_sha256": fingerprint(before),
            },
        )
    historical = read(out / "observed/historical.json")
    check(
        read(out / "observed/rejected_rollback.json") == historical,
        "failed_rollback_changed_active_project",
    )


def collect(bundle, project, out, *, development=False):
    check(os.name == "nt", "native_windows_required")
    new_paths(bundle, project, out)
    rel, origin = identity(bundle, development)
    out.mkdir(parents=True)
    project.mkdir(parents=True)
    phase = "historical_install"
    header = {
        "schema_version": "1.0.0",
        "release_manifest_sha256": rel["manifest_sha256"],
        "verifier_sha256": sha(HERE),
        "supporting_verifier_sha256": sha(HELPER),
        "verifier_binding": origin,
        "started_utc": now(),
        "release_ready": False,
        "administrator_role": bool(ctypes.windll.shell32.IsUserAnAdmin()),
        "host_journey": "not_evaluated",
        "scientific_accuracy": "not_evaluated",
    }
    try:
        record(
            out / "observed/runner.json",
            {
                "administrator_role": header["administrator_role"],
                "pid": os.getpid(),
                "token_probe": "shell32.IsUserAnAdmin",
                "windows_build": sys.getwindowsversion().build,
                "started_utc": header["started_utc"],
            },
        )
        cfg = project / ".codex/config.toml"
        write(cfg, up.USER_CONFIG)
        print("historical native install", flush=True)
        installed = native(bundle, project, out, phase)
        check(installed["ok"], "historical_install_failed")
        up.activation(project, rel)
        source = up.fixture(project, installed["python_executable"])
        record(out / "science/fixture.json", source)
        before = analyse(project, out, "historical", source, rel)
        record(out / "science/historical.json", before)
        capture(out, project, rel, source, before, "historical")
        phase = "rollback_write_fault"
        rejected_rollback(bundle, project, out, rel, source, before)
        phase = "historical_rollback"
        check(native(bundle, project, out, phase)["ok"], "historical_rollback_failed")
        phase = "protected_uninstall"
        check(native(bundle, project, out, phase)["ok"], "protected_uninstall_failed")
        capture(out, project, rel, source, before, "uninstalled")
        inactive = status(project, rel)
        check(
            inactive["active_pointer"] is None and inactive["runtime_marker"] is None,
            "cannot_inject_into_active_or_staged_project",
        )
        phase = "blocked_activation"
        print("fresh provisioning with the verifier's project SQLite write lock", flush=True)
        database = up.owned_file(project, project / ".autotem/runs/harness.sqlite3")
        with closing(sqlite3.connect(database.as_uri() + "?mode=rw", uri=True)) as writer:
            writer.execute("BEGIN IMMEDIATE")
            check(writer.in_transaction, "verifier_write_lock")
            contest = competing_writer(database)
            check(contest["result"] == "busy", "independent_writer_must_wait")
            locked_utc = now()
            blocked = native(bundle, project, out, phase)
            check(
                not blocked["ok"] and blocked["state"] == "blocked",
                "activation_succeeded_under_lock",
            )
            snapshot = journal(project, blocked["installation_id"])
            write(out / "observed/pending-transaction.json", snapshot.read_bytes())
            write(
                out / "observed/pending-inventory.json", snapshot.with_name(INVENTORY).read_bytes()
            )
            capture(out, project, rel, source, before, "blocked")
            check(writer.in_transaction, "verifier_transaction_ended_early")
            writer.rollback()
        unlocked = competing_writer(database)
        check(unlocked["result"] == "acquired_and_rolled_back", "private_writer_still_owns_project")
        record(
            out / "observed/write-lock.json",
            {
                "pid": os.getpid(),
                "database": str(database),
                "locked_utc": locked_utc,
                "released_utc": now(),
                "during": contest,
                "after": unlocked,
            },
        )
        phase = "retry"
        print("native retry after releasing the verifier's lock", flush=True)
        check(native(bundle, project, out, phase)["ok"], "native_retry_failed")
        capture(out, project, rel, source, before, "reactivated")
        phase = "recovered_science"
        after = analyse(project, out, "recovered", source, rel)
        record(out / "science/recovered.json", after)
        capture(out, project, rel, source, before, "final")
        details = derive(out, project, rel)
        report = {**header, **details, "finished_utc": now(), "case_verdict": "passed"}
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        sqlite3.Error,
        subprocess.SubprocessError,
    ) as exc:
        report = {
            **header,
            "finished_utc": now(),
            "case_verdict": "failed",
            "failed_phase": phase,
            "error": str(exc),
        }
    record(out / "recovery.json", report)
    return report


def derive(out, project, rel):
    raw = native_evidence(out, project, rel)
    runner = read(out / "observed/runner.json")
    check(
        isinstance(runner["administrator_role"], bool)
        and runner["token_probe"] == "shell32.IsUserAnAdmin"
        and runner["windows_build"] > 0,
        "unknown_windows_runner_token",
    )
    lock = read(out / "observed/write-lock.json")
    blocked_native = read(out / "native/blocked_activation.process.json")
    retry_native = read(out / "native/retry.process.json")
    check(
        lock["pid"] == runner["pid"]
        and int(lock["pid"]) > 0
        and Path(lock["database"]) == project / ".autotem/runs/harness.sqlite3"
        and lock["during"]["result"] == "busy"
        and "database is locked" in lock["during"]["error"].lower()
        and lock["during"]["sqlite_errorcode"] in (None, 5)
        and lock["after"] == {"result": "acquired_and_rolled_back"}
        and runner["started_utc"]
        <= lock["locked_utc"]
        <= blocked_native["started_utc"]
        <= blocked_native["ended_utc"]
        <= lock["released_utc"]
        <= retry_native["started_utc"],
        "owned_project_activation_write_lock",
    )
    source = read(out / "science/fixture.json")
    check(
        {s["operation_id"] for s in source["plan"]["steps"]} == up.J01
        and sha(up.owned_file(project, source["input"])) == source["sha256"],
        "fixture_bytes_plan",
    )
    ids, jobs, runs = [], [], {}
    for phase in ("historical", "recovered"):
        row = read(out / "science" / (phase + ".json"))
        rid = row["run_id"]
        check(up.evidence(project, rid, rel, source, up.J01) == row, "science_bytes_" + phase)
        for i, tool in enumerate(("submit_analysis", "export_report"), 1):
            prefix = out / "science" / f"{phase}-{i}"
            req = read(prefix.with_suffix(".request.json"))
            reply = read(prefix.with_suffix(".response.json"))
            check(
                req["tool"] == reply["tool"] == tool
                and req["request_id"] == reply["request_id"] == i
                and reply["ok"] is True,
                "science_protocol_" + phase,
            )
            if i == 1:
                check(
                    req["arguments"]["plan"] == source["plan"]
                    and req["arguments"]["idempotency_key"] == "upgrade-" + phase
                    and reply["result"]["run_id"] == rid
                    and reply["result"]["status"] == "completed",
                    "submitted_run_" + phase,
                )
            else:
                check(
                    req["arguments"] == {"run_id": rid, "formats": ["json", "html"]}
                    and reply["result"]["completeness"]["required_complete"],
                    "report_run_" + phase,
                )
        runs[phase] = {
            "run_id": rid,
            "job_id": row["job_id"],
            "artifacts": len(row["artifacts"]),
            "d_spacings_nm": row["d_spacings_nm"],
            "reports": row["reports"],
        }
        ids.append(rid)
        jobs.append(row["job_id"])
    check(len(set(ids)) == len(set(jobs)) == 2, "two_distinct_analytic_runs")
    original = fingerprint(read(out / "science/historical.json"))
    seen = {}
    for stage in (
        "historical",
        "rejected_rollback",
        "uninstalled",
        "blocked",
        "reactivated",
        "final",
    ):
        obs = read(out / "observed" / (stage + ".json"))
        state = obs["state"]
        check(
            obs["historical_run_sha256"] == original
            and state["completed"] == (sorted(ids) if stage == "final" else ids[:1]),
            "preserved_history_" + stage,
        )
        seen[stage] = state
    lease = read(out / "observed/lease-fault.json")
    check(
        seen["rejected_rollback"] == seen["historical"]
        and lease["original_science_sha256"] == original
        and lease["previous"] == lease["restored"]
        and lease["previous"][0] is None
        and lease["on_failure"][:2] == [runner["pid"], LEASE_IDENTITY]
        and LEASE_FAULT in lease["trigger_sql"]
        and LEASE_IDENTITY in lease["trigger_sql"],
        "rejected_lease_write_preserved_active_project",
    )
    config_hash = hashlib.sha256(up.USER_CONFIG).hexdigest()
    for name in ("uninstalled", "blocked"):
        check(
            seen[name]["active_pointer"] is None
            and seen[name]["adapter_sha256"] is None
            and seen[name]["user_config_sha256"] == config_hash,
            "inactive_user_project_" + name,
        )
    marker = seen["blocked"]["runtime_marker"]
    rt = project / ".autotem/runtimes" / rel["runtime_id"]
    check(
        seen["uninstalled"]["runtime_marker"] is None
        and marker["state"] == "ready"
        and marker["managed_by"] == "autotem.bootstrap"
        and marker["profile"] == PROFILE
        and Path(marker["project"]) == project
        and Path(marker["runtime"]) == rt
        and marker["manifest_sha256"] == rel["manifest_sha256"]
        and READY_CHECKS <= set(marker["checks"]),
        "first_attempt_staged_owned_runtime",
    )
    tids = [
        raw[name]["installation_id"]
        for name in ("historical_install", "blocked_activation", "retry")
    ]
    check(
        len(set(tids)) == 3
        and seen["historical"]["active_pointer"]["installation_id"] == tids[0]
        and seen["reactivated"]["active_pointer"]["installation_id"] == tids[2]
        and seen["final"] == status(project, rel)
        and seen["final"]["active_pointer"] == seen["reactivated"]["active_pointer"]
        and seen["historical"]["user_config_sha256"] == seen["final"]["user_config_sha256"],
        "current_and_historical_activation",
    )
    up.activation(project, rel)
    pending = read(out / "observed/pending-transaction.json")
    recovered = journal(project, tids[1])
    inv_file = recovered.with_name(INVENTORY)
    inv = read(inv_file)
    check(
        pending["state"] == "preparing"
        and pending["previous"] is None
        and pending["installation_id"] == tids[1]
        and pending["manifest_sha256"] == rel["manifest_sha256"]
        and pending["profile"] == PROFILE
        and not pending["attempted_assets"]
        and not pending["applied_assets"]
        and read(recovered) == {**pending, "state": "recovered"}
        and sha(inv_file)
        == sha(out / "observed/pending-inventory.json")
        == pending["runtime_inventory_sha256"]
        and inv["installation_id"] == tids[1]
        and inv["manifest_sha256"] == rel["manifest_sha256"]
        and Path(inv["project"]) == project,
        "recovered_pending_transaction_and_ownership",
    )
    for name in ("Scripts/python.exe", "Lib/site-packages/autotem/harness/worker.py"):
        check(
            sha(up.owned_file(project, rt / name)) == inv["files"][name], "staged_installed_bytes"
        )
    final_journal = read(journal(project, tids[2]))
    check(
        final_journal["state"] == "host_setup_required"
        and final_journal["previous"] is None
        and final_journal["manifest_sha256"] == rel["manifest_sha256"]
        and set(final_journal["applied_assets"])
        == set(seen["final"]["active_pointer"]["host_installations"].values()),
        "successful_retry_transaction",
    )
    files = {
        p.relative_to(out).as_posix(): sha(p)
        for folder in ("observed", "native", "science")
        for p in sorted((out / folder).iterdir())
        if p.is_file()
    }
    return {
        "administrator_role": runner["administrator_role"],
        "windows_build": runner["windows_build"],
        "started_utc": runner["started_utc"],
        "fault_trigger": "owned_project_activation_write_lock",
        "rollback_lease_write_failure": "active_pointer_assets_and_science_preserved",
        "failure_code": raw["blocked_activation"]["error"]["code"],
        "pending_transaction_id": tids[1],
        "retried_transaction_id": tids[2],
        "recovery_transition": ["preparing", "recovered"],
        "runs": runs,
        "evidence_sha256": files,
    }


def recheck(report_path, bundle, project, *, development=False):
    """Reopen native transcripts, transactions, both database jobs and every scientific artifact."""
    path, bundle, project = (
        Path(report_path).resolve(),
        Path(bundle).resolve(),
        Path(project).resolve(),
    )
    rel, origin = identity(bundle, development)
    doc = read(path)
    check(
        doc["schema_version"] == "1.0.0"
        and doc["case_verdict"] == "passed"
        and doc["release_manifest_sha256"] == rel["manifest_sha256"]
        and doc["verifier_sha256"] == sha(HERE)
        and doc["supporting_verifier_sha256"] == sha(HELPER)
        and doc["verifier_binding"] == origin
        and doc["release_ready"] is False
        and doc["host_journey"] == doc["scientific_accuracy"] == "not_evaluated",
        "invalid_recovery_certificate",
    )
    result = derive(path.parent, project, rel)
    check(
        all(doc.get(k) == val for k, val in result.items()),
        "record_does_not_match_physical_evidence",
    )
    return {
        "case_verdict": "passed",
        "release_manifest_sha256": rel["manifest_sha256"],
        "verifier_binding": origin,
        "runs": result["runs"],
        "pending_transaction_id": result["pending_transaction_id"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True, help="New, dedicated project")
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    parser.add_argument(
        "--verify", action="store_true", help="Recheck retained originals read-only"
    )
    parser.add_argument(
        "--development-verifier",
        action="store_true",
        help="Record a development run against an older release that omitted this verifier",
    )
    args = parser.parse_args()
    try:
        if args.verify:
            result = recheck(
                args.output / "recovery.json",
                args.bundle,
                args.project,
                development=args.development_verifier,
            )
        else:
            result = collect(
                args.bundle.resolve(),
                args.project.absolute(),
                args.output.absolute(),
                development=args.development_verifier,
            )
        print(json.dumps({k: result[k] for k in ("case_verdict", "error", "runs") if k in result}))
        sys.exit(0 if result["case_verdict"] == "passed" else 2)
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, zipfile.BadZipFile) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
