"""Install a verified offline release into a versioned project runtime."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sqlite3
import sys
import time
import uuid
import zipfile
from contextlib import contextmanager

from autotem import bootstrap as boot
from autotem.harness.hosts import rollback_host
from autotem.installation import host_reconnect
from autotem.installation import uninstall as removal
from autotem.installation.host_conflict import HostFileConflict, child_conflicts
from autotem.installation.host_entry import runtime_directory, runtime_python
from autotem.installation.release import (
    PROFILE,
    PROFILE_EXTRAS,
    ReleaseError,
    sha256,
    verify_release,
)


@contextmanager
def project_lock(control):
    # OS releases the lock even on process death. Never unlink/recreate the locked inode.
    with boot._inside(control, "install.lock").open("a+b") as stream:
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise boot.BootstrapError("INSTALLATION_BUSY", "Another installer is active.") from exc
        yield


@contextmanager
def switch_window(control):
    """Reserve the store during file publication, with all lease writes committed first."""
    database = boot._inside(control, "runs/harness.sqlite3")
    database.parent.mkdir(exist_ok=True)
    db = sqlite3.connect(database, timeout=5)

    def reserve():
        db.execute("BEGIN IMMEDIATE")
        version = db.execute("PRAGMA user_version").fetchone()[0]
        # This bootstrap zipapp has no installed Harness. It knows the lease columns of
        # v0 and v1 only; the runtime owns migration and its mandatory pre-migration backup.
        if version not in (0, 1):
            raise boot.BootstrapError(
                "STORE_MIGRATION_REQUIRED", "请保留项目数据，并使用支持此数据库版本的发行包。"
            )
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if not tables and version == 0:
            return version, None
        required = {"runs", "steps", "artifacts", "scheduler"}
        if version == 1:
            required.add("jobs")
        if not required <= tables:
            raise boot.BootstrapError("STORE_UNAVAILABLE", "The project database is incompatible.")
        active = db.execute(
            "SELECT 1 FROM runs WHERE status IN ('preparing','queued','running') LIMIT 1"
        ).fetchone()
        if not active and "jobs" in tables:
            active = db.execute(
                "SELECT 1 FROM jobs WHERE status IN ('queued','running') LIMIT 1"
            ).fetchone()
        if active:
            raise boot.BootstrapError(
                "WAIT_ACTIVE_RUNS", "Allow active jobs to finish or cancel them through the host."
            )
        columns = {row[1] for row in db.execute("PRAGMA table_info(scheduler)")}
        if not {"singleton", "pid", "heartbeat"} <= columns or (
            version == 1 and "identity" not in columns
        ):
            raise boot.BootstrapError("STORE_UNAVAILABLE", "The scheduler schema is incompatible.")
        if "identity" in columns:
            lease = db.execute("SELECT pid,identity FROM scheduler WHERE singleton=1").fetchone()
            clear = "UPDATE scheduler SET pid=NULL,identity=NULL WHERE singleton=1"
        else:
            lease = db.execute("SELECT pid FROM scheduler WHERE singleton=1").fetchone()
            clear = "UPDATE scheduler SET pid=NULL WHERE singleton=1"
        if lease is None and version == 1:
            raise boot.BootstrapError("STORE_UNAVAILABLE", "The scheduler record is missing.")
        return version, clear if lease and any(value is not None for value in lease) else None

    try:
        try:
            version, clear = reserve()
            if clear:
                db.execute(clear)
                db.commit()
                # A submission, migration or scheduler may win between these reservations.
                # Only the second, still-held reservation can authorize changing files.
                locked_version, renewed_lease = reserve()
                if locked_version != version:
                    raise boot.BootstrapError(
                        "STORE_VERSION_CHANGED",
                        "项目数据库版本已更改。请保留项目并重新开始本次操作。",
                    )
                if renewed_lease:
                    raise boot.BootstrapError(
                        "PROJECT_DATABASE_BUSY",
                        "项目调度器已重新开始工作。请等待后在同一项目重试。",
                    )
        except sqlite3.Error as exc:
            code = getattr(exc, "sqlite_errorcode", None)
            # Python 3.10 lacks sqlite_errorcode. SQLite may name the locked table or schema.
            busy = (
                ((code & 0xFF) in (5, 6))
                if code is not None
                else (
                    str(exc).partition(":")[0].lower()
                    in {
                        "database is locked",
                        "database is busy",
                        "database table is locked",
                        "database schema is locked",
                    }
                )
            )
            if busy:
                raise boot.BootstrapError(
                    "PROJECT_DATABASE_BUSY",
                    f"项目数据库正被占用（SQLite: {exc}）。请保留项目文件，"
                    "等待占用此项目的写入结束，然后在同一项目重试刚才的操作。",
                ) from exc
            raise boot.BootstrapError(
                "STORE_UNAVAILABLE",
                f"项目数据库未能完成切换前检查（SQLite: {exc}）。"
                "请保留项目文件，检查数据库后重试。",
            ) from exc
        yield version
    finally:
        # This final reservation has only read state. close releases it without submitting
        # any SQLite write or commit after a pointer or host file may have been published.
        db.close()


def _record_path(control, installation_id):
    if not isinstance(installation_id, str) or not re.fullmatch(r"[a-f0-9]{32}", installation_id):
        raise ValueError("Invalid installation transaction identifier.")
    return boot._inside(control, f"installations/{installation_id}/transaction.json")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _pointer(control):
    path = boot._inside(control, "active-runtime.json")
    entry = _read(path)
    if entry is not None:
        _activation_journal(control, entry)
    return entry


def _activation_journal(control, entry):
    runtime_directory(control, entry)
    journal = _read(_record_path(control, entry["installation_id"]))
    if (
        not journal
        or any(journal.get(key) != entry[key] for key in ("installation_id", "manifest_sha256"))
        or journal.get("profile", PROFILE) != entry.get("profile", PROFILE)
    ):
        raise ValueError("The activation pointer does not match its installation journal.")
    return journal


def _restore_assets(project, transaction, *, preview=False, recover=True):
    ids = reversed(transaction["attempted_assets"])
    existing = []
    for item in ids:
        if not re.fullmatch(r"install-[a-f0-9]{32}", item):
            raise ValueError("Invalid recorded host asset transaction.")
        path = boot._inside(project, f".autotem/host-installations/{item}/manifest.json")
        if path.is_file():
            existing.append(item)
        elif item in transaction.get("applied_assets", []):
            raise boot.BootstrapError(
                "HOST_RECEIPT_MISSING", "Restore the recorded host receipt before rollback."
            )
    for apply in (False,) if preview else (False, True):
        for item in existing:
            result = rollback_host(project, item, dry_run=not apply, recover=recover)
            if not result["ok"]:
                raise HostFileConflict(result)


def _pending_transactions(control):
    for path in sorted((control / "installations").glob("*/transaction.json")):
        boot._no_links(path)
        record = _read(path)
        if record["state"] in ("preparing", "switching", "reverting"):
            yield path, record


def _recover(control):
    reverted = None
    for path, record in _pending_transactions(control):
        current = _pointer(control)
        if record["state"] == "reverting":
            if record.get("host_reconnect_recovery"):
                _finish_reconnect_rollback(control, path, record, raise_blocked=True)
            else:
                _revert(control, record)
            reverted = record
        elif current and current["installation_id"] == record["installation_id"]:
            record["state"] = _active_state(current.get("profile", PROFILE))
        elif current == record["previous"]:
            _restore_assets(control.parent, record)
            record["state"] = "recovered"
        else:
            raise boot.BootstrapError(
                "ACTIVATION_CONFLICT", "Review the pending activation journal."
            )
        boot._atomic_json(path, record)
    return reverted


def _revert(control, transaction):
    previous = transaction["previous"]
    current = _pointer(control)
    if current == previous:
        transaction["state"] = "rolled_back"
        return
    if not current or current["installation_id"] != transaction["installation_id"]:
        raise boot.BootstrapError("ACTIVATION_CONFLICT", "The active installation changed.")
    report = transaction.get("host_reconnect_recovery", {}).get("report")
    if previous:
        if report:
            journal = _activation_journal(control, previous)
            if journal["state"] != _active_state(previous.get("profile", PROFILE)):
                raise boot.BootstrapError(
                    "INVALID_INSTALL_RECEIPT", "Review the predecessor's activation journal."
                )
        try:
            predecessor = runtime_python(control, previous)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise boot.BootstrapError(
                "ROLLBACK_RUNTIME_FAILED", "Repair the predecessor runtime."
            ) from exc
        probe = boot._run(
            [
                str(predecessor),
                "-I",
                "-m",
                "autotem.harness.cli",
                "doctor",
                "--project",
                str(control.parent),
                "--host",
                report["host"] if report else "codex",
                "--timeout",
                "60",
            ],
            environment=boot._child_environment(False),
            cwd=control.parent,
            deadline=time.monotonic() + 240,
        )
        try:
            health = json.loads(probe.stdout)["runtime"]
            healthy = health["status"] == "ready" and (
                "mcp" not in PROFILE_EXTRAS[previous.get("profile", PROFILE)]
                or health["local_mcp"]["status"] == "ready"
            )
        except (KeyError, TypeError, ValueError):
            healthy = False
        if probe.returncode not in (0, 2) or not healthy:
            raise boot.BootstrapError("ROLLBACK_RUNTIME_FAILED", "Repair the predecessor runtime.")
    with switch_window(control) as db_version:
        if db_version not in (transaction["db_version"], current.get("store_schema_version")):
            raise boot.BootstrapError("STORE_VERSION_CHANGED", "Review run database compatibility.")
        recovering = transaction["state"] == "reverting"
        _restore_assets(control.parent, transaction, preview=True, recover=recovering)
        if previous and report:
            host = report["host"]
            host_reconnect.verify_restored_assets(control, transaction, current, host)
        transaction["state"] = "reverting"
        boot._atomic_json(_record_path(control, transaction["installation_id"]), transaction)
        _restore_assets(control.parent, transaction, recover=recovering)
        pointer_path = boot._inside(control, "active-runtime.json")
        if previous:
            boot._atomic_json(pointer_path, previous)
        else:
            pointer_path.unlink()
        transaction["state"] = "rolled_back"


def _finish_reconnect_rollback(control, path, record, *, raise_blocked=False):
    recovery = record["host_reconnect_recovery"]
    outcome = recovery["rollback"]
    try:
        uninstall = _read(boot._inside(control, removal.JOURNAL))
        if uninstall and uninstall.get("state") in removal.BUSY_STATES:
            raise boot.BootstrapError("UNINSTALL_PENDING", "Finish the pending uninstall first.")
        if any(other != path for other, _ in _pending_transactions(control)):
            raise boot.BootstrapError(
                "INSTALLATION_BUSY", "Finish the pending installation before retrying this report."
            )
        _revert(control, record)
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
        outcome.update(
            status="blocked",
            error={"code": getattr(exc, "code", "INSTALLATION_FAILED"), "message": str(exc)},
        )
        if isinstance(exc, HostFileConflict) and exc.resolution:
            outcome["conflict_resolution"] = exc.resolution
        else:
            outcome.pop("conflict_resolution", None)
        # Keep the reason for a blocked attempt after a subsequent successful retry.
        recovery["last_blocked_error"] = outcome["error"]
        boot._atomic_json(path, record)
        if raise_blocked:
            raise
        return
    outcome.update(status="rolled_back")
    outcome.pop("error", None)
    outcome.pop("conflict_resolution", None)
    boot._atomic_json(path, record)


def _handle_reconnect(control, result, report, digest):
    """Apply only to the reported activation; never let stale input recover unrelated work."""
    key = report["installation_id"]
    path = _record_path(control, key)
    record = _read(path)
    current = _pointer(control)
    relative = path.relative_to(control.parent).as_posix()
    result.update(installation_id=key, active_runtime=current)
    if report["host"] != result["host"] or report["release_id"] != result["release_id"]:
        raise boot.BootstrapError(
            "HOST_RECONNECT_TARGET_MISMATCH", "Select the reported host and release bundle."
        )
    if not record or record.get("installation_id") != key:
        raise boot.BootstrapError(
            "ACTIVATION_CONFLICT", "The reported activation is not registered in this project."
        )
    if record.get("manifest_sha256") != result["manifest_sha256"] or (
        "release_id" in record and record["release_id"] != report["release_id"]
    ):
        raise boot.BootstrapError(
            "HOST_RECONNECT_TARGET_MISMATCH", "The release bundle does not match this activation."
        )
    recovery = record.get("host_reconnect_recovery")
    if recovery and (recovery["report_sha256"] != digest or recovery["report"] != report):
        raise boot.BootstrapError(
            "HOST_RECONNECT_TARGET_MISMATCH",
            "A different failure is already recorded. Retry with the original report.",
        )

    def receipt():
        rollback = recovery["rollback"]
        ok = rollback["status"] == "rolled_back"
        return {
            **result,
            "ok": ok,
            "state": "rolled_back" if ok else "blocked",
            "active_runtime": _pointer(control),
            "host_reconnect_failure": report,
            "report_sha256": digest,
            "journal_path": relative,
            "host_connection": "reported_failed",
            "restored_host_connection": "not_evaluated",
            "rollback": rollback,
            "next_action": (
                "Reconnect and verify the active host entry."
                if ok
                else rollback["error"]["message"]
            ),
            **{k: rollback[k] for k in ("error", "conflict_resolution") if k in rollback},
        }

    if recovery and recovery["rollback"]["status"] == "rolled_back":
        return receipt()
    previous = record.get("previous")
    # A crash may have published exactly the intended predecessor before finalizing this journal.
    finalizing = recovery and record["state"] == "reverting" and previous and current == previous
    if not finalizing and (not current or current["installation_id"] != key):
        raise boot.BootstrapError(
            "ACTIVATION_CONFLICT", "The reported installation is no longer active."
        )
    if not previous or previous["manifest_sha256"] == result["manifest_sha256"]:
        raise boot.BootstrapError(
            "HOST_RECONNECT_NO_PREDECESSOR", "This activation has no preceding release to restore."
        )
    host = report["host"]
    if (
        not previous["host_installations"].get(host)
        or not finalizing
        and not current["host_installations"].get(host)
    ):
        raise boot.BootstrapError(
            "HOST_RECONNECT_TARGET_MISMATCH",
            "This host has no registered adapter on both activations.",
        )
    if not recovery and record["state"] not in (
        _active_state(record.get("profile", PROFILE)),
        "switching",
    ):
        raise boot.BootstrapError(
            "ACTIVATION_CONFLICT",
            "The reported activation is not available for reconnect recovery.",
        )
    if not recovery:
        recovery = record["host_reconnect_recovery"] = {
            "report": report,
            "report_sha256": digest,
            "observed_by": "operator",
            "received_at": time.time(),
            "rollback": {
                "status": "pending",
                "from_installation_id": key,
                "to_installation_id": previous["installation_id"],
            },
        }
        boot._atomic_json(path, record)
    _finish_reconnect_rollback(control, path, record)
    return receipt()


def _record_timeout_inventory(control, path, record, staged, release, bundle, old_base_caches):
    # This call is limited to a directory that did not exist before this provisioning attempt.
    # A later retry must never inventory an existing tree as though it were freshly installed.
    checks = {
        "dependency_install_passed",
        "installed_autotem_wheel_bytes_passed",
        "private_package_provenance_passed",
        "pip_check_passed",
        "installed_profile_versions_and_origins_passed",
        "capabilities_passed",
    }
    diagnostic = record.get("runtime_diagnostic", {})
    if (
        record["error"].get("code") != "MCP_HANDSHAKE_FAILED"
        or diagnostic.get("status") != "ready"
        or diagnostic.get("local_mcp", {}).get("code") != "MCP_PROBE_TIMEOUT"
        or not checks.issubset(staged["checks"])
    ):
        return "not_established"
    digest = record["manifest_sha256"]
    try:
        runtime = boot._inside(control, f"runtimes/rt-{digest[:20]}")
        marker = _read(boot._inside(runtime, boot.MARKER_NAME))
        expected = {
            "managed_by": "autotem.bootstrap",
            "schema_version": boot.BOOTSTRAP_VERSION,
            "project": str(control.parent),
            "runtime": str(runtime),
            "state": "failed",
            "manifest_sha256": digest,
            "with_mcp": True,
            "wheel_sha256": staged.get("wheel_sha256"),
        }
        if (
            not isinstance(marker, dict)
            or any(marker.get(k) != v for k, v in expected.items())
            or not re.fullmatch("[a-f0-9]{64}", staged.get("wheel_sha256") or "")
            or sha256(release.wheel) != staged["wheel_sha256"]
            or not checks.issubset(marker.get("checks", ()))
            or (marker.get("error") or {}).get("code") != "MCP_HANDSHAKE_FAILED"
        ):
            return "not_established"
        removal.record_new_runtime(control, path, record, fresh=True)
        removal.record_new_base_caches(
            control, record, old_base_caches, bundle=bundle, release=release, fresh=True
        )
        boot._atomic_json(path, record)
    except (
        OSError,
        TypeError,
        LookupError,
        ValueError,
        csv.Error,
        zipfile.BadZipFile,
        boot.BootstrapError,
    ):
        # Keep the original MCP diagnosis and failed journal. Unproven files stay protected.
        return "not_established"
    return "recorded"


def _active_state(profile):
    return "host_setup_required" if "mcp" in PROFILE_EXTRAS[profile] else "local_cli_ready"


def manage(
    project,
    bundle,
    *,
    host=None,
    profile=None,
    apply=False,
    rollback=False,
    uninstall=False,
    host_reconnect_failure=None,
    timeout=1800,
):
    result = {
        "schema_version": "1.0.0",
        "ok": False,
        "state": "blocked",
        "runtime": "not_evaluated",
        "local_transport": "not_evaluated",
        "host_assets": "not_evaluated",
        "host_connection": "not_evaluated",
        "host_journey": "not_evaluated",
        "release_ready": False,
    }
    try:
        root = boot._project_path(project)
        control = boot._inside(root, ".autotem")
        if host_reconnect_failure is not None:
            if apply or rollback or uninstall:
                raise ValueError("Use --host-reconnect-failure as a separate maintenance action.")
            report, digest = host_reconnect.read_report(host_reconnect_failure)
            result.update(
                host_reconnect_failure=report,
                report_sha256=digest,
                restored_host_connection="not_evaluated",
            )
            if host is None:
                host = report["host"]
        elif host is None:
            host = "codex"
        if host not in boot.HOSTS or type(timeout) not in (float, int) or not 0 < timeout <= 3600:
            raise ValueError("Invalid host or installation timeout.")
        release = verify_release(bundle, profile=profile)
        runtime_id = f"rt-{release.manifest_sha256[:20]}"
        result.update(
            release_id=release.release_id,
            manifest_sha256=release.manifest_sha256,
            runtime_id=runtime_id,
            host=host,
            profile=release.profile,
            transport="mcp" if release.with_mcp else "jsonl",
        )
        if host_reconnect_failure is not None:
            if not control.is_dir():
                raise boot.BootstrapError(
                    "ACTIVATION_CONFLICT", "This project has no registered installation."
                )
            with project_lock(control):
                return _handle_reconnect(control, result, report, digest)
        if rollback and uninstall:
            raise ValueError("Choose either rollback or uninstall.")
        if uninstall:

            def remove():
                return {
                    **result,
                    **removal.remove(
                        control,
                        bundle,
                        release,
                        apply=apply,
                        pointer=_pointer(control),
                        switch_window=switch_window,
                    ),
                }

            if not apply or not control.exists():
                return remove()
            with project_lock(control):
                _recover(control)
                return remove()
        if (control / "runtime").exists() and not (control / "active-runtime.json").exists():
            raise boot.BootstrapError(
                "LEGACY_MIGRATION_REQUIRED",
                "This project has a fixed-path installation. Drain its sessions and review "
                "its host configuration before migration, or select a new project.",
            )
        if not apply:
            return {
                **result,
                "ok": True,
                "state": "planned",
                "project": str(root),
                "action": "rollback" if rollback else "install",
                "next_action": "Apply to verify the runtime and prepare project host files.",
            }
        control.mkdir(exist_ok=True)
        with project_lock(control):
            uninstall_journal = _read(boot._inside(control, removal.JOURNAL))
            if uninstall_journal and uninstall_journal.get("state") in removal.BUSY_STATES:
                raise boot.BootstrapError(
                    "UNINSTALL_PENDING", "Finish the pending uninstall first."
                )
            recovered_rollback = _recover(control)
            previous = _pointer(control)
            if rollback:
                record = recovered_rollback
                if not record:
                    if not previous:
                        raise boot.BootstrapError("NO_ACTIVE_RUNTIME", "No release is active.")
                    path = _record_path(control, previous["installation_id"])
                    record = _read(path)
                    expected = _active_state(previous.get("profile", PROFILE))
                    if not record or record["state"] != expected:
                        raise ValueError("The active installation journal is unavailable.")
                    if record.get("host_reconnect_recovery"):
                        _finish_reconnect_rollback(control, path, record, raise_blocked=True)
                    else:
                        _revert(control, record)
                        boot._atomic_json(path, record)
                return {
                    **result,
                    "ok": True,
                    "state": "rolled_back",
                    "active_runtime": record["previous"],
                    "staged_runtime_retained": True,
                    "next_action": "Reconnect the host to its restored project configuration.",
                }

            if previous and previous["manifest_sha256"] == release.manifest_sha256:
                try:
                    runtime_python(control, previous)
                except (OSError, ValueError, TypeError) as exc:
                    raise boot.BootstrapError(
                        "ACTIVE_RUNTIME_DAMAGED",
                        "Roll back the damaged activation before repairing its private runtime.",
                    ) from exc
            if (
                shutil.disk_usage(root).free
                < sum(p.stat().st_size for p in release.wheelhouse.glob("*.whl")) * 3
            ):
                raise boot.BootstrapError(
                    "DISK_SPACE_LOW", "Choose a project with more free space."
                )
            transaction_id = uuid.uuid4().hex
            path = _record_path(control, transaction_id)
            path.parent.mkdir(parents=True)
            record = {
                "schema_version": "1.0.0",
                "installation_id": transaction_id,
                "created": time.time(),
                "release_id": release.release_id,
                "manifest_sha256": release.manifest_sha256,
                "profile": release.profile,
                "previous": previous,
                "state": "preparing",
                "attempted_assets": [],
                "applied_assets": [],
            }
            boot._atomic_json(path, record)
            result["installation_id"] = transaction_id
            fresh_runtime = not boot._inside(control, f"runtimes/{runtime_id}").exists()
            old_base_caches = removal.base_caches(control)
            staged = boot._provision(
                root,
                release.wheel,
                host=host,
                wheelhouse=release.wheelhouse,
                with_mcp=release.with_mcp,
                allow_network=False,
                index_url=None,
                timeout=timeout,
                dry_run=False,
                release=release,
            )
            if not staged["ok"]:
                record.update(state="failed", error=staged["error"])
                readiness = staged.get("readiness", {}).get("runtime", {})
                if readiness:
                    record["runtime_diagnostic"] = readiness
                    result["runtime_diagnostic"] = readiness
                result["checks"] = staged["checks"]
                boot._atomic_json(path, record)
                if fresh_runtime:
                    result["staged_file_ownership"] = _record_timeout_inventory(
                        control, path, record, staged, release, bundle, old_base_caches
                    )
                raise boot.BootstrapError(staged["error"]["code"], staged["error"]["message"])
            result.update(
                runtime="passed",
                local_transport="passed",
                local_mcp="passed" if release.with_mcp else "not_in_profile",
                checks=staged["checks"],
                python_executable=staged["python_executable"],
                readiness=staged["readiness"],
            )
            removal.record_new_runtime(control, path, record, fresh=fresh_runtime)
            boot._atomic_json(path, record)
            owners = dict(previous["host_installations"]) if previous else {}
            hosts = sorted(set(owners) | {host})
            command = [
                staged["python_executable"],
                "-I",
                "-m",
                "autotem.harness.cli",
                "--root",
                str(control / "runs"),
                "install-host",
            ]
            environment = boot._child_environment(False)
            with switch_window(control) as db_version:
                target_version = staged["store_schema_version"]
                if db_version > target_version or (
                    previous and previous.get("store_schema_version", db_version) != target_version
                ):
                    raise boot.BootstrapError(
                        "STORE_MIGRATION_REQUIRED",
                        "Review database compatibility before activation.",
                    )
                record.update(state="switching", db_version=db_version)
                boot._atomic_json(path, record)
                try:
                    for name in hosts:
                        install_id = "install-" + uuid.uuid4().hex
                        record["attempted_assets"].append(install_id)
                        boot._atomic_json(path, record)
                        args = [
                            name,
                            "--project",
                            str(root),
                            "--managed",
                            "--apply",
                            "--installation-id",
                            install_id,
                        ]
                        if owners.get(name):
                            args.extend(["--previous-installation", owners[name]])
                        if release.with_mcp and name == "codex":
                            args.append("--with-mcp")
                        elif not release.with_mcp:
                            args.append("--without-mcp")
                        proc = boot._run(
                            command + args,
                            environment=environment,
                            cwd=root,
                            deadline=time.monotonic() + 60,
                        )
                        try:
                            data = json.loads(proc.stdout)
                        except ValueError:
                            data = {}
                        if not isinstance(data, dict):
                            data = {}
                        if not data.get("ok") or proc.returncode:
                            conflict = child_conflicts(root, name, owners.get(name), data)
                            if conflict:
                                raise HostFileConflict(conflict)
                            raise boot.BootstrapError(
                                "HOST_ASSET_INSTALL_FAILED",
                                "安装器未能验证宿主文件。请检查项目路径和项目写入权限后重试。",
                            )
                        if data.get("installation_id"):
                            if data["installation_id"] != install_id:
                                raise ValueError("Unexpected host installation receipt.")
                            record["applied_assets"].append(install_id)
                            boot._atomic_json(path, record)
                            owners[name] = install_id
                        else:
                            owners.setdefault(name, None)
                    removal.record_new_base_caches(
                        control,
                        record,
                        old_base_caches,
                        bundle=bundle,
                        release=release,
                        fresh=fresh_runtime,
                    )
                    boot._atomic_json(path, record)
                    if (
                        previous
                        and previous["manifest_sha256"] == release.manifest_sha256
                        and owners == previous["host_installations"]
                    ):
                        record["state"] = "already_active"
                    else:
                        pointer = {
                            "schema_version": "1.0.0",
                            "runtime_id": runtime_id,
                            "manifest_sha256": release.manifest_sha256,
                            "profile": release.profile,
                            "installation_id": transaction_id,
                            "store_schema_version": target_version,
                            "host_installations": owners,
                        }
                        boot._atomic_json(control / "active-runtime.json", pointer)
                        record["state"] = _active_state(release.profile)
                except Exception:
                    _restore_assets(root, record)
                    record["state"] = "failed"
                    boot._atomic_json(path, record)
                    raise
            boot._atomic_json(path, record)
            return {
                **result,
                "ok": True,
                "state": _active_state(release.profile),
                "host_assets": "passed",
                "activation": record["state"],
                "next_action": boot.host_install_next_action(
                    host,
                    mcp_available=release.with_mcp,
                    codex_configured=host == "codex" and release.with_mcp,
                ),
            }
    except (
        ReleaseError,
        boot.BootstrapError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        sqlite3.Error,
    ) as exc:
        error = {"code": getattr(exc, "code", "INSTALLATION_FAILED"), "message": str(exc)}
        if host_reconnect_failure is not None and "host_reconnect_failure" in result:
            result["rollback"] = {"status": "not_attempted", "error": error}
        return {
            **result,
            "error": error,
            "next_action": str(exc),
            **({"conflict_resolution": exc.resolution}
               if isinstance(exc, HostFileConflict) and exc.resolution else {}),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument(
        "--host", choices=boot.HOSTS,
        help="Installation defaults to codex. Reconnect reports identify their host.",
    )
    parser.add_argument("--profile", choices=PROFILE_EXTRAS,
                        help="Require this profile; defaults to the bundle's only profile.")
    parser.add_argument("--apply", action="store_true")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--rollback", action="store_true")
    actions.add_argument("--uninstall", action="store_true")
    actions.add_argument(
        "--host-reconnect-failure", metavar="REPORT.json",
        help="Report a failed new host connection and restore the preceding release.",
    )
    args = parser.parse_args(argv)
    result = manage(
        args.project,
        args.release,
        host=args.host,
        profile=args.profile,
        apply=args.apply,
        rollback=args.rollback,
        uninstall=args.uninstall,
        host_reconnect_failure=args.host_reconnect_failure,
    )
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
