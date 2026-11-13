"""Check native recovery qualification rejects weaker failures and unsafe workspaces."""

import hashlib
import json
import sqlite3
import zipfile

import pytest

from tools.setup import verify_recovery as recovery

pytestmark = pytest.mark.smoke


def raw(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_collection_requires_new_nonoverlapping_project_and_evidence(tmp_path):
    bundle, project, out = [tmp_path / x for x in ("bundle", "project", "evidence")]
    bundle.mkdir()
    recovery.new_paths(bundle, project, out)
    for a, b in ((bundle / "p", out), (project, project / "nested"), (project, project)):
        with pytest.raises(ValueError, match="overlapping_paths"):
            recovery.new_paths(bundle, a, b)
    with pytest.raises(ValueError, match="noncanonical"):
        recovery.new_paths(bundle, tmp_path / "new" / ".." / "bundle" / "p", out)
    project.mkdir()
    with pytest.raises(ValueError, match="new_project"):
        recovery.new_paths(bundle, project, out)
    receipt = out / "native.json"
    recovery.record(receipt, {"ok": False})
    with pytest.raises(FileExistsError):
        recovery.record(receipt, {"ok": True})
    assert recovery.read(receipt) == {"ok": False}


def test_replay_requires_matching_archived_maintainer_code(tmp_path):
    bundle = tmp_path / "bundle"
    archive = bundle / "source/autotem-source.zip"
    archive.parent.mkdir(parents=True)
    exe = bundle / "AutoTEM-Setup.exe"
    exe.write_bytes(b"test launcher")

    def issue(driver, helper):
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("tools/setup/verify_upgrade.py", helper)
            if driver is not None:
                z.writestr("tools/setup/verify_recovery.py", driver)
        raw(
            bundle / "release-manifest.json",
            {
                "release_id": "fixture",
                "package_version": "0.1.0",
                "profiles": {recovery.PROFILE: {}},
                "artifacts": [{"path": exe.name, "sha256": recovery.sha(exe)}],
                "source": {
                    "archive": "source/autotem-source.zip",
                    "archive_sha256": recovery.sha(archive),
                },
            },
        )

    issue(None, recovery.HELPER.read_bytes())
    with pytest.raises(ValueError, match="missing_from_release"):
        recovery.identity(bundle, False)
    assert recovery.identity(bundle, True)[1] == "development_after_release_freeze"
    issue(recovery.HERE.read_bytes(), recovery.HELPER.read_bytes())
    assert recovery.identity(bundle, False)[1] == "release_source_archive"
    issue(b"changed verifier", recovery.HELPER.read_bytes())
    with pytest.raises(ValueError, match="changed_after_freeze"):
        recovery.identity(bundle, True)
    issue(recovery.HERE.read_bytes(), b"wrong scientific helper")
    with pytest.raises(ValueError, match="wrong_supporting_verifier"):
        recovery.identity(bundle, False)


@pytest.fixture
def receipts(tmp_path):
    out, project = tmp_path / "evidence", tmp_path / "project"
    rel = {"runtime_id": "rt-" + "a" * 20, "manifest_sha256": "a" * 64, "launcher_sha256": "b" * 64}
    states = {
        "historical_install": "host_setup_required",
        "rollback_write_fault": "blocked",
        "historical_rollback": "rolled_back",
        "protected_uninstall": "uninstalled",
        "blocked_activation": "blocked",
        "retry": "host_setup_required",
    }
    for action, state in states.items():
        succeeded = action not in {"blocked_activation", "rollback_write_fault"}
        prefix = out / "native" / action
        recovery.write(prefix.with_suffix(".stderr"), b"")
        raw(
            prefix.with_suffix(".process.json"),
            {
                "exit_code": 0 if succeeded else 2,
                "launcher_sha256": rel["launcher_sha256"],
                "project": str(project),
                "flags": list(recovery.ACTIONS[action]),
            },
        )
        note = {
            "ok": succeeded,
            "state": state,
            "profile": recovery.PROFILE,
            "release_ready": False,
            "manifest_sha256": rel["manifest_sha256"],
            "runtime_id": rel["runtime_id"],
            "checks": sorted(recovery.READY_CHECKS),
            "runtime": "passed",
            "local_transport": "passed",
            "local_mcp": "passed",
            "host_assets": "passed" if succeeded else "not_evaluated",
            "host_journey": "not_evaluated",
            "host_connection": "not_evaluated",
        }
        if succeeded:
            note["activation"] = state
        elif action == "rollback_write_fault":
            note["error"] = {"code": "INSTALLATION_FAILED", "message": recovery.LEASE_FAULT}
        else:
            message = (
                "项目数据库正被占用（SQLite: database is locked）。请保留项目文件，"
                "等待占用此项目的写入结束，然后在同一项目重试刚才的操作。"
            )
            note["error"] = {"code": "PROJECT_DATABASE_BUSY", "message": message}
            note["next_action"] = message
        raw(prefix.with_suffix(".stdout.json"), note)
    return out, project, rel


@pytest.mark.parametrize(
    ("file_suffix", "changes", "gate"),
    [
        ("stdout", {"ok": True}, "native_result"),
        (
            "stdout",
            {"error": {"code": "VENV_FAILED", "message": "database is locked"}},
            "write_lock_failure",
        ),
        (
            "stdout",
            {"error": {"code": "PROJECT_DATABASE_BUSY", "message": "disk full"}},
            "write_lock_failure",
        ),
        (
            "stdout",
            {"error": {"code": "INSTALLATION_FAILED", "message": "database is locked"}},
            "write_lock_failure",
        ),
        ("stdout", {"next_action": "database is locked"}, "actionable_same_project_retry"),
        ("stdout", {"next_action": "请删除项目后重试"}, "actionable_same_project_retry"),
        ("stdout", {"activation": "host_setup_required"}, "write_lock_failure"),
        ("stdout", {"host_assets": "passed"}, "write_lock_failure"),
        ("stdout", {"checks": []}, "staged_runtime"),
        ("stdout", {"manifest_sha256": "e" * 64}, "native_result"),
        ("process", {"exit_code": 0}, "native_result"),
        ("process", {"project": "unrelated"}, "native_result"),
        ("process", {"flags": ["--rollback", "--apply"]}, "native_result"),
    ],
)
def test_requires_native_staged_but_unactivated_failure(receipts, file_suffix, changes, gate):
    out, project, rel = receipts
    assert recovery.native_evidence(out, project, rel)["blocked_activation"]["ok"] is False
    path = out / "native" / f"blocked_activation.{file_suffix}.json"
    raw(path, {**recovery.read(path), **changes})
    with pytest.raises(ValueError, match=gate):
        recovery.native_evidence(out, project, rel)


def test_saves_inactive_observation_without_writing_to_scientific_database(tmp_path):
    project = tmp_path / "project"
    db = project / ".autotem/runs/harness.sqlite3"
    db.parent.mkdir(parents=True)
    recovery.write(project / ".codex/config.toml", recovery.up.USER_CONFIG)
    with sqlite3.connect(db) as connection:
        for table in ("runs", "jobs"):
            connection.execute(f"CREATE TABLE {table} (run_id, job_id, status)")
            connection.execute(f"INSERT INTO {table} VALUES ('old','job','completed')")
    rel = {"runtime_id": "rt-" + "8" * 20}
    before = recovery.sha(db)
    obs = recovery.status(project, rel)
    assert obs["active_pointer"] is obs["runtime_marker"] is obs["adapter_sha256"] is None
    assert obs["completed"] == ["old"]
    assert obs["user_config_sha256"] == hashlib.sha256(recovery.up.USER_CONFIG).hexdigest()
    assert recovery.sha(db) == before
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE jobs SET status='running'")
    with pytest.raises(ValueError, match="active_jobs"):
        recovery.status(project, rel)


def test_actual_scoped_sqlite_busy_clears_after_transaction_without_changing_history(tmp_path):
    database = tmp_path / "project.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE history (id)")
        db.execute("INSERT INTO history VALUES (123)")
    before = recovery.sha(database)
    with sqlite3.connect(database) as db:
        db.execute("BEGIN IMMEDIATE")
        locked = recovery.competing_writer(database)
        assert locked["result"] == "busy" and "database is locked" in locked["error"].lower()
        assert db.execute("SELECT id FROM history").fetchall() == [(123,)]
        db.rollback()
    assert recovery.competing_writer(database) == {"result": "acquired_and_rolled_back"}
    assert recovery.sha(database) == before


def test_pass_string_alone_cannot_substitute_for_native_records_or_store(monkeypatch, tmp_path):
    rel = {"manifest_sha256": "e" * 64}
    monkeypatch.setattr(
        recovery, "identity", lambda bundle, development: (rel, "release_source_archive")
    )
    path = tmp_path / "recovery.json"
    summary = {
        "schema_version": "1.0.0",
        "case_verdict": "passed",
        "release_manifest_sha256": rel["manifest_sha256"],
        "verifier_sha256": recovery.sha(recovery.HERE),
        "supporting_verifier_sha256": recovery.sha(recovery.HELPER),
        "verifier_binding": "release_source_archive",
        "release_ready": False,
        "host_journey": "not_evaluated",
        "scientific_accuracy": "not_evaluated",
    }
    raw(path, summary)
    with pytest.raises(FileNotFoundError):
        recovery.recheck(path, tmp_path / "bundle", tmp_path / "project")
    raw(path, {**summary, "release_manifest_sha256": "f" * 64})
    with pytest.raises(ValueError, match="invalid_recovery_certificate"):
        recovery.recheck(path, tmp_path / "bundle", tmp_path / "project")


@pytest.mark.parametrize("identifier", ["..", "../elsewhere", "", "a" * 33, "Q" * 32, None])
def test_invalid_transaction_cannot_select_another_project(tmp_path, identifier):
    with pytest.raises(ValueError, match="invalid_transaction_id"):
        recovery.journal(tmp_path, identifier)
