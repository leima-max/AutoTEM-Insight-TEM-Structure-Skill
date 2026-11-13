"""Fault injection for activation, runtime binding and rollback of managed releases."""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from autotem import bootstrap
from autotem.harness import HarnessError
from autotem.harness.hosts import install_host
from autotem.harness.runtime_binding import selected_runtime
from autotem.harness.store import Store
from autotem.installation import managed
from autotem.installation.release import PROFILE, VerifiedRelease

pytestmark = pytest.mark.smoke


@pytest.fixture
def release_project(tmp_path, monkeypatch):
    root = tmp_path / "selected project"
    root.mkdir()
    wheelhouse = tmp_path / "pool"
    wheelhouse.mkdir()
    staged = []

    def verify(bundle, profile):
        issued = {"cli": "traditional-cli", "sim": "traditional-simulation"}.get(
            bundle.split("-")[0], PROFILE
        )
        assert profile is None or profile == issued
        return VerifiedRelease(
            wheelhouse / "test.whl",
            wheelhouse,
            tmp_path / "lock",
            bundle,
            hashlib.sha256(bundle.encode()).hexdigest(),
            issued != "traditional-cli",
            "25.0.1",
            issued,
        )

    def provision(project, wheel, *, release, **options):
        control = project / ".autotem"
        runtime = control / "runtimes" / f"rt-{release.manifest_sha256[:20]}"
        exe = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        exe.parent.mkdir(exist_ok=True, parents=True)
        exe.write_bytes(b"test runtime")
        bootstrap._atomic_json(
            runtime / bootstrap.MARKER_NAME,
            {
                "project": str(project),
                "runtime": str(runtime),
                "managed_by": "autotem.bootstrap",
                "manifest_sha256": release.manifest_sha256,
                "profile": release.profile,
                "state": "ready",
            },
        )
        staged.append(exe)
        with Store(control / "runs").connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
        return {
            "python_executable": str(exe),
            "store_schema_version": version,
            "checks": ["stub"],
            "readiness": {},
            "ok": True,
        }

    def run(args, **options):
        if "doctor" in args:
            marker = Path(args[0]).parent.parent / bootstrap.MARKER_NAME
            profile = json.loads(marker.read_bytes())["profile"]
            status = "dependency_unavailable" if profile == "traditional-cli" else "ready"
            health = {"runtime": {"status": "ready", "local_mcp": {"status": status}}}
            return subprocess.CompletedProcess(args, 0, json.dumps(health), "")

        def arg(flag):
            return args[args.index(flag) + 1] if flag in args else None

        answer = install_host(
            root,
            arg("install-host"),
            managed=True,
            with_mcp="--with-mcp" in args,
            mcp_available="--without-mcp" not in args,
            python_executable=args[0],
            harness_root=root / ".autotem/runs",
            previous_installation=arg("--previous-installation"),
            installation_id=arg("--installation-id"),
            dry_run=False,
        )
        return subprocess.CompletedProcess(args, 0 if answer["ok"] else 2, json.dumps(answer), "")

    monkeypatch.setattr(managed, "verify_release", verify)
    monkeypatch.setattr(bootstrap, "_provision", provision)
    monkeypatch.setattr(bootstrap, "_run", run)
    return root, staged


@pytest.mark.parametrize("host", ["codex", "openclaw", "workbuddy"])
def test_cli_edition_activates_project_commands_without_a_connector(release_project, host):
    root, _ = release_project
    result = managed.manage(root, "cli-first", apply=True, host=host)
    assert result["ok"] and result["state"] == "local_cli_ready"
    assert result["transport"] == "jsonl" and result["host_connection"] == "not_evaluated"
    adapter = json.loads((root / f"autotem-host/{host}/adapter.json").read_bytes())
    assert adapter["preferred_transport"] == "jsonl" and "mcp" not in adapter
    assert adapter["cli_command"][-1] == str(root / ".autotem/host-entry.py")
    assert adapter["jsonl_command"] == [*adapter["cli_command"], "stdio"]
    assert not (root / ".codex/config.toml").exists()
    action = result["next_action"]
    assert f"autotem-host/{host}/adapter.json" in action
    assert "cli_command" in action and "jsonl_command" in action
    assert "mcp" not in action.lower() and "授权" not in action
    repaired = managed.manage(root, "cli-first", apply=True, host=host)
    assert repaired["activation"] == "already_active"
    assert repaired["next_action"] == action
    lower = install_host(
        root, host, dry_run=False, managed=True, mcp_available=False,
        harness_root=root / ".autotem/runs", python_executable=result["python_executable"],
    )
    assert lower["installation_id"] is None and lower["next_action"] == action


@pytest.mark.parametrize("bundle", ["first", "sim-first"])
def test_host_onboarding_uses_the_selected_hosts_real_files(release_project, bundle):
    root, _ = release_project
    first = {}
    # When Codex or OpenClaw is selected, WorkBuddy is still the last owner refreshed.
    for host in ("workbuddy", "codex", "openclaw", "codex", "workbuddy", "openclaw"):
        result = managed.manage(root, bundle, host=host, apply=True)
        action = result["next_action"]
        assert result["state"] == "host_setup_required"
        assert result["host_connection"] == "not_evaluated"
        assert action.index("授权") < action.index("list_capabilities")
        adapter_path = f"autotem-host/{host}/adapter.json"
        adapter = json.loads((root / adapter_path).read_bytes())
        server = adapter["mcp"]["mcpServers"]["autotem"]
        assert server["args"] == ["-I", str(root / ".autotem/host-entry.py"), "mcp"]
        if host == "codex":
            assert "信任" in action and ".codex/config.toml" in action
            assert "mcp_servers.autotem" in action
            assert "[mcp_servers.autotem]" in (root / ".codex/config.toml").read_text("utf-8")
        else:
            assert adapter_path in action and "mcp.mcpServers.autotem" in action
            if host == "workbuddy":
                path = "autotem-host/workbuddy/autotem-skill.zip"
                assert path in action
                with zipfile.ZipFile(root / path) as archive:
                    assert "SKILL.md" in archive.namelist()
            else:
                assert "工作区" in action and "网关运行正常" in action
                assert (root / "skills/autotem/SKILL.md").is_file()
        if host in first:
            assert result["activation"] == "already_active" and action == first[host]
        else:
            assert result["activation"] == "host_setup_required"
            first[host] = action
        lower = install_host(
            root, host, dry_run=False, managed=True, with_mcp=host == "codex",
            harness_root=root / ".autotem/runs", python_executable=result["python_executable"],
        )
        assert lower["installation_id"] is None and lower["next_action"] == action


def test_switching_profile_restores_exact_codex_config_and_rolls_back(release_project):
    root, _ = release_project
    config = root / ".codex/config.toml"
    config.parent.mkdir()
    original = b'# existing project settings\r\nmodel = "user-model"\r\n'
    config.write_bytes(original)
    core = managed.manage(root, "first", apply=True)
    with_mcp = config.read_bytes()
    assert b"[mcp_servers.autotem]" in with_mcp
    assert managed.manage(root, "second", apply=True)["ok"]
    cli = managed.manage(root, "cli-first", apply=True)
    assert cli["state"] == "local_cli_ready" and config.read_bytes() == original
    assert core["runtime_id"] != cli["runtime_id"]
    assert managed.manage(root, "third", apply=True)["state"] == "host_setup_required"
    assert managed.manage(root, "third", apply=True, rollback=True)["ok"]
    assert config.read_bytes() == original
    assert managed.manage(root, "cli-first", apply=True, rollback=True)["ok"]
    assert config.read_bytes() == with_mcp


def test_cli_switch_preserves_user_modified_codex_config(release_project):
    root, _ = release_project
    assert managed.manage(root, "first", apply=True)["ok"]
    pointer = root / ".autotem/active-runtime.json"
    old_pointer = pointer.read_bytes()
    config = root / ".codex/config.toml"
    edited = config.read_bytes() + b"\n# User settings changed\n"
    config.write_bytes(edited)
    result = managed.manage(root, "cli-first", apply=True)
    assert not result["ok"]
    assert config.read_bytes() == edited and pointer.read_bytes() == old_pointer


def test_busy_project_database_preserves_activation_and_explains_retry(release_project):
    root, _ = release_project
    assert managed.manage(root, "first", apply=True)["ok"]
    control = root / ".autotem"
    pointer = control / "active-runtime.json"
    previous = pointer.read_bytes()
    writer = sqlite3.connect(control / "runs/harness.sqlite3")
    try:
        writer.execute("BEGIN IMMEDIATE")
        blocked = managed.manage(root, "first", apply=True, rollback=True)
        assert blocked["state"] == "blocked" and not blocked["ok"]
        assert blocked["error"]["code"] == "PROJECT_DATABASE_BUSY"
        assert "database is locked" in blocked["error"]["message"]
        assert "等待" in blocked["next_action"] and "同一项目重试" in blocked["next_action"]
        assert pointer.read_bytes() == previous
    finally:
        writer.close()
    assert managed.manage(root, "first", apply=True, rollback=True)["state"] == "rolled_back"


def test_upgrade_switches_once_and_rollback_preserves_data_and_config(release_project):
    root, staged = release_project
    assert managed.manage(root, "first")["state"] == "planned"
    assert not list(root.iterdir())
    assert managed.manage(root, "first", apply=True)["state"] == "host_setup_required"
    pointer = root / ".autotem/active-runtime.json"
    first = pointer.read_bytes()
    config = root / ".codex/config.toml"
    host_config = config.read_bytes()
    data = root / ".autotem/runs/keep.raw"
    data.write_bytes(b"existing science")
    assert managed.manage(root, "first", apply=True)["activation"] == "already_active"
    assert pointer.read_bytes() == first
    assert managed.manage(root, "second", apply=True)["ok"]
    assert pointer.read_bytes() != first
    assert config.read_bytes() == host_config
    # An unusable promoted version must still allow rollback to its healthy predecessor.
    staged[-1].unlink()
    restored = managed.manage(root, "second", apply=True, rollback=True)
    assert restored["state"] == "rolled_back"
    assert pointer.read_bytes() == first
    assert config.read_bytes() == host_config
    assert staged[0].is_file()
    assert data.read_bytes() == b"existing science"


def test_conflicts_and_active_jobs_preserve_previous_pointer(release_project):
    root, _ = release_project
    assert managed.manage(root, "first", apply=True)["ok"]
    control = root / ".autotem"
    pointer = control / "active-runtime.json"
    before = pointer.read_bytes()
    store = Store(control / "runs")
    with store.connect() as db:
        db.execute(
            "INSERT INTO jobs(job_id, run_id, created, status) VALUES('test','run',0,'running')"
        )
    assert managed.manage(root, "second", apply=True)["error"]["code"] == "WAIT_ACTIVE_RUNS"
    with store.connect() as db:
        db.execute("UPDATE jobs SET status='completed' WHERE job_id='test'")
    asset = root / "autotem-host/codex/adapter.json"
    prior = asset.read_bytes()
    asset.write_text("user changed this", encoding="utf-8")
    result = managed.manage(root, "second", apply=True)
    assert result["error"]["code"] == "HOST_FILE_CONFLICT"
    assert "adapter.json" in result["error"]["message"]
    assert pointer.read_bytes() == before and asset.read_bytes() == b"user changed this"
    assert (
        managed.manage(root, "first", apply=True, rollback=True)["error"]["code"]
        == "HOST_FILE_CONFLICT"
    )
    assert pointer.read_bytes() == before
    asset.write_bytes(prior)
    assert managed.manage(root, "second", apply=True)["ok"]


def test_recover_after_pointer_commit_and_reject_stale_session(release_project, monkeypatch):
    root, staged = release_project
    first = managed.manage(root, "first", apply=True)
    actual_exe = staged[-1]
    control = root / ".autotem"
    path = managed._record_path(control, first["installation_id"])
    journal = json.loads(path.read_bytes())
    journal["state"] = "switching"
    bootstrap._atomic_json(path, journal)
    assert managed.manage(root, "first", apply=True)["activation"] == "already_active"
    assert json.loads(path.read_bytes())["state"] == "host_setup_required"
    with pytest.raises(HarnessError, match="Reconnect"):
        selected_runtime(control / "runs")
    monkeypatch.setattr(sys, "executable", str(actual_exe))
    monkeypatch.setattr(sys, "prefix", str(actual_exe.parent.parent))
    assert selected_runtime(control / "runs") == first["runtime_id"]
    assert selected_runtime(Path("other")) is None
    assert managed.manage(root, "first", apply=True, rollback=True)["ok"]
    with pytest.raises(HarnessError, match="Reconnect"):
        selected_runtime(control / "runs")


def test_refuses_missing_applied_receipt(release_project):
    root, _ = release_project
    assert managed.manage(root, "first", apply=True)["ok"]
    pointer = root / ".autotem/active-runtime.json"
    before = pointer.read_bytes()
    receipt = json.loads(before)["host_installations"]["codex"]
    (root / ".autotem/host-installations" / receipt / "manifest.json").unlink()
    denied = managed.manage(root, "first", apply=True, rollback=True)
    assert denied["error"]["code"] == "HOST_RECEIPT_MISSING"
    assert pointer.read_bytes() == before


def test_installed_source_drift_including_extra_files(tmp_path):
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("autotem/module.py", b"scale = 1")
    package = tmp_path / "Lib/site-packages/autotem"
    package.mkdir(parents=True)
    module = package / "module.py"
    module.write_bytes(b"scale = 1")
    bootstrap._verify_release_files(tmp_path, wheel)
    module.write_bytes(b"scale = 2")
    with pytest.raises(bootstrap.BootstrapError, match="differ"):
        bootstrap._verify_release_files(tmp_path, wheel)
    module.write_bytes(b"scale = 1")
    (package / "extra.py").write_bytes(b"# local")
    with pytest.raises(bootstrap.BootstrapError, match="differ"):
        bootstrap._verify_release_files(tmp_path, wheel)
