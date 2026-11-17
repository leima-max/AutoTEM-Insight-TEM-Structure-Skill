"""Project connector defaults and ownership during the legacy startup migration."""

import json
import sys

import pytest
from test_release_activation import release_project  # noqa: F401
from test_release_host_reconnect import report_file

from autotem.harness import hosts
from autotem.installation import managed

CONFIG = ".codex/config.toml"
ORIGINAL_CONFIG = (
    b'# project settings\r\nmodel = "example"\r\n[mcp_servers.other]\r\ncommand = "other"'
)


def legacy_config(project):
    command = json.dumps(sys._base_executable, ensure_ascii=False)
    args = json.dumps(["-I", str(project / ".autotem/host-entry.py"), "mcp"], ensure_ascii=False)
    cwd = json.dumps(str(project), ensure_ascii=False)
    section = (
        f"\n\n[mcp_servers.autotem]\ncommand = {command}\nargs = {args}\n"
        f"cwd = {cwd}\nstartup_timeout_sec = 60\n"
    )
    return ORIGINAL_CONFIG + section.encode()


def read_config(path):
    return hosts._read_toml(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("managed_install", [False, True])
def test_codex_connector_waits_at_startup_and_reinstall_is_idempotent(tmp_path, managed_install):
    project = tmp_path / "project"
    config = project / CONFIG
    config.parent.mkdir(parents=True)
    config.write_bytes(ORIGINAL_CONFIG)
    options = dict(
        dry_run=False, with_mcp=True, managed=managed_install, harness_root=".autotem/runs"
    )
    installed = hosts.install_host(project, "codex", **options)
    assert installed["ok"] and installed["installation_id"]
    installed_config = config.read_bytes()
    parsed = read_config(config)
    entry = parsed["mcp_servers"]["autotem"]
    assert entry["required"] is True
    assert entry["startup_timeout_sec"] == 180
    assert parsed["mcp_servers"]["other"] == {"command": "other"}
    assert parsed["model"] == "example"
    assert installed_config.startswith(ORIGINAL_CONFIG)

    repeated = hosts.install_host(project, "codex", **options)
    assert repeated["ok"] and repeated["installation_id"] is None
    assert config.read_bytes() == installed_config
    assert hosts.rollback_host(project, installed["installation_id"], dry_run=False)["ok"]
    assert config.read_bytes() == ORIGINAL_CONFIG


@pytest.fixture
def owned_legacy_config(release_project, monkeypatch):  # noqa: F811
    root, _ = release_project
    config = root / CONFIG
    config.parent.mkdir()
    config.write_bytes(ORIGINAL_CONFIG)
    expected = legacy_config(root)
    planned_files = hosts._planned_files

    def old_version(*args, **kwargs):
        # Model the old release's config bytes while retaining the real transaction and receipt.
        files = planned_files(*args, **kwargs)
        assert CONFIG in files
        return {**files, CONFIG: expected}

    with monkeypatch.context() as patch:
        patch.setattr(hosts, "_planned_files", old_version)
        first = managed.manage(root, "first", apply=True)
    assert first["ok"], first.get("error")
    assert config.read_bytes() == expected
    return root, config, expected, first


@pytest.mark.parametrize("reconnect", [False, True])
def test_managed_codex_migrates_legacy_config_and_restores_it_on_rollback(
    owned_legacy_config, reconnect
):
    root, config, old_config, first = owned_legacy_config
    pointer = root / ".autotem/active-runtime.json"
    old_pointer = pointer.read_bytes()
    upgraded = managed.manage(root, "second", apply=True)
    assert upgraded["ok"], upgraded.get("error")
    entry = read_config(config)["mcp_servers"]["autotem"]
    assert entry["required"] is True
    assert entry["startup_timeout_sec"] == 180
    assert config.read_bytes().startswith(ORIGINAL_CONFIG)
    assert upgraded["installation_id"] != first["installation_id"]
    if reconnect:
        report, _ = report_file(
            root,
            upgraded,
            "connect",
            error_code="MCP_STARTUP_TIMEOUT",
            message="AutoTEM MCP startup timed out.",
        )
        reverted = managed.manage(root, "second", host_reconnect_failure=report)
        assert reverted["host_connection"] == "reported_failed"
        assert reverted["restored_host_connection"] == "not_evaluated"
        assert reverted["release_ready"] is False
    else:
        reverted = managed.manage(root, "second", apply=True, rollback=True)
    assert reverted["ok"] and reverted["state"] == "rolled_back"
    assert pointer.read_bytes() == old_pointer
    assert config.read_bytes() == old_config


@pytest.mark.parametrize(
    "change", [b"startup_timeout_sec = 120\n", b"startup_timeout_sec = 60\nrequired = false\n"]
)
def test_managed_upgrade_refuses_user_startup_setting_changes(owned_legacy_config, change):
    root, config, old_config, _ = owned_legacy_config
    pointer = root / ".autotem/active-runtime.json"
    old_pointer = pointer.read_bytes()
    edited = old_config.replace(b"startup_timeout_sec = 60\n", change)
    config.write_bytes(edited)
    blocked = managed.manage(root, "second", apply=True)
    assert blocked["error"]["code"] == "HOST_FILE_CONFLICT"
    issue = blocked["conflict_resolution"]["files"][0]
    assert issue["path"] == CONFIG and issue["condition"] == "recorded_file"
    assert (root / issue["comparison_path"]).read_bytes() == old_config
    assert config.read_bytes() == edited and pointer.read_bytes() == old_pointer


def test_first_managed_install_refuses_matching_legacy_user_entry(release_project):  # noqa: F811
    root, _ = release_project
    config = root / CONFIG
    config.parent.mkdir()
    owned_by_user = legacy_config(root)
    config.write_bytes(owned_by_user)
    blocked = managed.manage(root, "first", apply=True)
    assert blocked["error"]["code"] == "HOST_FILE_CONFLICT"
    assert config.read_bytes() == owned_by_user
    assert not (root / ".autotem/active-runtime.json").exists()


def test_codex_installer_protects_user_entry_with_nonboolean_required(tmp_path):
    config = tmp_path / CONFIG
    config.parent.mkdir()
    # The full legacy launch command matches. Only `required` has an invalid TOML type.
    user_config = legacy_config(tmp_path).replace(
        b"startup_timeout_sec = 60\n", b"required = 1\nstartup_timeout_sec = 180\n"
    )
    config.write_bytes(user_config)
    blocked = hosts.install_host(
        tmp_path,
        "codex",
        dry_run=False,
        managed=True,
        with_mcp=True,
        harness_root=".autotem/runs",
    )
    assert not blocked["ok"] and blocked["conflicts"] == [CONFIG]
    assert config.read_bytes() == user_config


@pytest.mark.parametrize("host", ["openclaw", "workbuddy"])
def test_other_host_install_leaves_codex_config_and_mcp_proposal_unchanged(tmp_path, host):
    config = tmp_path / CONFIG
    config.parent.mkdir()
    config.write_bytes(ORIGINAL_CONFIG)
    installed = hosts.install_host(
        tmp_path, host, dry_run=False, managed=True, harness_root=".autotem/runs"
    )
    assert installed["ok"]
    assert config.read_bytes() == ORIGINAL_CONFIG
    adapter = json.loads((tmp_path / f"autotem-host/{host}/adapter.json").read_bytes())
    assert set(adapter["mcp"]["mcpServers"]["autotem"]) == {"command", "args"}
