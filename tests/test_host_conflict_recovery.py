"""User-visible conflict recovery through managed transactions and real project files."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import test_release_activation

from autotem.harness.hosts import install_host, rollback_host
from autotem.installation import host_conflict, managed

pytestmark = pytest.mark.smoke
release_project = test_release_activation.release_project
ADAPTER = "autotem-host/codex/adapter.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def conflict(result, operation, name):
    assert not result["ok"] and result["state"] == "blocked"
    assert result["error"]["code"] == "HOST_FILE_CONFLICT"
    info = result["conflict_resolution"]
    assert info["operation"] == operation
    assert [row["id"] for row in info["choices"]] == ["choose_new_project", "restore_then_retry"]
    assert "空项目" in result["next_action"] and "私人备份" in result["next_action"]
    return next(r for r in info["files"] if r["path"] == name)


def test_first_occupied_skill_keeps_user_file_and_exposes_verified_candidate(release_project):  # noqa: F811
    root, _ = release_project
    name = ".agents/skills/autotem/SKILL.md"
    owned = root / name
    owned.parent.mkdir(parents=True)
    owned.write_bytes(b"my private skill")
    result = managed.manage(root, "first", apply=True)
    row = conflict(result, "install", name)
    assert row["current_sha256"] == digest(b"my private skill")
    comparison = root / row["comparison_path"]
    assert row["expected_sha256"] == row["comparison_sha256"] == digest(comparison.read_bytes())
    assert row["condition"] == "available_path" and row["receipt_path"] is None
    assert owned.read_bytes() == b"my private skill"
    assert not (root / ".autotem/active-runtime.json").exists()
    owned.rename(root / "private-skill.md")
    assert managed.manage(root, "first", apply=True)["ok"]
    assert (root / "private-skill.md").read_bytes() == b"my private skill"


def test_first_toml_conflict_offers_only_standalone_managed_section(release_project):  # noqa: F811
    root, _ = release_project
    name = ".codex/config.toml"
    config = root / name
    config.parent.mkdir()
    raw = b'private_key="secret-first-file"\n[mcp_servers.autotem]\ncommand="user-server"\n'
    config.write_bytes(raw)
    result = managed.manage(root, "first", apply=True)
    row = conflict(result, "install", name)
    assert row["condition"] == "compatible_toml" and row["expected_sha256"] is None
    assert row["current_sha256"] == digest(raw) and row["comparison_kind"] == "managed_toml_section"
    section = (root / row["comparison_path"]).read_bytes()
    assert digest(section) == row["comparison_sha256"] and b"[mcp_servers.autotem]" in section
    assert b"secret-first-file" not in section and b"user-server" not in section
    assert "secret-first-file" not in json.dumps(result) and config.read_bytes() == raw


def test_upgrade_and_rollback_can_be_retried_from_their_own_after_images(release_project):  # noqa: F811
    root, _ = release_project
    assert managed.manage(root, "first", apply=True)["ok"]
    pointer = root / ".autotem/active-runtime.json"
    first = pointer.read_bytes()
    asset = root / ADAPTER
    before = asset.read_bytes()
    raw = root / ".autotem/runs/existing-user-raw.npy"
    raw.write_bytes(b"previous science")
    asset.write_bytes(b"private edit")
    row = conflict(managed.manage(root, "second", apply=True), "upgrade", ADAPTER)
    assert row["expected_sha256"] == digest(before) and row["current_sha256"] == digest(
        b"private edit"
    )
    assert row["receipt_path"] and row["comparison_kind"] == "installed_file"
    assert pointer.read_bytes() == first and asset.read_bytes() == b"private edit"
    shutil.copyfile(root / row["comparison_path"], asset)
    assert managed.manage(root, "second", apply=True)["ok"]
    second, updated = pointer.read_bytes(), asset.read_bytes()
    assert second != first and before != updated
    asset.write_bytes(b"next private edit")
    row = conflict(managed.manage(root, "second", apply=True, rollback=True), "rollback", ADAPTER)
    assert (root / row["comparison_path"]).read_bytes() == updated
    assert row["expected_sha256"] == row["comparison_sha256"] == digest(updated)
    assert pointer.read_bytes() == second and asset.read_bytes() == b"next private edit"
    shutil.copyfile(root / row["comparison_path"], asset)
    assert managed.manage(root, "second", apply=True, rollback=True)["ok"]
    assert pointer.read_bytes() == first and asset.read_bytes() == before
    assert raw.read_bytes() == b"previous science"


@pytest.mark.parametrize("reference", ["legacy", "before_backup", "damaged"])
def test_legacy_or_damaged_copy_never_presents_before_as_after(release_project, reference):  # noqa: F811
    root, _ = release_project
    assert managed.manage(root, "first", apply=True)["ok"]
    assert managed.manage(root, "second", apply=True)["ok"]
    pointer = root / ".autotem/active-runtime.json"
    active = pointer.read_bytes()
    key = json.loads(active)["host_installations"]["codex"]
    directory = root / ".autotem/host-installations" / key
    receipt = directory / "manifest.json"
    record = json.loads(receipt.read_bytes())
    asset_row = next(r for r in record["files"] if r["path"] == ADAPTER)
    assert digest((directory / asset_row["backup"]).read_bytes()) == asset_row["before_sha256"]
    assert asset_row["before_sha256"] != asset_row["after_sha256"]
    if reference == "legacy":
        del asset_row["after_copy"]
    elif reference == "before_backup":
        asset_row["after_copy"] = asset_row["backup"]
    else:
        (directory / asset_row["after_copy"]).write_bytes(b"damaged")
    receipt.write_text(json.dumps(record), encoding="utf-8")
    (root / ADAPTER).write_bytes(b"changed")
    res = managed.manage(root, "second", apply=True, rollback=True)
    row = conflict(res, "rollback", ADAPTER)
    assert row["expected_sha256"] == asset_row["after_sha256"]
    assert row["comparison_path"] is None and row["comparison_sha256"] is None
    assert "没有可核验副本" in res["next_action"] and "backups/" not in res["next_action"]
    assert pointer.read_bytes() == active and (root / ADAPTER).read_bytes() == b"changed"


def test_missing_active_resource_is_not_assumed_to_be_an_interrupted_rollback(release_project):  # noqa: F811
    root, _ = release_project
    assert managed.manage(root, "first", apply=True)["ok"]
    pointer = root / ".autotem/active-runtime.json"
    first = pointer.read_bytes()
    (root / ADAPTER).unlink()
    failed = managed.manage(root, "first", apply=True, rollback=True)
    row = conflict(failed, "rollback", ADAPTER)
    assert row["current_state"] == "missing" and row["current_sha256"] is None
    assert pointer.read_bytes() == first
    row = conflict(managed.manage(root, "second", apply=True), "upgrade", ADAPTER)
    assert row["current_state"] == "missing" and row["condition"] == "recorded_file"
    shutil.copyfile(root / row["comparison_path"], root / ADAPTER)
    assert managed.manage(root, "first", apply=True, rollback=True)["ok"]


def test_private_merged_toml_is_preserved_and_comparison_stays_project_local(release_project):  # noqa: F811
    root, _ = release_project
    config = root / ".codex/config.toml"
    config.parent.mkdir()
    config.write_bytes(b'private_key="pre-existing-secret"\n')
    assert managed.manage(root, "first", apply=True)["ok"]
    expected = config.read_bytes()
    edited = expected + b"\n# second-private-token\n"
    config.write_bytes(edited)
    res = managed.manage(root, "second", apply=True)
    row = conflict(res, "upgrade", ".codex/config.toml")
    assert row["current_sha256"] == digest(edited)
    assert row["comparison_kind"] == "installed_project_config"
    assert Path(row["comparison_path"]).parts[0] == ".autotem"
    assert (root / row["comparison_path"]).read_bytes() == expected
    assert config.read_bytes() == edited
    assert all(s not in json.dumps(res) for s in ("pre-existing-secret", "second-private-token"))


def test_project_path_limit(tmp_path):
    root = tmp_path / ("p" * max(1, 166 - len(str(tmp_path))))
    root.mkdir()
    key = "install-" + "1" * 32
    base = root / ".autotem/host-installations" / key
    old_path = base / "after/0002-autotem_atomic_strain.toml"
    assert len(str(old_path)) >= 260
    result = install_host(root, "codex", installation_id=key, dry_run=False)
    assert result["ok"]
    receipt = json.loads((base / "manifest.json").read_bytes())
    for i, row in enumerate(receipt["files"]):
        reference = base / row["after_copy"]
        assert row["after_copy"] == f"a/{i:04d}"
        assert len(str(reference)) < len(str(base / "manifest.json"))
        assert row["after_sha256"] == digest(reference.read_bytes())
    asset = root / ADAPTER
    asset.write_bytes(b"user-edited")
    answer = rollback_host(root, key, dry_run=False)
    row = next(item for item in answer["conflict_resolution"]["files"] if item["path"] == ADAPTER)
    assert asset.read_bytes() == b"user-edited"
    shutil.copyfile(root / row["comparison_path"], asset)
    assert rollback_host(root, key, dry_run=False)["ok"]


def test_directory_in_place_of_toml_is_protected(release_project):  # noqa: F811
    root, _ = release_project
    name = ".codex/config.toml"
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "private").write_bytes(b"private")
    result = managed.manage(root, "first", apply=True)
    row = conflict(result, "install", name)
    assert row["current_state"] == "directory" and row["current_sha256"] is None
    assert "移到" in result["next_action"] and "文本编辑器" not in result["next_action"]
    assert (directory / "private").read_bytes() == b"private"


def test_real_cli_conflict(tmp_path):
    root = tmp_path / "中文 项目"
    name = ".agents/skills/autotem/SKILL.md"
    occupied = root / name
    occupied.parent.mkdir(parents=True)
    occupied.write_bytes(b"private edits")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "autotem.harness.cli",
            "--root",
            str(root / ".autotem/runs"),
            "install-host",
            "codex",
            "--project",
            str(root),
            "--managed",
            "--apply",
        ],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 2, proc.stderr
    payload = json.loads(proc.stdout)
    row = host_conflict.child_conflicts(root, "codex", None, payload)["conflict_resolution"][
        "files"
    ][0]
    assert row["path"] == name and row["current_sha256"] == digest(b"private edits")
    assert row["expected_sha256"] == row["comparison_sha256"]
    assert row["comparison_sha256"] == digest((root / row["comparison_path"]).read_bytes())
    assert b"private edits" not in proc.stdout and occupied.read_bytes() == b"private edits"


def test_identical_user_file_is_never_adopted_during_upgrade(tmp_path, monkeypatch):
    from autotem.harness import hosts

    original = hosts._resource_files()
    name = ".agents/skills/autotem/SKILL.md"
    skill = tmp_path / name
    skill.parent.mkdir(parents=True)
    skill.write_bytes(original["SKILL.md"])
    first = install_host(tmp_path, "codex", dry_run=False)
    monkeypatch.setattr(hosts, "_resource_files", lambda: {**original, "SKILL.md": b"new version"})
    later = install_host(
        tmp_path, "codex", dry_run=False, previous_installation=first["installation_id"]
    )
    row = next(r for r in later["conflict_resolution"]["files"] if r["path"] == name)
    assert not later["ok"] and row["condition"] == "available_path"
    assert row["expected_sha256"] == digest(b"new version") and row["receipt_path"] is None
    assert skill.read_bytes() == original["SKILL.md"]
