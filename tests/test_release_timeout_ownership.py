"""Recover a failed cold MCP probe while preserving the original file-ownership boundary."""

import hashlib
import json
from pathlib import Path

import pytest
import test_release_activation

from autotem import bootstrap
from autotem.installation import managed, uninstall

pytestmark = pytest.mark.smoke
release_project = test_release_activation.release_project

TRUSTED = [
    "dependency_install_passed",
    "installed_autotem_wheel_bytes_passed",
    "private_package_provenance_passed",
    "pip_check_passed",
    "installed_profile_versions_and_origins_passed",
    "capabilities_passed",
]


def install_timeout(project, monkeypatch, *, missing_gate=False, wrong_wheel=False):
    real = bootstrap._provision
    checks = TRUSTED[:-1] if missing_gate else TRUSTED
    error = {"code": "MCP_HANDSHAKE_FAILED", "message": "fixture MCP timeout"}

    def timeout(*args, **kwargs):
        staged = real(*args, **kwargs)
        wheel = kwargs["release"].wheel
        wheel.write_bytes(b"verified fixture wheel")
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        runtime = Path(staged["python_executable"]).parent.parent
        pkg = runtime / "Lib/site-packages/sample"
        pkg.mkdir(parents=True)
        (pkg / "science.py").write_text("value = 1\n", encoding="utf-8")
        records = pkg.parent / "sample-1.dist-info"
        records.mkdir()
        (records / "RECORD").write_text(
            "sample/science.py,,\nsample-1.dist-info/RECORD,,\n", encoding="utf-8"
        )
        marker = runtime / bootstrap.MARKER_NAME
        data = json.loads(marker.read_bytes())
        data.update(
            schema_version=bootstrap.BOOTSTRAP_VERSION,
            state="failed",
            error=error,
            checks=checks,
            wheel_sha256=digest,
            with_mcp=True,
        )
        bootstrap._atomic_json(marker, data)
        if wrong_wheel:
            wheel.write_bytes(b"changed wheel")
        return {
            **staged,
            "ok": False,
            "checks": checks,
            "error": error,
            "wheel_sha256": digest,
            "readiness": {
                "runtime": {
                    "status": "ready",
                    "local_mcp": {
                        "status": "blocked", "code": "MCP_PROBE_TIMEOUT",
                        "diagnostic": {"stage": "tools/call", "last_response": "tools/list"},
                    },
                }
            },
        }

    with monkeypatch.context() as once:
        once.setattr(bootstrap, "_provision", timeout)
        failed = managed.manage(project, "release", apply=True)
    assert not failed["ok"] and failed["error"]["code"] == "MCP_HANDSHAKE_FAILED"
    assert not (project / ".autotem/active-runtime.json").exists()
    return failed


def test_verified_timeout_remains_removable_after_transport_retry(release_project, monkeypatch):
    project, staged = release_project
    called = []
    original = uninstall.record_new_base_caches

    def observe(control, record, before, **kwargs):
        called.append((record["state"], kwargs["fresh"]))
        return original(control, record, before, **kwargs)

    monkeypatch.setattr(uninstall, "record_new_base_caches", observe)
    failed = install_timeout(project, monkeypatch)
    assert failed["staged_file_ownership"] == "recorded" and ("failed", True) in called
    control = project / ".autotem"
    txn = managed._record_path(control, failed["installation_id"])
    inventory = txn.with_name(uninstall.INVENTORY)
    before = inventory.read_bytes()
    assert (
        json.loads(txn.read_bytes())["runtime_inventory_sha256"]
        == hashlib.sha256(before).hexdigest()
    )
    assert managed.manage(project, "release", apply=True)["ok"]
    assert inventory.read_bytes() == before
    removed = managed.manage(project, "release", apply=True, uninstall=True)
    assert removed["state"] == "uninstalled", removed
    assert not staged[0].parent.parent.exists()


def test_user_changes_between_failure_and_repair_are_not_adopted(release_project, monkeypatch):
    project, staged = release_project
    assert install_timeout(project, monkeypatch)["staged_file_ownership"] == "recorded"
    root = staged[0].parent.parent
    added = root / "my-notes.raw"
    changed = root / "Lib/site-packages/sample/science.py"
    added.write_bytes(b"my raw evidence")
    changed.write_text("value = 2\n", encoding="utf-8")
    assert managed.manage(project, "release", apply=True)["ok"]
    result = managed.manage(project, "release", apply=True, uninstall=True)
    assert result["error"]["code"] == "UNINSTALL_RETAINED_FILES"
    assert result["retained"][str(added)] == "unregistered_file"
    assert result["retained"][str(changed)] == "changed"
    assert added.read_bytes() == b"my raw evidence" and changed.read_text() == "value = 2\n"


def test_incomplete_inventory_preserves_original_transport_diagnosis(release_project, monkeypatch):
    project, _ = release_project

    def interrupted_inventory(*args, **kwargs):
        raise OSError("Cannot finish the local receipt")

    monkeypatch.setattr(uninstall, "record_new_runtime", interrupted_inventory)
    failure = install_timeout(project, monkeypatch)
    assert failure["staged_file_ownership"] == "not_established"
    txn = managed._record_path(project / ".autotem", failure["installation_id"])
    stored = json.loads(txn.read_bytes())
    assert "runtime_inventory_sha256" not in stored
    assert failure["runtime_diagnostic"]["local_mcp"]["code"] == "MCP_PROBE_TIMEOUT"
    assert stored["runtime_diagnostic"] == failure["runtime_diagnostic"]
    assert stored["runtime_diagnostic"]["local_mcp"]["diagnostic"]["stage"] == "tools/call"


@pytest.mark.parametrize("options", [{"missing_gate": True}, {"wrong_wheel": True}])
def test_unproven_first_installation_is_not_retroactively_owned(
    release_project, monkeypatch, options
):
    project, staged = release_project
    failure = install_timeout(project, monkeypatch, **options)
    assert failure["staged_file_ownership"] == "not_established"
    txn = managed._record_path(project / ".autotem", failure["installation_id"])
    assert "runtime_inventory_sha256" not in json.loads(txn.read_bytes())
    assert managed.manage(project, "release", apply=True)["ok"]
    removed = managed.manage(project, "release", apply=True, uninstall=True)
    assert removed["retained"][str(staged[0].parent.parent)] == "no_verified_file_inventory"
    assert staged[0].is_file()
