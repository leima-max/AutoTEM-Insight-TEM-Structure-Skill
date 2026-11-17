"""Exercise removal through the managed installer's real transaction and host-asset paths."""

import hashlib
import importlib.util
import json
import os
import py_compile
import shutil
import subprocess
import sys
import venv
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_release_activation

from autotem import bootstrap
from autotem.harness.store import Store
from autotem.installation import managed, uninstall

pytestmark = pytest.mark.smoke
release_project = test_release_activation.release_project


def apply(root, release="first", **flags):
    return managed.manage(root, release, apply=True, **flags)


def remove(root, release="first"):
    return apply(root, release, uninstall=True)


def test_project_uninstall_restores_multiple_hosts_and_preserves_science(release_project):  # noqa: F811
    root, staged = release_project
    config = root / ".codex/config.toml"
    config.parent.mkdir()
    original = b'model = "local"\r\n'
    config.write_bytes(original)
    assert apply(root)["ok"] and apply(root, "second", host="openclaw")["ok"]
    db = root / ".autotem/runs/harness.sqlite3"
    run = db.parent / "history"
    run.mkdir()
    files = [root / "original.raw", run / "report.html", run / "observations.sqlite3"]
    for item in files:
        item.write_bytes(b"scientific evidence")
    before = config.read_bytes()
    preview = managed.manage(root, "second", uninstall=True)
    assert preview["state"] == "planned" and preview["inventoried_runtimes"] == 2
    assert config.read_bytes() == before and all(path.exists() for path in staged)
    result = remove(root, "second")
    assert result["state"] == "uninstalled", result
    assert all(not path.exists() for path in staged)
    assert not (root / ".autotem/active-runtime.json").exists()
    assert not (root / "skills/autotem/SKILL.md").exists()
    assert config.read_bytes() == original and db.exists()
    assert all(item.read_bytes() == b"scientific evidence" for item in files)
    assert remove(root)["ok"]
    second_cycle = apply(root)
    assert second_cycle["ok"] and remove(root)["state"] == "uninstalled"
    completed = json.loads((root / ".autotem" / uninstall.JOURNAL).read_bytes())
    assert second_cycle["installation_id"] in completed["retired_installations"]


def test_retired_receipt_cannot_claim_recreated_runtime(release_project):  # noqa: F811
    root, staged = release_project
    assert apply(root)["ok"]
    exe = staged[-1]
    marker = exe.parent.parent / bootstrap.MARKER_NAME
    binary, marker_bytes = exe.read_bytes(), marker.read_bytes()
    assert remove(root)["state"] == "uninstalled"
    # A user restored this directory independently, even with the historical marker.
    exe.parent.mkdir(parents=True)
    exe.write_bytes(binary)
    marker.write_bytes(marker_bytes)
    denied = remove(root)
    assert denied["error"]["code"] == "UNINSTALL_RETAINED_FILES"
    assert denied["removed_file_count"] == 0 and exe.read_bytes() == binary


def test_missing_current_ownership_marker_preserves_whole_tree(release_project):  # noqa: F811
    root, staged = release_project
    assert apply(root)["ok"]
    exe = staged[-1]
    runtime = exe.parent.parent
    (runtime / bootstrap.MARKER_NAME).unlink()
    result = remove(root)
    assert result["retained"][str(runtime)] == "ownership_marker_missing_or_changed"
    assert exe.exists() and result["removed_file_count"] == 0


def test_changed_and_unregistered_runtime_files_survive_reinstall(release_project):  # noqa: F811
    root, staged = release_project
    assert apply(root)["ok"]
    exe = staged[-1]
    extra = exe.parent / "notes.raw"
    extra.write_bytes(b"my data")
    assert apply(root)["ok"]
    exe.write_bytes(b"changed after install")
    result = remove(root)
    assert not result["ok"] and result["error"]["code"] == "UNINSTALL_RETAINED_FILES"
    assert result["retained"][str(exe)] == "changed"
    assert result["retained"][str(extra)] == "unregistered_file"
    assert exe.read_bytes() == b"changed after install" and extra.read_bytes() == b"my data"
    assert apply(root)["error"]["code"] == "UNINSTALL_PENDING"
    exe.unlink()
    extra.unlink()
    assert remove(root)["state"] == "uninstalled"


def test_active_worker_and_changed_host_preserve_active_pointer(release_project):  # noqa: F811
    root, _ = release_project
    assert apply(root)["ok"]
    control = root / ".autotem"
    pointer = control / "active-runtime.json"
    active = pointer.read_bytes()
    with Store(control / "runs").connect() as db:
        db.execute(
            "INSERT INTO jobs(job_id, run_id, created, status) VALUES('job','run',0,'running')"
        )
    result = remove(root)
    assert result["error"]["code"] == "WAIT_ACTIVE_RUNS" and pointer.read_bytes() == active
    with Store(control / "runs").connect() as db:
        db.execute("UPDATE jobs SET status='completed' WHERE job_id='job'")
    asset = root / "autotem-host/codex/adapter.json"
    asset.write_text("local settings", encoding="utf-8")
    result = remove(root)
    assert result["error"]["code"] == "HOST_FILE_CONFLICT"
    assert asset.read_text() == "local settings" and pointer.read_bytes() == active
    assert not (control / uninstall.JOURNAL).exists()


def test_retained_configuration_then_live_sessions_block_deletion(release_project, monkeypatch):  # noqa: F811
    root, staged = release_project
    assert apply(root)["ok"]
    connector = root / ".mcp.json"
    connector.write_text(json.dumps({"command": str(staged[-1])}), encoding="utf-8")
    result = remove(root)
    assert result["error"]["code"] == "RUNTIME_REFERENCED"
    assert str(connector) in result["configuration_references"] and staged[-1].is_file()
    assert not (root / ".autotem/active-runtime.json").exists()
    connector.unlink()
    monkeypatch.setattr(uninstall, "_runtime_users", lambda roots: [123])
    result = remove(root)
    assert result["error"]["code"] == "RUNTIME_IN_USE" and result["active_pids"] == [123]
    assert staged[-1].is_file()
    monkeypatch.setattr(uninstall, "_runtime_users", lambda roots: [])
    assert remove(root)["state"] == "uninstalled"


def test_legacy_inventory_never_adopted_and_empty_project_stays_empty(release_project):  # noqa: F811
    root, staged = release_project
    assert remove(root)["state"] == "not_installed" and not list(root.iterdir())
    installed = apply(root)
    record = managed._record_path(root / ".autotem", installed["installation_id"])
    doc = json.loads(record.read_bytes())
    doc.pop("runtime_inventory_sha256")
    bootstrap._atomic_json(record, doc)
    assert apply(root)["ok"]
    result = remove(root)
    assert result["error"]["code"] == "UNINSTALL_RETAINED_FILES"
    assert result["retained"][str(staged[-1].parent.parent)] == "no_verified_file_inventory"
    assert staged[-1].exists()


def test_interrupted_host_detachment_resumes_before_any_runtime_removal(
    release_project,
    monkeypatch,  # noqa: F811
):
    root, staged = release_project
    assert apply(root)["ok"] and apply(root, "second", host="openclaw")["ok"]
    # Simulate process death after a receipt was restored but before checkpointing it.
    original = uninstall.rollback_host

    def stop(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(uninstall, "rollback_host", stop)
    with pytest.raises(KeyboardInterrupt):
        remove(root)
    assert all(path.exists() for path in staged)
    assert apply(root)["error"]["code"] == "UNINSTALL_PENDING"
    monkeypatch.setattr(uninstall, "rollback_host", original)
    assert remove(root)["state"] == "uninstalled"


@pytest.mark.skipif(os.name != "nt", reason="Windows process image inspection")
def test_windows_process_check_sees_executing_interpreter(tmp_path):
    import ctypes
    from ctypes import wintypes

    # A Windows venv can report its launcher in sys.executable while this PID runs the base image.
    get_image = ctypes.WinDLL("kernel32", use_last_error=True).GetModuleFileNameW
    get_image.argtypes = (wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD)
    get_image.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(32768)
    assert 0 < get_image(None, buffer, len(buffer)) < len(buffer)
    image = Path(buffer.value)
    assert os.getpid() in uninstall._windows_users([image])

    same_name = tmp_path / image.name
    shutil.copyfile(image, same_name)
    assert os.getpid() not in uninstall._windows_users([same_name])


@pytest.mark.skipif(os.name != "nt", reason="Windows venv executable forwards to a base process")
def test_live_venv_launcher_and_base_process_are_both_protected(tmp_path):
    runtime = tmp_path / "private-venv"
    venv.EnvBuilder(with_pip=False).create(runtime)
    exe = runtime / "Scripts/python.exe"
    code = "import os,sys; print(os.getpid(), flush=True); sys.stdin.readline()"
    with subprocess.Popen(
        [str(exe), "-I", "-B", "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    ) as proc:
        try:
            child_pid = int(proc.stdout.readline())
            assert child_pid != proc.pid
            users = uninstall._runtime_users([runtime, Path(sys.base_prefix)])
            assert child_pid in users and proc.pid in users
        finally:
            proc.stdin.close()
            proc.wait(timeout=10)


def test_registered_future_cache_matches_cpython_and_rejects_changed_bytes(tmp_path):
    control = tmp_path / ".autotem"
    digest = "a" * 64
    runtime = control / "runtimes" / ("rt-" + digest[:20])
    site = runtime / "Lib/site-packages"
    package = site / "sample"
    package.mkdir(parents=True)
    src = package / "future.py"
    src.write_text("value = sum(range(4))\n", encoding="utf-8")
    dist = site / "sample-1.dist-info"
    dist.mkdir()
    (dist / "RECORD").write_text("sample/future.py,,\nsample-1.dist-info/RECORD,,\n")
    (runtime / bootstrap.MARKER_NAME).write_text("{}")
    key = "b" * 32
    txn = control / f"installations/{key}/transaction.json"
    txn.parent.mkdir(parents=True)
    row = {"installation_id": key, "manifest_sha256": digest}
    uninstall.record_new_runtime(control, txn, row, fresh=True)
    inventory = json.loads(txn.with_name(uninstall.INVENTORY).read_bytes())
    cache = Path(importlib.util.cache_from_source(str(src)))
    rel = cache.relative_to(runtime).as_posix()
    assert not cache.exists() and rel in inventory["files"]
    py_compile.compile(
        str(src), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP
    )
    assert uninstall.sha256(cache) == inventory["files"][rel]
    cache.write_bytes(cache.read_bytes() + b"local")
    result = {"retained": {}, "removed_file_count": 0}
    uninstall._remove_tree(runtime, inventory, result, {})
    assert result["retained"][str(cache)] == "changed" and cache.read_bytes().endswith(b"local")


def test_private_python_only_removes_original_archive_members_and_receipted_cache(tmp_path):
    control = tmp_path / "project/.autotem"
    bundle = tmp_path / "bundle"
    archive = bundle / "runtime/python.nupkg"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("tools/Lib/example.py", "zero = 0\n")
        zip_file.writestr("tools/python.exe", "fixture interpreter")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = bundle / "release-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "runtime": {
                    "status": "nuget_archive_included",
                    "archive": "runtime/python.nupkg",
                    "sha256": digest,
                },
                "target": {"python_version": "3.12.10"},
            }
        )
    )
    root = control / ("python/cpython-3.12.10-" + digest[:12])
    (root / "Lib").mkdir(parents=True)
    (root / ".autotem-python").write_text(digest)
    (root / "Lib/example.py").write_bytes(b"zero = 0\n")
    exe = root / "python.exe"
    exe.write_bytes(b"locally changed")
    unowned = root / "notes.raw"
    unowned.write_bytes(b"mine")
    release = SimpleNamespace(manifest_sha256=uninstall.sha256(manifest))
    source = root / "Lib/example.py"
    cache = Path(importlib.util.cache_from_source(str(source)))
    relative = cache.relative_to(root).as_posix()
    py_compile.compile(
        str(source), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP
    )
    older = cache.read_bytes()
    # The same shared base can receive legitimate bytecode from another install
    # transaction. The most recent transaction need not sort last by UUID.
    data = source.stat()
    os.utime(source, (data.st_mtime + 20, data.st_mtime + 20))
    py_compile.compile(
        str(source), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP
    )
    newer = cache.read_bytes()
    assert newer != older

    def receipt(timestamp, content):
        return {
            "created": timestamp,
            "base_cache_runtime": root.name,
            "base_cache_hashes": {relative: hashlib.sha256(content).hexdigest()},
        }

    records = {"current": receipt(2, newer), "previous": receipt(1, older)}
    target, inventory = uninstall._base_inventory(control, bundle, release, records)
    approved = inventory["files"][relative]
    for expected in (older, newer):
        cache.write_bytes(expected)
        assert uninstall._remove_file(cache, approved, {}) is None
    cache.write_bytes(b"unrecorded change")
    report = {"retained": {}, "removed_file_count": 0}
    uninstall._remove_tree(target, inventory, report, {})
    assert report["removed_file_count"] == 1
    assert report["retained"][str(exe)] == "changed"
    assert report["retained"][str(cache)] == "changed"
    assert report["retained"][str(unowned)] == "unregistered_file"
    assert (root / ".autotem-python").exists() and exe.read_bytes() == b"locally changed"
