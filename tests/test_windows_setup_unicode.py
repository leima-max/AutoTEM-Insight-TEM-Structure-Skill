"""Exercise the real Windows launcher, its redirected streams and a CPython zipapp."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

from tools.setup import build_setup, verify_cli_profile, verify_install


@pytest.fixture
def native_bundle(tmp_path, monkeypatch):
    if os.name != "nt":
        pytest.skip("The release launcher is a Windows x64 executable")
    # A small stdlib venv redirector permits an actual isolated CPython child in offline CI.
    env = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False).create(env)
    bundle = tmp_path / "发行 𠀋"
    archive = bundle / "runtime/python.3.12.10.nupkg"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as package:
        package.write(env / "Scripts/python.exe", "tools/python.exe")
        package.write(env / "pyvenv.cfg", "tools/pyvenv.cfg")

    source = tmp_path / "source"
    for filename in (
        "autotem/__init__.py",
        "autotem/bootstrap.py",
        "autotem/harness/__init__.py",
        "autotem/harness/hosts.py",
        "autotem/installation/__init__.py",
    ):
        path = source / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    # Emit literal UTF-8 on both pipes so ASCII JSON escaping cannot hide a bad decoder.
    (source / "autotem/installation/managed.py").write_text(
        "import json, sys\n"
        "def main():\n"
        "    project = sys.argv[sys.argv.index('--project') + 1]\n"
        "    ok = '--apply' not in sys.argv\n"
        "    print('診斷 𠀋 🔬', file=sys.stderr)\n"
        "    print(json.dumps({'project': project, 'ok': ok, 'note': '接收 🔬', "
        "'utf8': sys.flags.utf8_mode, 'isolated': sys.flags.isolated}, ensure_ascii=False))\n"
        "    return 0 if ok else 2\n",
        encoding="utf-8",
    )
    (source / "tools/setup").mkdir(parents=True)
    for filename in ("Setup.cs", "SetupPreview.cs"):
        shutil.copyfile(
            build_setup.ROOT / "tools/setup" / filename, source / "tools/setup" / filename
        )
    policy = source / "configs/release/traditional-core-mcp.json"
    policy.parent.mkdir(parents=True)
    policy.write_text(
        json.dumps(
            {
                "python_version": "3.12.10",
                "python_nupkg": {"sha256": hashlib.sha256(archive.read_bytes()).hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(build_setup, "ROOT", source)
    build_setup.build(bundle)
    artifacts = [
        {
            "path": path.relative_to(bundle).as_posix(),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in (archive, bundle / "AutoTEM-install.pyz")
    ]
    (bundle / "release-manifest.json").write_text(
        json.dumps(
            {
                "profiles": {"traditional-core-mcp": {}},
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return bundle / "AutoTEM-Setup.exe"


def test_native_and_python_responses_round_trip_unicode(native_bundle, tmp_path):
    project = tmp_path / "电镜 项目 𠀋 🔬"
    project.mkdir()
    temporary = tmp_path / "临时 𠀋"
    temporary.mkdir()
    env = {
        **os.environ,
        "TEMP": str(temporary),
        "TMP": str(temporary),
        "PYTHONIOENCODING": "cp1252",
        "PYTHONUTF8": "0",
    }

    def reply(exe, flags, exit_code):
        result = subprocess.run(
            [exe, "--project", project, "--host", "codex", *flags],
            cwd=project,
            env=env,
            capture_output=True,
            timeout=90,
        )
        assert result.returncode == exit_code, (result.stdout, result.stderr)
        assert not result.stdout.startswith(b"\xef\xbb\xbf")
        return json.loads(result.stdout.decode("utf-8"))

    preview = reply(native_bundle, [], 0)
    assert preview["state"] == "planned" and preview["project"] == str(project)
    assert Path(preview["python"]).is_relative_to(project)
    for flags, code in [(["--uninstall"], 0), (["--uninstall", "--apply"], 2)]:
        child = reply(native_bundle, flags, code)
        assert child == {
            "ok": code == 0,
            "project": str(project),
            "note": "接收 🔬",
            "utf8": 1,
            "isolated": 1,
        }
    assert not list(project.iterdir()) and not list(temporary.iterdir())
    broken = tmp_path / "缺失 🔬" / native_bundle.name
    broken.parent.mkdir()
    shutil.copyfile(native_bundle, broken)
    failed = reply(broken, [], 2)
    assert failed["error"]["code"] == "SETUP_FAILED"
    assert str(broken.parent / "release-manifest.json") in failed["error"]["message"]


def test_native_preview_reports_project_paths_and_space_without_project_writes(
    native_bundle, tmp_path
):
    bundle = native_bundle.parent
    project = tmp_path / "待安装 项目"
    project.mkdir()
    manifest_path = bundle / "release-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    wheel = bundle / "wheelhouse/traditional-core-mcp/example-1-py3-none-any.whl"
    wheel.parent.mkdir(parents=True)
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as file:
        file.writestr("example/payload", b"x" * 1024)
    manifest["artifacts"].append(
        {
            "path": wheel.relative_to(bundle).as_posix(),
            "size": wheel.stat().st_size,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with zipfile.ZipFile(bundle / "runtime/python.3.12.10.nupkg") as archive:
        base_size = sum(row.file_size for row in archive.infolist())
    for host, owned in [
        ("codex", ".codex/config.toml"),
        ("openclaw", "skills/autotem/"),
        ("workbuddy", "autotem-host/workbuddy/"),
    ]:
        plan = json.loads(
            subprocess.check_output(
                [native_bundle, "--project", project, "--host", host], cwd=project, timeout=60
            )
        )
        assert plan["state"] == "planned" and plan["active_runtime_id"] is None
        assert plan["existing_project_control"] is False
        assert plan["estimated_new_payload_bytes"] == base_size + 1024
        assert (
            plan["available_bytes"]
            >= plan["minimum_free_bytes"]
            == base_size + 3 * wheel.stat().st_size
        )
        assert owned in plan["affected_project_paths"] and "授权" in plan["host_authorization"]
        assert (
            plan["target_runtime_id"]
            == "rt-" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:20]
        )
        assert not list(project.iterdir())

    control = project / ".autotem"
    control.mkdir()
    active = control / "active-runtime.json"
    data = {
        "runtime_id": "rt-recorded",
        "profile": "traditional-core-mcp",
        "host_installations": {"codex": "install-recorded"},
    }
    active.write_text(json.dumps(data), encoding="utf-8")
    before = active.read_bytes()
    reused = json.loads(
        subprocess.check_output([native_bundle, "--project", project, "--host", "openclaw"])
    )
    assert reused["active_runtime_id"] == "rt-recorded"
    assert {".codex/config.toml", "skills/autotem/"} <= set(reused["affected_project_paths"])
    assert active.read_bytes() == before and list(control.iterdir()) == [active]
    marker = control / "runtimes" / reused["target_runtime_id"] / ".autotem-runtime.json"
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"{")
    partial = json.loads(subprocess.check_output([native_bundle, "--project", project]))
    assert partial["state"] == "planned"
    assert partial["estimated_new_payload_bytes"] == base_size + 1024
    assert marker.read_bytes() == b"{"


def test_install_verifiers_reject_non_utf8_and_preserve_raw_response(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "release-manifest.json").write_text(
        json.dumps({"profiles": {"traditional-core-mcp": {}}}), encoding="utf-8"
    )
    payload = '{"state":"planned","project":"电镜"}'.encode("gbk")

    def bad_wire(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, payload, b"")

    monkeypatch.setattr(subprocess, "run", bad_wire)
    out = tmp_path / "evidence"
    failed = verify_install.run(bundle, out, extended=False, keep_active=False)
    assert failed["case_verdict"] == "failed" and "UTF-8" in failed["error"]
    assert (out / "invalid-response.stdout").read_bytes() == payload
    with pytest.raises(UnicodeDecodeError):
        verify_cli_profile.execute([sys.executable], tmp_path, {}, 5)
