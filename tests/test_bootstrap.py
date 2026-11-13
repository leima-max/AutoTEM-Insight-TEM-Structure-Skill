"""Installer contracts; fake subprocess tests are not host/science acceptance."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from autotem import bootstrap

pytestmark = pytest.mark.smoke


@pytest.fixture
def installation(tmp_path):
    project = tmp_path / "项目 空格"
    project.mkdir()
    wheel = tmp_path / "autotem_insight-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "autotem_insight-0.1.0.dist-info/METADATA", "Name: autotem-insight\nVersion: 0.1.0\n"
        )
    house = tmp_path / "wheelhouse"
    house.mkdir()
    dependency = house / "numpy-1.0-py3-none-any.whl"
    dependency.write_bytes(b"fixture only, not an installable dependency")
    return project, wheel, house


class FakeProcesses:
    def __init__(self, *, fail_stage=None, configuration="", mcp_status="ready", host_ready=True):
        self.calls = []
        self.fail_stage = fail_stage
        self.configuration = configuration
        self.mcp_status = mcp_status
        self.host_ready = host_ready

    def __call__(self, command, *, environment, cwd, deadline):
        self.calls.append((list(command), dict(environment)))
        if "venv" in command:
            runtime = Path(command[-1])
            executable = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.write_bytes(b"fake interpreter")
            (runtime / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
            stage, stdout = "venv", ""
        elif "config" in command:
            stage, stdout = "config", self.configuration
        elif "install" in command:
            stage, stdout = "install", ""
        elif "check" in command:
            stage, stdout = "check", ""
        elif "-c" in command:
            stage, stdout = "provenance", json.dumps({"ok": True})
        elif command[-1] == "capabilities":
            stage, stdout = (
                "capabilities",
                json.dumps(
                    {"ok": True, "result": {"operations": [{"operation_id": "fixture_only"}]}}
                ),
            )
        elif "doctor" in command:
            stage, stdout = (
                "doctor",
                json.dumps(
                    {
                        "ok": self.host_ready,
                        "runtime": {"status": "ready", "local_mcp": {"status": self.mcp_status}},
                        "host_policy": {"status": "not_evaluated"},
                    }
                ),
            )
            if not self.host_ready:
                return subprocess.CompletedProcess(command, 2, stdout, "")
        else:
            stage, stdout = (
                "host",
                json.dumps(
                    {
                        "ok": True,
                        "dry_run": "--apply" not in command,
                        "host_llm_e2e_verified": False,
                    }
                ),
            )
        return subprocess.CompletedProcess(
            command,
            int(stage == self.fail_stage),
            stdout,
            "https://user:private-password@example.invalid",
        )


def test_default_offline_refuses_missing_wheelhouse_before_mutation(installation, monkeypatch):
    project, wheel, _ = installation
    processes = FakeProcesses()
    monkeypatch.setattr(bootstrap, "_run", processes)
    result = bootstrap.bootstrap_runtime(project, wheel, dry_run=False)
    assert result["error"]["code"] == "WHEELHOUSE_REQUIRED"
    assert not result["runtime_ready"] and not (project / ".autotem").exists()
    assert processes.calls == []


def test_preview_is_read_only_and_not_ready(installation, monkeypatch):
    project, wheel, house = installation
    processes = FakeProcesses()
    monkeypatch.setattr(bootstrap, "_run", processes)
    result = bootstrap.bootstrap_runtime(project, wheel, wheelhouse=house)
    assert result["ok"] and result["dry_run"] and not result["runtime_ready"]
    assert result["wheelhouse_integrity"] == "not_verified"
    assert processes.calls == [] and list(project.iterdir()) == []


def test_authorized_network_override_is_child_only_and_preserves_enterprise_config(
    installation, monkeypatch
):
    project, wheel, _ = installation
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("PIP_CONFIG_FILE", "enterprise-pip.ini")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.invalid")
    monkeypatch.setenv("PYTHONPATH", "untrusted/source")
    before = os.environ.copy()
    processes = FakeProcesses()
    monkeypatch.setattr(bootstrap, "_run", processes)
    result = bootstrap.bootstrap_runtime(project, wheel, allow_network=True, dry_run=False)
    assert result["runtime_ready"] and result["ok"]
    assert os.environ == before
    for _, environment in processes.calls:
        assert environment["PIP_NO_INDEX"] == "0"
        assert environment["PIP_CONFIG_FILE"] == "enterprise-pip.ini"
        assert environment["HTTPS_PROXY"] == "https://proxy.invalid"
        assert "PYTHONPATH" not in environment
    assert "--no-deps" not in processes.calls[2][0]
    assert result["host_probe"] == result["host_llm_e2e"] == "not_evaluated"
    assert result["scientific_e2e"] == "not_evaluated"


def test_offline_installs_dependencies_and_only_applies_codex_project_files(
    installation, monkeypatch
):
    project, wheel, house = installation
    processes = FakeProcesses()
    monkeypatch.setattr(bootstrap, "_run", processes)
    result = bootstrap.bootstrap_runtime(project, wheel, wheelhouse=house, dry_run=False)
    assert result["runtime_ready"]
    install = next(command for command, _ in processes.calls if "install" in command)
    assert "--no-index" in install and "--find-links" in install
    assert "--only-binary=:all:" in install
    assert "--no-deps" not in install and install[-1].endswith("[mcp]")
    assert all(environment["PIP_NO_INDEX"] == "1" for _, environment in processes.calls)
    host = processes.calls[-1][0]
    assert host[0] == result["python_executable"] and host[1] == "-I"
    assert "--apply" in host and "--with-mcp" in host
    assert "start" not in host


@pytest.mark.parametrize("host", ["workbuddy", "openclaw"])
def test_other_hosts_apply_project_assets_with_connector_proposals(installation, monkeypatch, host):
    project, wheel, house = installation
    processes = FakeProcesses()
    monkeypatch.setattr(bootstrap, "_run", processes)
    result = bootstrap.bootstrap_runtime(project, wheel, wheelhouse=house, host=host, dry_run=False)
    assert result["runtime_ready"]
    assert result["host_install_mode"] == "apply_project_assets_connector_proposal"
    assert "--apply" in processes.calls[-1][0]
    assert "--with-mcp" not in processes.calls[-1][0]


@pytest.mark.parametrize("mcp_status", ["blocked", "dependency_unavailable", "not_evaluated"])
def test_missing_or_broken_mcp_handshake_never_marks_runtime_ready(
    installation, monkeypatch, mcp_status
):
    project, wheel, house = installation
    processes = FakeProcesses(mcp_status=mcp_status)
    monkeypatch.setattr(bootstrap, "_run", processes)
    result = bootstrap.bootstrap_runtime(project, wheel, wheelhouse=house, dry_run=False)
    assert not result["runtime_ready"] and result["error"]["code"] == "MCP_HANDSHAKE_FAILED"
    assert not any("install-host" in command for command, _ in processes.calls)


def test_host_not_ready_is_separate_from_verified_private_runtime(installation, monkeypatch):
    project, wheel, house = installation
    processes = FakeProcesses(host_ready=False)
    monkeypatch.setattr(bootstrap, "_run", processes)
    result = bootstrap.bootstrap_runtime(project, wheel, wheelhouse=house, dry_run=False)
    assert result["runtime_ready"] and result["ok"]
    assert result["readiness"]["ok"] is False
    assert result["host_llm_e2e"] == "not_evaluated"


def test_without_mcp_allows_verified_json_cli_runtime(installation, monkeypatch):
    project, wheel, house = installation
    processes = FakeProcesses(mcp_status="dependency_unavailable")
    monkeypatch.setattr(bootstrap, "_run", processes)
    result = bootstrap.bootstrap_runtime(
        project, wheel, wheelhouse=house, with_mcp=False, dry_run=False
    )
    assert result["runtime_ready"] and "local_mcp_handshake_passed" not in result["checks"]
    assert "--with-mcp" not in processes.calls[-1][0]


@pytest.mark.parametrize(
    "stage,code",
    [
        ("install", "DEPENDENCIES_UNAVAILABLE"),
        ("check", "DEPENDENCY_CHECK_FAILED"),
        ("capabilities", "CAPABILITY_CHECK_FAILED"),
        ("provenance", "RUNTIME_PROVENANCE_FAILED"),
    ],
)
def test_failed_verification_never_publishes_ready_or_raw_secrets(
    installation, monkeypatch, stage, code
):
    project, wheel, house = installation
    monkeypatch.setattr(bootstrap, "_run", FakeProcesses(fail_stage=stage))
    result = bootstrap.bootstrap_runtime(project, wheel, wheelhouse=house, dry_run=False)
    assert not result["runtime_ready"] and result["error"]["code"] == code
    state = json.loads((project / ".autotem/runtime" / bootstrap.MARKER_NAME).read_text())
    assert state["state"] == "failed"
    assert "private-password" not in json.dumps(result) + json.dumps(state)
    assert not (project / ".autotem/runtime-bootstrap.lock").exists()


def test_timeout_has_failed_diagnostics_and_no_ready(installation, monkeypatch):
    project, wheel, house = installation

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("private", 1, output="private-password")

    monkeypatch.setattr(subprocess, "run", timeout)
    result = bootstrap.bootstrap_runtime(project, wheel, wheelhouse=house, dry_run=False)
    assert result["error"]["code"] == "BOOTSTRAP_TIMEOUT" and not result["runtime_ready"]
    assert "private-password" not in json.dumps(result)


def test_idempotent_owned_ready_runtime_does_not_reinstall_or_require_wheelhouse(
    installation, monkeypatch
):
    project, wheel, house = installation
    processes = FakeProcesses()
    monkeypatch.setattr(bootstrap, "_run", processes)
    first = bootstrap.bootstrap_runtime(project, wheel, wheelhouse=house, dry_run=False)
    second = bootstrap.bootstrap_runtime(project, wheel, dry_run=False)
    assert first["runtime_ready"] and second["runtime_ready"] and second["reuse_owned_runtime"]
    assert sum("venv" in command for command, _ in processes.calls) == 1
    assert sum("install" in command for command, _ in processes.calls) == 1
    assert first["python_executable"] == second["python_executable"]


def test_unowned_existing_runtime_is_never_overwritten(installation, monkeypatch):
    project, wheel, house = installation
    runtime = project / ".autotem/runtime"
    runtime.mkdir(parents=True)
    sentinel = runtime / "user-file.txt"
    sentinel.write_text("preserve")
    processes = FakeProcesses()
    monkeypatch.setattr(bootstrap, "_run", processes)
    result = bootstrap.bootstrap_runtime(project, wheel, wheelhouse=house, dry_run=False)
    assert result["error"]["code"] == "UNOWNED_RUNTIME"
    assert sentinel.read_text() == "preserve" and processes.calls == []


@pytest.mark.parametrize("change", ["wheel", "mcp"])
def test_ready_environment_upgrade_requires_review_and_preserves_old_runtime(
    installation, monkeypatch, change
):
    project, wheel, house = installation
    processes = FakeProcesses()
    monkeypatch.setattr(bootstrap, "_run", processes)
    first = bootstrap.bootstrap_runtime(project, wheel, wheelhouse=house, dry_run=False)
    assert first["runtime_ready"]
    marker = project / ".autotem/runtime" / bootstrap.MARKER_NAME
    before = marker.read_bytes()
    kwargs = {"with_mcp": False} if change == "mcp" else {}
    if change == "wheel":
        with zipfile.ZipFile(wheel, "a") as archive:
            archive.writestr("new-resource.txt", "different release build")
    calls_before = len(processes.calls)
    result = bootstrap.bootstrap_runtime(project, wheel, wheelhouse=house, dry_run=False, **kwargs)
    assert result["error"]["code"] == "UPGRADE_REQUIRES_REVIEW"
    assert marker.read_bytes() == before and len(processes.calls) == calls_before


def test_home_root_and_symlink_escape_are_refused(installation, monkeypatch, tmp_path):
    project, wheel, house = installation
    home_result = bootstrap.bootstrap_runtime(Path.home(), wheel, wheelhouse=house)
    assert home_result["error"]["code"] == "UNSAFE_PROJECT"
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (project / ".autotem").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip(
            "OS account cannot create symlinks; reparse paths are still guarded in implementation."
        )
    result = bootstrap.bootstrap_runtime(project, wheel, wheelhouse=house, dry_run=False)
    assert result["error"]["code"] == "UNSAFE_PATH" and list(outside.iterdir()) == []


@pytest.mark.parametrize(
    "setting",
    [
        "global.target='outside'",
        "install.prefix='outside'",
        ":env:.root='outside'",
        "global.user='true'",
    ],
)
def test_pip_redirect_is_refused_before_dependency_install(installation, monkeypatch, setting):
    project, wheel, house = installation
    processes = FakeProcesses(configuration=setting)
    monkeypatch.setattr(bootstrap, "_run", processes)
    result = bootstrap.bootstrap_runtime(project, wheel, wheelhouse=house, dry_run=False)
    assert result["error"]["code"] == "PIP_DESTINATION_UNSAFE"
    assert not any("install" in command for command, _ in processes.calls)


def test_offline_refuses_configured_network_find_links(installation, monkeypatch):
    project, wheel, house = installation
    monkeypatch.setattr(
        bootstrap,
        "_run",
        FakeProcesses(configuration="global.find-links='https://private-password@example.invalid'"),
    )
    result = bootstrap.bootstrap_runtime(project, wheel, wheelhouse=house, dry_run=False)
    assert result["error"]["code"] == "OFFLINE_CONFIGURATION_NETWORK"
    assert "private-password" not in json.dumps(result)


@pytest.mark.parametrize(
    "url",
    [
        "http://packages.invalid/simple",
        "https://user:secret@packages.invalid/simple",
        "https://packages.invalid/simple?token=secret",
    ],
)
def test_index_url_rejects_credentials_without_echoing_values(installation, url):
    project, wheel, _ = installation
    result = bootstrap.bootstrap_runtime(project, wheel, allow_network=True, index_url=url)
    assert result["error"]["code"] == "INDEX_URL_INVALID"
    assert "secret" not in json.dumps(result)


def write_manifest(house):
    wheels = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in house.glob("*.whl")
    }
    data = {
        "schema_version": "1.0.0",
        "target": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": sys.platform,
            "architecture": platform.machine(),
        },
        "wheels": wheels,
    }
    (house / "wheelhouse-manifest.json").write_text(json.dumps(data))
    return data


def test_wheelhouse_manifest_verifies_hashes_and_target(installation):
    project, wheel, house = installation
    write_manifest(house)
    result = bootstrap.bootstrap_runtime(project, wheel, wheelhouse=house)
    assert result["ok"] and result["wheelhouse_integrity"] == "manifest_hashes_and_target_verified"
    next(house.glob("*.whl")).write_bytes(b"changed")
    result = bootstrap.bootstrap_runtime(project, wheel, wheelhouse=house)
    assert result["error"]["code"] == "WHEELHOUSE_INTEGRITY_FAILED"
    assert list(project.iterdir()) == []


def test_manifest_wrong_platform_fails_before_venv(installation):
    project, wheel, house = installation
    data = write_manifest(house)
    data["target"]["platform"] = "unsupported-target"
    (house / "wheelhouse-manifest.json").write_text(json.dumps(data))
    result = bootstrap.bootstrap_runtime(project, wheel, wheelhouse=house, dry_run=False)
    assert result["error"]["code"] == "WHEELHOUSE_TARGET_MISMATCH"
    assert list(project.iterdir()) == []


def test_manifest_duplicate_json_keys_are_refused(installation):
    project, wheel, house = installation
    (house / "wheelhouse-manifest.json").write_text(
        '{"schema_version":"1.0.0","schema_version":"1.0.0"}'
    )
    result = bootstrap.bootstrap_runtime(project, wheel, wheelhouse=house, dry_run=False)
    assert result["error"]["code"] == "WHEELHOUSE_MANIFEST_INVALID"
    assert list(project.iterdir()) == []


def test_offline_sdist_only_wheelhouse_fails_before_venv(installation):
    project, wheel, house = installation
    next(house.glob("*.whl")).unlink()
    (house / "unhashed-source.tar.gz").write_bytes(b"sdist fixture")
    result = bootstrap.bootstrap_runtime(project, wheel, wheelhouse=house, dry_run=False)
    assert result["error"]["code"] == "WHEELHOUSE_INCOMPLETE"
    assert list(project.iterdir()) == []


def test_stdlib_import_works_without_site_packages():
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "from autotem.bootstrap import main; raise SystemExit(main(['--help']))",
        ],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0 and "--allow-network" in result.stdout
