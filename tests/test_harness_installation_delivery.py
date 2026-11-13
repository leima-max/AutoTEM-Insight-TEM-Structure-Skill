"""Cold-start maintenance remains usable without scientific site packages."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from autotem.harness.hosts import _planned_files

PROJECT = Path(__file__).resolve().parents[1]


def test_doctor_returns_structured_missing_runtime_without_scientific_dependencies(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "autotem.harness.cli",
            "doctor",
            "--host",
            "workbuddy",
            "--project",
            str(tmp_path),
            "--python",
            str(tmp_path / "missing-python.exe"),
            "--timeout",
            "1",
        ],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert report["runtime"]["status"] == "blocked"
    assert report["host_llm_e2e"] == "not_evaluated"
    assert "Traceback" not in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_bootstrap_help_runs_without_any_scientific_site_packages():
    result = subprocess.run(
        [sys.executable, "-S", "-m", "autotem.bootstrap", "--help"],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--wheelhouse" in result.stdout


def test_host_commands_preserve_selected_venv_symlink_path(tmp_path):
    selected = tmp_path / "venv-python"
    try:
        selected.symlink_to(sys.executable)
    except OSError:
        # Windows developer-mode restrictions can prevent symlinks; use an
        # existing explicit interpreter for the argv contract in that case.
        selected = Path(sys.executable)
    files = _planned_files(
        tmp_path,
        "openclaw",
        with_mcp=False,
        harness_root="runs",
        python_executable=str(selected),
    )
    adapter = json.loads(files["autotem-host/openclaw/adapter.json"])
    command = adapter["mcp"]["mcpServers"]["autotem"]
    assert command["command"] == str(selected)
    assert command["args"][0] == "-I"
    assert adapter["jsonl_command"] == [command["command"], *command["args"][:-1], "stdio"]
    assert adapter["runtime_readiness"] == "not_evaluated"
