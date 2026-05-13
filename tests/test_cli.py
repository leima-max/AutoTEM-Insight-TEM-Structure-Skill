from pathlib import Path
from unittest.mock import patch

import numpy as np

from autotem.cli import EXIT_CONFIG_ERROR, EXIT_FILE_ERROR, EXIT_RUNTIME_ERROR, main


def test_cli_help(capsys) -> None:
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Run AutoTEM analysis workflows." in captured.out


def test_cli_analyze_generates_report(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "cli_input.npy"
    np.save(input_path, np.zeros((16, 16), dtype=np.float32))
    output_dir = tmp_path / "cli_outputs"

    exit_code = main(["analyze", str(input_path), "--output", str(output_dir)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "A2 baseline workflow completed." in captured.out
    assert (output_dir / "analysis_report.json").exists()
    assert (output_dir / "workflow_used.yaml").exists()


def test_cli_returns_config_error_code_for_invalid_workflow(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "cli_input.npy"
    np.save(input_path, np.zeros((16, 16), dtype=np.float32))
    workflow_path = tmp_path / "invalid.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: invalid",
                "  parameters:",
                "    preprocessing:",
                "      clip_percentile_low: 99",
                "      clip_percentile_high: 1",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["analyze", str(input_path), "--workflow", str(workflow_path)])
    captured = capsys.readouterr()

    assert exit_code == EXIT_CONFIG_ERROR
    assert "configuration error" in captured.err


def test_cli_returns_file_error_code_for_missing_workflow(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "cli_input.npy"
    np.save(input_path, np.zeros((16, 16), dtype=np.float32))
    missing = tmp_path / "missing.yaml"

    exit_code = main(["analyze", str(input_path), "--workflow", str(missing)])
    captured = capsys.readouterr()

    assert exit_code == EXIT_FILE_ERROR
    assert "file error" in captured.err


def test_cli_returns_runtime_error_code(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "cli_input.npy"
    np.save(input_path, np.zeros((16, 16), dtype=np.float32))

    with patch("autotem.cli.TEMAnalysisSkill.run", side_effect=RuntimeError("boom")):
        exit_code = main(["analyze", str(input_path)])
    captured = capsys.readouterr()

    assert exit_code == EXIT_RUNTIME_ERROR
    assert "runtime error" in captured.err


def test_cli_analyze_detect_atoms_flag(monkeypatch, tmp_path: Path, capsys) -> None:
    class DummySkill:
        def run(self, **kwargs):
            assert kwargs["detect_atoms_step"] is True
            return {"status": "ok", "atom_count": 3}

    monkeypatch.setattr("autotem.cli.TEMAnalysisSkill", DummySkill)
    input_path = tmp_path / "sample.npy"
    np.save(input_path, np.zeros((8, 8), dtype=np.float32))

    exit_code = main(["analyze", str(input_path), "--detect-atoms"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "ok" in captured.out
    assert "Detected atoms: 3" in captured.out


def test_cli_analyze_without_detect_atoms_flag(monkeypatch, tmp_path: Path, capsys) -> None:
    class DummySkill:
        def run(self, **kwargs):
            assert kwargs["detect_atoms_step"] is None
            return {"status": "ok"}

    monkeypatch.setattr("autotem.cli.TEMAnalysisSkill", DummySkill)
    input_path = tmp_path / "sample.npy"
    np.save(input_path, np.zeros((8, 8), dtype=np.float32))

    exit_code = main(["analyze", str(input_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "ok" in captured.out
