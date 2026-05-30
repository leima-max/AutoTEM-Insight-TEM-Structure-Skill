from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autotem import TEMAnalysisSkill, __version__
from autotem.cli import main as cli_main
from autotem.workflows.pipeline import _RunLogger


@pytest.mark.smoke
def test_smoke_package_imports() -> None:
    assert __version__ == "0.1.0"
    assert TEMAnalysisSkill is not None


@pytest.mark.smoke
def test_smoke_minimal_workflow_run(tmp_path: Path) -> None:
    input_path = tmp_path / "smoke_input.npy"
    np.save(input_path, np.zeros((16, 16), dtype=np.float32))

    workflow_path = tmp_path / "smoke_workflow.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: smoke_minimal",
                "  steps:",
                "    - load",
                "    - report",
                "outputs:",
                "  formats: [json]",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "outputs"
    result = TEMAnalysisSkill().run(
        input_path=input_path,
        workflow_path=workflow_path,
        output_dir=output_dir,
    )

    assert result["status"] == "A2 baseline workflow completed."
    assert (output_dir / "analysis_report.json").exists()
    assert (output_dir / "workflow_used.yaml").exists()
    assert (output_dir / "run_summary.json").exists()
    assert (output_dir / "run.log").exists()


@pytest.mark.smoke
def test_smoke_cli_analyze_with_minimal_workflow(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "cli_smoke.npy"
    np.save(input_path, np.zeros((8, 8), dtype=np.float32))

    workflow_path = tmp_path / "cli_smoke_workflow.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: cli_smoke",
                "  steps: [load, report]",
                "outputs:",
                "  formats: [json]",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "cli_outputs"
    exit_code = cli_main(
        [
            "analyze",
            str(input_path),
            "--workflow",
            str(workflow_path),
            "--output",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "A2 baseline workflow completed." in captured.out
    assert (output_dir / "analysis_report.json").exists()


def test_run_logger_writes_entries_immediately(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    logger = _RunLogger(log_path)

    logger.info("before long-running step")

    text = log_path.read_text(encoding="utf-8")
    assert "before long-running step" in text
    assert len(text.strip().splitlines()) == 1

    logger.warning("step warning", context={"stage": "simulate"})
    logger.flush()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "step warning" in lines[1]
