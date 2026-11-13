from pathlib import Path

import numpy as np
import pytest

from autotem import TEMAnalysisSkill, __version__


def test_package_imports() -> None:
    assert __version__ == "0.1.0"
    assert TEMAnalysisSkill is not None


def test_workflow_runs_from_real_file(tmp_path) -> None:
    source = tmp_path / "sample.npy"
    np.save(source, np.cos(np.mgrid[:32, :32][1] * np.pi / 4))
    recipe = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    result = TEMAnalysisSkill().run(
        str(source), workflow_path=recipe, output_dir=tmp_path / "report"
    )
    assert result["status"] == "A2 baseline workflow completed."
    assert result["input_path"] == str(source)
    assert result["report_path"].endswith("analysis_report.json")


def test_missing_file_cannot_produce_successful_scaffold(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        TEMAnalysisSkill().run(str(tmp_path / "missing.tif"))
