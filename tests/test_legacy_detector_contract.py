from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from autotem import TEMAnalysisSkill
from autotem.atomic import detect_atoms
from autotem.core import AnalysisReport
from autotem.harness import Harness
from autotem.harness.contracts import HarnessError

pytestmark = pytest.mark.smoke


def test_python_overlay_hook_keeps_the_legacy_result_and_warning(tmp_path):
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8, 12] = 200
    image[22, 17] = 120
    source = tmp_path / "image.tif"
    Image.fromarray(image).save(source)
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({"workflow": {"steps": ["load", "report"]}}))
    expected = detect_atoms(image, return_overlay_array=True)
    warning = next(item for item in expected.warnings if "return_overlay_array" in item)
    assert len(expected.coordinates) > 0

    result = TEMAnalysisSkill().run(
        source, recipe, tmp_path / "result", detect_atoms_step=True,
        atom_detection_options={"return_overlay_array": True},
    )
    assert result["execution_status"] == "completed"
    report = AnalysisReport.model_validate_json(Path(result["report_path"]).read_bytes())
    step = next(step for step in report.workflow.steps if step.name == "detect_atoms")
    assert warning in step.warnings and warning in report.atom_lattice.warnings
    assert "overlay_array" not in result and report.atom_lattice.overlay is None
    np.testing.assert_array_equal(
        [[atom.y, atom.x] for atom in report.atom_lattice.atoms], expected.coordinates,
    )
    np.testing.assert_array_equal(
        [atom.intensity.value for atom in report.atom_lattice.atoms], expected.intensities,
    )


def test_harness_rejects_zero_detector_threshold_before_execution(tmp_path):
    harness = Harness(tmp_path / "runs")
    source = tmp_path / "image.npy"
    np.save(source, np.eye(8))
    detection = {"id": "atoms", "operation_id": "atomic.detect", "parameters": {}}
    plan = {
        "input_path": str(source),
        "steps": [{"id": "raw", "operation_id": "input.load"}, detection],
    }
    for threshold in (1e-9, 1.0):
        detection["parameters"]["threshold_rel"] = threshold
        validated = harness.validate_plan(plan)
        assert validated["plan"]["steps"][1]["parameters"]["threshold_rel"] == threshold
    detection["parameters"]["threshold_rel"] = 0.0
    with pytest.raises(HarnessError) as error:
        harness.validate_plan(plan)
    assert error.value.code == "INVALID_PARAMETERS"
    assert any(part["loc"] == ("threshold_rel",) for part in error.value.details)
