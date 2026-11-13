import hashlib
from pathlib import Path

import numpy as np
import pytest

from autotem import TEMAnalysisSkill
from autotem.atomic import detect_atoms
from autotem.cli import main as cli
from autotem.core import AnalysisReport
from autotem.services.data_quality import read_selected_input

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize(("suffix", "delimiter"), [
    ("npy", None), ("txt", " "), ("csv", ","), ("csv", " "),
])
def test_legacy_cli_detects_raw_numeric_file(tmp_path, suffix, delimiter):
    pixels = np.zeros((72, 80), dtype=np.float64)
    pixels[20, 30] = 120.0000003
    pixels[50, 60] = 80.0000002
    source = tmp_path / f"atoms.{suffix}"
    if delimiter is None:
        np.save(source, pixels)
    else:
        np.savetxt(source, pixels, fmt="%.12f", delimiter=delimiter,
                   header="numeric image, values in a.u.")
    initial_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    recipe = tmp_path / "measure.yaml"
    recipe.write_text(
        "workflow:\n  steps: [load, detect_atoms, report]\n"
        "  parameters:\n    detect_atoms: {mode: dog, max_atoms: 2}\n"
        "outputs:\n  formats: [json, csv]\n",
        encoding="utf-8",
    )
    output = tmp_path / "result"
    assert cli(["analyze", str(source), "--workflow", str(recipe), "--output", str(output)]) == 0
    report = AnalysisReport.model_validate_json((output / "analysis_report.json").read_bytes())
    assert report.provenance["run_outcome"]["execution_status"] == "completed"
    assert report.atom_lattice is not None
    expected = detect_atoms(pixels, mode="dog", max_atoms=2)
    assert len(expected.coordinates) == 2
    atoms = report.atom_lattice.atoms
    np.testing.assert_array_equal([(a.y, a.x) for a in atoms], expected.coordinates)
    np.testing.assert_array_equal([a.intensity.value for a in atoms], expected.intensities)
    assert {(a.y, a.x, a.intensity.value) for a in atoms} == {
        (20, 30, 120.0000003), (50, 60, 80.0000002),
    }
    source_record = report.atom_lattice.provenance["detector_provenance"]["source"]
    assert Path(source_record["path"]).resolve() == source.resolve()
    registered_raw, _ = read_selected_input(source, {}, dataset_id="test")
    np.testing.assert_array_equal(registered_raw, pixels)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == initial_hash


def test_detector_slice_cannot_override_a_workflow_plane(tmp_path):
    source = tmp_path / "stack.npy"
    images = np.zeros((2, 32, 32))
    images[0, 8, 9] = 97
    images[1, 17, 18] = 201
    np.save(source, images)
    recipe = tmp_path / "selected.yaml"
    recipe.write_text(
        "workflow:\n  steps: [load, detect_atoms, report]\n"
        "  parameters:\n    load: {frame: 1}\n",
        encoding="utf-8",
    )
    result = TEMAnalysisSkill().run(
        source, recipe, tmp_path / "result",
        atom_detection_options={"max_atoms": 1, "slice_index": 17},
    )
    assert result["execution_status"] == "completed"
    report = AnalysisReport.model_validate_json(Path(result["report_path"]).read_bytes())
    step = next(s for s in report.workflow.steps if s.name == "detect_atoms")
    lattice = report.atom_lattice
    assert [(a.y, a.x, a.intensity.value) for a in lattice.atoms] == [(17, 18, 201)]
    assert lattice.provenance["selection"]["frame"] == 1
    assert step.parameters["slice_index"] == lattice.provenance["parameters"]["slice_index"] == 0
    assert any("slice_index=17" in message for message in step.warnings)
