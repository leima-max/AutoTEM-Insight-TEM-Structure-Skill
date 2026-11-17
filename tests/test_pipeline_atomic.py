from pathlib import Path

import numpy as np
import pytest
import yaml

from autotem.atomic import detect_atoms
from autotem.core.models import AtomLattice
from autotem.harness.contracts import HarnessError
from autotem.workflows.pipeline import TEMAnalysisSkill


def _synthetic_atoms_image(size: int = 128) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    image = np.zeros((size, size), dtype=np.float64)
    centers = [(32, 32), (32, 96), (64, 64), (96, 32), (96, 96)]
    for cy, cx in centers:
        image += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.4**2)))
    image += 0.04 * np.random.default_rng(0).normal(size=(size, size))
    return image


def _write_tif(path: Path, image: np.ndarray) -> None:
    tifffile = pytest.importorskip("tifffile")
    tifffile.imwrite(str(path), image.astype(np.float32))


def test_pipeline_detect_atoms_from_workflow_yaml(tmp_path: Path) -> None:
    img = _synthetic_atoms_image()
    image_path = tmp_path / "atoms.tif"
    _write_tif(image_path, img)

    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        "workflow:\n"
        "  name: c0\n"
        "  steps:\n"
        "    - load\n"
        "    - detect_atoms\n",
        encoding="utf-8",
    )

    skill = TEMAnalysisSkill()
    result = skill.run(
        input_path=image_path,
        workflow_path=workflow_path,
        atom_detection_options={
            "mode": "log",
            "gaussian_sigma": 0.8,
            "min_distance": 4,
            "threshold_rel": 0.2,
        },
    )

    assert "atom_lattice" in result
    assert isinstance(result["atom_lattice"], AtomLattice)
    assert result["atom_count"] > 0


def test_pipeline_detect_atoms_can_be_forced_without_workflow(tmp_path: Path) -> None:
    img = _synthetic_atoms_image()
    image_path = tmp_path / "atoms.tif"
    _write_tif(image_path, img)

    skill = TEMAnalysisSkill()
    result = skill.run(
        input_path=image_path,
        detect_atoms_step=True,
        atom_detection_options={
            "mode": "dog",
            "gaussian_sigma": 0.8,
            "min_distance": 4,
            "threshold_rel": 0.2,
        },
    )

    assert "atom_lattice" in result
    assert isinstance(result["atom_lattice"], AtomLattice)
    assert result["atom_count"] > 0


@pytest.mark.smoke
def test_legacy_yaml_atom_options_and_explicit_override(tmp_path: Path) -> None:
    from autotem.core import AnalysisReport

    img = _synthetic_atoms_image().astype(np.float32)
    source = tmp_path / "raw.tiff"
    _write_tif(source, img)
    recipe = tmp_path / "detect.yaml"
    params = {"mode": "dog", "min_distance": 10, "threshold_rel": 0.6, "max_atoms": 2}
    recipe.write_text(yaml.safe_dump({"workflow": {
        "steps": ["load", "detect_atoms"], "parameters": {
            "atom_detection": {"mode": "dog", "min_distance": 10},
            "detect_atoms": {"threshold_rel": 0.6, "max_atoms": 2},
        },
    }}), encoding="utf-8")
    for cap in (2, 1):
        folder = tmp_path / f"cap-{cap}"
        override = {"max_atoms": cap, "overlay_path": folder / "custom.png"} if cap == 1 else None
        out = TEMAnalysisSkill().run(source, recipe, folder, atom_detection_options=override)
        expected = detect_atoms(img, **{**params, "max_atoms": cap})
        assert out["execution_status"] == "completed" and out["atom_count"] == cap
        report = AnalysisReport.model_validate_json(Path(out["report_path"]).read_bytes())
        assert report.atom_lattice.provenance["image_role"] == "raw"
        assert report.atom_lattice.provenance["parameters"] == expected.parameters
        np.testing.assert_array_equal(
            [[a.y, a.x] for a in report.atom_lattice.atoms], expected.coordinates
        )
        np.testing.assert_array_equal(
            [a.intensity.value for a in report.atom_lattice.atoms], expected.intensities
        )
        workflow_used = yaml.safe_load(Path(out["workflow_used_path"]).read_bytes())
        used = workflow_used["workflow"]["parameters"]
        assert used["detect_atoms"]["max_atoms"] == cap and "atom_detection" not in used
        assert report.workflow.steps[1].parameters["threshold_rel"] == 0.6
        if override:
            assert (folder / "custom.png").is_file()
            recorded = report.workflow.steps[1].parameters["overlay_path"]
            assert recorded == str(override["overlay_path"])


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("steps", "groups", "message"),
    [
        (["load", "detect_atoms"], {"detect_atoms": {"max_atom": 1}}, "max_atom"),
        (["load", "detect_atoms"], {"detect_atoms": []}, "must be a mapping"),
        (["load", "detect_atoms"], {
            "detect_atoms": {"max_atoms": 2}, "atom_detection": {"max_atoms": 1}
        }, "max_atoms"),
        (["load", "detect_atoms"], {"atom_detection": {"max_atoms": False}}, "max_atoms"),
        (["load", "detect_atoms"], {"detect_atoms": {"threshold_rel": 0}}, "threshold_rel"),
        (
            ["load", "detect_atoms"], {"detect_atoms": {"image_role": "processed"}},
            "registered workflow",
        ),
        (["load", "report"], {"detect_atoms": {"max_atoms": 1}}, "require a detect_atoms step"),
    ],
)
def test_legacy_atom_options_reject_before_analysis(tmp_path, steps, groups, message):
    recipe = tmp_path / "detect.yaml"
    recipe.write_text(
        yaml.safe_dump({"workflow": {"steps": steps, "parameters": groups}}), encoding="utf-8"
    )
    out = tmp_path / "output"
    with pytest.raises((ValueError, HarnessError), match=message):
        TEMAnalysisSkill().run(tmp_path / "missing.tiff", recipe, out)
    assert not (out / "analysis_report.json").exists()
