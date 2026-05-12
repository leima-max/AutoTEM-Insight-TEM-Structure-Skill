from pathlib import Path

import numpy as np
import pytest

from autotem.core.models import AtomLattice
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
