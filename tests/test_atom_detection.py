from pathlib import Path

import numpy as np
import pytest

from autotem.atomic.atom_detection import AtomDetectionResult, detect_atoms
from autotem.core.models import AtomLattice


def _synthetic_atoms_image(size: int = 128) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    image = np.zeros((size, size), dtype=np.float64)
    centers = [(32, 32), (32, 96), (64, 64), (96, 32), (96, 96)]
    for cy, cx in centers:
        image += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.4**2)))
    image += 0.04 * np.random.default_rng(0).normal(size=(size, size))
    return image


def test_detect_atoms_from_ndarray_basic() -> None:
    image = _synthetic_atoms_image()
    result = detect_atoms(image, mode="auto", gaussian_sigma=0.8, threshold_rel=0.2, min_distance=4)

    assert isinstance(result, AtomDetectionResult)
    assert result.coordinates.ndim == 2
    assert result.coordinates.shape[1] == 2
    assert len(result.coordinates) >= 4
    assert len(result.intensities) == len(result.coordinates)
    assert 0.0 <= result.confidence <= 1.0
    assert "source" in result.provenance


def test_detect_atoms_from_tif(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    image = _synthetic_atoms_image()
    input_path = tmp_path / "atoms.tif"
    overlay_path = tmp_path / "overlay.png"
    tifffile.imwrite(str(input_path), image.astype(np.float32))

    result = detect_atoms(
        str(input_path),
        mode="log",
        gaussian_sigma=1.0,
        threshold_rel=0.2,
        min_distance=4,
        overlay_path=overlay_path,
    )
    assert len(result.coordinates) >= 4
    assert result.overlay_path is not None
    assert overlay_path.exists()


def test_invalid_mode_raises() -> None:
    image = _synthetic_atoms_image()
    with pytest.raises(ValueError):
        detect_atoms(image, mode="invalid")


def test_no_atom_warning() -> None:
    image = np.zeros((96, 96), dtype=np.float64)
    result = detect_atoms(image, threshold_rel=0.95)
    assert len(result.coordinates) == 0
    assert any("No atoms detected" in w for w in result.warnings)


def test_slice_index_for_stack_input() -> None:
    image2d = _synthetic_atoms_image(size=96)
    stack = np.stack([np.zeros_like(image2d), image2d], axis=0)
    result = detect_atoms(stack, slice_index=1, threshold_rel=0.2, min_distance=4)
    assert len(result.coordinates) >= 4


def test_slice_index_out_of_range_raises() -> None:
    stack = np.zeros((2, 48, 48), dtype=np.float64)
    with pytest.raises(ValueError):
        detect_atoms(stack, slice_index=3)


def test_to_atom_lattice_contract_mapping() -> None:
    image = _synthetic_atoms_image()
    result = detect_atoms(image, mode="log", gaussian_sigma=0.8, min_distance=4, threshold_rel=0.2)
    lattice = result.to_atom_lattice(coordinate_unit="pixel")

    assert isinstance(lattice, AtomLattice)
    assert lattice.coordinate_unit == "pixel"
    assert len(lattice.atoms) == len(result.coordinates)
    assert lattice.confidence == result.confidence
    assert lattice.provenance["coordinate_convention_input"] == "yx"

    if len(lattice.atoms) > 0:
        atom0 = lattice.atoms[0]
        y0, x0 = result.coordinates[0]
        assert atom0.x == pytest.approx(float(x0))
        assert atom0.y == pytest.approx(float(y0))
        assert atom0.intensity is not None
        assert atom0.intensity.value == pytest.approx(float(result.intensities[0]))
        assert atom0.intensity.unit == "a.u."


def test_to_atom_lattice_invalid_unit_raises() -> None:
    image = _synthetic_atoms_image()
    result = detect_atoms(image)
    with pytest.raises(ValueError):
        result.to_atom_lattice(coordinate_unit="px")
