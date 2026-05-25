import numpy as np
import pytest

from autotem.atomic.displacement import compute_atomic_displacement_from_basis


def _rect_lattice(nx: int = 6, ny: int = 5, ax: float = 12.0, by: float = 9.0) -> np.ndarray:
    coords: list[list[float]] = []
    for j in range(ny):
        for i in range(nx):
            y = 20.0 + j * by
            x = 30.0 + i * ax
            coords.append([y, x])
    return np.asarray(coords, dtype=np.float64)


def test_atomic_displacement_is_zero_for_perfect_lattice() -> None:
    coords = _rect_lattice()
    base_idx = 0
    a_idx = 1
    b_idx = 6

    out = compute_atomic_displacement_from_basis(
        coords,
        base_index=base_idx,
        a_index=a_idx,
        b_index=b_idx,
    )

    assert out.confidence > 0.95
    assert np.all(out.valid_atom_mask)
    assert np.max(out.displacement_magnitude) < 1e-6
    assert np.allclose(out.coordinates_ideal_yx, coords, atol=1e-6)
    assert np.all(out.lattice_indices_ij[0] == np.array([0, 0]))


def test_atomic_displacement_recovers_shifted_atom_with_reference_mask() -> None:
    coords = _rect_lattice()
    shifted_idx = 14
    coords[shifted_idx] += np.array([0.35, -0.22], dtype=np.float64)
    base_idx = 0
    a_idx = 1
    b_idx = 6
    reference_mask = np.ones(len(coords), dtype=bool)
    reference_mask[shifted_idx] = False

    out = compute_atomic_displacement_from_basis(
        coords,
        base_index=base_idx,
        a_index=a_idx,
        b_index=b_idx,
        reference_mask=reference_mask,
    )

    disp = out.displacement_yx[shifted_idx]
    assert disp[0] == pytest.approx(0.35, abs=0.05)
    assert disp[1] == pytest.approx(-0.22, abs=0.05)
    assert out.provenance["fit_statistics"]["fit_atom_count"] == int(np.sum(reference_mask))
    assert np.mean(out.displacement_magnitude[reference_mask]) < 0.05


def test_atomic_displacement_rejects_collinear_basis() -> None:
    coords = _rect_lattice()
    with pytest.raises(ValueError, match="collinear"):
        compute_atomic_displacement_from_basis(
            coords,
            base_index=0,
            a_index=1,
            b_index=2,
        )
