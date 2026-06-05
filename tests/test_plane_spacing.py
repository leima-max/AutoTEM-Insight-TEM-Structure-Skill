import numpy as np
import pytest

from autotem.atomic.plane_spacing import (
    compute_atomic_plane_spacing_segments,
    compute_atomic_plane_spacing_segments_from_basis,
)


def _lattice(nx: int = 4, ny: int = 3, ax: float = 10.0, by: float = 8.0):
    coords: list[list[float]] = []
    indices: list[list[int]] = []
    for j in range(ny):
        for i in range(nx):
            coords.append([20.0 + j * by, 30.0 + i * ax])
            indices.append([i, j])
    return np.asarray(coords, dtype=np.float64), np.asarray(indices, dtype=np.int32)


def test_plane_spacing_segments_connect_a_and_b_neighbors() -> None:
    coords, indices = _lattice()

    out = compute_atomic_plane_spacing_segments(coords, indices)

    assert len(out.length_px) == (3 * (4 - 1)) + ((3 - 1) * 4)
    assert np.sum(out.direction == "a") == 9
    assert np.sum(out.direction == "b") == 8
    assert np.allclose(out.length_px[out.direction == "a"], 10.0)
    assert np.allclose(out.length_px[out.direction == "b"], 8.0)
    assert out.confidence > 0.8


def test_plane_spacing_segments_respect_valid_mask() -> None:
    coords, indices = _lattice()
    valid = np.ones(len(coords), dtype=bool)
    valid[1] = False

    out = compute_atomic_plane_spacing_segments(coords, indices, valid_atom_mask=valid)

    assert not np.any(out.source_atom_index == 1)
    assert not np.any(out.target_atom_index == 1)
    assert len(out.length_px) < 17


def test_plane_spacing_segments_can_index_directly_from_basis() -> None:
    coords, _ = _lattice()

    out = compute_atomic_plane_spacing_segments_from_basis(
        coords,
        base_index=0,
        a_index=1,
        b_index=4,
        lattice_index_tolerance=0.25,
    )

    assert out.provenance["method"] == "direct_short_segments_from_custom_ab_basis"
    assert np.sum(out.direction == "a") == 9
    assert np.sum(out.direction == "b") == 8
    assert np.allclose(out.length_px[out.direction == "a"], 10.0)
    assert np.allclose(out.length_px[out.direction == "b"], 8.0)


def test_plane_spacing_segments_validate_direction() -> None:
    coords, indices = _lattice()

    with pytest.raises(ValueError, match="Unsupported directions"):
        compute_atomic_plane_spacing_segments(coords, indices, directions=("c",))
