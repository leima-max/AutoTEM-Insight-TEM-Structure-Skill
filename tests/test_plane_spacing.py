import numpy as np
import pytest

from autotem.atomic.plane_spacing import (
    compute_atomic_plane_spacing_segments,
    compute_atomic_plane_spacing_segments_by_angle,
    compute_atomic_plane_spacing_segments_from_basis,
    compute_atomic_plane_spacing_segments_from_lattice_angle,
    compute_atomic_plane_spacing_segments_from_picked_direction,
)


def _lattice(nx: int = 4, ny: int = 3, ax: float = 10.0, by: float = 8.0):
    coords: list[list[float]] = []
    indices: list[list[int]] = []
    for j in range(ny):
        for i in range(nx):
            coords.append([20.0 + j * by, 30.0 + i * ax])
            indices.append([i, j])
    return np.asarray(coords, dtype=np.float64), np.asarray(indices, dtype=np.int32)


def _rotated_lattice(
    nx: int = 5,
    ny: int = 4,
    ax: float = 10.0,
    by: float = 8.0,
    angle_deg: float = 17.0,
):
    theta = np.deg2rad(angle_deg)
    a_vec = np.array([np.sin(theta), np.cos(theta)], dtype=np.float64) * ax
    b_vec = np.array([np.sin(theta + np.pi / 2.0), np.cos(theta + np.pi / 2.0)]) * by
    origin = np.array([50.0, 60.0], dtype=np.float64)
    coords: list[np.ndarray] = []
    for j in range(ny):
        for i in range(nx):
            coords.append(origin + i * a_vec + j * b_vec)
    return np.asarray(coords, dtype=np.float64)


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


def test_plane_spacing_segments_can_infer_directions_from_lattice_angle_only() -> None:
    coords = _rotated_lattice(angle_deg=17.0)

    out = compute_atomic_plane_spacing_segments_from_lattice_angle(
        coords,
        lattice_angle_deg=90.0,
        angle_tolerance_deg=5.0,
        min_spacing_px=6.0,
        max_spacing_px=12.0,
    )

    assert out.provenance["method"] == "auto_direction_segments_from_lattice_angle"
    stats = out.provenance["fit_statistics"]
    assert stats["axis_separation_deg"] == pytest.approx(90.0)
    assert set(out.direction.tolist()) == {"angle_only_axis_1", "angle_only_axis_2"}
    assert np.sum(out.direction == "angle_only_axis_1") > 0
    assert np.sum(out.direction == "angle_only_axis_2") > 0
    assert np.all((out.length_px >= 6.0) & (out.length_px <= 12.0))


def test_plane_spacing_segments_by_manual_angle_connects_rows() -> None:
    coords, _ = _lattice()

    out = compute_atomic_plane_spacing_segments_by_angle(
        coords,
        angle_deg=0.0,
        angle_tolerance_deg=5.0,
        min_spacing_px=6.0,
        max_spacing_px=12.0,
    )

    assert out.provenance["method"] == "nearest_neighbor_segments_along_image_angle"
    assert len(out.length_px) == 3 * (4 - 1)
    assert set(out.direction.tolist()) == {"angle_0.00deg"}
    assert np.allclose(out.length_px, 10.0)


def test_plane_spacing_segments_by_angle_treats_opposite_axis_as_same_direction() -> None:
    coords, _ = _lattice()

    out_90 = compute_atomic_plane_spacing_segments_by_angle(
        coords,
        angle_deg=90.0,
        angle_tolerance_deg=5.0,
        min_spacing_px=6.0,
        max_spacing_px=10.0,
    )
    out_270 = compute_atomic_plane_spacing_segments_by_angle(
        coords,
        angle_deg=270.0,
        angle_tolerance_deg=5.0,
        min_spacing_px=6.0,
        max_spacing_px=10.0,
    )

    assert len(out_90.length_px) == (3 - 1) * 4
    assert len(out_270.length_px) == len(out_90.length_px)
    assert np.allclose(out_90.length_px, 8.0)
    assert np.allclose(np.sort(out_270.length_px), np.sort(out_90.length_px))


def test_plane_spacing_segments_from_picked_direction_uses_two_atoms() -> None:
    coords, _ = _lattice()

    out = compute_atomic_plane_spacing_segments_from_picked_direction(
        coords,
        start_index=0,
        end_index=1,
        angle_tolerance_deg=5.0,
        min_spacing_px=6.0,
        max_spacing_px=12.0,
    )

    assert out.provenance["method"] == "nearest_neighbor_segments_from_two_picked_atoms"
    assert out.provenance["picked_direction"]["angle_deg"] == pytest.approx(0.0)
    assert len(out.length_px) == 3 * (4 - 1)
    assert np.allclose(out.length_px, 10.0)


def test_plane_spacing_segments_validate_direction() -> None:
    coords, indices = _lattice()

    with pytest.raises(ValueError, match="Unsupported directions"):
        compute_atomic_plane_spacing_segments(coords, indices, directions=("c",))


def test_plane_spacing_segments_by_angle_validate_distance_range() -> None:
    coords, _ = _lattice()

    with pytest.raises(ValueError, match="max_spacing_px"):
        compute_atomic_plane_spacing_segments_by_angle(
            coords,
            angle_deg=0.0,
            min_spacing_px=10.0,
            max_spacing_px=5.0,
        )
