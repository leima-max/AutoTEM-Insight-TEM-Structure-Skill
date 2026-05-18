import numpy as np
import pytest

from autotem.core.models import StrainResult
from autotem.strain.atom_strain import compute_atom_strain_fields, run_atom_strain


def _synthetic_lattice_coordinates(
    *,
    nx: int = 14,
    ny: int = 12,
    spacing_x: float = 11.0,
    spacing_y: float = 9.0,
    exx: float = 0.015,
    eyy: float = -0.008,
    exy: float = 0.006,
) -> tuple[np.ndarray, tuple[int, int]]:
    xs = np.arange(nx, dtype=np.float64) * spacing_x + 24.0
    ys = np.arange(ny, dtype=np.float64) * spacing_y + 28.0
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    points_xy = np.column_stack([xx.ravel(), yy.ravel()])
    center_xy = np.mean(points_xy, axis=0)
    strain = np.array(
        [
            [1.0 + exx, exy],
            [exy, 1.0 + eyy],
        ],
        dtype=np.float64,
    )
    observed_xy = center_xy + (points_xy - center_xy) @ strain.T
    coords_yx = observed_xy[:, ::-1]
    height = int(np.ceil(np.max(coords_yx[:, 0]) + 30.0))
    width = int(np.ceil(np.max(coords_yx[:, 1]) + 30.0))
    return coords_yx.astype(np.float64), (height, width)


def _synthetic_piecewise_strain_coordinates(
    *,
    exx: float = 0.015,
    eyy: float = -0.008,
    exy: float = 0.006,
) -> tuple[np.ndarray, tuple[int, int], float]:
    coords, image_shape = _synthetic_lattice_coordinates(exx=0.0, eyy=0.0, exy=0.0)
    points_xy = coords[:, ::-1].copy()
    mid_x = float(np.median(points_xy[:, 0]))
    center_y = float(np.mean(points_xy[:, 1]))
    strain = np.array(
        [
            [exx, exy],
            [exy, eyy],
        ],
        dtype=np.float64,
    )
    right_mask = points_xy[:, 0] > mid_x
    rel = points_xy[right_mask] - np.array([mid_x, center_y], dtype=np.float64)
    points_xy[right_mask] = points_xy[right_mask] + rel @ strain.T
    return points_xy[:, ::-1], image_shape, mid_x


def test_compute_atom_strain_fields_recovers_piecewise_affine_strain() -> None:
    coords, image_shape, mid_x = _synthetic_piecewise_strain_coordinates()
    out = compute_atom_strain_fields(
        coordinates=coords,
        image_shape=image_shape,
        reference_region=(20, image_shape[0] - 20, 20, int(mid_x - 5.0)),
        neighbor_k=10,
        direction_tolerance_deg=18.0,
        interpolation_grid_step=1,
    )

    valid = out.valid_atom_mask
    assert np.any(valid)
    assert out.confidence > 0.75
    right_mask = valid & (coords[:, 1] > mid_x + 3.0)
    left_mask = valid & (coords[:, 1] < mid_x - 8.0)
    assert float(np.mean(out.atom_strain_xx[right_mask])) == pytest.approx(0.015, abs=0.005)
    assert float(np.mean(out.atom_strain_yy[right_mask])) == pytest.approx(-0.008, abs=0.005)
    assert float(np.mean(out.atom_strain_xy[right_mask])) == pytest.approx(0.006, abs=0.005)
    assert abs(float(np.mean(out.atom_strain_xx[left_mask]))) < 0.004
    assert out.strain_xx_map.shape == image_shape
    assert out.rotation_xy_map.shape == image_shape


def test_run_atom_strain_returns_contract_with_reference_mask() -> None:
    coords, image_shape = _synthetic_lattice_coordinates(exx=0.01, eyy=0.0, exy=0.0)
    mask = np.zeros(image_shape, dtype=bool)
    mask[30 : image_shape[0] - 20, 30 : image_shape[1] - 20] = True

    out = run_atom_strain(
        coordinates=coords,
        image_shape=image_shape,
        reference_mask=mask,
        neighbor_k=10,
        direction_tolerance_deg=20.0,
        interpolation_grid_step=2,
        map_inline_mode="full",
    )

    assert isinstance(out, StrainResult)
    assert out.strain_xx is not None
    assert out.strain_yy is not None
    assert out.rotation_xy is not None
    assert out.reference_region is not None
    assert out.provenance["method"] == "atom_ai_stem"
    assert out.provenance["map_stats"]["valid_atom_count"] > 0


def test_atom_strain_warns_on_too_few_atoms() -> None:
    coords = np.array(
        [
            [10.0, 10.0],
            [20.0, 10.0],
            [10.0, 20.0],
            [20.0, 20.0],
        ],
        dtype=np.float64,
    )
    out = run_atom_strain(
        coordinates=coords,
        image_shape=(64, 64),
        neighbor_k=4,
        direction_tolerance_deg=18.0,
        map_inline_mode="preview",
    )

    assert out.confidence == pytest.approx(0.0)
    assert any("need at least 6 atom positions" in w.lower() for w in out.warnings)
