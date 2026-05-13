import numpy as np

from autotem.atomic.gaussian_refinement import refine_atom_coordinates
from autotem.atomic.lattice_constraint import apply_lattice_constraint


def _single_peak_image(
    shape: tuple[int, int] = (96, 96),
    center: tuple[float, float] = (41.35, 54.72),
    sigma: float = 2.0,
) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    cy, cx = center
    img = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))
    img += 0.01 * np.random.default_rng(0).normal(size=shape)
    return img


def test_refinement_com_improves_subpixel_error() -> None:
    true_yx = np.array([[41.35, 54.72]], dtype=np.float64)
    image = _single_peak_image(center=(true_yx[0, 0], true_yx[0, 1]))
    coarse = np.array([[41.0, 55.0]], dtype=np.float64)

    res = refine_atom_coordinates(
        image=image,
        coordinates=coarse,
        method="com",
        window_radius=4,
        max_shift=2.0,
    )

    err_before = float(np.linalg.norm(coarse[0] - true_yx[0]))
    err_after = float(np.linalg.norm(res.coordinates_refined[0] - true_yx[0]))
    assert err_after < err_before


def test_refinement_gaussian_improves_subpixel_error() -> None:
    true_yx = np.array([[41.35, 54.72]], dtype=np.float64)
    image = _single_peak_image(center=(true_yx[0, 0], true_yx[0, 1]))
    coarse = np.array([[41.0, 55.0]], dtype=np.float64)

    res = refine_atom_coordinates(
        image=image,
        coordinates=coarse,
        method="gaussian",
        window_radius=4,
        max_shift=2.0,
    )

    err_before = float(np.linalg.norm(coarse[0] - true_yx[0]))
    err_after = float(np.linalg.norm(res.coordinates_refined[0] - true_yx[0]))
    assert err_after < err_before


def test_lattice_constraint_disabled_keeps_coordinates() -> None:
    coords = np.array([[0.0, 0.0], [0.0, 5.0], [5.0, 0.0], [5.0, 5.0]], dtype=np.float64)
    out = apply_lattice_constraint(coords, enabled=False)
    assert np.allclose(out.coordinates_regularized, coords)
    assert out.enabled is False


def test_lattice_constraint_enabled_reduces_outlier_offset() -> None:
    coords = np.array(
        [
            [0.0, 0.0],
            [0.0, 5.0],
            [5.0, 0.0],
            [5.0, 5.0],
            [9.0, 9.0],  # outlier
        ],
        dtype=np.float64,
    )
    center = np.mean(coords[:-1], axis=0)
    dist_before = float(np.linalg.norm(coords[-1] - center))

    out = apply_lattice_constraint(
        coords,
        enabled=True,
        k_neighbors=3,
        strength=0.4,
        iterations=2,
        max_shift_per_iter=1.0,
    )
    dist_after = float(np.linalg.norm(out.coordinates_regularized[-1] - center))

    assert out.enabled is True
    assert dist_after < dist_before
