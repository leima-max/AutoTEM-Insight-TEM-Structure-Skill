import numpy as np
import pytest

from autotem.core import DiffractionResult, GVector, Measurement
from autotem.strain import run_gpa


def _synthetic_lattice_with_linear_strain(
    shape: tuple[int, int] = (128, 128),
    g1: tuple[float, float] = (0.085, 0.0),
    g2: tuple[float, float] = (0.0, 0.082),
    exx: float = 0.012,
) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float64)
    cx = shape[1] / 2.0

    ux = exx * (xx - cx)
    uy = np.zeros_like(ux)

    phase1 = 2.0 * np.pi * (g1[0] * (xx - ux) + g1[1] * (yy - uy))
    phase2 = 2.0 * np.pi * (g2[0] * (xx - ux) + g2[1] * (yy - uy))

    image = np.cos(phase1) + np.cos(phase2)
    image += 0.03 * np.random.default_rng(0).normal(size=shape)
    return image.astype(np.float32)


def _make_diffraction_with_gvectors_nm(
    g1_px: tuple[float, float],
    g2_px: tuple[float, float],
    pixel_size_nm: float,
) -> DiffractionResult:
    g1_nm = (g1_px[0] / pixel_size_nm, g1_px[1] / pixel_size_nm)
    g2_nm = (g2_px[0] / pixel_size_nm, g2_px[1] / pixel_size_nm)
    return DiffractionResult(
        g_vectors=[
            GVector(
                gx=Measurement(value=g1_nm[0], unit="1/nm", method="test"),
                gy=Measurement(value=g1_nm[1], unit="1/nm", method="test"),
            ),
            GVector(
                gx=Measurement(value=g2_nm[0], unit="1/nm", method="test"),
                gy=Measurement(value=g2_nm[1], unit="1/nm", method="test"),
            ),
        ]
    )


def test_run_gpa_b3_with_diffraction_vectors() -> None:
    pixel_size_nm = 0.02
    g1 = (0.085, 0.0)
    g2 = (0.0, 0.082)
    image = _synthetic_lattice_with_linear_strain(g1=g1, g2=g2, exx=0.012)
    diffraction = _make_diffraction_with_gvectors_nm(g1, g2, pixel_size_nm=pixel_size_nm)

    out = run_gpa(
        image=image,
        method="gpa_b3",
        diffraction_result=diffraction,
        pixel_size_nm=pixel_size_nm,
        reference_region=(32, 96, 32, 96),
    )

    assert out.strain_xx is not None
    assert out.strain_yy is not None
    assert out.strain_xy is not None
    assert out.rotation_xy is not None
    assert out.reference_region == (32, 96, 32, 96)
    assert out.provenance["method"] == "gpa_b3"
    assert len(out.g_vectors) == 2
    assert 0.0 <= (out.confidence or 0.0) <= 1.0
    mean_exx = float(out.provenance["map_stats"]["mean_exx"])
    assert abs(mean_exx - 0.012) < 0.02


def test_run_gpa_with_manual_vectors_in_pixel_unit() -> None:
    image = _synthetic_lattice_with_linear_strain(exx=0.008)
    out = run_gpa(
        image=image,
        method="gpa_b3",
        g_vectors=[(0.085, 0.0), (0.0, 0.082)],
        g_vector_unit="1/pixel",
    )
    assert out.provenance["method"] == "gpa_b3"
    assert out.strain_xx is not None
    assert len(out.g_vectors) == 2


def test_run_ppa_ai_stem_method() -> None:
    image = _synthetic_lattice_with_linear_strain(exx=0.006)
    out = run_gpa(
        image=image,
        method="ppa_ai_stem",
        reference_region=(40, 88, 40, 88),
        ppa_window_radius=7,
        ppa_grid_step=6,
    )
    assert out.displacement_x is not None
    assert out.displacement_y is not None
    assert out.strain_xx is not None
    assert out.provenance["method"] == "ppa_ai_stem"
    assert out.provenance["method_metadata"]["sample_count"] > 10
    assert 0.0 <= (out.confidence or 0.0) <= 1.0


def test_run_gpa_invalid_method_raises() -> None:
    image = np.ones((32, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        run_gpa(image=image, method="unknown")
