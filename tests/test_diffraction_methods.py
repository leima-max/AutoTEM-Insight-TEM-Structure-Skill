from __future__ import annotations

from pathlib import Path

import numpy as np

from autotem.core import ArrayReference, CalibrationResult, Measurement, PreprocessingResult
from autotem.diffraction.basic import run_diffraction_analysis


def _build_inputs(
    tmp_path: Path,
    image: np.ndarray,
) -> tuple[PreprocessingResult, CalibrationResult]:
    array = image.astype(np.float32)
    input_path = tmp_path / "diffraction_input.npy"
    np.save(input_path, array)

    preprocessing = PreprocessingResult(
        image=ArrayReference(
            shape=tuple(int(v) for v in array.shape),
            dtype=str(array.dtype),
            uri=str(input_path),
            inline=array[:8, :8].tolist(),
        )
    )
    calibration = CalibrationResult(
        pixel_size=Measurement(value=0.1, unit="nm"),
    )
    return preprocessing, calibration


def test_peak_threshold_strategy_controls_peak_count(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260514)
    image = rng.normal(0.0, 0.5, size=(128, 128)).astype(np.float32)
    image[64, 76] += 8.0
    image[64, 52] += 8.0
    image[44, 64] += 5.0
    image[84, 64] += 5.0

    preprocessing, calibration = _build_inputs(tmp_path, image)
    loose = run_diffraction_analysis(
        preprocessing,
        calibration,
        parameters={
            "top_k_peaks": 40,
            "peak_threshold_strategy": "relative_max",
            "peak_threshold_value": 0.05,
        },
    )
    strict = run_diffraction_analysis(
        preprocessing,
        calibration,
        parameters={
            "top_k_peaks": 40,
            "peak_threshold_strategy": "relative_max",
            "peak_threshold_value": 0.45,
        },
    )

    assert len(strict.peaks) <= len(loose.peaks)
    used = strict.provenance.get("used_parameters", {})
    assert used.get("peak_threshold_strategy") == "relative_max"


def test_subpixel_refinement_enables_fractional_peak_coordinates(tmp_path: Path) -> None:
    size = 128
    yy, xx = np.mgrid[0:size, 0:size]
    image = (
        np.sin(2.0 * np.pi * (5.4 * xx / size + 3.7 * yy / size))
        + 0.6 * np.sin(2.0 * np.pi * (11.2 * xx / size - 4.8 * yy / size))
    ).astype(np.float32)
    image += np.random.default_rng(11).normal(0.0, 0.03, size=image.shape).astype(np.float32)

    preprocessing, calibration = _build_inputs(tmp_path, image)
    no_subpixel = run_diffraction_analysis(
        preprocessing,
        calibration,
        parameters={
            "top_k_peaks": 24,
            "peak_threshold_strategy": "relative_max",
            "peak_threshold_value": 0.15,
            "subpixel_refine": False,
        },
    )
    with_subpixel = run_diffraction_analysis(
        preprocessing,
        calibration,
        parameters={
            "top_k_peaks": 24,
            "peak_threshold_strategy": "relative_max",
            "peak_threshold_value": 0.15,
            "subpixel_refine": True,
            "subpixel_window_radius": 2,
            "subpixel_max_shift": 1.5,
        },
    )

    assert no_subpixel.peaks
    assert with_subpixel.peaks
    assert all(
        abs(peak.x - round(peak.x)) < 1e-9 and abs(peak.y - round(peak.y)) < 1e-9
        for peak in no_subpixel.peaks
    )
    assert any(
        abs(peak.x - round(peak.x)) > 1e-3 or abs(peak.y - round(peak.y)) > 1e-3
        for peak in with_subpixel.peaks
    )


def test_lattice_fit_indexing_emits_residual_and_basis_metadata(tmp_path: Path) -> None:
    size = 128
    yy, xx = np.mgrid[0:size, 0:size]
    image = (
        np.sin(2.0 * np.pi * (6.0 * xx / size + 4.0 * yy / size))
        + 0.8 * np.sin(2.0 * np.pi * (10.0 * xx / size - 3.0 * yy / size))
    ).astype(np.float32)
    image += np.random.default_rng(5).normal(0.0, 0.02, size=image.shape).astype(np.float32)

    preprocessing, calibration = _build_inputs(tmp_path, image)
    result = run_diffraction_analysis(
        preprocessing,
        calibration,
        parameters={
            "top_k_peaks": 30,
            "peak_threshold_strategy": "relative_max",
            "peak_threshold_value": 0.15,
            "indexing_mode": "lattice_fit",
            "subpixel_refine": True,
        },
    )

    assert result.indexed_spots
    assert any(spot.error is not None for spot in result.indexed_spots)
    indexing_meta = result.provenance.get("indexing", {})
    assert indexing_meta.get("indexing_method") in {"lattice_fit", "lattice_fit_fallback"}
