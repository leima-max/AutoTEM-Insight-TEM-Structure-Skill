from __future__ import annotations

import numpy as np

from autotem.core import ArrayReference, CalibrationResult, Measurement, PreprocessingResult
from autotem.diffraction.basic import run_diffraction_analysis


def _synthetic_preprocessing_result(
    *,
    size: int = 64,
    radius_px: int = 10,
) -> PreprocessingResult:
    spectrum = np.zeros((size, size), dtype=np.complex128)
    cy = size // 2
    cx = size // 2
    spectrum[cy, cx] = 1200.0
    for dy, dx, amp in [
        (0, radius_px, 240.0),
        (0, -radius_px, 240.0),
        (radius_px, 0, 220.0),
        (-radius_px, 0, 220.0),
    ]:
        spectrum[cy + dy, cx + dx] = amp
    image = np.fft.ifft2(np.fft.ifftshift(spectrum)).real.astype(np.float32)
    return PreprocessingResult(
        image=ArrayReference(
            shape=tuple(int(v) for v in image.shape),
            dtype=str(image.dtype),
            inline=image.tolist(),
            description="Synthetic inverse-FFT image for diffraction tests.",
        ),
        provenance={"source": "test.synthetic_preprocessing_result"},
    )


def _calibration_with_scale(
    scale: float,
    *,
    method: str,
    uncertainty: float = 0.0,
) -> CalibrationResult:
    return CalibrationResult(
        diffraction_scale=Measurement(
            value=scale,
            unit="1/nm/pixel",
            uncertainty=uncertainty,
            method=method,
            confidence=0.95,
        ),
        confidence=0.9,
        provenance={"source": "test.calibration"},
    )


def _nearest_nm_error(d_spacings: list[Measurement], target_nm: float) -> float:
    values = [
        float(item.value)
        for item in d_spacings
        if isinstance(item.value, (int, float)) and item.unit == "nm"
    ]
    if not values:
        return float("inf")
    return min(abs(value - target_nm) for value in values)


def test_diffraction_metrics_scale_drives_d_spacing() -> None:
    expected_d_nm = 0.2
    radius_px = 10
    scale = 1.0 / (radius_px * expected_d_nm)
    preprocessing = _synthetic_preprocessing_result(radius_px=radius_px)
    calibration = _calibration_with_scale(
        scale,
        method="known_d_spacing.single_pair",
    )

    result = run_diffraction_analysis(
        preprocessing,
        calibration,
        parameters={
            "top_k_peaks": 12,
            "min_peak_distance": 3,
            "center_exclusion_radius": 4,
            "peak_threshold_strategy": "relative_max",
            "peak_threshold_value": 0.15,
        },
    )

    assert result.peaks
    assert result.d_spacings
    assert _nearest_nm_error(result.d_spacings, expected_d_nm) < 0.025
    assert result.provenance.get("diffraction_scale_source") == "known_d_spacing.single_pair"
    consistency = result.provenance.get("b3_b4_consistency", {})
    assert isinstance(consistency, dict)
    assert "indexed_nonzero_fraction" in consistency


def test_diffraction_metrics_calibrated_scale_reduces_d_spacing_error() -> None:
    expected_d_nm = 0.2
    radius_px = 10
    calibrated_scale = 1.0 / (radius_px * expected_d_nm)
    drifted_scale = calibrated_scale * 0.83

    preprocessing = _synthetic_preprocessing_result(radius_px=radius_px)
    result_drifted = run_diffraction_analysis(
        preprocessing,
        _calibration_with_scale(drifted_scale, method="pixel_size_fft_fallback"),
        parameters={
            "top_k_peaks": 12,
            "min_peak_distance": 3,
            "center_exclusion_radius": 4,
            "peak_threshold_strategy": "relative_max",
            "peak_threshold_value": 0.15,
        },
    )
    result_calibrated = run_diffraction_analysis(
        preprocessing,
        _calibration_with_scale(
            calibrated_scale,
            method="known_d_spacing.multi_reference",
            uncertainty=0.002,
        ),
        parameters={
            "top_k_peaks": 12,
            "min_peak_distance": 3,
            "center_exclusion_radius": 4,
            "peak_threshold_strategy": "relative_max",
            "peak_threshold_value": 0.15,
        },
    )

    drifted_error = _nearest_nm_error(result_drifted.d_spacings, expected_d_nm)
    calibrated_error = _nearest_nm_error(result_calibrated.d_spacings, expected_d_nm)

    assert calibrated_error < drifted_error
    assert calibrated_error < 0.025


def test_diffraction_metrics_propagates_scale_uncertainty() -> None:
    expected_d_nm = 0.2
    radius_px = 10
    scale = 1.0 / (radius_px * expected_d_nm)
    preprocessing = _synthetic_preprocessing_result(radius_px=radius_px)
    calibration = _calibration_with_scale(
        scale,
        method="known_d_spacing.multi_reference",
        uncertainty=0.003,
    )

    result = run_diffraction_analysis(
        preprocessing,
        calibration,
        parameters={
            "top_k_peaks": 12,
            "min_peak_distance": 3,
            "center_exclusion_radius": 4,
            "peak_threshold_strategy": "relative_max",
            "peak_threshold_value": 0.15,
        },
    )

    assert result.g_vectors
    assert result.d_spacings
    assert any(
        vector.magnitude is not None
        and vector.magnitude.uncertainty is not None
        and vector.magnitude.uncertainty > 0
        for vector in result.g_vectors
    )
    assert any(
        item.uncertainty is not None and item.uncertainty > 0 for item in result.d_spacings
    )
    assert any(
        vector.angle is not None
        and vector.angle.uncertainty is not None
        and vector.angle.uncertainty > 0
        for vector in result.g_vectors
    )


def test_diffraction_metrics_position_and_center_uncertainty_affect_d_uncertainty() -> None:
    expected_d_nm = 0.2
    radius_px = 10
    scale = 1.0 / (radius_px * expected_d_nm)
    preprocessing = _synthetic_preprocessing_result(radius_px=radius_px)
    calibration = _calibration_with_scale(
        scale,
        method="known_d_spacing.multi_reference",
        uncertainty=0.0,
    )

    result_lo = run_diffraction_analysis(
        preprocessing,
        calibration,
        parameters={
            "top_k_peaks": 12,
            "min_peak_distance": 3,
            "center_exclusion_radius": 4,
            "peak_threshold_strategy": "relative_max",
            "peak_threshold_value": 0.15,
            "direct_beam_center_uncertainty_px": 0.0,
        },
    )
    result_hi = run_diffraction_analysis(
        preprocessing,
        calibration,
        parameters={
            "top_k_peaks": 12,
            "min_peak_distance": 3,
            "center_exclusion_radius": 4,
            "peak_threshold_strategy": "relative_max",
            "peak_threshold_value": 0.15,
            "direct_beam_center_uncertainty_px": 0.8,
        },
    )

    unc_lo = [item.uncertainty or 0.0 for item in result_lo.d_spacings]
    unc_hi = [item.uncertainty or 0.0 for item in result_hi.d_spacings]
    assert unc_lo
    assert unc_hi
    assert max(unc_hi) > max(unc_lo)
    model_text = (
        result_hi.provenance.get("reciprocal_metrics", {}).get("uncertainty_model", "")
    )
    assert "sigma_g^2" in model_text


def test_diffraction_metrics_warning_grading_for_low_peak_count() -> None:
    expected_d_nm = 0.2
    radius_px = 10
    scale = 1.0 / (radius_px * expected_d_nm)
    preprocessing = _synthetic_preprocessing_result(radius_px=radius_px)
    calibration = _calibration_with_scale(scale, method="known_d_spacing.single_pair")

    result = run_diffraction_analysis(
        preprocessing,
        calibration,
        parameters={
            "top_k_peaks": 2,
            "min_peak_distance": 6,
            "center_exclusion_radius": 4,
            "peak_threshold_strategy": "relative_max",
            "peak_threshold_value": 0.20,
        },
    )
    assert any(
        item.startswith("WARNING: Few diffraction peaks detected")
        for item in result.warnings
    )
