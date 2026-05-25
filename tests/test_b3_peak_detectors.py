from __future__ import annotations

import numpy as np

from autotem.core import ArrayReference, CalibrationResult, Measurement, PreprocessingResult
from autotem.diffraction.basic import run_diffraction_analysis


def _synthetic_preprocessing_result(
    *,
    size: int = 64,
    radius_px: int = 9,
) -> PreprocessingResult:
    spectrum = np.zeros((size, size), dtype=np.complex128)
    cy = size // 2
    cx = size // 2
    spectrum[cy, cx] = 1800.0
    for dy, dx, amp in [
        (0, radius_px, 260.0),
        (0, -radius_px, 260.0),
        (radius_px, 0, 240.0),
        (-radius_px, 0, 240.0),
    ]:
        spectrum[cy + dy, cx + dx] = amp
    image = np.fft.ifft2(np.fft.ifftshift(spectrum)).real.astype(np.float32)
    return PreprocessingResult(
        image=ArrayReference(
            shape=tuple(int(v) for v in image.shape),
            dtype=str(image.dtype),
            inline=image.tolist(),
            description="synthetic inverse FFT image",
        ),
    )


def _calibration(scale: float) -> CalibrationResult:
    return CalibrationResult(
        diffraction_scale=Measurement(
            value=scale,
            unit="1/nm/pixel",
            uncertainty=0.002,
            method="known_d_spacing.multi_reference",
            confidence=0.95,
        ),
        confidence=0.9,
    )


def test_b3_peak_detectors_localmax_log_dog() -> None:
    radius_px = 9
    expected_d_nm = 0.2
    scale = 1.0 / (radius_px * expected_d_nm)
    preprocessing = _synthetic_preprocessing_result(radius_px=radius_px)
    calibration = _calibration(scale)

    for detector in ("local_max", "log", "dog"):
        result = run_diffraction_analysis(
            preprocessing,
            calibration,
            parameters={
                "peak_detector": detector,
                "detector_sigma": 1.2,
                "dog_sigma_ratio": 1.8,
                "top_k_peaks": 16,
                "min_peak_distance": 3,
                "center_exclusion_radius": 4,
                "peak_threshold_strategy": "relative_max",
                "peak_threshold_value": 0.15,
            },
        )
        assert result.peaks
        assert result.g_vectors
        assert result.d_spacings
        assert result.provenance.get("peak_detection", {}).get("detector")


def test_b3_fft_array_reference_saved_when_source_uri_exists(tmp_path) -> None:
    size = 64
    radius_px = 10
    expected_d_nm = 0.22
    scale = 1.0 / (radius_px * expected_d_nm)
    spectrum = np.zeros((size, size), dtype=np.complex128)
    cy = size // 2
    cx = size // 2
    spectrum[cy, cx] = 1500.0
    spectrum[cy, cx + radius_px] = 220.0
    spectrum[cy, cx - radius_px] = 220.0
    image = np.fft.ifft2(np.fft.ifftshift(spectrum)).real.astype(np.float32)
    input_path = tmp_path / "b3_source.npy"
    np.save(input_path, image)

    preprocessing = PreprocessingResult(
        image=ArrayReference(
            shape=tuple(int(v) for v in image.shape),
            dtype=str(image.dtype),
            uri=str(input_path),
        )
    )
    result = run_diffraction_analysis(
        preprocessing,
        _calibration(scale),
        parameters={
            "top_k_peaks": 12,
            "min_peak_distance": 3,
            "center_exclusion_radius": 4,
            "peak_threshold_strategy": "relative_max",
            "peak_threshold_value": 0.10,
        },
    )
    assert result.fft_image is not None
    assert result.fft_image.uri is not None
    assert result.fft_image.uri.endswith("_fft_magnitude.npy")


def test_b3_peak_adapter_fallback_when_dependency_missing() -> None:
    radius_px = 9
    expected_d_nm = 0.2
    scale = 1.0 / (radius_px * expected_d_nm)
    preprocessing = _synthetic_preprocessing_result(radius_px=radius_px)
    calibration = _calibration(scale)

    result = run_diffraction_analysis(
        preprocessing,
        calibration,
        parameters={
            "peak_adapter_engine": "py4dstem",
            "peak_detector": "local_max",
            "top_k_peaks": 16,
            "min_peak_distance": 3,
            "center_exclusion_radius": 4,
            "peak_threshold_strategy": "relative_max",
            "peak_threshold_value": 0.15,
        },
    )
    assert result.peaks
    assert result.provenance.get("peak_adapter_engine_requested") == "py4dstem"
    assert result.provenance.get("peak_adapter_engine_status") == "internal_fallback"
    assert any("Open-source peak adapter failed" in item for item in result.warnings)


def test_b3_auto_peak_tuning_populates_provenance_and_detects_peaks() -> None:
    radius_px = 9
    expected_d_nm = 0.2
    scale = 1.0 / (radius_px * expected_d_nm)
    preprocessing = _synthetic_preprocessing_result(radius_px=radius_px)
    calibration = _calibration(scale)

    result = run_diffraction_analysis(
        preprocessing,
        calibration,
        parameters={
            "auto_peak_tuning": True,
            "peak_threshold_strategy": "auto",
        },
    )
    assert result.peaks
    auto_diag = result.provenance.get("auto_peak_tuning", {})
    assert isinstance(auto_diag, dict)
    assert auto_diag.get("enabled") is True
    peak_diag = result.provenance.get("peak_detection", {})
    assert isinstance(peak_diag, dict)
    assert str(peak_diag.get("threshold_method", "")).startswith("auto_quantile_")
    assert "auto_snr_proxy_z99" in peak_diag


def test_b3_max_peak_radius_filters_far_outliers() -> None:
    size = 64
    radius_px = 9
    spectrum = np.zeros((size, size), dtype=np.complex128)
    cy = size // 2
    cx = size // 2
    spectrum[cy, cx] = 1800.0
    for dy, dx, amp in [
        (0, radius_px, 260.0),
        (0, -radius_px, 260.0),
        (radius_px, 0, 240.0),
        (-radius_px, 0, 240.0),
        (size // 2 - 3, size // 2 - 3, 400.0),
        (-(size // 2 - 3), -(size // 2 - 3), 400.0),
    ]:
        spectrum[cy + dy, cx + dx] = amp
    image = np.fft.ifft2(np.fft.ifftshift(spectrum)).real.astype(np.float32)
    preprocessing = PreprocessingResult(
        image=ArrayReference(
            shape=tuple(int(v) for v in image.shape),
            dtype=str(image.dtype),
            inline=image.tolist(),
            description="synthetic inverse FFT image with outliers",
        ),
    )
    calibration = _calibration(scale=1.0 / (radius_px * 0.2))
    result = run_diffraction_analysis(
        preprocessing,
        calibration,
        parameters={
            "top_k_peaks": 12,
            "min_peak_distance": 3,
            "center_exclusion_radius": 4,
            "max_peak_radius_px": 15,
            "peak_threshold_strategy": "relative_max",
            "peak_threshold_value": 0.10,
        },
    )
    center_x, center_y = result.center
    radii = [
        float(np.hypot(peak.x - center_x, peak.y - center_y))
        for peak in result.peaks
    ]
    assert radii
    assert max(radii) <= 15.5
    peak_diag = result.provenance.get("peak_detection", {})
    assert peak_diag.get("max_peak_radius_px") == 15.0


def test_b3_radius_outlier_filter_removes_far_pair_without_radius_cap() -> None:
    size = 128
    radius_px = 14
    spectrum = np.zeros((size, size), dtype=np.complex128)
    cy = size // 2
    cx = size // 2
    spectrum[cy, cx] = 2000.0
    for ring_radius, ring_amp, count in [
        (14, 260.0, 4),
        (22, 220.0, 4),
        (30, 180.0, 4),
        (38, 150.0, 4),
    ]:
        for index in range(count):
            angle = (2.0 * np.pi * index / count) + (0.2 if ring_radius == 22 else 0.0)
            dx = int(round(ring_radius * np.cos(angle)))
            dy = int(round(ring_radius * np.sin(angle)))
            spectrum[cy + dy, cx + dx] = ring_amp
    for dy, dx, amp in [
        (size // 2 - 8, size // 2 - 10, 460.0),
        (-(size // 2 - 8), -(size // 2 - 10), 460.0),
    ]:
        spectrum[cy + dy, cx + dx] = amp
    image = np.fft.ifft2(np.fft.ifftshift(spectrum)).real.astype(np.float32)
    preprocessing = PreprocessingResult(
        image=ArrayReference(
            shape=tuple(int(v) for v in image.shape),
            dtype=str(image.dtype),
            inline=image.tolist(),
            description="synthetic inverse FFT image with far outlier pair",
        ),
    )
    calibration = _calibration(scale=1.0 / (radius_px * 0.2))
    result_no_filter = run_diffraction_analysis(
        preprocessing,
        calibration,
        parameters={
            "top_k_peaks": 24,
            "min_peak_distance": 3,
            "center_exclusion_radius": 4,
            "max_peak_radius_px": 5000,
            "radius_outlier_filter": False,
            "peak_threshold_strategy": "relative_max",
            "peak_threshold_value": 0.08,
        },
    )
    radii_no_filter = sorted(
        float(np.hypot(peak.x - result_no_filter.center[0], peak.y - result_no_filter.center[1]))
        for peak in result_no_filter.peaks
    )
    assert radii_no_filter[-1] > 40.0

    result_filtered = run_diffraction_analysis(
        preprocessing,
        calibration,
        parameters={
            "top_k_peaks": 24,
            "min_peak_distance": 3,
            "center_exclusion_radius": 4,
            "max_peak_radius_px": 5000,
            "radius_outlier_filter": True,
            "radius_outlier_mad_factor": 4.0,
            "peak_threshold_strategy": "relative_max",
            "peak_threshold_value": 0.08,
        },
    )
    radii_filtered = sorted(
        float(np.hypot(peak.x - result_filtered.center[0], peak.y - result_filtered.center[1]))
        for peak in result_filtered.peaks
    )
    assert radii_filtered[-1] < radii_no_filter[-1]
    peak_diag = result_filtered.provenance.get("peak_detection", {})
    assert peak_diag.get("radius_outlier_removed_count", 0) >= 1
