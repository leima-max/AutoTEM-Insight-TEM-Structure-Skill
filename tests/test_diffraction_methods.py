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


def test_b4_mvp_indexing_mode_emits_phase_rank_and_hkl_assignments(tmp_path: Path) -> None:
    size = 128
    yy, xx = np.mgrid[0:size, 0:size]
    image = (
        np.sin(2.0 * np.pi * (6.0 * xx / size))
        + 0.9 * np.sin(2.0 * np.pi * (6.0 * yy / size))
        + 0.8 * np.sin(2.0 * np.pi * (6.0 * (xx + yy) / size))
    ).astype(np.float32)
    image += np.random.default_rng(77).normal(0.0, 0.03, size=image.shape).astype(np.float32)

    preprocessing, calibration = _build_inputs(tmp_path, image)
    result = run_diffraction_analysis(
        preprocessing,
        calibration,
        parameters={
            "top_k_peaks": 24,
            "peak_threshold_strategy": "relative_max",
            "peak_threshold_value": 0.12,
            "indexing_mode": "b4_mvp",
            "max_hkl": 4,
            "d_spacing_tolerance_rel": 0.12,
            "angle_tolerance_deg": 15.0,
            "candidate_phase_params": [
                {
                    "name": "phase_good",
                    "a": 2.13,
                    "b": 2.13,
                    "c": 2.13,
                    "alpha": 90,
                    "beta": 90,
                    "gamma": 90,
                    "space_group": "Pm-3m",
                },
                {
                    "name": "phase_bad",
                    "a": 0.62,
                    "b": 0.62,
                    "c": 0.62,
                    "alpha": 90,
                    "beta": 90,
                    "gamma": 90,
                    "space_group": "Pm-3m",
                },
            ],
        },
    )

    assert result.phase_candidates
    phase_names = [str(item.name) for item in result.phase_candidates]
    assert "phase_good" in phase_names
    assert result.indexed_spots
    assert any((spot.h, spot.k, spot.l) != (0, 0, 0) for spot in result.indexed_spots)
    indexing_meta = result.provenance.get("indexing", {})
    assert indexing_meta.get("indexing_method") == "b4_mvp"
    assert isinstance(indexing_meta.get("b4"), dict)


def _make_synthetic_fft_real_image(
    *,
    size: int,
    positive_offsets: list[tuple[int, int]],
    center_amp: float = 1200.0,
    base_bg: float = 8.0,
    streak_amp: float = 28.0,
    rng_seed: int = 123,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    cy = size // 2
    cx = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    rr = np.hypot(xx - cx, yy - cy)
    mag = base_bg / (1.0 + (rr / max(8.0, size * 0.08)) ** 1.4)
    mag[cy, cx] += center_amp

    true_points: list[tuple[float, float]] = []
    for dy, dx in positive_offsets:
        for sign in (1, -1):
            y = cy + sign * dy
            x = cx + sign * dx
            if not (2 <= y < size - 2 and 2 <= x < size - 2):
                continue
            amp = 120.0 if abs(dy) + abs(dx) < 40 else 65.0
            blob = np.exp(-(((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * 1.15**2)))
            mag += amp * blob
            true_points.append((float(x), float(y)))

    mag[cy - 1 : cy + 2, :] += streak_amp
    mag[:, cx - 1 : cx + 2] += streak_amp
    rng = np.random.default_rng(rng_seed)
    mag += rng.normal(0.0, 2.2, size=mag.shape)
    mag = np.clip(mag, 0.0, None).astype(np.float64)

    phase = rng.uniform(0.0, 2.0 * np.pi, size=(size, size))
    f_shift = mag * np.exp(1j * phase)
    f = np.fft.ifftshift(f_shift)
    image = np.fft.ifft2(f).real.astype(np.float32)
    image = (image - float(np.mean(image))).astype(np.float32)
    return image, true_points


def test_robust_fft_peak_detection_v2_recall_and_precision(tmp_path: Path) -> None:
    size = 256
    offsets = [(10, 18), (16, 27), (24, 36), (30, 12), (34, 28)]
    image, true_points = _make_synthetic_fft_real_image(
        size=size,
        positive_offsets=offsets,
        center_amp=1800.0,
        base_bg=12.0,
        streak_amp=34.0,
        rng_seed=20260515,
    )
    preprocessing, calibration = _build_inputs(tmp_path, image)
    result = run_diffraction_analysis(
        preprocessing,
        calibration,
        parameters={
            "top_k_peaks": 96,
            "max_peaks": 96,
            "min_peak_distance": 3.0,
            "min_snr": 2.0,
            "center_mask_radius": 6.0,
            "axis_streak_mask_width": 2.0,
            "border_mask": 2.0,
            "radial_bin_width": 2.0,
            "peak_threshold_strategy": "hybrid",
            "peak_threshold_value": 3.5,
            "fft_window": "hann",
            "fft_compression": "log1p",
            "background_subtraction_method": "hybrid",
            "gaussian_background_sigma": 12.0,
            "return_peak_debug": True,
        },
    )

    detected = [(float(p.x), float(p.y)) for p in result.peaks]
    assert detected
    tol_px = 2.0
    hits = 0
    matched_err: list[float] = []
    for tx, ty in true_points:
        best = min(np.hypot(dx - tx, dy - ty) for dx, dy in detected)
        if best <= tol_px:
            hits += 1
            matched_err.append(float(best))
    recall = hits / max(1, len(true_points))
    assert recall > 0.9
    assert len(detected) <= len(true_points) + 40
    assert matched_err
    assert float(np.mean(matched_err)) < 0.5


def test_robust_fft_peak_detection_v2_d_spacing_error_and_strict_guard(tmp_path: Path) -> None:
    size = 224
    pixel_size_nm = 0.1
    offsets = [(14, 21), (22, 30), (28, 12)]
    image, true_points = _make_synthetic_fft_real_image(
        size=size,
        positive_offsets=offsets,
        center_amp=1500.0,
        base_bg=10.0,
        streak_amp=25.0,
        rng_seed=88,
    )
    preprocessing, calibration = _build_inputs(tmp_path, image)
    calibration = calibration.model_copy(
        update={"pixel_size": Measurement(value=pixel_size_nm, unit="nm")}
    )
    result = run_diffraction_analysis(
        preprocessing,
        calibration,
        parameters={
            "top_k_peaks": 80,
            "max_peaks": 80,
            "min_peak_distance": 3.0,
            "min_snr": 2.2,
            "center_mask_radius": 5.0,
            "axis_streak_mask_width": 1.5,
            "border_mask": 2.0,
            "radial_bin_width": 2.0,
            "peak_threshold_strategy": "radial_mad",
            "peak_threshold_value": 3.8,
            "fft_window": "hann",
            "background_subtraction_method": "radial_mad",
            "return_peak_debug": True,
        },
    )

    detected = [(float(p.x), float(p.y), idx) for idx, p in enumerate(result.peaks)]
    assert detected
    if result.center is None:
        cx, cy = size / 2.0, size / 2.0
    else:
        cx, cy = float(result.center[0]), float(result.center[1])
    d_rel_errors: list[float] = []
    for tx, ty in true_points:
        best = min(detected, key=lambda row: np.hypot(row[0] - tx, row[1] - ty))
        if np.hypot(best[0] - tx, best[1] - ty) > 2.0:
            continue
        idx = int(best[2])
        g = result.g_vectors[idx]
        g_norm = float(getattr(g.magnitude, "value", 0.0) or 0.0)
        if g_norm <= 1e-8:
            continue
        d_est = 1.0 / g_norm
        r_true = float(np.hypot(tx - cx, ty - cy))
        g_true = r_true / (size * pixel_size_nm)
        d_true = 1.0 / max(g_true, 1e-8)
        d_rel_errors.append(abs(d_est - d_true) / d_true)
    assert d_rel_errors
    assert float(np.mean(d_rel_errors)) < 0.02

    # strict-indexing guard: only one Friedel pair -> should skip strict B4.
    sparse_image, _ = _make_synthetic_fft_real_image(
        size=192,
        positive_offsets=[(16, 26)],
        center_amp=1800.0,
        base_bg=1.0,
        streak_amp=0.0,
        rng_seed=7,
    )
    sparse_pre, sparse_cal = _build_inputs(tmp_path, sparse_image)
    sparse = run_diffraction_analysis(
        sparse_pre,
        sparse_cal,
        parameters={
            "top_k_peaks": 30,
            "max_peaks": 8,
            "min_snr": 3.5,
            "center_mask_radius": 6.0,
            "axis_streak_mask_width": 0.0,
            "peak_threshold_strategy": "hybrid",
            "peak_threshold_value": 5.0,
            "indexing_mode": "b4_strict",
            "min_pair_count_for_strict": 3,
            "candidate_phase_params": [
                {
                    "name": "test_phase",
                    "a": 0.352,
                    "b": 0.352,
                    "c": 0.352,
                    "alpha": 90,
                    "beta": 90,
                    "gamma": 90,
                    "space_group": "Fm-3m",
                }
            ],
        },
    )
    assert any("peak_detection_insufficient" in item for item in sparse.warnings)
    indexing_meta = sparse.provenance.get("indexing", {})
    b4_meta = indexing_meta.get("b4", {}) if isinstance(indexing_meta, dict) else {}
    assert isinstance(b4_meta, dict)
    assert b4_meta.get("reason") == "peak_detection_insufficient"
