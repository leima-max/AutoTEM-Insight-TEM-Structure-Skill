"""B1 quality assessment metrics.

Metrics are deterministic, lightweight, and inspired by common microscopy
practice: signal-to-noise, blur/focus, drift proxy, contamination risk,
astigmatism proxy, and undersampling risk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from autotem.core import Measurement, QualityAssessment, TEMDataset


def _read_thresholds(parameters: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(parameters, dict):
        return {}

    thresholds: dict[str, float] = {}
    raw_thresholds = parameters.get("thresholds")
    if isinstance(raw_thresholds, dict):
        source = raw_thresholds
    else:
        source = parameters

    for key in (
        "snr_min",
        "blur_risk_max",
        "drift_px_max",
        "contamination_risk_max",
        "astigmatism_risk_max",
        "undersampling_risk_max",
        "overall_score_min",
    ):
        value = source.get(key)
        if value is None:
            continue
        try:
            thresholds[key] = float(value)
        except (TypeError, ValueError):
            continue
    return thresholds


def _apply_quality_threshold_warnings(
    warnings: list[str],
    *,
    thresholds: dict[str, float],
    snr_value: float,
    blur_risk: float,
    drift_px: float | None,
    contamination_value: float,
    astig_risk: float,
    undersampling_risk: float,
    overall: float,
) -> list[str]:
    def _warn_if_below(name: str, value: float, threshold: float) -> None:
        if value < threshold:
            warnings.append(
                f"Quality threshold not met: {name}={value:.4g} < {threshold:.4g}."
            )

    def _warn_if_above(name: str, value: float, threshold: float) -> None:
        if value > threshold:
            warnings.append(
                f"Quality threshold not met: {name}={value:.4g} > {threshold:.4g}."
            )

    if "snr_min" in thresholds:
        _warn_if_below("snr", snr_value, thresholds["snr_min"])
    if "blur_risk_max" in thresholds:
        _warn_if_above("blur_risk", blur_risk, thresholds["blur_risk_max"])
    if "drift_px_max" in thresholds and drift_px is not None:
        _warn_if_above("drift_px", drift_px, thresholds["drift_px_max"])
    if "contamination_risk_max" in thresholds:
        _warn_if_above(
            "contamination_risk",
            contamination_value,
            thresholds["contamination_risk_max"],
        )
    if "astigmatism_risk_max" in thresholds:
        _warn_if_above(
            "astigmatism_risk",
            astig_risk,
            thresholds["astigmatism_risk_max"],
        )
    if "undersampling_risk_max" in thresholds:
        _warn_if_above(
            "undersampling_risk",
            undersampling_risk,
            thresholds["undersampling_risk_max"],
        )
    if "overall_score_min" in thresholds:
        _warn_if_below("overall_score", overall, thresholds["overall_score_min"])
    return warnings


def load_dataset_array(dataset: TEMDataset) -> np.ndarray | None:
    """Load pixel array from dataset URI for quality and preprocessing steps."""
    uri = dataset.data.uri
    if uri is None:
        return None
    path = Path(uri)
    if not path.exists():
        return None
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False).astype(np.float32)
    if path.suffix.lower() in {".txt", ".csv"}:
        return np.loadtxt(path).astype(np.float32)
    return None


def _phase_correlation_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float]:
    f_ref = np.fft.fft2(reference)
    f_mov = np.fft.fft2(moving)
    cross_power = f_ref * np.conj(f_mov)
    denom = np.abs(cross_power)
    cross_power /= np.maximum(denom, 1e-12)
    corr = np.fft.ifft2(cross_power)
    max_pos = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    shift_y = float(max_pos[0])
    shift_x = float(max_pos[1])
    if shift_y > reference.shape[0] / 2.0:
        shift_y -= float(reference.shape[0])
    if shift_x > reference.shape[1] / 2.0:
        shift_x -= float(reference.shape[1])
    return shift_y, shift_x


def _estimate_drift_px(image: np.ndarray) -> float | None:
    if image.ndim != 2:
        return None
    if image.shape[0] < 8 or image.shape[1] < 8:
        return None
    half = image.shape[0] // 2
    top = image[:half, :]
    bottom = image[half : half * 2, :]
    if top.shape != bottom.shape:
        return None
    dy, dx = _phase_correlation_shift(top, bottom)
    return float(np.hypot(dx, dy))


def _estimate_astigmatism(image: np.ndarray) -> float:
    if image.ndim != 2:
        return 0.5
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image)))
    spectrum = np.asarray(spectrum, dtype=np.float64)
    total = float(np.sum(spectrum))
    if total <= 0:
        return 0.5
    h, w = spectrum.shape
    yy, xx = np.mgrid[0:h, 0:w]
    yy = yy - (h - 1.0) / 2.0
    xx = xx - (w - 1.0) / 2.0
    cov_xx = float(np.sum(spectrum * xx * xx) / total)
    cov_yy = float(np.sum(spectrum * yy * yy) / total)
    cov_xy = float(np.sum(spectrum * xx * yy) / total)
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    if det <= 0 or trace <= 0:
        return 0.5
    delta = max(trace * trace - 4.0 * det, 0.0)
    eig1 = 0.5 * (trace + np.sqrt(delta))
    eig2 = 0.5 * (trace - np.sqrt(delta))
    if eig1 <= 0 or eig2 <= 0:
        return 0.5
    ratio = float(np.sqrt(max(eig1, eig2) / min(eig1, eig2)))
    return max(0.0, min(1.0, (ratio - 1.0) / 2.0))


def assess_image_quality(
    dataset: TEMDataset,
    *,
    parameters: dict[str, Any] | None = None,
) -> QualityAssessment:
    """Compute B1-style quality metrics for downstream gating and warnings."""
    image = load_dataset_array(dataset)
    warnings = list(dataset.warnings)
    thresholds = _read_thresholds(parameters)
    metadata = dataset.metadata if isinstance(dataset.metadata, dict) else {}
    metadata_norm = (
        metadata.get("normalized")
        if isinstance(metadata.get("normalized"), dict)
        else {}
    )
    metadata_completeness = metadata_norm.get("metadata_completeness")
    if metadata_completeness is None:
        metadata_completeness = 0.7 if metadata else 0.3
    metadata_completeness = float(np.clip(float(metadata_completeness), 0.0, 1.0))

    if image is None or image.size == 0:
        warnings.append("Quality metrics are approximate because pixel data was not loaded.")
        return QualityAssessment(
            overall_score=0.3,
            metadata_completeness=metadata_completeness,
            confidence=0.35,
            warnings=warnings,
            provenance={"method": "b1_quality_no_pixels"},
        )

    finite = np.isfinite(image)
    valid = image[finite]
    if valid.size == 0:
        warnings.append("Image contains no finite values.")
        return QualityAssessment(
            overall_score=0.1,
            metadata_completeness=metadata_completeness,
            confidence=0.1,
            warnings=warnings,
            provenance={"method": "b1_quality_invalid_pixels"},
        )

    image = image.astype(np.float32)
    low_p = float(np.percentile(valid, 5))
    high_p = float(np.percentile(valid, 95))
    signal_span = max(high_p - low_p, 1e-12)

    smooth = ndimage.gaussian_filter(image, sigma=1.0)
    residual = image - smooth
    noise_std = float(np.std(residual[np.isfinite(residual)]))
    snr_value = signal_span / max(noise_std, 1e-12)

    lap = ndimage.laplace(image.astype(np.float64))
    lap_var = float(np.var(lap[np.isfinite(lap)]))
    image_var = float(np.var(valid))
    blur_risk = float(np.exp(-lap_var / max(image_var, 1e-12)))

    drift_px = _estimate_drift_px(image)
    if drift_px is None:
        warnings.append("Drift proxy unavailable: image too small or unsupported dimensionality.")
        drift_risk = 0.5
    else:
        drift_risk = max(0.0, min(1.0, drift_px / 3.0))
        if drift_px > 1.5:
            warnings.append("Potential drift detected from phase-correlation proxy.")

    high_clip_ratio = float(np.mean(valid >= np.percentile(valid, 99.8)))
    contamination_value = max(0.0, min(1.0, high_clip_ratio * 10.0))
    if contamination_value > 0.4:
        warnings.append("Potential contamination/hot-pixel risk detected.")

    astig_risk = _estimate_astigmatism(image)
    if astig_risk > 0.35:
        warnings.append("FFT anisotropy indicates potential astigmatism.")

    nyquist_band = np.abs(np.fft.fftshift(np.fft.fft2(image)))
    h, w = nyquist_band.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(yy - (h - 1.0) / 2.0, xx - (w - 1.0) / 2.0)
    rr_norm = rr / max(float(np.max(rr)), 1e-12)
    hf_energy = float(np.mean(nyquist_band[rr_norm > 0.8]))
    lf_energy = float(np.mean(nyquist_band[rr_norm < 0.4]))
    if lf_energy <= 0:
        undersampling_risk = 0.5
    else:
        undersampling_risk = float(np.clip((hf_energy / lf_energy - 0.6) / 0.8, 0.0, 1.0))
    if undersampling_risk > 0.5:
        warnings.append("High-frequency energy suggests potential undersampling/aliasing risk.")

    snr_score = float(np.clip(np.tanh(snr_value / 8.0), 0.0, 1.0))
    sharpness_score = float(np.clip(1.0 - blur_risk, 0.0, 1.0))
    drift_score = float(np.clip(1.0 - drift_risk, 0.0, 1.0))
    contamination_score = float(np.clip(1.0 - contamination_value, 0.0, 1.0))
    astig_score = float(np.clip(1.0 - astig_risk, 0.0, 1.0))
    undersampling_score = float(np.clip(1.0 - undersampling_risk, 0.0, 1.0))
    overall = (
        0.24 * snr_score
        + 0.20 * sharpness_score
        + 0.14 * drift_score
        + 0.14 * contamination_score
        + 0.14 * astig_score
        + 0.14 * undersampling_score
    )
    overall = float(np.clip(0.85 * overall + 0.15 * metadata_completeness, 0.0, 1.0))

    if snr_value < 3.0:
        warnings.append("Low SNR detected; downstream peak detection may be unstable.")

    warnings = _apply_quality_threshold_warnings(
        warnings,
        thresholds=thresholds,
        snr_value=snr_value,
        blur_risk=blur_risk,
        drift_px=drift_px,
        contamination_value=contamination_value,
        astig_risk=astig_risk,
        undersampling_risk=undersampling_risk,
        overall=overall,
    )

    confidence = float(np.clip(0.5 + 0.4 * metadata_completeness, 0.0, 1.0))
    return QualityAssessment(
        overall_score=overall,
        snr=Measurement(value=snr_value, unit="a.u.", method="p95_p5_over_residual_std"),
        drift=Measurement(value=drift_px, unit="px", method="phase_correlation_proxy")
        if drift_px is not None
        else None,
        blur=Measurement(value=blur_risk, unit="risk", method="laplacian_variance_norm"),
        contamination=Measurement(
            value=contamination_value,
            unit="risk",
            method="bright_tail_ratio",
            parameters={"percentile": 99.8},
        ),
        astigmatism=Measurement(
            value=astig_risk,
            unit="risk",
            method="fft_second_moment_ellipticity",
        ),
        undersampling_risk=undersampling_risk,
        metadata_completeness=metadata_completeness,
        confidence=confidence,
        warnings=warnings,
        provenance={
            "method": "b1_quality_metrics",
            "thresholds": thresholds,
            "components": {
                "snr_score": snr_score,
                "sharpness_score": sharpness_score,
                "drift_score": drift_score,
                "contamination_score": contamination_score,
                "astig_score": astig_score,
                "undersampling_score": undersampling_score,
            },
        },
    )
