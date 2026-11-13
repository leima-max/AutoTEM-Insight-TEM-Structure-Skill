"""B1 quality assessment: SNR, drift, blur, contamination, astigmatism, and risk summary."""

from __future__ import annotations

import warnings as pywarnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from autotem.core import Measurement, QualityAssessment, TEMDataset
from autotem.io.readers import read_numeric_file

try:
    from skimage.measure import blur_effect as _skimage_blur_effect
except Exception:  # noqa: BLE001
    _skimage_blur_effect = None

try:
    from skimage.registration import phase_cross_correlation as _phase_cross_correlation
except Exception:  # noqa: BLE001
    _phase_cross_correlation = None

try:
    from skimage.restoration import estimate_sigma as _estimate_sigma
except Exception:  # noqa: BLE001
    _estimate_sigma = None

_BASE_THRESHOLDS: dict[str, float] = {
    "snr_min": 6.0,
    "drift_px_max": 2.0,
    "blur_risk_max": 0.70,
    "contamination_risk_max": 0.70,
    "astigmatism_risk_max": 0.70,
    "undersampling_risk_max": 0.80,
    "overall_score_min": 0.35,
}
_MODALITY_THRESHOLDS: dict[str, dict[str, float]] = {
    "HRTEM": {
        "snr_min": 8.0,
        "drift_px_max": 1.6,
        "blur_risk_max": 0.62,
        "astigmatism_risk_max": 0.60,
        "undersampling_risk_max": 0.72,
    },
    "HAADF-STEM": {
        "snr_min": 7.0,
        "drift_px_max": 1.8,
        "contamination_risk_max": 0.62,
        "undersampling_risk_max": 0.70,
    },
    "ABF-STEM": {
        "snr_min": 7.0,
        "drift_px_max": 1.8,
        "blur_risk_max": 0.64,
        "undersampling_risk_max": 0.72,
    },
    "SAED": {
        "snr_min": 5.0,
        "drift_px_max": 3.5,
        "blur_risk_max": 0.75,
        "undersampling_risk_max": 0.85,
    },
    "NBED": {
        "snr_min": 5.5,
        "drift_px_max": 3.0,
        "blur_risk_max": 0.74,
        "undersampling_risk_max": 0.82,
    },
    "4D-STEM": {
        "snr_min": 5.5,
        "drift_px_max": 2.6,
        "blur_risk_max": 0.72,
        "undersampling_risk_max": 0.80,
    },
}
_BASE_SCORE_WEIGHTS: dict[str, float] = {
    "snr": 1.0,
    "drift": 1.0,
    "blur": 1.0,
    "contamination": 1.0,
    "astigmatism": 1.0,
    "undersampling": 1.0,
    "metadata": 0.8,
}
_MODALITY_SCORE_WEIGHTS: dict[str, dict[str, float]] = {
    "HRTEM": {"blur": 1.4, "astigmatism": 1.3, "drift": 1.2},
    "HAADF-STEM": {"contamination": 1.4, "undersampling": 1.3, "snr": 1.1},
    "ABF-STEM": {"blur": 1.3, "undersampling": 1.2},
    "SAED": {"undersampling": 1.5, "metadata": 1.0, "drift": 0.8},
    "NBED": {"undersampling": 1.4, "drift": 0.9},
    "4D-STEM": {"undersampling": 1.4, "drift": 0.9},
}
_WEIGHT_KEYS: tuple[str, ...] = (
    "snr",
    "drift",
    "blur",
    "contamination",
    "astigmatism",
    "undersampling",
    "metadata",
)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        import re

        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value.strip())
        if match:
            return float(match.group(0))
    return None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _load_from_uri(uri: str | None) -> np.ndarray | None:
    if not uri:
        return None
    path = Path(uri)
    if not path.exists() or not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix in {".npy", ".txt", ".csv"}:
        array = read_numeric_file(path)
        return np.asarray(array, dtype=np.float64) if array.ndim == 2 else None
    if suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        try:
            from PIL import Image
        except ImportError:
            return None
        with Image.open(path) as handle:
            image = np.asarray(handle)
        if image.ndim == 3:
            image = image.mean(axis=2)
        return np.asarray(image, dtype=np.float64) if image.ndim == 2 else None
    # Fallback for microscopy formats handled by autotem.io readers
    # (e.g., dm3/dm4/mrc/h5/emd/ser) that are not directly loadable here.
    try:
        from autotem.io.readers import read_image
    except Exception:  # noqa: BLE001
        return None
    try:
        read_result = read_image(path)
    except Exception:  # noqa: BLE001
        return None
    array = np.asarray(read_result.array, dtype=np.float64)
    if array.ndim == 3:
        array = array.mean(axis=2)
    return array if array.ndim == 2 else None
    return None


def load_dataset_array(dataset: TEMDataset) -> np.ndarray | None:
    """Load 2D array from dataset inline payload or URI."""
    data_ref = dataset.data
    if data_ref.inline is not None:
        array = np.asarray(data_ref.inline, dtype=np.float64)
        expected_shape = tuple(int(v) for v in data_ref.shape)
        if expected_shape and array.size == int(np.prod(expected_shape)):
            array = array.reshape(expected_shape)
        if array.ndim == 2:
            return array
    return _load_from_uri(data_ref.uri)


def _estimate_noise_sigma(image: np.ndarray) -> float:
    if _estimate_sigma is not None:
        try:
            sigma = float(_estimate_sigma(image, average_sigmas=True, channel_axis=None))
        except Exception:  # noqa: BLE001
            sigma = 0.0
        if sigma > 0:
            return sigma
    residual = image - ndi.gaussian_filter(image, sigma=1.2)
    mad = float(np.median(np.abs(residual - np.median(residual))))
    return max(1e-12, 1.4826 * mad)


def _estimate_drift_px(image: np.ndarray) -> float:
    h = image.shape[0]
    if h < 8:
        return 0.0
    top = image[: h // 2, :]
    bottom = image[h // 2 :, :]
    if top.shape != bottom.shape:
        min_h = min(top.shape[0], bottom.shape[0])
        top = top[:min_h, :]
        bottom = bottom[:min_h, :]
    if float(np.std(top)) < 1e-12 or float(np.std(bottom)) < 1e-12:
        return 0.0
    if _phase_cross_correlation is not None:
        try:
            with pywarnings.catch_warnings():
                pywarnings.filterwarnings("ignore", category=UserWarning)
                shift, _, _ = _phase_cross_correlation(top, bottom, upsample_factor=20)
            return float(np.hypot(float(shift[0]), float(shift[1])))
        except Exception:  # noqa: BLE001
            pass

    f0 = np.fft.fft2(top)
    f1 = np.fft.fft2(bottom)
    product = f0 * np.conj(f1)
    denom = np.abs(product) + 1e-12
    corr = np.fft.ifft2(product / denom)
    y_idx, x_idx = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    dy = float(y_idx if y_idx <= corr.shape[0] / 2 else y_idx - corr.shape[0])
    dx = float(x_idx if x_idx <= corr.shape[1] / 2 else x_idx - corr.shape[1])
    return float(np.hypot(dy, dx))


def _estimate_blur_risk(image: np.ndarray) -> float:
    if _skimage_blur_effect is not None:
        try:
            return _clip01(float(_skimage_blur_effect(image)))
        except Exception:  # noqa: BLE001
            pass
    lap = ndi.laplace(image)
    lap_var = float(np.var(lap))
    # Higher variance => sharper => lower blur risk.
    return _clip01(1.0 / (1.0 + lap_var / 0.01))


def _estimate_contamination_risk(image: np.ndarray) -> float:
    p995 = float(np.percentile(image, 99.5))
    p999 = float(np.percentile(image, 99.9))
    dynamic = float(np.percentile(image, 99.0) - np.percentile(image, 1.0)) + 1e-12
    bright_mask = image >= p999
    bright_fraction = float(np.mean(bright_mask))
    spike = max(0.0, (p999 - p995) / dynamic)
    return _clip01(0.7 * spike + 0.3 * min(1.0, bright_fraction * 400.0))


def _estimate_astigmatism_risk(fft_magnitude: np.ndarray) -> float:
    log_magnitude = np.log1p(fft_magnitude)
    threshold = float(np.quantile(log_magnitude, 0.985))
    mask = log_magnitude >= threshold
    ys, xs = np.where(mask)
    if len(xs) < 16:
        return 0.5
    x = xs.astype(np.float64) - float(np.mean(xs))
    y = ys.astype(np.float64) - float(np.mean(ys))
    cov = np.cov(np.vstack([x, y]))
    eigvals = np.linalg.eigvalsh(cov)
    lam_min = max(1e-12, float(np.min(eigvals)))
    lam_max = max(lam_min, float(np.max(eigvals)))
    anisotropy = abs(np.log(lam_max / lam_min))
    return _clip01(anisotropy / 2.5)


def _estimate_high_freq_ratio(fft_mag: np.ndarray) -> float:
    h, w = fft_mag.shape
    yy, xx = np.indices((h, w), sparse=True)
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    rr = np.hypot(yy - cy, xx - cx)
    max_r = max(1.0, float(np.max(rr)))
    high_mask = rr >= 0.75 * max_r
    total = float(np.sum(fft_mag)) + 1e-12
    return float(np.sum(fft_mag[high_mask]) / total)


def _metadata_completeness(dataset: TEMDataset) -> float:
    metadata = dataset.metadata.get("normalized", {}) if isinstance(dataset.metadata, dict) else {}
    if isinstance(metadata, dict):
        complete = _to_float(metadata.get("metadata_completeness"))
        if complete is not None:
            return _clip01(complete)
    fields_present = 0
    fields_total = 3
    if dataset.pixel_size is not None:
        fields_present += 1
    if isinstance(metadata, dict) and metadata.get("accelerating_voltage_kv") is not None:
        fields_present += 1
    if isinstance(metadata, dict) and metadata.get("camera_length_mm") is not None:
        fields_present += 1
    return fields_present / fields_total


def _canonical_modality(modality: str | None) -> str:
    if not modality:
        return "unknown"
    normalized = str(modality).strip().upper()
    if normalized in {"TEM", "HRTEM", "STEM", "HAADF-STEM", "ABF-STEM", "SAED", "NBED", "4D-STEM"}:
        return normalized
    normalized = normalized.replace("_", "-")
    if normalized in _MODALITY_THRESHOLDS:
        return normalized
    return "unknown"


def _get_modality_mapping(
    mapping: dict[str, Any],
    modality: str,
) -> dict[str, Any] | None:
    if not isinstance(mapping, dict):
        return None
    if modality in mapping and isinstance(mapping.get(modality), dict):
        return dict(mapping[modality])
    modality_lower = modality.lower()
    for key, value in mapping.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        if key.lower() == modality_lower:
            return dict(value)
    return None


def _thresholds_from_parameters(
    parameters: dict[str, Any] | None,
    *,
    modality: str,
) -> tuple[dict[str, float], str]:
    merged = dict(_BASE_THRESHOLDS)
    source = "base"
    modality_defaults = _MODALITY_THRESHOLDS.get(modality)
    if modality_defaults:
        merged.update(modality_defaults)
        source = f"base+modality:{modality}"

    if not parameters:
        return merged, source

    raw_global = parameters.get("thresholds", parameters)
    if isinstance(raw_global, dict):
        for key in _BASE_THRESHOLDS:
            value = _to_float(raw_global.get(key))
            if value is None:
                continue
            merged[key] = float(value)
            source = f"{source}+global_override"

    raw_by_modality = parameters.get("thresholds_by_modality")
    if isinstance(raw_by_modality, dict):
        row = _get_modality_mapping(raw_by_modality, modality)
        if row:
            for key in _BASE_THRESHOLDS:
                value = _to_float(row.get(key))
                if value is None:
                    continue
                merged[key] = float(value)
                source = f"{source}+modality_override:{modality}"
    return merged, source


def _score_weights_from_parameters(
    parameters: dict[str, Any] | None,
    *,
    modality: str,
) -> tuple[dict[str, float], str]:
    merged = dict(_BASE_SCORE_WEIGHTS)
    source = "base"
    modality_defaults = _MODALITY_SCORE_WEIGHTS.get(modality)
    if modality_defaults:
        merged.update(modality_defaults)
        source = f"base+modality:{modality}"

    if parameters:
        raw_global = parameters.get("metric_weights")
        if isinstance(raw_global, dict):
            for key in _WEIGHT_KEYS:
                value = _to_float(raw_global.get(key))
                if value is None or value <= 0:
                    continue
                merged[key] = float(value)
                source = f"{source}+global_override"
        raw_by_modality = parameters.get("metric_weights_by_modality")
        if isinstance(raw_by_modality, dict):
            row = _get_modality_mapping(raw_by_modality, modality)
            if row:
                for key in _WEIGHT_KEYS:
                    value = _to_float(row.get(key))
                    if value is None or value <= 0:
                        continue
                    merged[key] = float(value)
                    source = f"{source}+modality_override:{modality}"

    total = float(sum(merged.values()))
    if total <= 0:
        normalized = dict(_BASE_SCORE_WEIGHTS)
        total = float(sum(normalized.values()))
        return {key: value / total for key, value in normalized.items()}, "fallback_base"
    return {key: float(value / total) for key, value in merged.items()}, source


def assess_image_quality(
    dataset: TEMDataset,
    *,
    parameters: dict[str, Any] | None = None,
) -> QualityAssessment:
    """Assess B1 quality metrics and return normalized quality summary."""
    warnings: list[str] = []
    image = load_dataset_array(dataset)
    modality = _canonical_modality(dataset.modality)
    thresholds, threshold_source = _thresholds_from_parameters(
        parameters,
        modality=modality,
    )
    score_weights, weight_source = _score_weights_from_parameters(
        parameters,
        modality=modality,
    )

    metadata_complete = _metadata_completeness(dataset)
    if image is None:
        warnings.append("Image pixels unavailable for quality assessment.")
        return QualityAssessment(
            overall_score=0.0,
            snr=Measurement(value=0.0, unit="dB", method="quality_missing_image"),
            drift=Measurement(value=0.0, unit="pixel", method="quality_missing_image"),
            blur=Measurement(value=1.0, unit="risk", method="quality_missing_image"),
            contamination=Measurement(value=1.0, unit="risk", method="quality_missing_image"),
            astigmatism=Measurement(value=1.0, unit="risk", method="quality_missing_image"),
            undersampling_risk=1.0,
            metadata_completeness=metadata_complete,
            confidence=0.1,
            provenance={
                "method": "assess_image_quality",
                "reason": "missing_image",
                "modality": modality,
                "threshold_source": threshold_source,
                "score_weight_source": weight_source,
                "thresholds": thresholds,
                "score_weights": score_weights,
            },
            warnings=warnings,
        )

    arr = np.asarray(image, dtype=np.float64)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, copy=False)
        warnings.append("Image contains non-finite values; converted with nan_to_num.")

    signal_std = float(np.std(arr))
    noise_sigma = _estimate_noise_sigma(arr)
    snr_linear = signal_std / max(noise_sigma, 1e-12)
    snr_db = float(20.0 * np.log10(max(1e-12, snr_linear)))
    drift_px = _estimate_drift_px(arr)
    blur_risk = _estimate_blur_risk(arr)
    contamination_risk = _estimate_contamination_risk(arr)
    fft_magnitude = np.abs(np.fft.fftshift(np.fft.fft2(arr)))
    astigmatism_risk = _estimate_astigmatism_risk(fft_magnitude)
    hf_ratio = _estimate_high_freq_ratio(fft_magnitude)

    pixel_size_nm = None
    if dataset.pixel_size is not None:
        pixel_size_nm = _to_float(dataset.pixel_size.value)
    metadata = dataset.metadata.get("normalized", {}) if isinstance(dataset.metadata, dict) else {}
    if pixel_size_nm is None and isinstance(metadata, dict):
        pixel_size_nm = _to_float(metadata.get("pixel_size_nm"))

    hf_risk = _clip01((hf_ratio - 0.12) / 0.35)
    if pixel_size_nm is None or pixel_size_nm <= 0:
        undersampling_risk = hf_risk
    else:
        nyquist_nm = 2.0 * pixel_size_nm
        geometric_risk = _clip01((nyquist_nm - 0.25) / 1.5)
        undersampling_risk = _clip01(0.6 * geometric_risk + 0.4 * hf_risk)

    snr_score = _clip01((snr_db - thresholds["snr_min"]) / 20.0)
    drift_score = _clip01(1.0 - drift_px / max(1e-6, thresholds["drift_px_max"]))
    score_components = {
        "snr": snr_score,
        "drift": drift_score,
        "blur": 1.0 - blur_risk,
        "contamination": 1.0 - contamination_risk,
        "astigmatism": 1.0 - astigmatism_risk,
        "undersampling": 1.0 - undersampling_risk,
        "metadata": metadata_complete,
    }
    overall_score = float(
        sum(score_weights[key] * score_components[key] for key in _WEIGHT_KEYS)
    )
    overall_score = _clip01(overall_score)

    if snr_db < thresholds["snr_min"]:
        warnings.append(
            f"Low SNR detected ({snr_db:.2f} dB < {thresholds['snr_min']:.2f} dB)."
        )
    if drift_px > thresholds["drift_px_max"]:
        warnings.append(
            "Possible drift risk "
            f"({drift_px:.2f}px > {thresholds['drift_px_max']:.2f}px)."
        )
    if blur_risk > thresholds["blur_risk_max"]:
        warnings.append(
            f"Blur risk is high ({blur_risk:.3f} > {thresholds['blur_risk_max']:.3f})."
        )
    if contamination_risk > thresholds["contamination_risk_max"]:
        warnings.append(
            "Contamination/bright-spike risk is high "
            f"({contamination_risk:.3f} > {thresholds['contamination_risk_max']:.3f})."
        )
    if astigmatism_risk > thresholds["astigmatism_risk_max"]:
        warnings.append(
            "Astigmatism risk is high "
            f"({astigmatism_risk:.3f} > {thresholds['astigmatism_risk_max']:.3f})."
        )
    if undersampling_risk > thresholds["undersampling_risk_max"]:
        warnings.append(
            "Undersampling risk is high "
            f"({undersampling_risk:.3f} > {thresholds['undersampling_risk_max']:.3f})."
        )
    if overall_score < thresholds["overall_score_min"]:
        warnings.append(
            "Overall quality score is below threshold "
            f"({overall_score:.3f} < {thresholds['overall_score_min']:.3f})."
        )

    confidence = _clip01(0.45 + 0.35 * metadata_complete + 0.2 * (1.0 - undersampling_risk))

    return QualityAssessment(
        overall_score=overall_score,
        snr=Measurement(
            value=snr_db,
            unit="dB",
            method="signal_std_over_noise_sigma",
            confidence=_clip01(0.8 - 0.2 * blur_risk),
        ),
        drift=Measurement(
            value=drift_px,
            unit="pixel",
            method=(
                "phase_cross_correlation"
                if _phase_cross_correlation is not None
                else "fft_phase_correlation_fallback"
            ),
            confidence=_clip01(0.7 + 0.2 * metadata_complete),
        ),
        blur=Measurement(
            value=blur_risk,
            unit="risk",
            method=("skimage.blur_effect" if _skimage_blur_effect is not None else "laplacian_var"),
            confidence=0.75,
        ),
        contamination=Measurement(
            value=contamination_risk,
            unit="risk",
            method="percentile_spike_fraction",
            confidence=0.7,
        ),
        astigmatism=Measurement(
            value=astigmatism_risk,
            unit="risk",
            method="fft_anisotropy_covariance",
            confidence=0.65,
        ),
        undersampling_risk=undersampling_risk,
        metadata_completeness=metadata_complete,
        confidence=confidence,
        provenance={
            "method": "assess_image_quality",
            "uses_skimage_estimate_sigma": _estimate_sigma is not None,
            "uses_skimage_phase_xcorr": _phase_cross_correlation is not None,
            "uses_skimage_blur_effect": _skimage_blur_effect is not None,
            "modality": modality,
            "threshold_source": threshold_source,
            "score_weight_source": weight_source,
            "thresholds": thresholds,
            "score_weights": score_weights,
            "score_components": score_components,
        },
        warnings=warnings,
    )
