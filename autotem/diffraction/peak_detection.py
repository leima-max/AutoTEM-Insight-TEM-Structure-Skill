"""B3 Bragg peak detection with local-max / LoG / DoG detectors."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from autotem.core import DiffractionPeak

SUPPORTED_PEAK_DETECTORS: set[str] = {"local_max", "log", "dog"}
SUPPORTED_THRESHOLD_STRATEGIES: set[str] = {
    "auto",
    "quantile",
    "std",
    "relative_max",
    "mad",
}


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _to_float(value: Any, default: float) -> float:
    if value is None or isinstance(value, bool):
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        import re

        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value.strip())
        if match:
            return float(match.group(0))
    return float(default)


def _to_int(value: Any, default: int) -> int:
    return int(round(_to_float(value, float(default))))


def suggest_peak_parameter_defaults(
    log_magnitude: np.ndarray,
) -> dict[str, float | int]:
    """Suggest deterministic defaults from FFT size for B3 auto tuning."""
    h, w = log_magnitude.shape
    base = float(max(1, min(h, w)))
    return {
        "top_k_peaks": int(_clip(round(base * 0.40), 16, 96)),
        "min_peak_distance": int(_clip(round(base / 32.0), 2, 8)),
        "center_exclusion_radius": _clip(base * 0.06, 3.0, 14.0),
        "detector_sigma": _clip(base / 96.0, 0.8, 2.2),
        "dog_sigma_ratio": 1.8,
    }


def _adaptive_auto_quantile(response: np.ndarray) -> tuple[float, float]:
    """Infer auto quantile from robust response SNR proxy."""
    med = float(np.median(response))
    mad = float(np.median(np.abs(response - med)))
    sigma = max(1e-12, 1.4826 * mad)
    p99 = float(np.quantile(response, 0.99))
    z99 = (p99 - med) / sigma
    if z99 >= 12.0:
        return 0.9988, float(z99)
    if z99 >= 9.0:
        return 0.9980, float(z99)
    if z99 >= 6.0:
        return 0.9970, float(z99)
    if z99 >= 4.0:
        return 0.9960, float(z99)
    return 0.9945, float(z99)


def _resolve_threshold(
    response: np.ndarray,
    *,
    strategy: str,
    value: float | None,
) -> tuple[float, str, dict[str, float]]:
    token = str(strategy).strip().lower()
    if token not in SUPPORTED_THRESHOLD_STRATEGIES:
        token = "auto"
    if token == "auto":
        q, z99 = _adaptive_auto_quantile(response)
        threshold = float(np.quantile(response, q))
        return threshold, f"auto_quantile_{q:.4f}", {
            "auto_quantile": float(q),
            "auto_snr_proxy_z99": float(z99),
        }
    if token == "quantile":
        q = float(max(0.0, min(1.0, 0.995 if value is None else value)))
        threshold = float(np.quantile(response, q))
        return threshold, f"quantile_{q:.4f}", {}
    if token == "std":
        k = max(0.0, float(2.8 if value is None else value))
        threshold = float(np.mean(response) + k * np.std(response))
        return threshold, f"mean_plus_{k:.2f}std", {}
    if token == "relative_max":
        ratio = max(0.0, min(1.0, float(0.24 if value is None else value)))
        threshold = float(np.max(response) * ratio)
        return threshold, f"relative_max_{ratio:.3f}", {}
    k = max(0.0, float(6.0 if value is None else value))
    med = float(np.median(response))
    mad = float(np.median(np.abs(response - med)))
    threshold = float(med + k * 1.4826 * mad)
    return threshold, f"median_plus_{k:.2f}mad", {}


def _build_response_map(
    log_magnitude: np.ndarray,
    *,
    detector: str,
    detector_sigma: float,
    dog_sigma_ratio: float,
) -> tuple[np.ndarray, str]:
    token = str(detector).strip().lower()
    if token not in SUPPORTED_PEAK_DETECTORS:
        token = "local_max"

    if token == "local_max":
        return np.asarray(log_magnitude, dtype=np.float64), "local_max"
    if token == "log":
        sigma = max(0.3, float(detector_sigma))
        response = -ndi.gaussian_laplace(np.asarray(log_magnitude, dtype=np.float64), sigma=sigma)
        return response, f"log_sigma_{sigma:.2f}"

    sigma1 = max(0.2, float(detector_sigma))
    ratio = max(1.2, float(dog_sigma_ratio))
    sigma2 = sigma1 * ratio
    blur_small = ndi.gaussian_filter(np.asarray(log_magnitude, dtype=np.float64), sigma=sigma1)
    blur_large = ndi.gaussian_filter(np.asarray(log_magnitude, dtype=np.float64), sigma=sigma2)
    response = blur_small - blur_large
    return response, f"dog_sigma_{sigma1:.2f}_{sigma2:.2f}"


def _refine_peak_subpixel(
    response: np.ndarray,
    *,
    y: int,
    x: int,
    window_radius: int,
    max_shift: float,
) -> tuple[float, float, float | None, float | None]:
    y0 = max(0, y - window_radius)
    y1 = min(response.shape[0], y + window_radius + 1)
    x0 = max(0, x - window_radius)
    x1 = min(response.shape[1], x + window_radius + 1)
    patch = response[y0:y1, x0:x1]
    if patch.size == 0:
        return float(x), float(y), None, None

    weights = patch - float(np.min(patch))
    total = float(np.sum(weights))
    if total <= 1e-12:
        return float(x), float(y), None, None
    yy, xx = np.indices(patch.shape)
    abs_y = yy + y0
    abs_x = xx + x0
    yc = float(np.sum(weights * abs_y) / total)
    xc = float(np.sum(weights * abs_x) / total)
    if abs(yc - y) > max_shift or abs(xc - x) > max_shift:
        return float(x), float(y), None, None
    sigma_y = float(np.sqrt(np.sum(weights * (abs_y - yc) ** 2) / total))
    sigma_x = float(np.sqrt(np.sum(weights * (abs_x - xc) ** 2) / total))
    return float(xc), float(yc), sigma_x, sigma_y


@lru_cache(maxsize=16)
def _disk_offsets(distance: int) -> tuple[tuple[int, int], ...]:
    radius = max(1, int(distance))
    out: list[tuple[int, int]] = []
    r2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= r2:
                out.append((dy, dx))
    return tuple(out)


def _suppress_disk(mask: np.ndarray, y: int, x: int, *, distance: int) -> None:
    offsets = _disk_offsets(distance)
    h, w = mask.shape
    for dy, dx in offsets:
        yy = y + dy
        xx = x + dx
        if 0 <= yy < h and 0 <= xx < w:
            mask[yy, xx] = True


def detect_bragg_peaks(
    log_magnitude: np.ndarray,
    *,
    center: tuple[float, float],
    top_k: int = 32,
    min_distance: int = 3,
    center_exclusion_radius: float = 4.0,
    threshold_strategy: str = "auto",
    threshold_value: float | None = None,
    subpixel_window_radius: int = 2,
    subpixel_max_shift: float = 1.5,
    detector: str = "local_max",
    detector_sigma: float = 1.2,
    dog_sigma_ratio: float = 1.8,
) -> tuple[list[DiffractionPeak], dict[str, Any], list[str]]:
    """Detect Bragg peaks from FFT log-magnitude map."""
    warnings: list[str] = []
    response, detector_name = _build_response_map(
        log_magnitude,
        detector=detector,
        detector_sigma=float(detector_sigma),
        dog_sigma_ratio=float(dog_sigma_ratio),
    )
    yy, xx = np.indices(response.shape)
    rr = np.hypot(xx - float(center[0]), yy - float(center[1]))
    threshold_response = response
    if float(center_exclusion_radius) > 0:
        annulus = response[rr >= float(center_exclusion_radius)]
        if annulus.size >= 32:
            threshold_response = np.asarray(annulus, dtype=np.float64)
    threshold, threshold_method, threshold_diag = _resolve_threshold(
        threshold_response,
        strategy=threshold_strategy,
        value=threshold_value,
    )

    std_value = float(np.std(response))
    if std_value <= 1e-10:
        warnings.append("Peak-response variation is near-zero; peak detection skipped.")
        return [], {
            "detector": detector_name,
            "threshold": threshold,
            "threshold_method": threshold_method,
            "std": std_value,
            "candidate_count": 0,
            "selected_count": 0,
        }, warnings

    distance = max(1, int(min_distance))
    footprint = 2 * distance + 1

    def _candidate_coords(
        response_map: np.ndarray,
        threshold_value_local: float,
    ) -> np.ndarray:
        max_filtered_local = ndi.maximum_filter(response_map, size=footprint, mode="nearest")
        local_max_local = np.isclose(
            response_map,
            max_filtered_local,
            rtol=1e-7,
            atol=1e-12,
        )
        mask_local = (
            local_max_local
            & (response_map >= threshold_value_local)
            & (rr >= float(center_exclusion_radius))
        )
        return np.argwhere(mask_local)

    coords = _candidate_coords(response, threshold)
    used_relaxed_threshold = False
    if coords.size == 0:
        if threshold_method.startswith("auto_quantile_"):
            for relaxed_q in (0.992, 0.990, 0.987, 0.985):
                relaxed_threshold = float(np.quantile(response, relaxed_q))
                relaxed_coords = _candidate_coords(response, relaxed_threshold)
                if relaxed_coords.size == 0:
                    continue
                threshold = relaxed_threshold
                threshold_method = f"auto_relaxed_quantile_{relaxed_q:.4f}"
                coords = relaxed_coords
                used_relaxed_threshold = True
                warnings.append(
                    "Auto threshold was relaxed to recover FFT peaks "
                    f"(quantile={relaxed_q:.4f})."
                )
                break
        if coords.size == 0 and detector_name == "local_max":
            fallback_response, _ = _build_response_map(
                log_magnitude,
                detector="dog",
                detector_sigma=max(0.8, float(detector_sigma)),
                dog_sigma_ratio=max(1.4, float(dog_sigma_ratio)),
            )
            fallback_threshold = float(np.max(fallback_response) * 0.12)
            fallback_coords = _candidate_coords(fallback_response, fallback_threshold)
            if fallback_coords.size > 0:
                response = fallback_response
                detector_name = "local_max_fallback_dog"
                threshold = fallback_threshold
                threshold_method = "fallback_relative_max_0.120"
                coords = fallback_coords
                warnings.append(
                    "Fallback detector path activated (DoG response) to recover FFT peaks."
                )
        if coords.size == 0:
            warnings.append(
                "No FFT peaks passed threshold; retry with lower peak_threshold_value or "
                "different detector/threshold strategy."
            )
            return [], {
                "detector": detector_name,
                "threshold": threshold,
                "threshold_method": threshold_method,
                "std": std_value,
                "candidate_count": 0,
                "selected_count": 0,
                "used_relaxed_threshold": used_relaxed_threshold,
            }, warnings

    scores = response[coords[:, 0], coords[:, 1]]
    order = np.argsort(scores)[::-1]
    selected: list[tuple[int, int]] = []
    suppressed = np.zeros(response.shape, dtype=bool)
    for idx in order.tolist():
        y, x = int(coords[idx, 0]), int(coords[idx, 1])
        if suppressed[y, x]:
            continue
        selected.append((y, x))
        _suppress_disk(suppressed, y, x, distance=distance)
        if len(selected) >= max(1, int(top_k)):
            break

    if not selected:
        warnings.append("No FFT peaks remained after non-maximum suppression.")
        return [], {
            "detector": detector_name,
            "threshold": threshold,
            "threshold_method": threshold_method,
            "std": std_value,
            "candidate_count": int(coords.shape[0]),
            "selected_count": 0,
        }, warnings

    selected_values = np.asarray([response[y, x] for y, x in selected], dtype=np.float64)
    vmin = float(np.min(selected_values))
    vmax = float(np.max(selected_values))
    denom = max(1e-12, vmax - vmin)

    peaks: list[DiffractionPeak] = []
    for y, x in selected:
        xf, yf, sigma_x, sigma_y = _refine_peak_subpixel(
            response,
            y=y,
            x=x,
            window_radius=max(1, _to_int(subpixel_window_radius, 2)),
            max_shift=max(0.0, float(subpixel_max_shift)),
        )
        value = float(response[y, x])
        confidence = 0.4 + 0.6 * ((value - vmin) / denom)
        peaks.append(
            DiffractionPeak(
                x=xf,
                y=yf,
                intensity=float(log_magnitude[y, x]),
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                confidence=max(0.0, min(1.0, confidence)),
            )
        )

    diagnostics: dict[str, Any] = {
        "detector": detector_name,
        "threshold": threshold,
        "threshold_method": threshold_method,
        "std": std_value,
        "response_p95": float(np.quantile(response, 0.95)),
        "response_p99": float(np.quantile(response, 0.99)),
        "candidate_count": int(coords.shape[0]),
        "selected_count": len(peaks),
        "min_distance_px": distance,
        "center_exclusion_radius_px": float(center_exclusion_radius),
        "used_relaxed_threshold": used_relaxed_threshold,
    }
    diagnostics.update(threshold_diag)
    return peaks, diagnostics, warnings
