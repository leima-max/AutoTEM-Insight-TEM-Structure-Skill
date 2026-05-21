"""B2 denoise operators: Gaussian, median, Wiener, and non-local means."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import wiener as scipy_wiener

try:
    from skimage.restoration import denoise_nl_means as _denoise_nl_means
    from skimage.restoration import estimate_sigma as _estimate_sigma
except Exception:  # noqa: BLE001
    _denoise_nl_means = None
    _estimate_sigma = None

SUPPORTED_DENOISE_METHODS: set[str] = {
    "auto",
    "none",
    "gaussian",
    "median",
    "wiener",
    "nlm",
}


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


def _normalize_method(method: str | None) -> str:
    token = "auto" if method is None else str(method).strip().lower()
    if token not in SUPPORTED_DENOISE_METHODS:
        return "auto"
    return token


def _resolve_auto_method(image: np.ndarray) -> str:
    std_value = float(np.std(image))
    if std_value < 0.02:
        return "none"
    if std_value > 0.30:
        return "wiener"
    return "gaussian"


def apply_denoise(
    image: np.ndarray,
    *,
    method: str | None = "auto",
    sigma: float | int | None = 0.8,
    median_size: int | None = 3,
    wiener_window: int | None = 5,
    wiener_noise: float | None = None,
    nlm_h: float | None = None,
    nlm_patch_size: int | None = 5,
    nlm_patch_distance: int | None = 6,
) -> tuple[np.ndarray, str, dict[str, float], list[str]]:
    """Apply one denoise method and return output + effective method + diagnostics."""
    warnings: list[str] = []
    input_array = np.asarray(image, dtype=np.float64)
    chosen = _normalize_method(method)
    if chosen == "auto":
        chosen = _resolve_auto_method(input_array)

    if chosen == "none":
        return (
            np.asarray(input_array, dtype=np.float32),
            "none",
            {"std_before": float(np.std(input_array)), "std_after": float(np.std(input_array))},
            warnings,
        )

    if chosen == "gaussian":
        sigma_value = max(0.0, _to_float(sigma, 0.8))
        denoised = np.asarray(ndi.gaussian_filter(input_array, sigma=sigma_value), dtype=np.float32)
        return (
            denoised,
            "gaussian",
            {
                "sigma": sigma_value,
                "std_before": float(np.std(input_array)),
                "std_after": float(np.std(denoised)),
            },
            warnings,
        )

    if chosen == "median":
        size = max(1, _to_int(median_size, 3))
        if size % 2 == 0:
            size += 1
        denoised = np.asarray(ndi.median_filter(input_array, size=size), dtype=np.float32)
        return (
            denoised,
            "median",
            {
                "median_size": float(size),
                "std_before": float(np.std(input_array)),
                "std_after": float(np.std(denoised)),
            },
            warnings,
        )

    if chosen == "wiener":
        window = max(3, _to_int(wiener_window, 5))
        if window % 2 == 0:
            window += 1
        noise_value = None if wiener_noise is None else max(0.0, _to_float(wiener_noise, 0.0))
        denoised = np.asarray(
            scipy_wiener(input_array, mysize=(window, window), noise=noise_value),
            dtype=np.float32,
        )
        if not np.isfinite(denoised).all():
            denoised = np.nan_to_num(denoised, copy=False)
            warnings.append("Wiener result contained non-finite values; converted with nan_to_num.")
        return (
            denoised,
            "wiener",
            {
                "wiener_window": float(window),
                "wiener_noise": float(noise_value) if noise_value is not None else -1.0,
                "std_before": float(np.std(input_array)),
                "std_after": float(np.std(denoised)),
            },
            warnings,
        )

    # chosen == "nlm"
    if _denoise_nl_means is None:
        sigma_value = max(0.0, _to_float(sigma, 0.8))
        fallback = np.asarray(ndi.gaussian_filter(input_array, sigma=sigma_value), dtype=np.float32)
        warnings.append("scikit-image is unavailable; NLM fell back to Gaussian denoise.")
        return (
            fallback,
            "nlm_fallback_gaussian",
            {
                "sigma": sigma_value,
                "std_before": float(np.std(input_array)),
                "std_after": float(np.std(fallback)),
            },
            warnings,
        )

    sigma_est = 0.0
    if _estimate_sigma is not None:
        try:
            sigma_est = float(_estimate_sigma(input_array, average_sigmas=True, channel_axis=None))
        except Exception:  # noqa: BLE001
            sigma_est = 0.0
    h_value = (
        max(1e-6, float(nlm_h))
        if nlm_h is not None
        else max(1e-6, 0.9 * max(sigma_est, 0.02))
    )
    patch_size = max(3, _to_int(nlm_patch_size, 5))
    if patch_size % 2 == 0:
        patch_size += 1
    patch_distance = max(1, _to_int(nlm_patch_distance, 6))
    denoised = _denoise_nl_means(
        input_array,
        h=h_value,
        fast_mode=True,
        patch_size=patch_size,
        patch_distance=patch_distance,
        channel_axis=None,
        preserve_range=True,
    )
    denoised = np.asarray(denoised, dtype=np.float32)
    return (
        denoised,
        "nlm",
        {
            "nlm_h": h_value,
            "nlm_patch_size": float(patch_size),
            "nlm_patch_distance": float(patch_distance),
            "noise_sigma_estimate": sigma_est,
            "std_before": float(np.std(input_array)),
            "std_after": float(np.std(denoised)),
        },
        warnings,
    )
