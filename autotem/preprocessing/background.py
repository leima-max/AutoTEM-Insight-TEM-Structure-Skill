"""B2 background estimation and subtraction operators."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

SUPPORTED_BACKGROUND_METHODS: set[str] = {"none", "gaussian", "rolling_ball"}


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
    token = "none" if method is None else str(method).strip().lower()
    if token not in SUPPORTED_BACKGROUND_METHODS:
        return "none"
    return token


def subtract_background(
    image: np.ndarray,
    *,
    method: str | None = "none",
    gaussian_sigma: float | int | None = 12.0,
    opening_size: int | None = 31,
    clip_to_zero: bool = False,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, float], list[str]]:
    """Subtract estimated background and return corrected + background arrays."""
    warnings: list[str] = []
    source = np.asarray(image, dtype=np.float64)
    chosen = _normalize_method(method)

    if chosen == "none":
        zeros = np.zeros_like(source, dtype=np.float32)
        return (
            np.asarray(source, dtype=np.float32),
            zeros,
            "none",
            {"std_before": float(np.std(source)), "std_after": float(np.std(source))},
            warnings,
        )

    if chosen == "gaussian":
        sigma = max(0.1, _to_float(gaussian_sigma, 12.0))
        background = np.asarray(ndi.gaussian_filter(source, sigma=sigma), dtype=np.float32)
        corrected = np.asarray(source - background, dtype=np.float32)
        if clip_to_zero:
            corrected = np.asarray(np.maximum(corrected, 0.0), dtype=np.float32)
        return (
            corrected,
            background,
            "gaussian",
            {
                "background_sigma": sigma,
                "std_before": float(np.std(source)),
                "std_after": float(np.std(corrected)),
            },
            warnings,
        )

    # chosen == "rolling_ball"
    size = max(3, _to_int(opening_size, 31))
    if size % 2 == 0:
        size += 1
    background = np.asarray(ndi.grey_opening(source, size=(size, size)), dtype=np.float32)
    corrected = np.asarray(source - background, dtype=np.float32)
    if clip_to_zero:
        corrected = np.asarray(np.maximum(corrected, 0.0), dtype=np.float32)
    return (
        corrected,
        background,
        "rolling_ball",
        {
            "opening_size": float(size),
            "std_before": float(np.std(source)),
            "std_after": float(np.std(corrected)),
        },
        warnings,
    )
