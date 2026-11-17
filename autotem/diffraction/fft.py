"""B2 FFT computation and direct-beam center localization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autotem.core import ArrayReference


@dataclass(slots=True)
class FFTAnalysis:
    """Computed FFT arrays and localized reciprocal-space center."""

    fft_complex: np.ndarray
    magnitude: np.ndarray
    log_magnitude: np.ndarray
    center: tuple[float, float]
    diagnostics: dict[str, Any]


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


def _weighted_centroid(
    image: np.ndarray,
    *,
    y: int,
    x: int,
    window: int,
) -> tuple[float, float]:
    y0 = max(0, y - window)
    y1 = min(image.shape[0], y + window + 1)
    x0 = max(0, x - window)
    x1 = min(image.shape[1], x + window + 1)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return float(x), float(y)
    weights = patch - float(np.min(patch))
    total = float(np.sum(weights))
    if total <= 1e-12:
        return float(x), float(y)
    yy, xx = np.indices(patch.shape)
    abs_y = yy + y0
    abs_x = xx + x0
    yc = float(np.sum(abs_y * weights) / total)
    xc = float(np.sum(abs_x * weights) / total)
    return xc, yc


def locate_direct_beam(
    magnitude: np.ndarray,
    *,
    center_mode: str = "auto",
    search_radius_px: float | None = None,
    refine_window: int = 3,
) -> tuple[tuple[float, float], dict[str, Any], list[str]]:
    """Locate direct beam from FFT magnitude near geometric center."""
    warnings: list[str] = []
    h, w = magnitude.shape
    geometric = (w / 2.0, h / 2.0)
    mode = str(center_mode).strip().lower()

    if mode == "geometric":
        return geometric, {"mode": "geometric", "offset_px": 0.0}, warnings

    radius = (
        max(2.0, float(search_radius_px))
        if search_radius_px is not None
        else max(4.0, min(h, w) * 0.15)
    )
    yy, xx = np.indices(magnitude.shape, sparse=True)
    rr = np.hypot(xx - geometric[0], yy - geometric[1])
    roi_mask = rr <= radius
    if not np.any(roi_mask):
        warnings.append("Direct-beam search ROI is empty; fallback to geometric center.")
        return geometric, {"mode": "auto", "offset_px": 0.0, "fallback": "empty_roi"}, warnings

    masked = np.asarray(magnitude, dtype=np.float64).copy()
    min_value = float(np.min(masked))
    masked[~roi_mask] = min_value - 1.0
    peak_flat = int(np.argmax(masked))
    y_peak, x_peak = np.unravel_index(peak_flat, masked.shape)
    xc, yc = _weighted_centroid(
        masked,
        y=int(y_peak),
        x=int(x_peak),
        window=max(1, int(refine_window)),
    )
    offset = float(np.hypot(xc - geometric[0], yc - geometric[1]))
    diagnostics = {
        "mode": "auto",
        "search_radius_px": radius,
        "refine_window": int(max(1, int(refine_window))),
        "geometric_center": [float(geometric[0]), float(geometric[1])],
        "direct_beam_center": [float(xc), float(yc)],
        "offset_px": offset,
    }
    return (float(xc), float(yc)), diagnostics, warnings


def compute_fft_analysis(
    image: np.ndarray,
    *,
    center_mode: str = "auto",
    search_radius_px: float | None = None,
    refine_window: int = 3,
) -> tuple[FFTAnalysis, list[str]]:
    """Compute shifted FFT magnitude and locate reciprocal-space center."""
    warnings: list[str] = []
    array = np.asarray(image, dtype=np.float64)
    fft_complex = np.fft.fftshift(np.fft.fft2(array))
    magnitude = np.abs(fft_complex).astype(np.float64)
    log_magnitude = np.log1p(magnitude).astype(np.float64)
    center, center_diag, center_warnings = locate_direct_beam(
        magnitude,
        center_mode=center_mode,
        search_radius_px=search_radius_px,
        refine_window=refine_window,
    )
    warnings.extend(center_warnings)
    diagnostics = {
        "center": center_diag,
        "image_shape": [int(v) for v in array.shape],
    }
    return (
        FFTAnalysis(
            fft_complex=fft_complex,
            magnitude=magnitude,
            log_magnitude=log_magnitude,
            center=center,
            diagnostics=diagnostics,
        ),
        warnings,
    )


def persist_fft_array_near_source(
    log_magnitude: np.ndarray,
    *,
    source_uri: str | None,
) -> str | None:
    """Persist FFT magnitude near preprocessed array when a URI exists."""
    if not source_uri:
        return None
    source = Path(source_uri)
    if not source.exists() or not source.is_file():
        return None
    if source.suffix.lower() != ".npy":
        return None
    target = source.with_name(f"{source.stem}_fft_magnitude.npy")
    np.save(target, np.asarray(log_magnitude, dtype=np.float32))
    return str(target)


def build_fft_reference(
    log_magnitude: np.ndarray,
    *,
    uri: str | None = None,
) -> ArrayReference:
    """Construct ArrayReference for FFT magnitude output."""
    fft_float = np.asarray(log_magnitude, dtype=np.float32)
    inline = fft_float.tolist() if fft_float.size <= 4096 else None
    return ArrayReference(
        shape=tuple(int(v) for v in fft_float.shape),
        dtype=str(fft_float.dtype),
        uri=uri,
        inline=inline,
        description="Log-scaled FFT magnitude from preprocessed image.",
    )
