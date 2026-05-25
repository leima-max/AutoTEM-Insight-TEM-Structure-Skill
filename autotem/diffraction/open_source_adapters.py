"""Optional open-source adapter hooks for B3 peak detection."""

from __future__ import annotations

from typing import Any

import numpy as np

from autotem.core import DiffractionPeak
from autotem.diffraction.peak_detection import detect_bragg_peaks

SUPPORTED_PEAK_ADAPTER_ENGINES: frozenset[str] = frozenset({"none", "pyxem", "py4dstem"})


def resolve_peak_adapter_engine(engine: str | None) -> str:
    """Normalize and validate optional B3 peak-adapter engine names."""
    resolved = (engine or "none").strip().lower()
    if resolved not in SUPPORTED_PEAK_ADAPTER_ENGINES:
        raise ValueError(
            "Unsupported peak adapter engine: "
            f"{engine}. Supported: {sorted(SUPPORTED_PEAK_ADAPTER_ENGINES)}"
        )
    return resolved


def _ensure_engine_dependency(engine: str) -> None:
    if engine == "pyxem":
        try:
            import pyxem  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("pyxem is not installed.") from exc
    elif engine == "py4dstem":
        try:
            import py4DSTEM  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("py4DSTEM is not installed.") from exc


def _engine_default_detector(engine: str, detector: str) -> str:
    detector_token = str(detector).strip().lower()
    if detector_token:
        return detector_token
    if engine == "py4dstem":
        return "dog"
    if engine == "pyxem":
        return "log"
    return "local_max"


def detect_peaks_with_adapter(
    *,
    engine: str | None,
    log_magnitude: np.ndarray,
    center: tuple[float, float],
    top_k: int,
    min_distance: int,
    center_exclusion_radius: float,
    max_peak_radius_px: float | None,
    threshold_strategy: str,
    threshold_value: float | None,
    subpixel_window_radius: int,
    subpixel_max_shift: float,
    detector: str,
    detector_sigma: float,
    dog_sigma_ratio: float,
    radius_outlier_filter: bool,
    radius_outlier_mad_factor: float,
) -> tuple[list[DiffractionPeak], dict[str, Any], list[str]]:
    """
    Adapter entry point for B3 peak detection.

    Current implementation runs the shared internal detector stack while validating
    optional engine dependencies so workflows can explicitly require pyxem/py4DSTEM
    and still preserve stable DiffractionResult contracts.
    """
    resolved = resolve_peak_adapter_engine(engine)
    if resolved == "none":
        raise RuntimeError("Adapter engine 'none' does not provide an external detection path.")

    _ensure_engine_dependency(resolved)
    chosen_detector = _engine_default_detector(resolved, detector)

    peaks, diagnostics, warnings = detect_bragg_peaks(
        np.asarray(log_magnitude, dtype=np.float64),
        center=center,
        top_k=max(1, int(top_k)),
        min_distance=max(1, int(min_distance)),
        center_exclusion_radius=max(0.0, float(center_exclusion_radius)),
        max_peak_radius_px=(
            None
            if max_peak_radius_px is None or float(max_peak_radius_px) <= 0
            else float(max_peak_radius_px)
        ),
        threshold_strategy=threshold_strategy,
        threshold_value=threshold_value,
        subpixel_window_radius=max(1, int(subpixel_window_radius)),
        subpixel_max_shift=max(0.0, float(subpixel_max_shift)),
        detector=chosen_detector,
        detector_sigma=max(0.2, float(detector_sigma)),
        dog_sigma_ratio=max(1.2, float(dog_sigma_ratio)),
        radius_outlier_filter=bool(radius_outlier_filter),
        radius_outlier_mad_factor=max(2.0, float(radius_outlier_mad_factor)),
    )

    warnings = list(warnings)
    warnings.append(
        "Open-source peak adapter is active in compatibility mode; "
        "core detection still uses internal algorithm with adapter-validated runtime."
    )
    diagnostics = {
        **diagnostics,
        "adapter_engine": resolved,
        "adapter_execution_mode": "compatibility_internal_detector",
    }
    return peaks, diagnostics, warnings
