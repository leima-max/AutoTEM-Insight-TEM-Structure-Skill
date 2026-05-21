"""Optional open-source adapter hooks for diffraction calibration."""

from __future__ import annotations

from typing import Any

import numpy as np

from autotem.calibration.pixel_calibration import convert_length

SUPPORTED_DIFFRACTION_CALIBRATION_ENGINES: frozenset[str] = frozenset(
    {"none", "pyxem", "py4dstem"}
)


def resolve_diffraction_calibration_engine(engine: str | None) -> str:
    """Normalize and validate diffraction calibration engine name."""
    resolved = (engine or "none").strip().lower()
    if resolved not in SUPPORTED_DIFFRACTION_CALIBRATION_ENGINES:
        raise ValueError(
            "Unsupported diffraction calibration engine: "
            f"{engine}. Supported: {sorted(SUPPORTED_DIFFRACTION_CALIBRATION_ENGINES)}"
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


def _inv_length_to_inv_nm(
    value: float,
    unit: str | None,
) -> float:
    if unit is None:
        return float(value)
    normalized = str(unit).strip().lower().replace(" ", "")
    normalized = normalized.replace("angstrom", "a")
    normalized = normalized.replace("^-1", "-1")
    if "a-1" in normalized or "1/a" in normalized:
        return float(value) * 10.0
    if "nm-1" in normalized or "1/nm" in normalized:
        return float(value)
    return float(value)


def _single_pair_scale_from_py4dstem(
    *,
    known_d_spacing: Any,
    measured_radius_px: Any,
    known_d_spacing_unit: str,
) -> float:
    import py4DSTEM

    q_meas = _to_float(measured_radius_px)
    if q_meas is None or q_meas <= 0:
        raise ValueError("measured_radius_px must be positive.")
    d_angstrom = convert_length(known_d_spacing, known_d_spacing_unit, "angstrom")
    if d_angstrom <= 0:
        raise ValueError("known_d_spacing must be positive.")

    calibration_mod = py4DSTEM.process.calibration.qpixelsize
    dq_value, dq_unit = calibration_mod.get_Q_pixel_size(
        q_meas=q_meas,
        q_known=d_angstrom,
        units="A",
    )
    dq = _to_float(dq_value)
    if dq is None or dq <= 0:
        raise RuntimeError("py4DSTEM returned an invalid diffraction calibration value.")
    return _inv_length_to_inv_nm(float(dq), str(dq_unit))


def _single_pair_scale_from_pyxem(
    *,
    known_d_spacing: Any,
    measured_radius_px: Any,
    known_d_spacing_unit: str,
) -> float:
    import pyxem

    from autotem.calibration.diffraction_calibration import diffraction_scale_from_known_d_spacing

    scale_inv_nm, _ = diffraction_scale_from_known_d_spacing(
        known_d_spacing,
        measured_radius_px,
        known_d_spacing_unit=known_d_spacing_unit,
    )
    scale_inv_a = float(scale_inv_nm) / 10.0

    signals = getattr(pyxem, "signals", None)
    if signals is None:
        raise RuntimeError("pyxem.signals is unavailable.")
    signal_cls = getattr(signals, "ElectronDiffraction2D", None)
    if signal_cls is None:
        signal_cls = getattr(signals, "Diffraction2D", None)
    if signal_cls is None:
        raise RuntimeError("pyxem diffraction signal class is unavailable.")

    signal = signal_cls(np.zeros((16, 16), dtype=np.float32))
    setter = getattr(signal, "set_diffraction_calibration", None)
    if not callable(setter):
        raise RuntimeError("pyxem diffraction signal does not expose set_diffraction_calibration.")
    setter(scale_inv_a, center=None)
    return float(scale_inv_nm)


def _robust_aggregate_scales(
    scales: list[float],
    *,
    outlier_sigma: float,
) -> float:
    if not scales:
        raise ValueError("No scale candidates to aggregate.")
    if len(scales) == 1:
        return float(scales[0])

    values = np.asarray(scales, dtype=np.float64)
    sigma = max(0.0, float(outlier_sigma))
    if len(values) >= 3 and sigma > 0:
        center = float(np.median(values))
        deviations = np.abs(values - center)
        mad = float(np.median(deviations))
        robust_sigma = 1.4826 * mad
        if robust_sigma > 0:
            keep = deviations <= max(1e-12, sigma * robust_sigma)
            if int(np.sum(keep)) >= 1:
                values = values[keep]
    return float(np.mean(values))


def scale_from_known_d_spacing_with_adapter(
    *,
    engine: str | None,
    known_d_spacing: Any,
    measured_radius_px: Any,
    known_d_spacing_unit: str = "nm",
    known_d_uncertainty_nm: Any = None,
    measured_radius_uncertainty_px: Any = None,
    reference_pairs: list[dict[str, Any]] | None = None,
    outlier_sigma: float | None = None,
) -> float:
    """
    Adapter entry point for scale calibration.

    Current implementation intentionally uses the same closed-form calibration equation as the
    internal fallback, while validating adapter dependencies so pipelines can explicitly require
    optional engines without changing the output contract.
    """
    resolved = resolve_diffraction_calibration_engine(engine)
    if resolved == "none":
        raise RuntimeError("Adapter engine 'none' does not provide an external calibration path.")
    _ensure_engine_dependency(resolved)

    sigma = 3.5 if outlier_sigma is None else max(0.0, float(outlier_sigma))
    candidates: list[float] = []

    if reference_pairs:
        for idx, pair in enumerate(reference_pairs):
            if not isinstance(pair, dict):
                continue
            pair_d = pair.get("known_d_spacing")
            pair_r = pair.get("measured_radius_px")
            if pair_d is None or pair_r is None:
                continue
            pair_unit = str(pair.get("known_d_spacing_unit", known_d_spacing_unit))
            try:
                if resolved == "py4dstem":
                    candidate = _single_pair_scale_from_py4dstem(
                        known_d_spacing=pair_d,
                        measured_radius_px=pair_r,
                        known_d_spacing_unit=pair_unit,
                    )
                else:
                    candidate = _single_pair_scale_from_pyxem(
                        known_d_spacing=pair_d,
                        measured_radius_px=pair_r,
                        known_d_spacing_unit=pair_unit,
                    )
            except Exception:  # noqa: BLE001
                if idx == len(reference_pairs) - 1 and not candidates:
                    raise
                continue
            if candidate > 0:
                candidates.append(float(candidate))

    if not candidates:
        if resolved == "py4dstem":
            candidate = _single_pair_scale_from_py4dstem(
                known_d_spacing=known_d_spacing,
                measured_radius_px=measured_radius_px,
                known_d_spacing_unit=known_d_spacing_unit,
            )
        else:
            candidate = _single_pair_scale_from_pyxem(
                known_d_spacing=known_d_spacing,
                measured_radius_px=measured_radius_px,
                known_d_spacing_unit=known_d_spacing_unit,
            )
        candidates = [float(candidate)]

    scale = _robust_aggregate_scales(candidates, outlier_sigma=sigma)
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Adapter returned an invalid diffraction calibration value.")
    _ = known_d_uncertainty_nm
    _ = measured_radius_uncertainty_px
    return float(scale)
