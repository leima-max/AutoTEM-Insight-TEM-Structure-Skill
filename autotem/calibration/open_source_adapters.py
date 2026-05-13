"""Optional open-source calibration adapters.

These adapters are intentionally optional. The default path remains the
deterministic in-house/internal implementation to avoid hard coupling to
GPL-licensed stacks in baseline deployments.
"""

from __future__ import annotations

import importlib
from typing import Any

from autotem.core import Measurement

SUPPORTED_DIFFRACTION_CALIBRATION_ENGINES = {"auto", "internal", "py4dstem", "pyxem"}


def resolve_diffraction_calibration_engine(
    *,
    parameters: dict[str, Any] | None = None,
    reference: dict[str, Any] | None = None,
) -> str:
    """Resolve adapter engine from parameters/reference with safe defaults."""
    for container in (parameters, reference):
        if not isinstance(container, dict):
            continue
        for key in ("diffraction_calibration_engine", "calibration_engine", "adapter"):
            raw = container.get(key)
            if raw is None:
                continue
            mode = str(raw).strip().lower()
            if mode in SUPPORTED_DIFFRACTION_CALIBRATION_ENGINES:
                return mode
    return "internal"


def _unit_scale_to_inv_nm(unit: str | None) -> float:
    if unit is None:
        return 10.0
    normalized = unit.strip().lower().replace(" ", "")
    if normalized in {"a^-1", "1/a", "angstrom^-1", "1/angstrom"}:
        return 10.0
    if normalized in {"nm^-1", "1/nm"}:
        return 1.0
    return 10.0


def _internal_known_d_scale(known_d_nm: float, measured_radius_px: float) -> float:
    return (1.0 / known_d_nm) / measured_radius_px


def scale_from_known_d_spacing_with_adapter(
    *,
    engine: str,
    known_d_nm: float,
    measured_radius_px: float,
    known_d_uncertainty_nm: float | None = None,
    radius_uncertainty_px: float | None = None,
) -> tuple[Measurement, list[str], dict[str, Any]]:
    """Compute diffraction scale via selected adapter or fallback."""
    warnings: list[str] = []
    adapter_meta: dict[str, Any] = {"requested_engine": engine}

    if engine == "auto":
        engine = "py4dstem"

    if engine == "py4dstem":
        try:
            qpixelsize = importlib.import_module("py4DSTEM.process.calibration.qpixelsize")
            fn = qpixelsize.get_Q_pixel_size
            raw = fn(q_meas=measured_radius_px, q_known=known_d_nm * 10.0, units="A")
            if isinstance(raw, tuple):
                raw_value = raw[0]
                raw_unit = raw[1] if len(raw) > 1 else None
            else:
                raw_value = raw
                raw_unit = None
            value_inv_nm_per_px = float(raw_value) * _unit_scale_to_inv_nm(
                str(raw_unit) if raw_unit is not None else None
            )
            adapter_meta.update(
                {
                    "used_engine": "py4dstem",
                    "raw_value": raw_value,
                    "raw_unit": raw_unit,
                }
            )
            return (
                Measurement(
                    value=value_inv_nm_per_px,
                    unit="1/nm/pixel",
                    method="py4dstem.get_Q_pixel_size",
                    uncertainty=_propagate_scale_uncertainty(
                        scale=value_inv_nm_per_px,
                        known_d_nm=known_d_nm,
                        measured_radius_px=measured_radius_px,
                        known_d_uncertainty_nm=known_d_uncertainty_nm,
                        radius_uncertainty_px=radius_uncertainty_px,
                    ),
                    confidence=0.86,
                ),
                warnings,
                adapter_meta,
            )
        except Exception as exc:  # noqa: BLE001 - adapter path must degrade gracefully.
            warnings.append(
                "py4DSTEM adapter unavailable/failed; fallback to internal calibration formula. "
                f"detail={exc!s}"
            )
            engine = "internal"

    if engine == "pyxem":
        try:
            importlib.import_module("pyxem")
            value_inv_nm_per_px = _internal_known_d_scale(known_d_nm, measured_radius_px)
            adapter_meta.update({"used_engine": "pyxem_proxy"})
            warnings.append(
                "pyxem adapter currently runs in proxy mode using the internal formula; "
                "for strict pyxem workflows, integrate project-specific calibration pipelines."
            )
            return (
                Measurement(
                    value=value_inv_nm_per_px,
                    unit="1/nm/pixel",
                    method="pyxem.proxy_known_d_spacing_formula",
                    uncertainty=_propagate_scale_uncertainty(
                        scale=value_inv_nm_per_px,
                        known_d_nm=known_d_nm,
                        measured_radius_px=measured_radius_px,
                        known_d_uncertainty_nm=known_d_uncertainty_nm,
                        radius_uncertainty_px=radius_uncertainty_px,
                    ),
                    confidence=0.8,
                ),
                warnings,
                adapter_meta,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                "pyxem adapter unavailable; fallback to internal calibration formula. "
                f"detail={exc!s}"
            )
            engine = "internal"

    value_inv_nm_per_px = _internal_known_d_scale(known_d_nm, measured_radius_px)
    adapter_meta.update({"used_engine": "internal"})
    return (
        Measurement(
            value=value_inv_nm_per_px,
            unit="1/nm/pixel",
            method="known_d_spacing_single_reference",
            uncertainty=_propagate_scale_uncertainty(
                scale=value_inv_nm_per_px,
                known_d_nm=known_d_nm,
                measured_radius_px=measured_radius_px,
                known_d_uncertainty_nm=known_d_uncertainty_nm,
                radius_uncertainty_px=radius_uncertainty_px,
            ),
            confidence=0.85,
        ),
        warnings,
        adapter_meta,
    )


def _propagate_scale_uncertainty(
    *,
    scale: float,
    known_d_nm: float,
    measured_radius_px: float,
    known_d_uncertainty_nm: float | None,
    radius_uncertainty_px: float | None,
) -> float | None:
    if known_d_uncertainty_nm is None and radius_uncertainty_px is None:
        return None
    rel_terms: list[float] = []
    if known_d_uncertainty_nm is not None and known_d_nm > 0:
        rel_terms.append((known_d_uncertainty_nm / known_d_nm) ** 2)
    if radius_uncertainty_px is not None and measured_radius_px > 0:
        rel_terms.append((radius_uncertainty_px / measured_radius_px) ** 2)
    if not rel_terms:
        return None
    return abs(scale) * float(sum(rel_terms) ** 0.5)
