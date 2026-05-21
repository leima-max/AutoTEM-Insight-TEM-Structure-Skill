"""B3 reciprocal metric helpers: g-vectors and d-spacing tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from autotem.core import DiffractionPeak, GVector, Measurement


@dataclass(slots=True)
class ReciprocalMetricsResult:
    """Computed g-vectors and per-peak d-spacing measurements."""

    g_vectors: list[GVector]
    d_by_peak: list[Measurement | None]
    warnings: list[str]
    diagnostics: dict[str, Any]


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        import re

        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value.strip())
        if match:
            return float(match.group(0))
    return None


def _measurement(
    value: float,
    *,
    unit: str,
    method: str,
    uncertainty: float | None = None,
    confidence: float | None = None,
) -> Measurement:
    return Measurement(
        value=float(value),
        unit=unit,
        uncertainty=uncertainty,
        method=method,
        confidence=confidence,
    )


def compute_gvectors_and_dspacing(
    peaks: list[DiffractionPeak],
    *,
    center: tuple[float, float],
    diffraction_scale_value: float | None,
    diffraction_scale_uncertainty: float | None,
    diffraction_scale_source: str,
    center_uncertainty_px: float | None = None,
) -> ReciprocalMetricsResult:
    """Compute reciprocal vectors and d-spacing values for detected peaks."""
    warnings: list[str] = []
    if diffraction_scale_value is None or diffraction_scale_value <= 0:
        warnings.append(
            "diffraction_scale is unavailable; g-vectors/d-spacings are reported in pixel units."
        )
        reciprocal_unit = "1/pixel"
        d_spacing_unit = "pixel"
        scale_value = None
        scale_uncertainty = None
    else:
        reciprocal_unit = "1/nm"
        d_spacing_unit = "nm"
        scale_value = float(diffraction_scale_value)
        scale_uncertainty = (
            None
            if diffraction_scale_uncertainty is None or diffraction_scale_uncertainty < 0
            else float(diffraction_scale_uncertainty)
        )
    center_sigma = (
        0.0
        if center_uncertainty_px is None or center_uncertainty_px < 0
        else float(center_uncertainty_px)
    )

    g_vectors: list[GVector] = []
    d_by_peak: list[Measurement | None] = []
    rel_uncertainties: list[float] = []
    high_rel_unc_count = 0
    for idx, peak in enumerate(peaks):
        dx_px = float(peak.x - center[0])
        dy_px = float(peak.y - center[1])
        radius_px = float(np.hypot(dx_px, dy_px))
        sigma_x_px = 0.0 if peak.sigma_x is None else max(0.0, float(peak.sigma_x))
        sigma_y_px = 0.0 if peak.sigma_y is None else max(0.0, float(peak.sigma_y))
        if radius_px > 1e-12:
            radial_sigma_from_peak = float(
                np.sqrt(
                    ((dx_px / radius_px) ** 2) * (sigma_x_px**2)
                    + ((dy_px / radius_px) ** 2) * (sigma_y_px**2)
                )
            )
        else:
            radial_sigma_from_peak = float(np.hypot(sigma_x_px, sigma_y_px))
        radial_sigma_px = float(np.hypot(radial_sigma_from_peak, center_sigma))

        if scale_value is None:
            gx = dx_px
            gy = dy_px
            g_mag = radius_px
            gx_unc = float(np.hypot(sigma_x_px, center_sigma))
            gy_unc = float(np.hypot(sigma_y_px, center_sigma))
            g_unc = radial_sigma_px
            g_method = "radius_in_pixels"
            gx_method = "peak_offset_x_px"
            gy_method = "peak_offset_y_px"
        else:
            gx = dx_px * scale_value
            gy = dy_px * scale_value
            g_mag = radius_px * scale_value
            scale_sigma = 0.0 if scale_uncertainty is None else scale_uncertainty
            gx_unc = float(
                np.hypot(
                    abs(dx_px) * scale_sigma,
                    abs(scale_value) * np.hypot(sigma_x_px, center_sigma),
                )
            )
            gy_unc = float(
                np.hypot(
                    abs(dy_px) * scale_sigma,
                    abs(scale_value) * np.hypot(sigma_y_px, center_sigma),
                )
            )
            g_unc = float(
                np.hypot(
                    abs(radius_px) * scale_sigma,
                    abs(scale_value) * radial_sigma_px,
                )
            )
            g_method = "radius_times_diffraction_scale"
            gx_method = "peak_offset_x_times_scale"
            gy_method = "peak_offset_y_times_scale"

        g_vectors.append(
            GVector(
                gx=_measurement(
                    gx,
                    unit=reciprocal_unit,
                    method=gx_method,
                    uncertainty=gx_unc,
                    confidence=peak.confidence,
                ),
                gy=_measurement(
                    gy,
                    unit=reciprocal_unit,
                    method=gy_method,
                    uncertainty=gy_unc,
                    confidence=peak.confidence,
                ),
                magnitude=_measurement(
                    g_mag,
                    unit=reciprocal_unit,
                    method=g_method,
                    uncertainty=g_unc,
                    confidence=peak.confidence,
                ),
                angle=_measurement(
                    float(np.degrees(np.arctan2(gy, gx))),
                    unit="deg",
                    method="atan2_gy_gx_error_propagation",
                    uncertainty=(
                        float(
                            np.degrees(
                                np.hypot(
                                    abs(gy) * gx_unc,
                                    abs(gx) * gy_unc,
                                )
                                / max(g_mag**2, 1e-12)
                            )
                        )
                        if g_mag > 1e-12
                        else None
                    ),
                    confidence=peak.confidence,
                ),
                source_peak_index=idx,
            )
        )

        if g_mag <= 1e-12:
            d_by_peak.append(None)
            continue
        d_value = 1.0 / g_mag
        d_unc = None if g_unc is None else abs(g_unc / max(g_mag**2, 1e-12))
        if d_unc is not None:
            rel_unc = abs(d_unc / max(abs(d_value), 1e-12))
            rel_uncertainties.append(float(rel_unc))
            if rel_unc > 0.5:
                high_rel_unc_count += 1
        d_by_peak.append(
            _measurement(
                d_value,
                unit=d_spacing_unit,
                method="inverse_g_magnitude_with_error_propagation",
                uncertainty=d_unc,
                confidence=peak.confidence,
            )
        )

    if rel_uncertainties:
        max_rel_unc = max(rel_uncertainties)
        if max_rel_unc > 0.5:
            warnings.append(
                "Large d-spacing uncertainty detected for some peaks "
                f"(max relative uncertainty={max_rel_unc:.3f})."
            )

    diagnostics = {
        "peak_count": len(peaks),
        "scale_value": scale_value,
        "scale_uncertainty": scale_uncertainty,
        "scale_source": diffraction_scale_source,
        "center_uncertainty_px": center_sigma,
        "uncertainty_model": (
            "sigma_g^2=(r*sigma_scale)^2+(scale*sigma_r)^2; "
            "sigma_r includes peak sigma + center sigma; sigma_d=sigma_g/g^2"
        ),
        "d_uncertainty_coverage": (
            (len(rel_uncertainties) / len(d_by_peak)) if d_by_peak else 0.0
        ),
        "d_uncertainty_max_relative": max(rel_uncertainties) if rel_uncertainties else 0.0,
        "d_uncertainty_mean_relative": (
            float(sum(rel_uncertainties) / len(rel_uncertainties))
            if rel_uncertainties
            else 0.0
        ),
        "d_uncertainty_high_relative_count": high_rel_unc_count,
        "reciprocal_unit": reciprocal_unit,
        "d_spacing_unit": d_spacing_unit,
    }
    return ReciprocalMetricsResult(
        g_vectors=g_vectors,
        d_by_peak=d_by_peak,
        warnings=warnings,
        diagnostics=diagnostics,
    )
