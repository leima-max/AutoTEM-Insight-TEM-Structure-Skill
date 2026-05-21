"""Diffraction calibration helpers for B1 reciprocal-space baseline."""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np

from autotem.calibration.pixel_calibration import convert_length


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


def electron_wavelength_nm(voltage_kv: float | int) -> float:
    """Relativistic electron wavelength in nm for accelerating voltage in kV."""
    voltage = _to_float(voltage_kv)
    if voltage is None or voltage <= 0:
        raise ValueError("voltage_kv must be positive.")
    volts = voltage * 1000.0
    wavelength_angstrom = 12.2639 / sqrt(volts * (1.0 + 0.97845e-6 * volts))
    return float(wavelength_angstrom * 0.1)


def _resolve_value(
    parameters: dict[str, Any],
    reference: dict[str, Any],
    key: str,
) -> Any:
    if key in parameters:
        return parameters.get(key)
    return reference.get(key)


def _collect_reference_pairs(
    parameters: dict[str, Any],
    reference: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    pairs: list[dict[str, Any]] = []
    warnings: list[str] = []

    raw_refs = _resolve_value(parameters, reference, "known_d_spacing_refs")
    if isinstance(raw_refs, list):
        for idx, row in enumerate(raw_refs):
            if isinstance(row, dict):
                pairs.append(dict(row))
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                pairs.append(
                    {
                        "known_d_spacing": row[0],
                        "measured_radius_px": row[1],
                        "label": f"known_d_spacing_refs[{idx}]",
                    }
                )
            else:
                warnings.append(
                    "Ignoring unsupported known_d_spacing_refs entry at "
                    f"index {idx}: expected mapping or [d, radius]."
                )

    known_d_spacings = _resolve_value(parameters, reference, "known_d_spacings")
    measured_radii = _resolve_value(parameters, reference, "measured_radii_px")
    if isinstance(known_d_spacings, (list, tuple)) and isinstance(measured_radii, (list, tuple)):
        count = min(len(known_d_spacings), len(measured_radii))
        if len(known_d_spacings) != len(measured_radii):
            warnings.append(
                "known_d_spacings and measured_radii_px length mismatch; "
                f"using first {count} paired entries."
            )
        units = _resolve_value(parameters, reference, "known_d_spacing_units")
        d_uncertainties = _resolve_value(parameters, reference, "known_d_uncertainties_nm")
        r_uncertainties = _resolve_value(parameters, reference, "measured_radii_uncertainties_px")
        labels = _resolve_value(parameters, reference, "known_d_spacing_labels")
        for idx in range(count):
            row: dict[str, Any] = {
                "known_d_spacing": known_d_spacings[idx],
                "measured_radius_px": measured_radii[idx],
                "label": (
                    labels[idx]
                    if isinstance(labels, (list, tuple)) and idx < len(labels)
                    else f"known_d_spacings[{idx}]"
                ),
            }
            if isinstance(units, (list, tuple)) and idx < len(units):
                row["known_d_spacing_unit"] = units[idx]
            if isinstance(d_uncertainties, (list, tuple)) and idx < len(d_uncertainties):
                row["known_d_uncertainty_nm"] = d_uncertainties[idx]
            if isinstance(r_uncertainties, (list, tuple)) and idx < len(r_uncertainties):
                row["measured_radius_uncertainty_px"] = r_uncertainties[idx]
            pairs.append(row)

    known_d_spacing = _resolve_value(parameters, reference, "known_d_spacing")
    measured_radius_px = _resolve_value(parameters, reference, "measured_radius_px")
    if known_d_spacing is not None and measured_radius_px is not None:
        pairs.append(
            {
                "known_d_spacing": known_d_spacing,
                "measured_radius_px": measured_radius_px,
                "known_d_spacing_unit": _resolve_value(
                    parameters,
                    reference,
                    "known_d_spacing_unit",
                ),
                "known_d_uncertainty_nm": _resolve_value(
                    parameters,
                    reference,
                    "known_d_uncertainty_nm",
                ),
                "measured_radius_uncertainty_px": _resolve_value(
                    parameters,
                    reference,
                    "measured_radius_uncertainty_px",
                ),
                "label": "known_d_spacing",
            }
        )

    return pairs, warnings


def reciprocal_scale_from_pixel_size(
    pixel_size_nm: float | int,
    fft_shape: int | tuple[int, ...] | None = None,
) -> float | None:
    """
    Estimate reciprocal calibration scale in 1/nm per pixel.

    For FFT coordinates:
      delta_g ~= 1 / (N * delta_x)
    where N is FFT size and delta_x is real-space pixel size.
    """
    pixel_nm = _to_float(pixel_size_nm)
    if pixel_nm is None or pixel_nm <= 0:
        return None

    if fft_shape is None:
        n = 1
    elif isinstance(fft_shape, int):
        n = int(fft_shape)
    else:
        if not fft_shape:
            n = 1
        else:
            n = int(max(fft_shape))

    if n <= 0:
        return None
    return float(1.0 / (n * pixel_nm))


def diffraction_scale_from_reference_pairs(
    reference_pairs: list[dict[str, Any]],
    *,
    default_known_d_spacing_unit: str = "nm",
    outlier_sigma: float = 3.5,
    min_inlier_pairs: int = 2,
) -> tuple[float, float | None, dict[str, Any], list[str]]:
    """
    Calibrate reciprocal scale from one or more known d-spacing references.

    Each pair requires `known_d_spacing` and `measured_radius_px`.
    """
    if not reference_pairs:
        raise ValueError("At least one reference pair is required.")

    warnings: list[str] = []
    parsed_pairs: list[dict[str, Any]] = []
    for idx, pair in enumerate(reference_pairs):
        if not isinstance(pair, dict):
            warnings.append(f"Skipping reference pair {idx}: not a mapping.")
            continue
        known_d_spacing = pair.get("known_d_spacing")
        measured_radius_px = pair.get("measured_radius_px")
        if known_d_spacing is None or measured_radius_px is None:
            warnings.append(
                f"Skipping reference pair {idx}: requires known_d_spacing and measured_radius_px."
            )
            continue
        unit = str(pair.get("known_d_spacing_unit", default_known_d_spacing_unit))
        try:
            known_d_nm = convert_length(known_d_spacing, unit, "nm")
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"Skipping reference pair {idx}: known_d_spacing conversion failed: {exc}"
            )
            continue
        radius_px = _to_float(measured_radius_px)
        if known_d_nm <= 0 or radius_px is None or radius_px <= 0:
            warnings.append(
                f"Skipping reference pair {idx}: known_d_spacing and radius must be positive."
            )
            continue
        d_unc = _to_float(pair.get("known_d_uncertainty_nm"))
        r_unc = _to_float(pair.get("measured_radius_uncertainty_px"))
        scale = float(1.0 / (known_d_nm * radius_px))
        variance = None
        if (d_unc is not None and d_unc > 0) or (r_unc is not None and r_unc > 0):
            rel_sq = 0.0
            if d_unc is not None and d_unc > 0:
                rel_sq += (d_unc / known_d_nm) ** 2
            if r_unc is not None and r_unc > 0:
                rel_sq += (r_unc / radius_px) ** 2
            variance = (scale * sqrt(max(0.0, rel_sq))) ** 2
        parsed_pairs.append(
            {
                "idx": idx,
                "label": str(pair.get("label", f"pair[{idx}]")),
                "known_d_nm": float(known_d_nm),
                "measured_radius_px": float(radius_px),
                "scale_inv_nm_per_px": scale,
                "variance": variance,
            }
        )

    if not parsed_pairs:
        raise ValueError("No valid reference pairs after parsing.")

    scales = np.asarray([row["scale_inv_nm_per_px"] for row in parsed_pairs], dtype=np.float64)
    inlier_mask = np.ones(len(parsed_pairs), dtype=bool)
    outlier_sigma_value = _to_float(outlier_sigma)
    outlier_sigma_eff = 3.5 if outlier_sigma_value is None else max(0.0, outlier_sigma_value)
    min_inlier = max(1, int(min_inlier_pairs))
    if len(parsed_pairs) >= 3 and outlier_sigma_eff > 0:
        center = float(np.median(scales))
        deviations = np.abs(scales - center)
        mad = float(np.median(deviations))
        robust_sigma = 1.4826 * mad
        if robust_sigma > 0:
            tolerance = max(1e-12, outlier_sigma_eff * robust_sigma)
            candidate_mask = deviations <= tolerance
            if int(candidate_mask.sum()) >= min_inlier:
                inlier_mask = candidate_mask
                dropped = int((~candidate_mask).sum())
                if dropped > 0:
                    warnings.append(
                        f"Excluded {dropped} outlier reference pair(s) using robust sigma filter."
                    )
        else:
            tolerance = max(1e-12, 0.05 * abs(center))
            candidate_mask = deviations <= tolerance
            if int(candidate_mask.sum()) >= min_inlier:
                inlier_mask = candidate_mask
                dropped = int((~candidate_mask).sum())
                if dropped > 0:
                    warnings.append(
                        f"Excluded {dropped} outlier reference pair(s) using median tolerance."
                    )

    inlier_scales = scales[inlier_mask]
    inlier_pairs = [pair for pair, keep in zip(parsed_pairs, inlier_mask, strict=False) if keep]
    variances = [pair["variance"] for pair in inlier_pairs]
    use_weighted = bool(variances) and all(
        variance is not None and variance > 0 for variance in variances
    )

    if use_weighted:
        weights = np.asarray([1.0 / float(variance) for variance in variances], dtype=np.float64)
        total_weight = float(np.sum(weights))
        scale = float(np.sum(weights * inlier_scales) / max(1e-24, total_weight))
        uncertainty = float(sqrt(1.0 / max(1e-24, total_weight)))
        aggregation = "weighted_inverse_variance"
    else:
        scale = float(np.mean(inlier_scales))
        if len(inlier_scales) >= 2:
            sem = float(np.std(inlier_scales, ddof=1) / sqrt(len(inlier_scales)))
            uncertainty = sem
        elif variances and variances[0] is not None:
            uncertainty = float(sqrt(float(variances[0])))
        else:
            uncertainty = None
        aggregation = "mean_sem"
        if any(variance is None for variance in variances):
            warnings.append(
                "At least one reference pair uncertainty is missing; using unweighted mean."
            )

    outlier_count = int(np.size(inlier_mask) - int(np.sum(inlier_mask)))
    details = {
        "pair_count": len(parsed_pairs),
        "inlier_count": int(np.sum(inlier_mask)),
        "outlier_count": outlier_count,
        "aggregation": aggregation,
        "pairs": [
            {
                "label": pair["label"],
                "known_d_nm": pair["known_d_nm"],
                "measured_radius_px": pair["measured_radius_px"],
                "scale_inv_nm_per_px": pair["scale_inv_nm_per_px"],
                "inlier": bool(keep),
            }
            for pair, keep in zip(parsed_pairs, inlier_mask, strict=False)
        ],
    }
    return scale, uncertainty, details, warnings


def diffraction_scale_from_known_d_spacing(
    known_d_spacing_nm: float | int | str,
    measured_radius_px: float | int,
    *,
    known_d_spacing_unit: str = "nm",
    known_d_uncertainty_nm: float | int | None = None,
    measured_radius_uncertainty_px: float | int | None = None,
) -> tuple[float, float | None]:
    """
    Calibrate reciprocal scale (1/nm per pixel) from known d-spacing.

    g = 1 / d, and g = scale * radius_px
      => scale = 1 / (d * radius_px)
    """
    scale, uncertainty, _, _ = diffraction_scale_from_reference_pairs(
        [
            {
                "known_d_spacing": known_d_spacing_nm,
                "known_d_spacing_unit": known_d_spacing_unit,
                "measured_radius_px": measured_radius_px,
                "known_d_uncertainty_nm": known_d_uncertainty_nm,
                "measured_radius_uncertainty_px": measured_radius_uncertainty_px,
                "label": "single_pair",
            }
        ],
        default_known_d_spacing_unit=known_d_spacing_unit,
        outlier_sigma=0.0,
        min_inlier_pairs=1,
    )
    return scale, uncertainty


def estimate_camera_length_mm(
    diffraction_scale_inv_nm_per_px: float | int,
    voltage_kv: float | int,
    detector_pixel_size_um: float | int,
) -> float | None:
    """
    Estimate camera length from reciprocal calibration.

    Small-angle approximation:
      theta ~= lambda * g
      r_detector = L * theta
      g_per_pixel = diffraction_scale_inv_nm_per_px
      r_per_pixel(mm) = detector_pixel_size_um * 1e-3
      => L(mm) = r_per_pixel / (lambda_nm * g_per_pixel)
    """
    scale = _to_float(diffraction_scale_inv_nm_per_px)
    detector_um = _to_float(detector_pixel_size_um)
    if scale is None or scale <= 0:
        return None
    if detector_um is None or detector_um <= 0:
        return None
    wavelength_nm = electron_wavelength_nm(voltage_kv)
    if wavelength_nm <= 0:
        return None
    detector_mm = detector_um * 1e-3
    return float(detector_mm / (wavelength_nm * scale))


def resolve_diffraction_calibration(
    *,
    pixel_size_nm: float | None = None,
    voltage_kv: float | None = None,
    parameters: dict[str, Any] | None = None,
    reference: dict[str, Any] | None = None,
    fft_shape: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Resolve diffraction calibration scale/camera length with fallbacks."""
    params = dict(parameters or {})
    ref = dict(reference or {})
    warnings: list[str] = []
    source = "unresolved"
    pair_details: dict[str, Any] | None = None

    scale = _to_float(params.get("diffraction_scale_inv_nm_per_px"))
    scale_uncertainty = None
    reference_pairs, pair_warnings = _collect_reference_pairs(params, ref)
    warnings.extend(pair_warnings)
    if scale is not None and scale > 0:
        source = "parameter.diffraction_scale_inv_nm_per_px"
    else:
        engine = str(params.get("diffraction_calibration_engine", "none")).strip().lower()
        if engine and engine != "none":
            try:
                from autotem.calibration.open_source_adapters import (
                    scale_from_known_d_spacing_with_adapter,
                )

                adapter_scale = scale_from_known_d_spacing_with_adapter(
                    engine=engine,
                    known_d_spacing=(
                        params.get("known_d_spacing")
                        if "known_d_spacing" in params
                        else ref.get("known_d_spacing")
                    ),
                    measured_radius_px=(
                        params.get("measured_radius_px")
                        if "measured_radius_px" in params
                        else ref.get("measured_radius_px")
                    ),
                    known_d_spacing_unit=str(
                        params.get(
                            "known_d_spacing_unit",
                            ref.get("known_d_spacing_unit", "nm"),
                        )
                    ),
                    known_d_uncertainty_nm=(
                        params.get("known_d_uncertainty_nm")
                        if "known_d_uncertainty_nm" in params
                        else ref.get("known_d_uncertainty_nm")
                    ),
                    measured_radius_uncertainty_px=(
                        params.get("measured_radius_uncertainty_px")
                        if "measured_radius_uncertainty_px" in params
                        else ref.get("measured_radius_uncertainty_px")
                    ),
                    reference_pairs=reference_pairs,
                    outlier_sigma=_to_float(
                        _resolve_value(params, ref, "multi_reference_outlier_sigma")
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    "Open-source diffraction adapter failed "
                    f"(engine={engine}): {exc}; fallback to internal calibration."
                )
            else:
                scale = _to_float(adapter_scale)
                if scale is not None and scale > 0:
                    source = f"adapter.{engine}"

    if scale is None or scale <= 0:
        if reference_pairs:
            try:
                scale, scale_uncertainty, pair_details, pair_internal_warnings = (
                    diffraction_scale_from_reference_pairs(
                        reference_pairs,
                        default_known_d_spacing_unit=str(
                            _resolve_value(params, ref, "known_d_spacing_unit") or "nm"
                        ),
                        outlier_sigma=float(
                            _to_float(_resolve_value(params, ref, "multi_reference_outlier_sigma"))
                            or 3.5
                        ),
                        min_inlier_pairs=int(
                            _to_float(_resolve_value(params, ref, "multi_reference_min_inliers"))
                            or 2
                        ),
                    )
                )
                warnings.extend(pair_internal_warnings)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Known-d-spacing diffraction calibration failed: {exc}")
                scale = None
            else:
                inlier_count = (
                    int(pair_details.get("inlier_count", 0))
                    if isinstance(pair_details, dict)
                    else 0
                )
                source = (
                    "known_d_spacing"
                    if inlier_count <= 1
                    else "known_d_spacing.multi_reference"
                )

    if (scale is None or scale <= 0) and pixel_size_nm is not None:
        fallback = reciprocal_scale_from_pixel_size(pixel_size_nm, fft_shape=fft_shape)
        if fallback is not None and fallback > 0:
            scale = fallback
            source = "pixel_size_fft_fallback"
            warnings.append(
                "Diffraction scale inferred from real-space pixel size fallback; "
                "provide known_d_spacing + measured_radius_px for stronger calibration."
            )

    if scale is None or scale <= 0:
        warnings.append(
            "Diffraction calibration is unresolved; provide either "
            "diffraction_scale_inv_nm_per_px or known_d_spacing + measured_radius_px."
        )
        scale = None

    detector_pixel_um = (
        params.get("detector_pixel_size_um")
        if "detector_pixel_size_um" in params
        else ref.get("detector_pixel_size_um")
    )
    camera_length_mm = None
    if scale is not None and voltage_kv is not None and detector_pixel_um is not None:
        camera_length_mm = estimate_camera_length_mm(scale, voltage_kv, detector_pixel_um)
        if camera_length_mm is None:
            warnings.append(
                "Camera length estimation failed from scale/voltage/detector pixel size."
            )

    return {
        "diffraction_scale_inv_nm_per_px": scale,
        "diffraction_scale_uncertainty": scale_uncertainty,
        "camera_length_mm": camera_length_mm,
        "source": source,
        "reference_pair_details": pair_details,
        "warnings": warnings,
    }
