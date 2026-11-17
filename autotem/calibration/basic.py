"""B1 calibration orchestration: pixel + diffraction + provenance."""

from __future__ import annotations

from typing import Any

from autotem.calibration.diffraction_calibration import resolve_diffraction_calibration
from autotem.calibration.pixel_calibration import (
    resolve_pixel_size_measurement,
    resolve_voltage_kv,
)
from autotem.core import CalibrationResult, Measurement, TEMDataset


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


def _merge_reference(
    reference: dict[str, object] | None,
    parameters: dict[str, object] | None,
) -> dict[str, object]:
    merged: dict[str, object] = {}
    if isinstance(reference, dict):
        merged.update(reference)
    if isinstance(parameters, dict):
        nested = parameters.get("reference")
        if isinstance(nested, dict):
            merged.update(nested)
    return merged


def _estimate_confidence(
    *,
    pixel_size: Measurement | None,
    voltage_kv: float | None,
    diffraction_scale: float | None,
    warning_count: int,
) -> float:
    score = 0.15
    if pixel_size is not None:
        score += 0.35
    if voltage_kv is not None:
        score += 0.2
    if diffraction_scale is not None:
        score += 0.2
    score -= min(0.3, 0.05 * warning_count)
    return max(0.0, min(1.0, score))


def calibrate_dataset(
    dataset: TEMDataset,
    *,
    method: str = "auto",
    reference: dict[str, object] | None = None,
    parameters: dict[str, object] | None = None,
) -> CalibrationResult:
    """Resolve B1 pixel/diffraction calibration with warning-forward behavior."""
    params = dict(parameters or {})
    ref = _merge_reference(reference, params)
    warnings: list[str] = []
    reference_measurements: list[Measurement] = []

    pixel_size, pixel_source, pixel_warnings = resolve_pixel_size_measurement(
        dataset,
        parameters=params,
        reference=ref,
    )
    warnings.extend(pixel_warnings)

    voltage_kv, voltage_source, voltage_warnings = resolve_voltage_kv(
        dataset,
        parameters=params,
        reference=ref,
    )
    warnings.extend(voltage_warnings)

    fft_shape = tuple(int(v) for v in dataset.data.shape) if dataset.data.shape else None
    direct_pattern = dataset.modality in {"SAED", "NBED"}
    fft_pixel_nm = None if direct_pattern or pixel_size is None else _to_float(pixel_size.value)
    diffraction_resolved = resolve_diffraction_calibration(
        pixel_size_nm=fft_pixel_nm,
        voltage_kv=voltage_kv,
        parameters=params,
        reference=ref,
        fft_shape=fft_shape,
    )
    warnings.extend(list(diffraction_resolved.get("warnings", [])))
    if direct_pattern and pixel_size is not None and diffraction_resolved["source"] == "unresolved":
        warnings.append(
            "Direct diffraction requires detector/reference calibration; FFT fallback withheld."
        )
    pair_details = diffraction_resolved.get("reference_pair_details")

    if isinstance(pair_details, dict) and isinstance(pair_details.get("pairs"), list):
        for idx, row in enumerate(pair_details.get("pairs", [])):
            if not isinstance(row, dict):
                continue
            known_d_nm = _to_float(row.get("known_d_nm"))
            radius_px = _to_float(row.get("measured_radius_px"))
            label = str(row.get("label", f"pair[{idx}]"))
            if known_d_nm is not None and known_d_nm > 0:
                reference_measurements.append(
                    Measurement(
                        value=float(known_d_nm),
                        unit="nm",
                        method=f"calibration.reference.{label}.known_d_nm",
                    )
                )
            if radius_px is not None and radius_px > 0:
                reference_measurements.append(
                    Measurement(
                        value=float(radius_px),
                        unit="pixel",
                        method=f"calibration.reference.{label}.measured_radius_px",
                    )
                )
    else:
        known_d_spacing = (
            params.get("known_d_spacing")
            if "known_d_spacing" in params
            else ref.get("known_d_spacing")
        )
        known_d_spacing_unit = str(
            params.get("known_d_spacing_unit", ref.get("known_d_spacing_unit", "nm"))
        )
        if known_d_spacing is not None:
            try:
                from autotem.calibration.pixel_calibration import convert_length

                known_d_nm = convert_length(known_d_spacing, known_d_spacing_unit, "nm")
            except Exception:  # noqa: BLE001
                warnings.append("known_d_spacing exists but cannot be converted to nm.")
            else:
                reference_measurements.append(
                    Measurement(
                        value=float(known_d_nm),
                        unit="nm",
                        method="calibration.reference.known_d_spacing",
                    )
                )

        measured_radius_px = (
            params.get("measured_radius_px")
            if "measured_radius_px" in params
            else ref.get("measured_radius_px")
        )
        radius_value = _to_float(measured_radius_px)
        if radius_value is not None and radius_value > 0:
            reference_measurements.append(
                Measurement(
                    value=float(radius_value),
                    unit="pixel",
                    method="calibration.reference.measured_radius_px",
                )
            )

    detector_pixel_um = (
        params.get("detector_pixel_size_um")
        if "detector_pixel_size_um" in params
        else ref.get("detector_pixel_size_um")
    )
    detector_value = _to_float(detector_pixel_um)
    if detector_value is not None and detector_value > 0:
        reference_measurements.append(
            Measurement(
                value=float(detector_value),
                unit="um/pixel",
                method="calibration.reference.detector_pixel_size_um",
            )
        )

    diffraction_scale_value = _to_float(
        diffraction_resolved.get("diffraction_scale_inv_nm_per_px")
    )
    diffraction_scale_uncertainty = _to_float(
        diffraction_resolved.get("diffraction_scale_uncertainty")
    )
    diffraction_scale = (
        None
        if diffraction_scale_value is None
        else Measurement(
            value=float(diffraction_scale_value),
            unit="1/nm/pixel",
            uncertainty=(
                None
                if diffraction_scale_uncertainty is None or diffraction_scale_uncertainty < 0
                else float(diffraction_scale_uncertainty)
            ),
            method=str(diffraction_resolved.get("source", "unresolved")),
            provenance={"source": str(diffraction_resolved.get("source", "unresolved"))},
            confidence=0.85 if diffraction_scale_value > 0 else 0.0,
        )
    )

    camera_length_value = _to_float(diffraction_resolved.get("camera_length_mm"))
    camera_length = (
        None
        if camera_length_value is None
        else Measurement(
            value=float(camera_length_value),
            unit="mm",
            method="camera_length_estimate",
            confidence=0.65,
        )
    )

    voltage_measurement = (
        None
        if voltage_kv is None
        else Measurement(
            value=float(voltage_kv),
            unit="kV",
            method=voltage_source,
            confidence=0.9 if voltage_source.startswith("parameter") else 0.75,
        )
    )

    confidence = _estimate_confidence(
        pixel_size=pixel_size,
        voltage_kv=voltage_kv,
        diffraction_scale=diffraction_scale_value,
        warning_count=len(warnings),
    )

    transform = {
        "method": method,
        "pixel_source": pixel_source,
        "voltage_source": voltage_source,
        "diffraction_source": str(diffraction_resolved.get("source", "unresolved")),
        "reference_pair_details": pair_details,
    }

    return CalibrationResult(
        pixel_size=pixel_size,
        pixel_unit="nm" if pixel_size is not None else "unknown",
        diffraction_scale=diffraction_scale,
        camera_length=camera_length,
        voltage=voltage_measurement,
        reference_measurements=reference_measurements,
        transform=transform,
        confidence=confidence,
        provenance={
            "method": method,
            "pixel_source": pixel_source,
            "voltage_source": voltage_source,
            "diffraction_source": str(diffraction_resolved.get("source", "unresolved")),
            "reference_pair_details": pair_details,
            "workflow_parameters": params,
            "reference": ref,
        },
        warnings=list(dict.fromkeys(warnings)),
    )
