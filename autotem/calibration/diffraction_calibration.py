"""B1 diffraction calibration utilities."""

from __future__ import annotations

import math
from typing import Any

from autotem.core import Measurement

from .open_source_adapters import (
    resolve_diffraction_calibration_engine,
    scale_from_known_d_spacing_with_adapter,
)
from .pixel_calibration import coerce_length_to_nm


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def electron_wavelength_nm(voltage_kv: float) -> float:
    """Return relativistic electron wavelength in nanometers for given kV."""
    voltage_v = max(float(voltage_kv), 1.0) * 1000.0
    # lambda(Angstrom) = 12.2639 / sqrt(V * (1 + 0.97845e-6 * V))
    wavelength_angstrom = 12.2639 / math.sqrt(voltage_v * (1.0 + 0.97845e-6 * voltage_v))
    return wavelength_angstrom * 0.1


def reciprocal_scale_from_pixel_size(
    pixel_size_nm: float,
    *,
    fft_size_px: int | None = None,
    pixel_size_uncertainty_nm: float | None = None,
) -> Measurement:
    """Compute reciprocal-space calibration in 1/nm per pixel."""
    if fft_size_px is not None and fft_size_px > 0:
        value = 1.0 / (float(fft_size_px) * max(float(pixel_size_nm), 1e-12))
        method = "fft_size_and_pixel_size"
    else:
        value = 1.0 / max(float(pixel_size_nm), 1e-12)
        method = "reciprocal_pixel_size"
    uncertainty = None
    if pixel_size_uncertainty_nm is not None and pixel_size_nm > 0:
        uncertainty = abs(value) * abs(pixel_size_uncertainty_nm / pixel_size_nm)
    return Measurement(
        value=value,
        unit="1/nm/pixel",
        method=method,
        uncertainty=uncertainty,
        confidence=0.6,
    )


def diffraction_scale_from_known_d_spacing(
    measured_radius_px: Any,
    known_d_spacing_value: Any,
    *,
    known_d_spacing_unit: str | None = "nm",
    known_d_uncertainty_nm: float | None = None,
    measured_radius_uncertainty_px: float | None = None,
    adapter_engine: str = "internal",
) -> tuple[Measurement | None, Measurement | None, list[str], dict[str, Any]]:
    """Infer reciprocal calibration using one known ring/spot spacing."""
    warnings: list[str] = []
    radius_px = _as_float(measured_radius_px)
    if radius_px is None or radius_px <= 0:
        return None, None, warnings, {}
    d_nm = coerce_length_to_nm(known_d_spacing_value, known_d_spacing_unit)
    if d_nm is None or d_nm <= 0:
        return None, None, warnings, {}
    if known_d_uncertainty_nm is None:
        known_d_uncertainty_nm = 0.01 * d_nm
    if measured_radius_uncertainty_px is None:
        measured_radius_uncertainty_px = 0.25

    scale_measurement, adapter_warnings, adapter_meta = scale_from_known_d_spacing_with_adapter(
        engine=adapter_engine,
        known_d_nm=d_nm,
        measured_radius_px=radius_px,
        known_d_uncertainty_nm=known_d_uncertainty_nm,
        radius_uncertainty_px=measured_radius_uncertainty_px,
    )
    warnings.extend(adapter_warnings)
    scale_measurement.parameters = {
        **scale_measurement.parameters,
        "measured_radius_px": radius_px,
        "known_d_nm": d_nm,
        "known_d_uncertainty_nm": known_d_uncertainty_nm,
        "measured_radius_uncertainty_px": measured_radius_uncertainty_px,
        **adapter_meta,
    }
    reference = Measurement(
        value=d_nm,
        unit="nm",
        uncertainty=known_d_uncertainty_nm,
        method="known_d_spacing",
        parameters={"measured_radius_px": radius_px},
        confidence=0.85,
    )
    return scale_measurement, reference, warnings, adapter_meta


def estimate_camera_length_mm(
    reciprocal_scale_inv_nm_per_px: float,
    *,
    detector_pixel_size_um: float,
    beam_energy_kv: float,
) -> Measurement | None:
    """Estimate camera length from reciprocal scale and detector geometry.

    Uses the small-angle approximation with g = R / (L * lambda), where R=1 px.
    """
    if reciprocal_scale_inv_nm_per_px <= 0:
        return None
    if detector_pixel_size_um <= 0 or beam_energy_kv <= 0:
        return None
    wavelength_nm = electron_wavelength_nm(beam_energy_kv)
    detector_pixel_mm = detector_pixel_size_um * 1e-3
    camera_length_mm = detector_pixel_mm / (reciprocal_scale_inv_nm_per_px * wavelength_nm)
    return Measurement(
        value=camera_length_mm,
        unit="mm",
        method="small_angle_camera_length_estimate",
        parameters={
            "detector_pixel_size_um": detector_pixel_size_um,
            "beam_energy_kv": beam_energy_kv,
            "electron_wavelength_nm": wavelength_nm,
        },
        confidence=0.45,
    )


def resolve_diffraction_calibration(
    *,
    pixel_size_nm: float,
    pixel_size_uncertainty_nm: float | None = None,
    reference: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    voltage_kv: float | None = None,
) -> tuple[Measurement, Measurement | None, list[Measurement], dict[str, Any], list[str], float]:
    """Resolve diffraction scale, camera length and calibration provenance."""
    warnings: list[str] = []
    refs: list[Measurement] = []
    transform: dict[str, Any] = {}
    confidence = 0.6
    container_chain = [("parameters", parameters), ("reference", reference)]
    adapter_engine = resolve_diffraction_calibration_engine(
        parameters=parameters,
        reference=reference,
    )
    transform["requested_adapter_engine"] = adapter_engine

    scale_measurement: Measurement | None = None
    for label, container in container_chain:
        if not isinstance(container, dict):
            continue

        known_d = container.get("known_d_spacing") or container.get("d_spacing")
        known_d_unit = container.get("known_d_spacing_unit") or container.get("d_spacing_unit")
        measured_radius_px = container.get("measured_radius_px") or container.get("measured_pixels")
        known_d_uncertainty_nm = _as_float(
            container.get("known_d_uncertainty_nm") or container.get("d_spacing_uncertainty_nm")
        )
        measured_radius_uncertainty_px = _as_float(
            container.get("measured_radius_uncertainty_px")
            or container.get("measured_pixels_uncertainty")
        )
        (
            scale_measurement,
            ref_measurement,
            adapter_warnings,
            adapter_meta,
        ) = diffraction_scale_from_known_d_spacing(
            measured_radius_px,
            known_d,
            known_d_spacing_unit=known_d_unit,
            known_d_uncertainty_nm=known_d_uncertainty_nm,
            measured_radius_uncertainty_px=measured_radius_uncertainty_px,
            adapter_engine=adapter_engine,
        )
        warnings.extend(adapter_warnings)
        if scale_measurement is not None:
            confidence = 0.85 if label == "parameters" else 0.8
            if ref_measurement is not None:
                refs.append(ref_measurement)
            transform["calibration_source"] = f"{label}.known_d_spacing"
            transform["adapter_meta"] = adapter_meta
            break

        explicit_scale = container.get("diffraction_scale_inv_nm_per_px")
        if explicit_scale is not None:
            scale_value = _as_float(explicit_scale)
            if scale_value is None:
                warnings.append(
                    f"Ignoring invalid diffraction scale from {label}: "
                    "diffraction_scale_inv_nm_per_px."
                )
                continue
            if scale_value > 0:
                explicit_uncertainty = _as_float(
                    container.get("diffraction_scale_uncertainty_inv_nm_per_px")
                )
                scale_measurement = Measurement(
                    value=scale_value,
                    unit="1/nm/pixel",
                    method=f"{label}.explicit_diffraction_scale",
                    uncertainty=explicit_uncertainty,
                    confidence=0.8 if label == "parameters" else 0.75,
                )
                confidence = float(scale_measurement.confidence or confidence)
                transform["calibration_source"] = f"{label}.explicit_diffraction_scale"
                transform["adapter_meta"] = {"used_engine": "explicit"}
                break

    if scale_measurement is None:
        fft_size = None
        if isinstance(parameters, dict):
            fft_size_raw = parameters.get("fft_size_px")
            if isinstance(fft_size_raw, (int, float)) and fft_size_raw > 0:
                fft_size = int(fft_size_raw)
        scale_measurement = reciprocal_scale_from_pixel_size(
            pixel_size_nm,
            fft_size_px=fft_size,
            pixel_size_uncertainty_nm=pixel_size_uncertainty_nm,
        )
        transform["calibration_source"] = "pixel_size_default"
        transform["adapter_meta"] = {"used_engine": "internal"}
        confidence = min(confidence, 0.65 if fft_size is not None else 0.6)

    camera_length: Measurement | None = None
    detector_pixel_size_um = None
    for container in (parameters, reference):
        if not isinstance(container, dict):
            continue
        value = container.get("detector_pixel_size_um")
        if value is not None:
            detector_pixel_size_um = _as_float(value)
            if detector_pixel_size_um is None:
                warnings.append(
                    "Ignoring invalid detector_pixel_size_um while estimating camera length."
                )
            break
    if (
        detector_pixel_size_um is not None
        and voltage_kv is not None
        and scale_measurement.value is not None
    ):
        camera_length = estimate_camera_length_mm(
            float(scale_measurement.value),
            detector_pixel_size_um=detector_pixel_size_um,
            beam_energy_kv=float(voltage_kv),
        )
        if camera_length is None:
            warnings.append("Camera length estimate failed due to invalid calibration inputs.")

    return scale_measurement, camera_length, refs, transform, warnings, confidence
