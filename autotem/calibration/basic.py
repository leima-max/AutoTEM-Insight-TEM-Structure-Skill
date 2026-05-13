"""Calibration orchestration utilities."""

from __future__ import annotations

from typing import Any

from autotem.calibration.diffraction_calibration import resolve_diffraction_calibration
from autotem.calibration.pixel_calibration import (
    resolve_pixel_size_measurement,
    resolve_voltage_kv,
)
from autotem.core import CalibrationResult, TEMDataset


def calibrate_dataset(
    dataset: TEMDataset,
    *,
    method: str = "auto",
    reference: dict[str, object] | None = None,
    parameters: dict[str, Any] | None = None,
) -> CalibrationResult:
    """Estimate pixel and diffraction calibration from metadata or explicit inputs."""
    parameters = parameters or {}
    pixel_size, warnings, pixel_confidence = resolve_pixel_size_measurement(
        dataset,
        reference=reference,
        parameters=parameters,
    )
    voltage, voltage_confidence = resolve_voltage_kv(
        dataset,
        reference=reference,
        parameters=parameters,
    )

    if method != "auto":
        warnings.append(
            f"Calibration method '{method}' requested; baseline solver still "
            "uses deterministic B1 path."
        )

    (
        diffraction_scale,
        camera_length,
        reference_measurements,
        transform,
        diff_warnings,
        diff_conf,
    ) = resolve_diffraction_calibration(
        pixel_size_nm=float(pixel_size.value),
        pixel_size_uncertainty_nm=pixel_size.uncertainty,
        reference=reference,
        parameters=parameters,
        voltage_kv=float(voltage.value),
    )
    warnings.extend(diff_warnings)
    confidence = float(
        max(
            0.0,
            min(
                1.0,
                0.45 * pixel_confidence + 0.2 * voltage_confidence + 0.35 * diff_conf,
            ),
        )
    )

    return CalibrationResult(
        pixel_size=pixel_size,
        pixel_unit="nm",
        diffraction_scale=diffraction_scale,
        camera_length=camera_length,
        voltage=voltage,
        reference_measurements=reference_measurements,
        transform=transform,
        confidence=confidence,
        warnings=warnings,
        provenance={
            "method": "b1_calibration",
            "requested_method": method,
            "used_parameters": {
                "pixel_size_nm": pixel_size.value,
                "pixel_size_uncertainty_nm": pixel_size.uncertainty,
                "voltage_kv": voltage.value,
                "voltage_uncertainty_kv": voltage.uncertainty,
                "camera_length_mm": camera_length.value if camera_length is not None else None,
                "diffraction_scale_inv_nm_per_px": diffraction_scale.value,
                "diffraction_scale_uncertainty_inv_nm_per_px": diffraction_scale.uncertainty,
            },
        },
    )
