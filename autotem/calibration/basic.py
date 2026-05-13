"""A2 baseline calibration utilities."""

from __future__ import annotations

from typing import Any

from autotem.core import CalibrationResult, Measurement, TEMDataset


def calibrate_dataset(
    dataset: TEMDataset,
    *,
    method: str = "auto",
    reference: dict[str, object] | None = None,
    parameters: dict[str, Any] | None = None,
) -> CalibrationResult:
    """Estimate pixel and diffraction calibration from metadata or defaults."""
    parameters = parameters or {}
    metadata = dataset.metadata
    warnings = list(dataset.warnings)

    pixel_nm = parameters.get("pixel_size_nm")
    if pixel_nm is None and isinstance(reference, dict):
        pixel_nm = reference.get("pixel_size_nm")
    if pixel_nm is None and isinstance(metadata, dict):
        pixel_nm = metadata.get("pixel_size_nm")
    if pixel_nm is None and dataset.pixel_size is not None and dataset.pixel_size.unit == "nm":
        pixel_nm = dataset.pixel_size.value
    if pixel_nm is None:
        pixel_nm = 0.1
        warnings.append("Pixel size was not provided; using 0.1 nm baseline.")
    else:
        pixel_nm = float(pixel_nm)

    voltage_kv = parameters.get("voltage_kv")
    if voltage_kv is None and isinstance(reference, dict):
        voltage_kv = reference.get("voltage_kv")
    if voltage_kv is None and isinstance(metadata, dict):
        voltage_kv = metadata.get("voltage_kv")
    if voltage_kv is None:
        voltage_kv = 200
    else:
        voltage_kv = float(voltage_kv)

    if method != "auto":
        warnings.append(
            f"Calibration method '{method}' requested; baseline solver still "
            "uses deterministic A2 path."
        )

    return CalibrationResult(
        pixel_size=Measurement(
            value=pixel_nm,
            unit="nm",
            method="metadata_or_default",
            confidence=(
                0.6
                if "Pixel size was not provided; using 0.1 nm baseline." not in warnings
                else 0.3
            ),
        ),
        pixel_unit="nm",
        diffraction_scale=Measurement(
            value=1.0 / max(pixel_nm, 1e-9),
            unit="1/nm/pixel",
            method="reciprocal_pixel_scale",
        ),
        voltage=Measurement(value=voltage_kv, unit="kV", method="metadata_or_default"),
        confidence=0.55,
        warnings=warnings,
        provenance={
            "method": "a2_baseline",
            "requested_method": method,
            "used_parameters": {
                "pixel_size_nm": pixel_nm,
                "voltage_kv": voltage_kv,
            },
        },
    )
