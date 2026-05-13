"""B1 pixel calibration utilities.

This module keeps calibration logic deterministic while supporting richer input
sources (manual values, metadata, and reference measurements).
"""

from __future__ import annotations

import re
from typing import Any

from autotem.core import Measurement, TEMDataset

_LENGTH_TO_NM: dict[str, float] = {
    "pm": 0.001,
    "nm": 1.0,
    "um": 1000.0,
    "mm": 1_000_000.0,
    "cm": 10_000_000.0,
    "m": 1_000_000_000.0,
    "a": 0.1,
    "ang": 0.1,
    "angstrom": 0.1,
}


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if match is None:
            return None
        return float(match.group(0))
    return None


def _normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    cleaned = unit.strip().lower()
    cleaned = cleaned.replace("ångström", "angstrom")
    cleaned = cleaned.replace("angstroms", "angstrom")
    cleaned = cleaned.replace("micrometer", "um")
    cleaned = cleaned.replace("micrometre", "um")
    if cleaned.endswith("/pixel"):
        cleaned = cleaned[: -len("/pixel")]
    if cleaned.endswith("/px"):
        cleaned = cleaned[: -len("/px")]
    return cleaned


def _parse_number_and_unit(value: Any) -> tuple[float | None, str | None]:
    if isinstance(value, str):
        match = re.search(
            r"(?P<num>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?P<unit>[A-Za-zµÅ]*)",
            value.strip(),
        )
        if match is None:
            return None, None
        number = float(match.group("num"))
        unit = _normalize_unit(match.group("unit") or None)
        return number, unit
    return _as_float(value), None


def convert_length(value: float, from_unit: str, to_unit: str = "nm") -> float:
    """Convert a scalar length value between supported microscopy units."""
    from_key = _normalize_unit(from_unit)
    to_key = _normalize_unit(to_unit)
    if from_key is None or to_key is None:
        msg = "Length unit cannot be empty."
        raise ValueError(msg)
    if from_key not in _LENGTH_TO_NM or to_key not in _LENGTH_TO_NM:
        msg = f"Unsupported length unit conversion: '{from_unit}' -> '{to_unit}'."
        raise ValueError(msg)
    value_nm = float(value) * _LENGTH_TO_NM[from_key]
    return value_nm / _LENGTH_TO_NM[to_key]


def coerce_length_to_nm(value: Any, unit: str | None = None) -> float | None:
    """Parse and convert heterogeneous scalar inputs to nanometers."""
    parsed_value, parsed_unit = _parse_number_and_unit(value)
    if parsed_value is None:
        return None
    resolved_unit = _normalize_unit(unit) or parsed_unit or "nm"
    if resolved_unit not in _LENGTH_TO_NM:
        return None
    return parsed_value * _LENGTH_TO_NM[resolved_unit]


def manual_pixel_size_from_reference(
    known_length_value: Any,
    *,
    known_length_unit: str | None = "nm",
    measured_distance_px: Any,
) -> float | None:
    """Compute pixel size in nm/px from a known real-space distance."""
    distance_px = _as_float(measured_distance_px)
    if distance_px is None or distance_px <= 0:
        return None
    known_nm = coerce_length_to_nm(known_length_value, known_length_unit)
    if known_nm is None or known_nm <= 0:
        return None
    return known_nm / distance_px


def _metadata_views(metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(metadata, dict):
        return []
    views: list[dict[str, Any]] = [metadata]
    normalized = metadata.get("normalized")
    if isinstance(normalized, dict):
        views.append(normalized)
    raw = metadata.get("raw")
    if isinstance(raw, dict):
        views.append(raw)
    return views


def _lookup_first(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for key in keys:
        if key in mapping:
            return mapping[key]
        lower_key = key.lower()
        if lower_key in lowered:
            return lowered[lower_key]
    return None


def _pixel_size_from_metadata(metadata: dict[str, Any] | None) -> tuple[float | None, str]:
    for view in _metadata_views(metadata):
        direct_nm = _lookup_first(view, ("pixel_size_nm", "pixel_nm"))
        value_nm = _as_float(direct_nm)
        if value_nm is not None and value_nm > 0:
            return value_nm, "metadata.pixel_size_nm"

        mixed = _lookup_first(view, ("pixel_size", "pixelsize", "sampling"))
        if mixed is not None:
            mixed_nm = coerce_length_to_nm(mixed)
            if mixed_nm is not None and mixed_nm > 0:
                return mixed_nm, "metadata.pixel_size"
    return None, "metadata.unavailable"


def _pixel_size_from_measurement(dataset: TEMDataset) -> tuple[float | None, str]:
    if dataset.pixel_size is None:
        return None, "dataset.pixel_size.unavailable"
    value_nm = coerce_length_to_nm(dataset.pixel_size.value, dataset.pixel_size.unit)
    if value_nm is None or value_nm <= 0:
        return None, "dataset.pixel_size.invalid"
    return value_nm, "dataset.pixel_size"


def _pixel_size_from_container(
    container: dict[str, Any] | None,
    *,
    label: str,
) -> tuple[float | None, str]:
    if not isinstance(container, dict):
        return None, f"{label}.unavailable"

    pixel_nm = _as_float(container.get("pixel_size_nm"))
    if pixel_nm is not None and pixel_nm > 0:
        return pixel_nm, f"{label}.pixel_size_nm"

    if "pixel_size" in container:
        mixed_nm = coerce_length_to_nm(
            container.get("pixel_size"),
            container.get("pixel_unit") or container.get("pixel_size_unit"),
        )
        if mixed_nm is not None and mixed_nm > 0:
            return mixed_nm, f"{label}.pixel_size"

    manual_nm = manual_pixel_size_from_reference(
        container.get("known_length")
        if "known_length" in container
        else container.get("known_distance"),
        known_length_unit=container.get("known_length_unit")
        or container.get("known_distance_unit")
        or container.get("d_spacing_unit"),
        measured_distance_px=container.get("measured_distance_px")
        if "measured_distance_px" in container
        else container.get("measured_pixels"),
    )
    if manual_nm is not None and manual_nm > 0:
        return manual_nm, f"{label}.manual_reference"
    return None, f"{label}.unavailable"


def _pixel_uncertainty_nm_from_container(
    container: dict[str, Any] | None,
) -> float | None:
    if not isinstance(container, dict):
        return None
    direct = _as_float(container.get("pixel_size_uncertainty_nm"))
    if direct is not None and direct >= 0:
        return direct
    mixed = container.get("pixel_size_uncertainty")
    if mixed is not None:
        value = coerce_length_to_nm(
            mixed,
            container.get("pixel_size_uncertainty_unit"),
        )
        if value is not None and value >= 0:
            return value
    return None


def resolve_pixel_size_measurement(
    dataset: TEMDataset,
    *,
    reference: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
) -> tuple[Measurement, list[str], float]:
    """Resolve pixel size measurement with deterministic priority order."""
    warnings = list(dataset.warnings)

    parameter_nm, parameter_source = _pixel_size_from_container(parameters, label="parameters")
    if parameter_nm is not None:
        uncertainty_nm = _pixel_uncertainty_nm_from_container(parameters)
        if uncertainty_nm is None:
            uncertainty_nm = 0.02 * parameter_nm
        return (
            Measurement(
                value=parameter_nm,
                unit="nm",
                method=parameter_source,
                uncertainty=uncertainty_nm,
                confidence=0.9,
            ),
            warnings,
            0.9,
        )

    reference_nm, reference_source = _pixel_size_from_container(reference, label="reference")
    if reference_nm is not None:
        uncertainty_nm = _pixel_uncertainty_nm_from_container(reference)
        if uncertainty_nm is None:
            uncertainty_nm = 0.03 * reference_nm
        return (
            Measurement(
                value=reference_nm,
                unit="nm",
                method=reference_source,
                uncertainty=uncertainty_nm,
                confidence=0.8,
            ),
            warnings,
            0.8,
        )

    metadata_nm, metadata_source = _pixel_size_from_metadata(dataset.metadata)
    if metadata_nm is not None:
        return (
            Measurement(
                value=metadata_nm,
                unit="nm",
                method=metadata_source,
                uncertainty=0.05 * metadata_nm,
                confidence=0.75,
            ),
            warnings,
            0.75,
        )

    dataset_nm, dataset_source = _pixel_size_from_measurement(dataset)
    if dataset_nm is not None:
        return (
            Measurement(
                value=dataset_nm,
                unit="nm",
                method=dataset_source,
                uncertainty=0.08 * dataset_nm,
                confidence=0.65,
            ),
            warnings,
            0.65,
        )

    warnings.append("Pixel size was not provided; using 0.1 nm baseline.")
    return (
        Measurement(
            value=0.1,
            unit="nm",
            method="fallback.default",
            uncertainty=0.02,
            confidence=0.3,
        ),
        warnings,
        0.3,
    )


def resolve_voltage_kv(
    dataset: TEMDataset,
    *,
    reference: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
) -> tuple[Measurement, float]:
    """Resolve accelerating voltage in kV from parameters/reference/metadata."""

    def _candidate(container: dict[str, Any] | None) -> float | None:
        if not isinstance(container, dict):
            return None
        for view in _metadata_views(container):
            for key in ("voltage_kv", "accelerating_voltage_kv", "beam_energy_kv", "voltage"):
                value = _as_float(view.get(key))
                if value is not None and value > 0:
                    return value
        return None

    def _candidate_uncertainty(container: dict[str, Any] | None) -> float | None:
        if not isinstance(container, dict):
            return None
        for view in _metadata_views(container):
            value = _as_float(
                view.get("voltage_uncertainty_kv")
                or view.get("accelerating_voltage_uncertainty_kv")
            )
            if value is not None and value >= 0:
                return value
        return None

    for source_name, container, confidence in (
        ("parameters", parameters, 0.9),
        ("reference", reference, 0.8),
        ("metadata", dataset.metadata, 0.7),
    ):
        value = _candidate(container)
        if value is not None:
            uncertainty = _candidate_uncertainty(container)
            if uncertainty is None:
                uncertainty = max(0.5, 0.01 * value)
            return (
                Measurement(
                    value=value,
                    unit="kV",
                    method=f"{source_name}.voltage",
                    uncertainty=uncertainty,
                    confidence=confidence,
                ),
                confidence,
            )

    return (
        Measurement(
            value=200.0,
            unit="kV",
            method="fallback.default",
            uncertainty=2.0,
            confidence=0.4,
        ),
        0.4,
    )
