"""Metadata parsing and normalization for microscopy I/O."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_ANGLE_TO_MRAD = {
    "mrad": 1.0,
    "rad": 1000.0,
    "deg": 17.453292519943295,
}

_LENGTH_TO_NM = {
    "nm": 1.0,
    "um": 1000.0,
    "mm": 1_000_000.0,
    "cm": 10_000_000.0,
    "m": 1_000_000_000.0,
    "pm": 0.001,
    "a": 0.1,
    "ang": 0.1,
    "angstrom": 0.1,
}

_VOLT_TO_KV = {
    "kv": 1.0,
    "v": 0.001,
}

_CAMERA_LENGTH_TO_MM = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "um": 0.001,
}

_FORMAT_ALIASES = {
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "tif": "image",
    "tiff": "image",
    "dm": "dm",
    "dm3": "dm",
    "dm4": "dm",
    "mrc": "mrc",
    "h5": "emd",
    "hdf5": "emd",
    "emd": "emd",
    "ser": "ser",
    "emi": "ser",
}

# Metadata mapping table for known format families.
# Each field contains prioritized key candidates and optional default units.
FORMAT_METADATA_MAPPINGS: dict[str, dict[str, list[dict[str, Any]]]] = {
    "default": {
        "pixel_size": [
            {"keys": ["pixel_size_nm"], "unit": "nm", "priority": 300},
            {"keys": ["pixel_size", "pixelsize"], "unit": None, "priority": 200},
            {"keys": ["sampling", "scale_nm_per_pixel"], "unit": "nm", "priority": 100},
        ],
        "voltage": [
            {"keys": ["accelerating_voltage_kv"], "unit": "kv", "priority": 300},
            {
                "keys": ["accelerating_voltage", "voltage", "beam_voltage"],
                "unit": None,
                "priority": 200,
            },
            {"keys": ["ht"], "unit": "kv", "priority": 100},
        ],
        "camera_length": [
            {"keys": ["camera_length_mm"], "unit": "mm", "priority": 300},
            {"keys": ["camera_length", "cam_length", "camera_len"], "unit": None, "priority": 200},
        ],
        "magnification": [
            {"keys": ["magnification", "mag"], "unit": None, "priority": 300},
        ],
        "detector_angle_inner": [
            {"keys": ["detector_inner_angle", "inner_angle"], "unit": None, "priority": 300},
            {"keys": ["collection_angle_inner"], "unit": "mrad", "priority": 200},
        ],
        "detector_angle_outer": [
            {"keys": ["detector_outer_angle", "outer_angle"], "unit": None, "priority": 300},
            {"keys": ["collection_angle_outer"], "unit": "mrad", "priority": 200},
        ],
        "detector_angle_generic": [
            {"keys": ["detector_angle", "detector_angles"], "unit": None, "priority": 200},
        ],
    },
    "dm": {
        "pixel_size": [
            {"keys": ["pixelSize.0"], "unit": "nm", "priority": 500},
            {"keys": ["pixelSize"], "unit": "nm", "priority": 400},
            {"keys": ["scale"], "unit": "nm", "priority": 300},
        ],
        "voltage": [
            {"keys": ["acqMicroscopeInfo.Voltage"], "unit": "kv", "priority": 500},
            {"keys": ["Voltage"], "unit": "kv", "priority": 300},
        ],
        "camera_length": [
            {"keys": ["acqMicroscopeInfo.CameraLength"], "unit": "mm", "priority": 500},
            {"keys": ["CameraLength"], "unit": "mm", "priority": 300},
        ],
        "magnification": [
            {"keys": ["acqMicroscopeInfo.IndicatedMagnification"], "unit": None, "priority": 500},
            {"keys": ["Magnification"], "unit": None, "priority": 300},
        ],
    },
    "mrc": {
        "pixel_size": [
            {"keys": ["pixel_size_x_angstrom"], "unit": "angstrom", "priority": 500},
            {"keys": ["voxel_size.x"], "unit": "angstrom", "priority": 400},
        ],
    },
    "emd": {
        "pixel_size": [
            {"keys": ["pixel_size"], "unit": None, "priority": 500},
            {"keys": ["sampling"], "unit": None, "priority": 300},
        ],
        "voltage": [
            {"keys": ["accelerating_voltage"], "unit": None, "priority": 500},
            {"keys": ["microscope.voltage"], "unit": None, "priority": 300},
        ],
        "camera_length": [
            {"keys": ["camera_length"], "unit": None, "priority": 500},
        ],
    },
    "ser": {
        "pixel_size": [
            {"keys": ["pixelSizeX"], "unit": "nm", "priority": 500},
            {"keys": ["pixelSize"], "unit": "nm", "priority": 400},
        ],
        "voltage": [
            {"keys": ["acceleratingVoltage"], "unit": "kv", "priority": 500},
            {"keys": ["highTension"], "unit": "kv", "priority": 400},
        ],
        "camera_length": [
            {"keys": ["cameraLength"], "unit": "mm", "priority": 500},
        ],
    },
}

WARNING_SEVERITY_BY_CODE: dict[str, str] = {
    "META_MISSING_PIXEL_SIZE": "warning",
    "META_INVALID_PIXEL_SIZE": "warning",
    "META_INVALID_XRES": "warning",
    "META_UNKNOWN_RESOLUTION_UNIT": "warning",
    "META_MISSING_VOLTAGE": "warning",
    "META_INVALID_VOLTAGE": "warning",
    "META_MISSING_CAMERA_LENGTH": "warning",
    "META_INVALID_CAMERA_LENGTH": "warning",
    "META_MISSING_MAGNIFICATION": "warning",
    "META_INVALID_MAGNIFICATION": "warning",
    "META_MISSING_DETECTOR_ANGLES": "warning",
    "META_INVALID_DETECTOR_ANGLES": "warning",
}

METADATA_WARNING_CATALOG: dict[str, str] = {
    "META_MISSING_PIXEL_SIZE": "Pixel size metadata is not found in the current source metadata.",
    "META_INVALID_PIXEL_SIZE": "Pixel size metadata exists but cannot be parsed or converted.",
    "META_INVALID_XRES": (
        "TIFF XResolution exists but cannot be interpreted as a positive numeric value."
    ),
    "META_UNKNOWN_RESOLUTION_UNIT": (
        "TIFF resolution unit is missing/unknown so pixel size is ambiguous."
    ),
    "META_MISSING_VOLTAGE": "Accelerating voltage metadata is not found.",
    "META_INVALID_VOLTAGE": (
        "Accelerating voltage metadata exists but cannot be parsed or converted."
    ),
    "META_MISSING_CAMERA_LENGTH": "Camera length metadata is not found.",
    "META_INVALID_CAMERA_LENGTH": (
        "Camera length metadata exists but cannot be parsed or converted."
    ),
    "META_MISSING_MAGNIFICATION": "Magnification metadata is not found.",
    "META_INVALID_MAGNIFICATION": (
        "Magnification metadata exists but cannot be parsed to numeric value."
    ),
    "META_MISSING_DETECTOR_ANGLES": "Detector angle metadata is not found.",
    "META_INVALID_DETECTOR_ANGLES": (
        "Detector angle metadata exists but cannot be parsed or converted."
    ),
}


def _mk_warning(
    code: str,
    message: str,
    field: str | None = None,
    severity: str | None = None,
) -> dict[str, Any]:
    warning: dict[str, Any] = {
        "code": code,
        "message": message,
        "severity": severity or WARNING_SEVERITY_BY_CODE.get(code, "warning"),
        "source": "metadata",
    }
    if field is not None:
        warning["field"] = field
    return warning


def _resolve_source_format(source_format: str | None) -> str:
    if source_format is None:
        return "default"
    normalized = source_format.strip().lower()
    normalized = normalized.lstrip(".")
    return _FORMAT_ALIASES.get(normalized, normalized)


def _normalize_unit_text(unit: str | None) -> str | None:
    if unit is None:
        return None
    normalized = unit.strip().lower()
    normalized = normalized.replace("micrometer", "um")
    normalized = normalized.replace("micrometre", "um")
    normalized = normalized.replace("angstroms", "angstrom")
    return normalized


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        numerator = _as_float(value[0])
        denominator = _as_float(value[1])
        if numerator is None or denominator in (None, 0):
            return None
        return numerator / denominator
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if match:
            return float(match.group(0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_number_with_unit(value: Any) -> tuple[float | None, str | None]:
    if isinstance(value, str):
        match = re.search(
            r"(?P<num>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?P<unit>[A-Za-z]*)",
            value.strip(),
        )
        if match:
            number = float(match.group("num"))
            unit = _normalize_unit_text(match.group("unit") or None)
            return number, unit
        return None, None
    return _as_float(value), None


def _convert_unit(
    number: float | None,
    unit: str | None,
    table: Mapping[str, float],
) -> float | None:
    if number is None:
        return None
    if unit is None or unit == "":
        return number
    factor = table.get(unit)
    if factor is None:
        return None
    return number * factor


def _extract_by_path(raw: Mapping[str, Any], path: str) -> Any:
    if path in raw:
        return raw[path]

    current: Any = raw
    for token in path.split("."):
        if isinstance(current, Mapping):
            lower_map = {str(k).lower(): v for k, v in current.items()}
            if token in current:
                current = current[token]
            elif token.lower() in lower_map:
                current = lower_map[token.lower()]
            else:
                return None
            continue
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            if not token.isdigit():
                return None
            index = int(token)
            if index < 0 or index >= len(current):
                return None
            current = current[index]
            continue
        return None
    return current


def _extract_value(raw: Mapping[str, Any], keys: list[str]) -> Any:
    lowered = {str(k).lower(): v for k, v in raw.items()}
    for key in keys:
        if "." in key:
            value = _extract_by_path(raw, key)
            if value is not None:
                return value
            continue
        if key in raw:
            return raw[key]
        value = lowered.get(key.lower())
        if value is not None:
            return value
    return None


def _mapping_candidates(source_format: str, field_name: str) -> list[dict[str, Any]]:
    candidates = list(FORMAT_METADATA_MAPPINGS.get(source_format, {}).get(field_name, []))
    candidates.extend(FORMAT_METADATA_MAPPINGS["default"].get(field_name, []))
    return sorted(candidates, key=lambda item: int(item.get("priority", 0)), reverse=True)


def _find_mapped_value(
    raw: Mapping[str, Any],
    source_format: str,
    field_name: str,
) -> tuple[Any, str | None, str | None]:
    for spec in _mapping_candidates(source_format, field_name):
        keys = [str(k) for k in spec.get("keys", [])]
        value = _extract_value(raw, keys)
        if value is not None:
            return value, spec.get("unit"), keys[0] if keys else None
    return None, None, None


def _pixel_size_from_tiff_tags(
    raw: Mapping[str, Any],
) -> tuple[float | None, list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    xres = _extract_value(raw, ["XResolution", "x_resolution", "282", "dpi"])
    if xres is None:
        return None, warnings

    dpi = _as_float(xres)
    if dpi is None or dpi <= 0:
        warnings.append(
            _mk_warning(
                "META_INVALID_XRES",
                "TIFF XResolution exists but is invalid; cannot infer pixel size.",
                field="pixel_size_nm",
            )
        )
        return None, warnings

    resolution_unit = _extract_value(raw, ["ResolutionUnit", "resolution_unit", "296"])
    unit_float = _as_float(resolution_unit)
    unit_numeric = int(unit_float) if unit_float is not None else None

    if unit_numeric == 2:
        return 25_400_000.0 / dpi, warnings
    if unit_numeric == 3:
        return 10_000_000.0 / dpi, warnings

    warnings.append(
        _mk_warning(
            "META_UNKNOWN_RESOLUTION_UNIT",
            "TIFF resolution unit is missing/unknown; cannot safely infer pixel size.",
            field="pixel_size_nm",
        )
    )
    return None, warnings


def _normalize_detector_angles_mrad(
    raw: Mapping[str, Any],
    source_format: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    out: dict[str, float] = {}

    inner, inner_unit, _ = _find_mapped_value(raw, source_format, "detector_angle_inner")
    outer, outer_unit, _ = _find_mapped_value(raw, source_format, "detector_angle_outer")
    generic, generic_unit, _ = _find_mapped_value(raw, source_format, "detector_angle_generic")

    def convert(value: Any, default_unit: str | None = None) -> float | None:
        number, parsed_unit = _parse_number_with_unit(value)
        unit = parsed_unit or default_unit or "mrad"
        return _convert_unit(number, unit, _ANGLE_TO_MRAD)

    if inner is not None:
        converted = convert(inner, inner_unit)
        if converted is not None:
            out["inner_mrad"] = converted
        else:
            warnings.append(
                _mk_warning(
                    "META_INVALID_DETECTOR_ANGLES",
                    "Detector inner angle metadata cannot be parsed.",
                    field="detector_angles_mrad",
                )
            )
    if outer is not None:
        converted = convert(outer, outer_unit)
        if converted is not None:
            out["outer_mrad"] = converted
        else:
            warnings.append(
                _mk_warning(
                    "META_INVALID_DETECTOR_ANGLES",
                    "Detector outer angle metadata cannot be parsed.",
                    field="detector_angles_mrad",
                )
            )
    if not out and generic is not None:
        if isinstance(generic, (tuple, list)) and len(generic) == 2:
            inner_val = convert(generic[0], generic_unit)
            outer_val = convert(generic[1], generic_unit)
            if inner_val is not None:
                out["inner_mrad"] = inner_val
            if outer_val is not None:
                out["outer_mrad"] = outer_val
        else:
            converted = convert(generic, generic_unit)
            if converted is not None:
                out["inner_mrad"] = converted
        if not out:
            warnings.append(
                _mk_warning(
                    "META_INVALID_DETECTOR_ANGLES",
                    "Detector angle metadata cannot be parsed.",
                    field="detector_angles_mrad",
                )
            )
    return out, warnings


def _compute_metadata_scores(
    normalized: Mapping[str, Any],
    warnings: list[dict[str, Any]],
) -> tuple[float, float]:
    core_fields = [
        normalized.get("pixel_size_nm") is not None,
        normalized.get("accelerating_voltage_kv") is not None,
        normalized.get("camera_length_mm") is not None,
        normalized.get("magnification") is not None,
        bool(normalized.get("detector_angles_mrad")),
    ]
    completeness = sum(1 for present in core_fields if present) / float(len(core_fields))

    invalid_codes = {
        "META_INVALID_PIXEL_SIZE",
        "META_INVALID_XRES",
        "META_UNKNOWN_RESOLUTION_UNIT",
        "META_INVALID_VOLTAGE",
        "META_INVALID_CAMERA_LENGTH",
        "META_INVALID_MAGNIFICATION",
        "META_INVALID_DETECTOR_ANGLES",
    }
    invalid_count = sum(1 for warning in warnings if warning.get("code") in invalid_codes)

    confidence = completeness - (0.1 * invalid_count)
    if (
        normalized.get("pixel_size_nm") is not None
        and normalized.get("accelerating_voltage_kv") is not None
    ):
        confidence += 0.05
    confidence = max(0.0, min(1.0, confidence))

    return round(completeness, 4), round(confidence, 4)


def normalize_metadata(
    raw: Mapping[str, Any] | None,
    source_format: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize partially structured metadata to a stable schema."""
    raw = raw or {}
    resolved_format = _resolve_source_format(
        source_format or _extract_value(raw, ["source_format", "format"])
    )
    warnings: list[dict[str, Any]] = []

    normalized: dict[str, Any] = {
        "pixel_size_nm": None,
        "accelerating_voltage_kv": None,
        "camera_length_mm": None,
        "magnification": None,
        "detector_angles_mrad": {},
        "source_format": resolved_format,
    }

    pixel_raw, pixel_default_unit, _ = _find_mapped_value(raw, resolved_format, "pixel_size")
    if pixel_raw is not None:
        pixel_number, pixel_unit = _parse_number_with_unit(pixel_raw)
        pixel_unit = pixel_unit or pixel_default_unit or "nm"
        normalized["pixel_size_nm"] = _convert_unit(pixel_number, pixel_unit, _LENGTH_TO_NM)
        if normalized["pixel_size_nm"] is None:
            warnings.append(
                _mk_warning(
                    "META_INVALID_PIXEL_SIZE",
                    "Pixel size metadata exists but cannot be parsed or converted.",
                    field="pixel_size_nm",
                )
            )
    else:
        pixel_size_from_tiff, tiff_warnings = _pixel_size_from_tiff_tags(raw)
        normalized["pixel_size_nm"] = pixel_size_from_tiff
        warnings.extend(tiff_warnings)

    if normalized["pixel_size_nm"] is None:
        warnings.append(
            _mk_warning(
                "META_MISSING_PIXEL_SIZE",
                "Pixel size metadata is missing.",
                field="pixel_size_nm",
            )
        )

    voltage_raw, voltage_default_unit, _ = _find_mapped_value(raw, resolved_format, "voltage")
    if voltage_raw is not None:
        voltage_number, voltage_unit = _parse_number_with_unit(voltage_raw)
        voltage_unit = voltage_unit or voltage_default_unit or "kv"
        normalized["accelerating_voltage_kv"] = _convert_unit(
            voltage_number,
            voltage_unit,
            _VOLT_TO_KV,
        )
        if normalized["accelerating_voltage_kv"] is None:
            warnings.append(
                _mk_warning(
                    "META_INVALID_VOLTAGE",
                    "Accelerating voltage metadata exists but cannot be parsed or converted.",
                    field="accelerating_voltage_kv",
                )
            )
    else:
        warnings.append(
            _mk_warning(
                "META_MISSING_VOLTAGE",
                "Accelerating voltage metadata is missing.",
                field="accelerating_voltage_kv",
            )
        )

    cam_raw, cam_default_unit, _ = _find_mapped_value(raw, resolved_format, "camera_length")
    if cam_raw is not None:
        camera_number, camera_unit = _parse_number_with_unit(cam_raw)
        camera_unit = camera_unit or cam_default_unit or "mm"
        normalized["camera_length_mm"] = _convert_unit(
            camera_number,
            camera_unit,
            _CAMERA_LENGTH_TO_MM,
        )
        if normalized["camera_length_mm"] is None:
            warnings.append(
                _mk_warning(
                    "META_INVALID_CAMERA_LENGTH",
                    "Camera length metadata exists but cannot be parsed or converted.",
                    field="camera_length_mm",
                )
            )
    else:
        warnings.append(
            _mk_warning(
                "META_MISSING_CAMERA_LENGTH",
                "Camera length metadata is missing.",
                field="camera_length_mm",
            )
        )

    mag_raw, _, _ = _find_mapped_value(raw, resolved_format, "magnification")
    if mag_raw is not None:
        normalized["magnification"] = _as_float(mag_raw)
        if normalized["magnification"] is None:
            warnings.append(
                _mk_warning(
                    "META_INVALID_MAGNIFICATION",
                    "Magnification metadata exists but cannot be parsed to numeric value.",
                    field="magnification",
                )
            )
    else:
        warnings.append(
            _mk_warning(
                "META_MISSING_MAGNIFICATION",
                "Magnification metadata is missing.",
                field="magnification",
            )
        )

    angles, angle_warnings = _normalize_detector_angles_mrad(raw, resolved_format)
    normalized["detector_angles_mrad"] = angles
    warnings.extend(angle_warnings)
    if not normalized["detector_angles_mrad"]:
        warnings.append(
            _mk_warning(
                "META_MISSING_DETECTOR_ANGLES",
                "Detector angle metadata is missing.",
                field="detector_angles_mrad",
            )
        )

    completeness, confidence = _compute_metadata_scores(normalized, warnings)
    normalized["metadata_completeness"] = completeness
    normalized["metadata_confidence"] = confidence

    return normalized, warnings
