"""Metadata parsing and normalization for microscopy I/O."""

from __future__ import annotations

import re
from collections.abc import Mapping
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


def _mk_warning(code: str, message: str, field: str | None = None) -> dict[str, Any]:
    warning: dict[str, Any] = {"code": code, "message": message}
    if field is not None:
        warning["field"] = field
    return warning


def _normalize_unit_text(unit: str | None) -> str | None:
    if unit is None:
        return None
    unit = unit.strip().lower()
    unit = unit.replace("micrometer", "um")
    unit = unit.replace("micrometre", "um")
    unit = unit.replace("angstroms", "angstrom")
    return unit


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
        pass
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


def _extract_value(raw: Mapping[str, Any], keys: list[str]) -> Any:
    lowered = {str(k).lower(): v for k, v in raw.items()}
    for key in keys:
        if key in raw:
            return raw[key]
        value = lowered.get(key.lower())
        if value is not None:
            return value
    return None


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
                field="pixel_size",
            )
        )
        return None, warnings

    resolution_unit = _extract_value(raw, ["ResolutionUnit", "resolution_unit", "296"])
    unit_float = _as_float(resolution_unit)
    unit_numeric = int(unit_float) if unit_float is not None else None

    if unit_numeric == 2:
        return 25_400_000.0 / dpi, warnings  # inches -> nm
    if unit_numeric == 3:
        return 10_000_000.0 / dpi, warnings  # centimeters -> nm

    warnings.append(
        _mk_warning(
            "META_UNKNOWN_RESOLUTION_UNIT",
            "TIFF resolution unit is missing/unknown; cannot safely infer pixel size.",
            field="pixel_size",
        )
    )
    return None, warnings


def _normalize_detector_angles_mrad(raw: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    inner = _extract_value(raw, ["detector_inner_angle", "inner_angle", "collection_angle_inner"])
    outer = _extract_value(raw, ["detector_outer_angle", "outer_angle", "collection_angle_outer"])
    generic = _extract_value(raw, ["detector_angle", "detector_angles"])

    def convert(value: Any) -> float | None:
        number, unit = _parse_number_with_unit(value)
        unit = unit or "mrad"
        return _convert_unit(number, unit, _ANGLE_TO_MRAD)

    if inner is not None:
        converted = convert(inner)
        if converted is not None:
            out["inner_mrad"] = converted
    if outer is not None:
        converted = convert(outer)
        if converted is not None:
            out["outer_mrad"] = converted
    if not out and generic is not None:
        if isinstance(generic, (tuple, list)) and len(generic) == 2:
            inner_val = convert(generic[0])
            outer_val = convert(generic[1])
            if inner_val is not None:
                out["inner_mrad"] = inner_val
            if outer_val is not None:
                out["outer_mrad"] = outer_val
        else:
            converted = convert(generic)
            if converted is not None:
                out["inner_mrad"] = converted
    return out


def normalize_metadata(
    raw: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize partially structured metadata to a stable schema."""
    raw = raw or {}
    warnings: list[dict[str, Any]] = []

    normalized: dict[str, Any] = {
        "pixel_size_nm": None,
        "accelerating_voltage_kv": None,
        "camera_length_mm": None,
        "magnification": None,
        "detector_angles_mrad": {},
    }

    pixel_raw = _extract_value(
        raw,
        [
            "pixel_size_nm",
            "pixel_size",
            "pixelsize",
            "sampling",
            "scale_nm_per_pixel",
            "x_scale",
        ],
    )
    if pixel_raw is not None:
        pixel_number, pixel_unit = _parse_number_with_unit(pixel_raw)
        pixel_unit = pixel_unit or "nm"
        normalized["pixel_size_nm"] = _convert_unit(pixel_number, pixel_unit, _LENGTH_TO_NM)
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

    voltage_raw = _extract_value(
        raw,
        ["accelerating_voltage_kv", "accelerating_voltage", "voltage", "ht", "beam_voltage"],
    )
    if voltage_raw is not None:
        voltage_number, voltage_unit = _parse_number_with_unit(voltage_raw)
        voltage_unit = voltage_unit or "kv"
        normalized["accelerating_voltage_kv"] = _convert_unit(
            voltage_number,
            voltage_unit,
            _VOLT_TO_KV,
        )
    else:
        warnings.append(
            _mk_warning(
                "META_MISSING_VOLTAGE",
                "Accelerating voltage metadata is missing.",
                field="accelerating_voltage_kv",
            )
        )

    cam_raw = _extract_value(
        raw,
        ["camera_length_mm", "camera_length", "cam_length", "camera_len"],
    )
    if cam_raw is not None:
        camera_number, camera_unit = _parse_number_with_unit(cam_raw)
        camera_unit = camera_unit or "mm"
        normalized["camera_length_mm"] = _convert_unit(
            camera_number,
            camera_unit,
            _CAMERA_LENGTH_TO_MM,
        )
    else:
        warnings.append(
            _mk_warning(
                "META_MISSING_CAMERA_LENGTH",
                "Camera length metadata is missing.",
                field="camera_length_mm",
            )
        )

    magnification_raw = _extract_value(raw, ["magnification", "mag"])
    if magnification_raw is not None:
        normalized["magnification"] = _as_float(magnification_raw)
    else:
        warnings.append(
            _mk_warning(
                "META_MISSING_MAGNIFICATION",
                "Magnification metadata is missing.",
                field="magnification",
            )
        )

    normalized["detector_angles_mrad"] = _normalize_detector_angles_mrad(raw)
    if not normalized["detector_angles_mrad"]:
        warnings.append(
            _mk_warning(
                "META_MISSING_DETECTOR_ANGLES",
                "Detector angle metadata is missing.",
                field="detector_angles_mrad",
            )
        )

    return normalized, warnings

