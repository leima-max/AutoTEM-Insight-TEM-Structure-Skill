"""Pixel calibration helpers for B1 calibration and quality workflows."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from autotem.core import Measurement, TEMDataset

OCRRow = tuple[str, float, float, float]
OCRReader = Callable[[np.ndarray], list[OCRRow]]

_LENGTH_TO_NM: dict[str, float] = {
    "nm": 1.0,
    "nanometer": 1.0,
    "nanometers": 1.0,
    "pm": 1e-3,
    "picometer": 1e-3,
    "picometers": 1e-3,
    "um": 1e3,
    "µm": 1e3,
    "μm": 1e3,
    "micrometer": 1e3,
    "micrometers": 1e3,
    "angstrom": 0.1,
    "angstroms": 0.1,
    "a": 0.1,
    "å": 0.1,
    "m": 1e9,
    "meter": 1e9,
}
_VOLT_TO_KV: dict[str, float] = {
    "kv": 1.0,
    "kilovolt": 1.0,
    "kilovolts": 1.0,
    "v": 1e-3,
    "volt": 1e-3,
    "volts": 1e-3,
}
_SCALEBAR_RE = re.compile(
    r"(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?P<unit>pm|nm|um|µm|μm|angstrom|a|å)\b",
    re.IGNORECASE,
)
_VOLT_RE = re.compile(
    r"(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?P<unit>k?V|kilovolts?|volts?)\b",
    re.IGNORECASE,
)
_EXTERNAL_OCR_READER: OCRReader | None = None


def _normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    normalized = unit.strip().lower()
    normalized = normalized.replace("μ", "µ")
    if normalized == "a":
        return "angstrom"
    if normalized == "å":
        return "angstrom"
    return normalized


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value.strip())
        if match:
            return float(match.group(0))
    return None


def _parse_length(value: Any, fallback_unit: str | None = None) -> tuple[float | None, str | None]:
    if isinstance(value, str):
        match = _SCALEBAR_RE.search(value.strip())
        if match:
            return float(match.group("value")), _normalize_unit(match.group("unit"))
        number = _to_float(value)
        return number, _normalize_unit(fallback_unit)
    return _to_float(value), _normalize_unit(fallback_unit)


def convert_length(value: float | int | str, from_unit: str, to_unit: str = "nm") -> float:
    """Convert one length value between common microscopy units."""
    number = _to_float(value)
    if number is None:
        raise ValueError("Length value must be numeric.")
    src = _normalize_unit(from_unit)
    dst = _normalize_unit(to_unit)
    if src not in _LENGTH_TO_NM:
        raise ValueError(f"Unsupported length unit: {from_unit}")
    if dst not in _LENGTH_TO_NM:
        raise ValueError(f"Unsupported length unit: {to_unit}")
    value_nm = number * _LENGTH_TO_NM[src]
    return value_nm / _LENGTH_TO_NM[dst]


def coerce_length_to_nm(value: Any, unit: str | None = None) -> float | None:
    """Parse value + unit and convert to nanometers when possible."""
    parsed_value, parsed_unit = _parse_length(value, unit)
    if parsed_value is None:
        return None
    if parsed_unit is None:
        return float(parsed_value)
    if parsed_unit not in _LENGTH_TO_NM:
        return None
    return float(parsed_value * _LENGTH_TO_NM[parsed_unit])


def manual_pixel_size_from_reference(
    *,
    reference_length: float | int | str,
    reference_pixels: float | int,
    reference_unit: str = "nm",
) -> float:
    """Compute pixel size from a known real-space reference length."""
    length_nm = coerce_length_to_nm(reference_length, reference_unit)
    pixels = _to_float(reference_pixels)
    if length_nm is None or length_nm <= 0:
        raise ValueError("reference_length must be a positive length.")
    if pixels is None or pixels <= 0:
        raise ValueError("reference_pixels must be positive.")
    return float(length_nm / pixels)


def _iter_metadata_strings(metadata: dict[str, Any] | None) -> list[tuple[str, str]]:
    if not metadata:
        return []
    rows: list[tuple[str, str]] = []
    stack: list[tuple[str, Any]] = [("metadata", metadata)]
    while stack:
        key, value = stack.pop()
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                stack.append((f"{key}.{child_key}", child_value))
            continue
        if isinstance(value, (list, tuple)):
            for idx, child in enumerate(value):
                stack.append((f"{key}[{idx}]", child))
            continue
        if isinstance(value, str):
            rows.append((key, value))
    return rows


def _extract_scalebar_length_nm_from_text(
    text: str,
    *,
    source_hint: str,
) -> tuple[float | None, str | None]:
    match = _SCALEBAR_RE.search(text)
    if not match:
        return None, None
    length_nm = coerce_length_to_nm(match.group("value"), match.group("unit"))
    if length_nm is None or length_nm <= 0:
        return None, None
    return float(length_nm), source_hint


def _parse_scalebar_text_source(
    metadata: dict[str, Any] | None,
    file_hint: str | None,
    image: np.ndarray,
) -> tuple[float | None, str | None, float | None, float | None]:
    for key, text in _iter_metadata_strings(metadata):
        length_nm, source = _extract_scalebar_length_nm_from_text(
            text,
            source_hint=f"scalebar.text (metadata:{key})",
        )
        if length_nm is not None:
            return length_nm, source, None, None

    if file_hint:
        length_nm, source = _extract_scalebar_length_nm_from_text(
            file_hint,
            source_hint="scalebar.text (file_hint)",
        )
        if length_nm is not None:
            return length_nm, source, None, None

    best_candidate: tuple[float, str, float, float, float] | None = None
    h = max(1.0, float(image.shape[0]))
    for text, confidence, x_center, y_center in _ocr_read_text_rows(image):
        length_nm, _ = _extract_scalebar_length_nm_from_text(
            text,
            source_hint="ocr",
        )
        if length_nm is None:
            continue
        confidence_clip = max(0.0, min(1.0, float(confidence)))
        y_norm = max(0.0, min(1.0, float(y_center) / h))
        score = 0.7 * confidence_clip + 0.3 * y_norm
        if best_candidate is None or score > best_candidate[0]:
            best_candidate = (
                score,
                f"scalebar.text (ocr:{confidence_clip:.3f})",
                float(length_nm),
                float(x_center),
                float(y_center),
            )
    if best_candidate is not None:
        _, source, length_nm, x_center, y_center = best_candidate
        return length_nm, source, x_center, y_center

    return None, None, None, None


def _find_longest_segment(mask_1d: np.ndarray) -> tuple[int, int] | None:
    if mask_1d.size == 0:
        return None
    # Use signed integer differences so falling edges remain -1 after np.diff.
    binary = mask_1d.astype(np.int16)
    padded = np.pad(binary, (1, 1), mode="constant")
    edges = np.diff(padded)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    if len(starts) == 0 or len(ends) == 0:
        return None
    best_start = 0
    best_end = 0
    best_len = -1
    for start, end in zip(starts, ends, strict=False):
        seg_len = int(end - start)
        if seg_len > best_len:
            best_len = seg_len
            best_start = int(start)
            best_end = int(end)
    if best_len <= 0:
        return None
    return best_start, best_end


def _estimate_scalebar_length_px(
    image: np.ndarray,
    *,
    x_hint: float | None = None,
    y_hint: float | None = None,
) -> tuple[float | None, str | None]:
    if image.ndim != 2 or image.size == 0:
        return None, None
    arr = np.asarray(image, dtype=np.float64)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, copy=False)

    h, w = arr.shape
    start = max(0, int(0.55 * h))
    roi = arr[start:, :]
    p_low = float(np.percentile(roi, 5.0))
    p_high = float(np.percentile(roi, 99.5))
    if p_high <= p_low + 1e-12:
        p_low = float(np.min(roi))
        p_high = float(np.max(roi))
        if p_high <= p_low + 1e-12:
            return None, None
    norm = np.clip((arr - p_low) / (p_high - p_low), 0.0, 1.0)
    roi_norm = norm[start:, :]

    bright_threshold = max(0.70, float(np.quantile(roi_norm, 0.985)))
    dark_threshold = min(0.30, float(np.quantile(roi_norm, 0.015)))
    masks: dict[str, np.ndarray] = {"bright": roi_norm >= bright_threshold}
    if float(np.median(roi_norm)) >= 0.55:
        masks["dark"] = roi_norm <= dark_threshold

    min_length = max(18, int(0.05 * max(h, w)))
    best_length: float | None = None
    best_mode: str | None = None
    best_score = -1e12
    for polarity, raw_mask in masks.items():
        coverage = float(np.mean(raw_mask))
        if coverage >= 0.45:
            continue
        horizontal_mask = ndi.binary_closing(raw_mask, structure=np.ones((1, 9), dtype=bool))
        vertical_mask = ndi.binary_closing(raw_mask, structure=np.ones((9, 1), dtype=bool))

        row_indices = list(range(horizontal_mask.shape[0]))
        if y_hint is not None:
            y_hint_local = int(round(y_hint)) - start
            near_rows = [idx for idx in row_indices if abs(idx - y_hint_local) <= 24]
            if near_rows:
                row_indices = near_rows + row_indices

        for local_y in row_indices:
            if local_y < 0 or local_y >= horizontal_mask.shape[0]:
                continue
            segment = _find_longest_segment(horizontal_mask[local_y])
            if segment is None:
                continue
            x0, x1 = segment
            length = float(x1 - x0)
            if length < min_length:
                continue
            absolute_y = start + local_y
            score = length + 0.45 * (absolute_y / max(1, h - 1))
            if y_hint is not None:
                score -= 0.10 * abs(absolute_y - y_hint)
            if x_hint is not None and not (x0 <= x_hint <= x1):
                score -= 0.08 * min(abs(x_hint - x0), abs(x_hint - x1))
            if score > best_score:
                best_score = score
                best_length = length
                best_mode = f"{polarity}.horizontal"

        col_indices = list(range(vertical_mask.shape[1]))
        if x_hint is not None:
            x_hint_int = int(round(x_hint))
            near_cols = [idx for idx in col_indices if abs(idx - x_hint_int) <= 24]
            if near_cols:
                col_indices = near_cols + col_indices

        for col_idx in col_indices:
            if col_idx < 0 or col_idx >= vertical_mask.shape[1]:
                continue
            segment = _find_longest_segment(vertical_mask[:, col_idx])
            if segment is None:
                continue
            y0, y1 = segment
            length = float(y1 - y0)
            if length < min_length:
                continue
            absolute_y0 = start + y0
            absolute_y1 = start + y1
            score = length
            if y_hint is not None and not (absolute_y0 <= y_hint <= absolute_y1):
                score -= 0.10 * min(abs(y_hint - absolute_y0), abs(y_hint - absolute_y1))
            if x_hint is not None:
                score -= 0.04 * abs(col_idx - x_hint)
            if score > best_score:
                best_score = score
                best_length = length
                best_mode = f"{polarity}.vertical"

    return best_length, best_mode


def register_scalebar_ocr_reader(reader: OCRReader | None) -> None:
    """Register or clear external OCR reader for scale-bar text detection."""
    global _EXTERNAL_OCR_READER
    _EXTERNAL_OCR_READER = reader


def _sanitize_ocr_rows(rows: Any) -> list[OCRRow]:
    sanitized: list[OCRRow] = []
    if not isinstance(rows, list):
        return sanitized
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            continue
        text = str(row[0])
        confidence = _to_float(row[1])
        x_center = _to_float(row[2])
        y_center = _to_float(row[3])
        if confidence is None or x_center is None or y_center is None:
            continue
        if not text.strip():
            continue
        sanitized.append((text, float(confidence), float(x_center), float(y_center)))
    return sanitized


def _default_tesseract_ocr_rows(image: np.ndarray) -> list[OCRRow]:
    try:
        import pytesseract
    except Exception:  # noqa: BLE001
        return []
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return []
    finite = np.isfinite(arr)
    if not finite.any():
        return []
    arr = np.nan_to_num(arr, copy=False)
    low = float(np.percentile(arr, 0.5))
    high = float(np.percentile(arr, 99.5))
    if high <= low + 1e-12:
        return []
    norm = np.clip((arr - low) / (high - low), 0.0, 1.0)
    image_u8 = np.asarray(np.round(norm * 255.0), dtype=np.uint8)
    try:
        ocr_payload = pytesseract.image_to_data(
            image_u8,
            config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:  # noqa: BLE001
        return []
    n_items = len(ocr_payload.get("text", []))
    rows: list[OCRRow] = []
    for idx in range(n_items):
        text = str(ocr_payload["text"][idx]).strip()
        if not text:
            continue
        confidence = _to_float(ocr_payload.get("conf", [None])[idx])
        if confidence is None:
            continue
        left = _to_float(ocr_payload.get("left", [None])[idx])
        top = _to_float(ocr_payload.get("top", [None])[idx])
        width = _to_float(ocr_payload.get("width", [None])[idx])
        height = _to_float(ocr_payload.get("height", [None])[idx])
        if (
            left is None
            or top is None
            or width is None
            or height is None
            or width <= 0
            or height <= 0
        ):
            continue
        x_center = left + 0.5 * width
        y_center = top + 0.5 * height
        rows.append((text, float(confidence), float(x_center), float(y_center)))
    return rows


def _ocr_read_text_rows(image: np.ndarray) -> list[OCRRow]:
    """OCR hook (text, confidence, x_center, y_center)."""
    if _EXTERNAL_OCR_READER is not None:
        try:
            return _sanitize_ocr_rows(_EXTERNAL_OCR_READER(image))
        except Exception:  # noqa: BLE001
            return []
    return _sanitize_ocr_rows(_default_tesseract_ocr_rows(image))


def infer_pixel_size_from_scalebar(
    image: np.ndarray,
    *,
    metadata: dict[str, Any] | None = None,
    file_hint: str | None = None,
) -> tuple[float | None, str, list[str]]:
    """Infer nm/pixel from scale-bar text and detected scale-bar width."""
    text_length_nm, text_source, x_hint, y_hint = _parse_scalebar_text_source(
        metadata,
        file_hint,
        image,
    )
    if text_length_nm is None:
        return (
            None,
            "scalebar.text.unavailable",
            ["Scale-bar text unavailable; cannot infer pixel size from scalebar."],
        )

    bar_length_px, bar_mode = _estimate_scalebar_length_px(
        np.asarray(image),
        x_hint=x_hint,
        y_hint=y_hint,
    )
    if bar_length_px is None or bar_length_px <= 0:
        return (
            None,
            "scalebar.bar.unavailable",
            ["Scale-bar text was found but bar segment detection failed."],
        )

    pixel_nm = float(text_length_nm / bar_length_px)
    if not np.isfinite(pixel_nm) or pixel_nm <= 0:
        return (
            None,
            "scalebar.unavailable",
            ["Computed pixel size from scalebar is invalid."],
        )

    mode = bar_mode or "unknown"
    return pixel_nm, f"{text_source} + scalebar.bar ({mode})", []


def _load_dataset_image(dataset: TEMDataset) -> np.ndarray | None:
    data_ref = dataset.data
    if data_ref.inline is not None:
        array = np.asarray(data_ref.inline, dtype=np.float64)
        expected_shape = tuple(int(v) for v in data_ref.shape)
        if expected_shape and array.size == int(np.prod(expected_shape)):
            array = array.reshape(expected_shape)
        if array.ndim == 2:
            return array

    if not data_ref.uri:
        return None
    path = Path(data_ref.uri)
    if not path.exists() or not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
        return np.asarray(array, dtype=np.float64) if array.ndim == 2 else None
    if suffix in {".txt", ".csv"}:
        array = np.loadtxt(path)
        return np.asarray(array, dtype=np.float64) if array.ndim == 2 else None
    if suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        try:
            from PIL import Image
        except ImportError:
            return None
        with Image.open(path) as handle:
            image = np.asarray(handle)
        if image.ndim == 3:
            image = image.mean(axis=2)
        return np.asarray(image, dtype=np.float64) if image.ndim == 2 else None
    return None


def resolve_pixel_size_measurement(
    dataset: TEMDataset,
    *,
    parameters: dict[str, Any] | None = None,
    reference: dict[str, Any] | None = None,
) -> tuple[Measurement | None, str, list[str]]:
    """Resolve pixel-size measurement with auto/manual/scalebar fallbacks."""
    params = dict(parameters or {})
    ref = dict(reference or {})
    warnings: list[str] = []

    explicit = _to_float(params.get("pixel_size_nm"))
    if explicit is not None and explicit > 0:
        uncertainty = _to_float(params.get("pixel_size_uncertainty_nm"))
        measurement = Measurement(
            value=float(explicit),
            unit="nm/pixel",
            uncertainty=(
                float(uncertainty)
                if uncertainty is not None and uncertainty >= 0
                else None
            ),
            method="parameter.pixel_size_nm",
            provenance={"source": "workflow.parameters.calibration.pixel_size_nm"},
            confidence=0.95,
        )
        return measurement, "parameter.pixel_size_nm", warnings

    if dataset.pixel_size is not None:
        unit = dataset.pixel_size.unit or "nm"
        pixel_nm = coerce_length_to_nm(dataset.pixel_size.value, unit.replace("/pixel", ""))
        if pixel_nm is not None and pixel_nm > 0:
            measurement = Measurement(
                value=float(pixel_nm),
                unit="nm/pixel",
                uncertainty=dataset.pixel_size.uncertainty,
                method=dataset.pixel_size.method or "dataset.pixel_size",
                provenance={"source": "dataset.pixel_size"},
                confidence=dataset.pixel_size.confidence,
            )
            return measurement, "dataset.pixel_size", warnings

    metadata = dataset.metadata.get("normalized", {}) if isinstance(dataset.metadata, dict) else {}
    if isinstance(metadata, dict):
        pixel_nm = _to_float(metadata.get("pixel_size_nm"))
        if pixel_nm is not None and pixel_nm > 0:
            measurement = Measurement(
                value=float(pixel_nm),
                unit="nm/pixel",
                method="metadata.normalized.pixel_size_nm",
                provenance={"source": "dataset.metadata.normalized.pixel_size_nm"},
                confidence=_to_float(metadata.get("metadata_confidence")),
            )
            return measurement, "metadata.normalized.pixel_size_nm", warnings

    ref_length = ref.get("reference_length", ref.get("known_length"))
    ref_pixels = ref.get("reference_pixels", ref.get("pixel_span"))
    if ref_length is not None and ref_pixels is not None:
        ref_unit = str(ref.get("reference_unit", ref.get("known_length_unit", "nm")))
        try:
            pixel_nm = manual_pixel_size_from_reference(
                reference_length=ref_length,
                reference_pixels=ref_pixels,
                reference_unit=ref_unit,
            )
        except ValueError as exc:
            warnings.append(f"Manual pixel calibration is invalid: {exc}")
        else:
            measurement = Measurement(
                value=float(pixel_nm),
                unit="nm/pixel",
                method="manual_reference_length",
                provenance={"source": "calibration.reference"},
                confidence=0.9,
            )
            return measurement, "manual_reference_length", warnings

    image = _load_dataset_image(dataset)
    if image is not None:
        pixel_nm, source, scale_warnings = infer_pixel_size_from_scalebar(
            image,
            metadata=dataset.metadata if isinstance(dataset.metadata, dict) else None,
            file_hint=dataset.source_path,
        )
        warnings.extend(scale_warnings)
        if pixel_nm is not None and pixel_nm > 0:
            measurement = Measurement(
                value=float(pixel_nm),
                unit="nm/pixel",
                method="scalebar",
                provenance={"source": source},
                confidence=0.75,
                warnings=list(scale_warnings),
            )
            return measurement, source, warnings

    warnings.append("Pixel size could not be resolved from parameters, metadata, or scalebar.")
    return None, "unresolved", warnings


def resolve_voltage_kv(
    dataset: TEMDataset,
    *,
    parameters: dict[str, Any] | None = None,
    reference: dict[str, Any] | None = None,
) -> tuple[float | None, str, list[str]]:
    """Resolve accelerating voltage in kV."""
    params = dict(parameters or {})
    ref = dict(reference or {})
    warnings: list[str] = []

    param_voltage = _to_float(params.get("voltage_kv"))
    if param_voltage is not None and param_voltage > 0:
        return float(param_voltage), "parameter.voltage_kv", warnings

    for key in ("voltage", "accelerating_voltage", "accelerating_voltage_kv"):
        if key not in ref:
            continue
        value = ref[key]
        if isinstance(value, str):
            match = _VOLT_RE.search(value)
            if match:
                unit = _normalize_unit(match.group("unit"))
                factor = _VOLT_TO_KV.get(unit or "", None)
                if factor is not None:
                    return float(match.group("value")) * factor, f"reference.{key}", warnings
        else:
            numeric = _to_float(value)
            if numeric is not None and numeric > 0:
                return float(numeric), f"reference.{key}", warnings

    metadata = dataset.metadata.get("normalized", {}) if isinstance(dataset.metadata, dict) else {}
    if isinstance(metadata, dict):
        voltage_kv = _to_float(metadata.get("accelerating_voltage_kv"))
        if voltage_kv is not None and voltage_kv > 0:
            return float(voltage_kv), "metadata.normalized.accelerating_voltage_kv", warnings

    raw_metadata = dataset.metadata.get("raw", {}) if isinstance(dataset.metadata, dict) else {}
    if isinstance(raw_metadata, dict):
        for key in ("accelerating_voltage_kv", "accelerating_voltage", "voltage", "ht"):
            if key not in raw_metadata:
                continue
            value = raw_metadata[key]
            if isinstance(value, str):
                match = _VOLT_RE.search(value)
                if match:
                    unit = _normalize_unit(match.group("unit"))
                    factor = _VOLT_TO_KV.get(unit or "", None)
                    if factor is not None:
                        return float(match.group("value")) * factor, f"metadata.raw.{key}", warnings
            else:
                numeric = _to_float(value)
                if numeric is not None and numeric > 0:
                    # Raw values are often already kV in EM metadata.
                    return float(numeric), f"metadata.raw.{key}", warnings

    warnings.append("Accelerating voltage could not be resolved.")
    return None, "unresolved", warnings
