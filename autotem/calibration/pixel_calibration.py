"""B1 pixel calibration utilities.

This module keeps calibration logic deterministic while supporting richer input
sources (manual values, metadata, and reference measurements).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

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

_SCALEBAR_VALUE_PATTERN = re.compile(
    r"(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?P<unit>nm|um|μm|µm|pm|a|å|angstrom)\b",
    flags=re.IGNORECASE,
)
_OCR_SCALE_TEXT_SANITIZE = str.maketrans(
    {
        "μ": "u",
        "µ": "u",
        "Å": "a",
        "å": "a",
        "O": "0",
        "o": "0",
    }
)


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


def _flatten_text_entries(
    payload: Any,
    *,
    prefix: str = "",
    depth: int = 0,
) -> list[tuple[str, str]]:
    if depth > 6:
        return []
    out: list[tuple[str, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            next_prefix = f"{prefix}.{key_text}" if prefix else key_text
            if isinstance(value, str):
                out.append((next_prefix, value))
            out.extend(_flatten_text_entries(value, prefix=next_prefix, depth=depth + 1))
        return out
    if isinstance(payload, (list, tuple)):
        for idx, item in enumerate(payload):
            next_prefix = f"{prefix}[{idx}]"
            if isinstance(item, str):
                out.append((next_prefix, item))
            out.extend(_flatten_text_entries(item, prefix=next_prefix, depth=depth + 1))
        return out
    if isinstance(payload, str):
        out.append((prefix or "text", payload))
    return out


def _extract_scalebar_value_nm(
    metadata: dict[str, Any] | None,
    *,
    file_hint: str | None = None,
    image: np.ndarray | None = None,
) -> tuple[float | None, str]:
    entries = _flatten_text_entries(metadata or {})
    if file_hint:
        entries.append(("file_hint", str(file_hint)))

    candidates: list[tuple[float, float, str]] = []
    for source, text in entries:
        if not text:
            continue
        lowered = text.lower()
        for match in _SCALEBAR_VALUE_PATTERN.finditer(text):
            value = _as_float(match.group("value"))
            unit = _normalize_unit(match.group("unit"))
            if value is None or value <= 0 or unit is None:
                continue
            if unit == "å":
                unit = "a"
            value_nm = coerce_length_to_nm(value, unit)
            if value_nm is None or value_nm <= 0:
                continue
            # Typical printed scale bar values are not absurdly tiny or huge.
            if not (0.01 <= value_nm <= 1_000_000.0):
                continue
            score = 1.0
            if "scale" in lowered or "bar" in lowered:
                score += 4.0
            if "image" in source.lower() and "description" in source.lower():
                score += 1.0
            candidates.append((score, value_nm, f"{source}:{match.group(0)}"))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, value_nm, source = candidates[0]
        return float(value_nm), f"scalebar.text ({source})"

    if image is not None:
        ocr_value_nm, ocr_source = _extract_scalebar_value_from_image_ocr(image)
        if ocr_value_nm is not None:
            return ocr_value_nm, ocr_source
    return None, "scalebar.text.unavailable"


def _ocr_read_text_rows(image: np.ndarray) -> list[tuple[str, float, float, float]]:
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except Exception:  # noqa: BLE001
        return []

    if image.ndim != 2:
        image = np.asarray(image).mean(axis=-1)
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return []
    arr = np.where(finite, arr, float(np.nanmedian(arr[finite])))
    p1, p99 = np.percentile(arr, [1, 99])
    span = max(float(p99 - p1), 1e-6)
    arr_u8 = np.clip((arr - p1) / span * 255.0, 0, 255).astype(np.uint8)
    h, w = arr_u8.shape
    y0 = int(h * 0.6)
    x1 = int(w * 0.62)
    roi = arr_u8[y0:, :x1]
    if roi.size == 0:
        return []
    roi_inv = 255 - roi
    roi_up = np.repeat(np.repeat(roi, 2, axis=0), 2, axis=1)
    roi_inv_up = 255 - roi_up

    engine = RapidOCR()
    rows: list[tuple[str, float, float, float]] = []
    variants = (
        ("raw", roi, 1.0),
        ("inv", roi_inv, 1.0),
        ("raw2x", roi_up, 2.0),
        ("inv2x", roi_inv_up, 2.0),
    )
    for _, img_variant, scale in variants:
        result, _ = engine(img_variant)
        if not result:
            continue
        for item in result:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            box, text, conf = item[0], item[1], item[2]
            if not isinstance(text, str):
                continue
            conf_f = _as_float(conf) or 0.0
            try:
                xs = [float(pt[0]) for pt in box]
                ys = [float(pt[1]) for pt in box]
            except Exception:  # noqa: BLE001
                continue
            cx = (min(xs) + max(xs)) * 0.5 / scale
            cy = (min(ys) + max(ys)) * 0.5 / scale
            rows.append((text, float(conf_f), float(cx), float(cy + y0)))
    return rows


def _extract_scalebar_value_from_image_ocr(image: np.ndarray) -> tuple[float | None, str]:
    rows = _ocr_read_text_rows(image)
    if not rows:
        return None, "scalebar.ocr.unavailable"

    h = float(np.asarray(image).shape[0])
    candidates: list[tuple[float, float, str]] = []
    for text, conf, cx, cy in rows:
        normalized = text.translate(_OCR_SCALE_TEXT_SANITIZE).lower().replace(" ", "")
        for match in _SCALEBAR_VALUE_PATTERN.finditer(normalized):
            value = _as_float(match.group("value"))
            unit = _normalize_unit(match.group("unit"))
            if value is None or value <= 0 or unit is None:
                continue
            value_nm = coerce_length_to_nm(value, unit)
            if value_nm is None or value_nm <= 0:
                continue
            y_bonus = 1.0 + 0.45 * (cy / max(h, 1.0))
            x_bonus = 1.0 + 0.2 * (1.0 - min(cx / 512.0, 1.0))
            score = float(conf) * y_bonus * x_bonus
            candidates.append((score, float(value_nm), f"ocr:{text.strip()}"))
    if not candidates:
        return None, "scalebar.ocr.no_value"
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, value_nm, source = candidates[0]
    return value_nm, f"scalebar.text ({source})"


def _row_segments(mask_row: np.ndarray, *, min_len: int, max_len: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    starts = np.where(np.diff(np.concatenate(([0], mask_row.astype(np.int8), [0]))) == 1)[0]
    ends = np.where(np.diff(np.concatenate(([0], mask_row.astype(np.int8), [0]))) == -1)[0]
    for x0, x1 in zip(starts, ends, strict=False):
        length = int(x1 - x0)
        if min_len <= length <= max_len:
            segments.append((int(x0), int(x1 - 1)))
    return segments


def _close_row_mask(mask_row: np.ndarray, *, gap_px: int) -> np.ndarray:
    if gap_px <= 0:
        return mask_row.astype(bool)
    kernel = np.ones(int(gap_px), dtype=np.int16)
    dilated = np.convolve(mask_row.astype(np.int16), kernel, mode="same") > 0
    return dilated


def _detect_scalebar_length_px(image: np.ndarray) -> tuple[float | None, str]:
    if image.ndim != 2:
        image = np.asarray(image).mean(axis=-1)
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        return None, "scalebar.bar.empty"
    finite = np.isfinite(arr)
    if not np.any(finite):
        return None, "scalebar.bar.invalid"
    arr = np.where(finite, arr, float(np.nanmedian(arr[finite])))
    h, w = arr.shape
    if h < 32 or w < 32:
        return None, "scalebar.bar.too_small"

    y0 = int(h * 0.68)
    roi = arr[y0:, :]
    p1, p99 = np.percentile(roi, [1, 99])
    span = max(float(p99 - p1), 1e-6)
    roi_n = np.clip((roi - p1) / span, 0.0, 1.0)

    min_len = max(20, int(w * 0.05))
    max_len = int(w * 0.7)
    roi_mean = float(np.mean(roi_n))
    roi_std = float(np.std(roi_n))
    bright_thr = max(float(np.percentile(roi_n, 99.5)), roi_mean + 2.6 * roi_std)
    dark_thr = min(float(np.percentile(roi_n, 0.5)), roi_mean - 2.6 * roi_std)
    masks = [
        ("bright", roi_n >= bright_thr),
        ("dark", roi_n <= dark_thr),
    ]

    groups: list[dict[str, Any]] = []
    for mask_name, mask in masks:
        for ry in range(mask.shape[0]):
            y_abs = y0 + ry
            segments = _row_segments(mask[ry], min_len=min_len, max_len=max_len)
            for x0, x1 in segments:
                length = float(x1 - x0 + 1)
                width_ratio = length / max(float(w), 1.0)
                if width_ratio > 0.85:
                    continue
                matched_group = None
                for group in groups:
                    if group["mask"] != mask_name:
                        continue
                    if abs(y_abs - group["last_y"]) > 3:
                        continue
                    gx0 = float(group["x0_sum"]) / float(group["count"])
                    gx1 = float(group["x1_sum"]) / float(group["count"])
                    overlap = max(0.0, min(gx1, x1) - max(gx0, x0) + 1.0)
                    min_len_overlap = max(1.0, min(float(gx1 - gx0 + 1.0), length))
                    if overlap / min_len_overlap < 0.55:
                        continue
                    matched_group = group
                    break

                if matched_group is None:
                    groups.append(
                        {
                            "mask": mask_name,
                            "x0_sum": float(x0),
                            "x1_sum": float(x1),
                            "count": 1,
                            "y_min": y_abs,
                            "y_max": y_abs,
                            "last_y": y_abs,
                        }
                    )
                else:
                    matched_group["x0_sum"] += float(x0)
                    matched_group["x1_sum"] += float(x1)
                    matched_group["count"] += 1
                    matched_group["y_min"] = min(int(matched_group["y_min"]), y_abs)
                    matched_group["y_max"] = max(int(matched_group["y_max"]), y_abs)
                    matched_group["last_y"] = y_abs

    best_len: float | None = None
    best_score = -1.0
    best_source = "scalebar.bar.unavailable"
    for group in groups:
        count = int(group["count"])
        if count <= 0:
            continue
        x0_avg = float(group["x0_sum"]) / float(count)
        x1_avg = float(group["x1_sum"]) / float(count)
        length = float(x1_avg - x0_avg + 1.0)
        thickness = int(group["y_max"]) - int(group["y_min"]) + 1
        if thickness > 18:
            continue
        y_center = 0.5 * (int(group["y_min"]) + int(group["y_max"]))
        y_weight = 1.0 + 0.4 * (y_center / max(h - 1, 1))
        x0_avg = float(group["x0_sum"]) / float(count)
        x1_avg = float(group["x1_sum"]) / float(count)
        edge_dist = min(x0_avg, max(float(w) - 1.0 - x1_avg, 0.0))
        edge_bonus = 1.25 if edge_dist <= 0.14 * float(w) else 1.0
        score = length * y_weight * (1.0 + 0.2 * min(thickness, 8)) * edge_bonus
        if score > best_score:
            best_score = score
            best_len = length
            best_source = (
                f"scalebar.bar.{group['mask']}@y{int(group['y_min'])}-{int(group['y_max'])}"
            )

    if best_len is not None:
        return best_len, best_source

    # Fallback: relaxed threshold + row-wise gap-closing, robust for anti-aliased
    # or outlined scale bars near image border.
    for q_hi, q_lo in ((98.0, 2.0), (97.0, 3.0), (96.0, 4.0)):
        hi = float(np.percentile(roi_n, q_hi))
        lo = float(np.percentile(roi_n, q_lo))
        for mode_name, mask in (
            ("bright_relaxed", roi_n >= hi),
            ("dark_relaxed", roi_n <= lo),
        ):
            for ry in range(mask.shape[0]):
                closed = _close_row_mask(mask[ry], gap_px=4)
                segments = _row_segments(closed, min_len=min_len, max_len=max_len)
                if not segments:
                    continue
                y_abs = y0 + ry
                y_weight = 1.0 + 0.45 * (y_abs / max(h - 1, 1))
                for x0, x1 in segments:
                    length = float(x1 - x0 + 1)
                    edge_dist = min(float(x0), max(float(w) - 1.0 - float(x1), 0.0))
                    edge_bonus = 1.35 if edge_dist <= 0.16 * float(w) else 1.0
                    score = length * y_weight * edge_bonus
                    if score > best_score:
                        best_score = score
                        best_len = length
                        best_source = f"scalebar.bar.{mode_name}@y{y_abs}"

    return best_len, best_source


def infer_pixel_size_from_scalebar(
    image: np.ndarray,
    *,
    metadata: dict[str, Any] | None = None,
    file_hint: str | None = None,
) -> tuple[float | None, str, list[str]]:
    """Infer nm/pixel from an embedded scale bar and nearby text metadata.

    Returns:
        pixel_size_nm, source, warnings
    """
    warnings: list[str] = []
    bar_px, bar_source = _detect_scalebar_length_px(image)
    if bar_px is None or bar_px <= 0:
        warnings.append("Scale bar line was not reliably detected in image.")
        return None, "scalebar.unavailable", warnings

    value_nm, value_source = _extract_scalebar_value_nm(
        metadata,
        file_hint=file_hint,
        image=image,
    )
    if value_nm is None or value_nm <= 0:
        warnings.append("Scale bar text value was not found in metadata/file hint/OCR.")
        return None, "scalebar.text.unavailable", warnings

    pixel_nm = float(value_nm) / float(bar_px)
    if not (1e-5 <= pixel_nm <= 1e6):
        warnings.append(
            f"Inferred scale from scalebar is implausible ({pixel_nm:.6g} nm/pixel)."
        )
        return None, "scalebar.implausible", warnings
    source = f"{value_source} + {bar_source}"
    return pixel_nm, source, warnings


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
