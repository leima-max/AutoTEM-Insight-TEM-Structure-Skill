"""Converters from reader payloads to core data models."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from autotem.core.models import ArrayReference, Measurement, TEMDataset

from .readers import ReadResult


def _warning_to_text(warning: dict[str, Any]) -> str:
    code = warning.get("code", "WARNING")
    severity = warning.get("severity", "warning")
    message = warning.get("message", "")
    field = warning.get("field")
    source = warning.get("source")
    pieces = [f"[{severity}]", code + ":"]
    if source:
        pieces.append(f"({source})")
    pieces.append(message)
    if field:
        pieces.append(f"[field={field}]")
    return " ".join(pieces).strip()


def _to_array_reference(result: ReadResult) -> ArrayReference:
    source_path = str(result.provenance.get("path", ""))
    inline_data = result.array.tolist() if result.array.size <= 4096 else None
    return ArrayReference(
        shape=tuple(int(dim) for dim in result.array.shape),
        dtype=str(result.array.dtype),
        uri=source_path or None,
        inline=inline_data,
        description="Loaded microscopy image array.",
    )


def _to_pixel_measurement(result: ReadResult) -> Measurement | None:
    pixel_size_nm = result.metadata_norm.get("pixel_size_nm")
    if pixel_size_nm is None:
        return None
    return Measurement(
        value=float(pixel_size_nm),
        unit="nm/pixel",
        method="metadata_normalization",
        provenance={"source": "autotem.io.metadata.normalize_metadata"},
    )


def _infer_modality(result: ReadResult) -> str:
    path = str(result.provenance.get("path", "")).lower()
    metadata_blob = str(result.metadata_raw).lower()

    if path.endswith((".ser", ".emi")):
        if any(token in metadata_blob for token in ["scan", "datacube", "4d"]):
            return "4D-STEM"
        return "STEM"
    if any(token in metadata_blob for token in ["haadf", "high angle annular"]):
        return "HAADF-STEM"
    if any(token in metadata_blob for token in ["annular bright", "abf"]):
        return "ABF-STEM"
    if any(token in metadata_blob for token in ["sted", "scanmode", "probe"]):
        return "STEM"
    if any(token in metadata_blob for token in ["saed", "selected area diffraction"]):
        return "SAED"
    if any(token in metadata_blob for token in ["nbed", "nanobeam"]):
        return "NBED"
    if path.endswith((".dm3", ".dm4", ".mrc", ".tif", ".tiff", ".png", ".jpg", ".jpeg")):
        return "TEM"
    return "unknown"


def to_temdataset(result: ReadResult, dataset_cls: Callable[..., Any] | None = None) -> Any:
    """Convert `ReadResult` into `TEMDataset` or a compatible override class."""
    warning_texts = [_warning_to_text(w) for w in result.warnings]
    source_path = str(result.provenance.get("path", ""))
    dataset_kwargs = {
        "data": _to_array_reference(result),
        "dataset_id": Path(source_path).stem if source_path else "",
        "source_path": source_path,
        "modality": _infer_modality(result),
        "pixel_size": _to_pixel_measurement(result),
        "pixel_unit": "nm" if result.metadata_norm.get("pixel_size_nm") is not None else "unknown",
        "metadata": {
            "normalized": result.metadata_norm,
            "raw": result.metadata_raw,
        },
        "provenance": result.provenance,
        "warnings": warning_texts,
    }

    if dataset_cls is not None:
        return dataset_cls(**dataset_kwargs)
    return TEMDataset(**dataset_kwargs)

