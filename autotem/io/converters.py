"""Converters from reader payloads to core data models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from autotem.core.models import ArrayReference, Measurement, TEMDataset

from .readers import ReadResult


def _warning_to_text(warning: dict[str, Any]) -> str:
    code = warning.get("code", "WARNING")
    message = warning.get("message", "")
    field = warning.get("field")
    suffix = f" [field={field}]" if field else ""
    return f"{code}: {message}{suffix}".strip()


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


def to_temdataset(result: ReadResult, dataset_cls: Callable[..., Any] | None = None) -> Any:
    """Convert `ReadResult` into `TEMDataset` or a compatible override class."""
    warning_texts = [_warning_to_text(w) for w in result.warnings]
    source_path = str(result.provenance.get("path", ""))
    dataset_kwargs = {
        "data": _to_array_reference(result),
        "dataset_id": Path(source_path).stem if source_path else "",
        "source_path": source_path,
        "modality": "unknown",
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

