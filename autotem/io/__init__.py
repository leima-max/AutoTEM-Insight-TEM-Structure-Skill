"""Input readers and metadata normalization."""

from pathlib import Path

from autotem.core import ArrayReference, TEMDataset
from autotem.io.converters import to_temdataset
from autotem.io.metadata import (
    FORMAT_METADATA_MAPPINGS,
    METADATA_WARNING_CATALOG,
    normalize_metadata,
)
from autotem.io.readers import (
    ReaderRegistration,
    ReaderRegistry,
    ReadResult,
    build_default_registry,
    read_image,
    read_series,
    read_series_as_temdataset,
)


def load_tem_dataset(input_path: str | Path) -> TEMDataset:
    """Load microscopy input and convert it to the shared TEMDataset model."""
    path = Path(input_path)
    if not path.exists():
        return TEMDataset(
            data=ArrayReference(shape=tuple(), dtype="unknown", uri=str(path)),
            dataset_id=path.stem,
            source_path=str(path),
            modality="unknown",
            metadata={},
            warnings=["Input path does not exist; metadata-only dataset scaffold returned."],
            provenance={"reader": "load_tem_dataset"},
        )
    result = read_image(path)
    return to_temdataset(result)


__all__ = [
    "ReadResult",
    "ReaderRegistration",
    "ReaderRegistry",
    "FORMAT_METADATA_MAPPINGS",
    "METADATA_WARNING_CATALOG",
    "build_default_registry",
    "load_tem_dataset",
    "normalize_metadata",
    "read_image",
    "read_series",
    "read_series_as_temdataset",
    "to_temdataset",
]

