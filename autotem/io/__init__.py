"""Input readers and metadata normalization."""

from autotem.io.converters import to_temdataset
from autotem.io.metadata import normalize_metadata
from autotem.io.readers import ReadResult, ReaderRegistry, build_default_registry, read_image

__all__ = [
    "ReadResult",
    "ReaderRegistry",
    "build_default_registry",
    "normalize_metadata",
    "read_image",
    "to_temdataset",
]
