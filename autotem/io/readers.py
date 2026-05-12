"""Multi-format reader registry and baseline image readers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, TiffTags

from .metadata import normalize_metadata

ReaderFn = Callable[[Path], "ReadResult"]


@dataclass
class ReadResult:
    """Normalized output for all readers."""

    array: np.ndarray
    metadata_raw: dict[str, Any]
    metadata_norm: dict[str, Any]
    warnings: list[dict[str, Any]]
    provenance: dict[str, Any]


class ReaderRegistry:
    """Registry dispatching file readers by extension."""

    def __init__(self) -> None:
        self._readers: dict[str, ReaderFn] = {}

    def register(self, extensions: str | list[str] | tuple[str, ...], reader: ReaderFn) -> None:
        if isinstance(extensions, str):
            extensions = [extensions]
        for ext in extensions:
            normalized = ext.lower().strip()
            if not normalized.startswith("."):
                normalized = "." + normalized
            self._readers[normalized] = reader

    def get_reader(self, path: str | Path) -> ReaderFn:
        suffix = Path(path).suffix.lower()
        if suffix not in self._readers:
            supported = ", ".join(sorted(self._readers))
            raise ValueError(f"Unsupported file extension: {suffix}. Supported: {supported}")
        return self._readers[suffix]

    def read(self, path: str | Path) -> ReadResult:
        reader = self.get_reader(path)
        return reader(Path(path))


def _tag_name(tag_id: int) -> str:
    return TiffTags.TAGS.get(tag_id, str(tag_id))


def _read_standard_image(path: Path) -> ReadResult:
    with Image.open(path) as img:
        metadata_raw: dict[str, Any] = dict(img.info)
        warnings: list[dict[str, Any]] = []

        if img.format == "TIFF" and hasattr(img, "tag_v2"):
            for tag_id, value in img.tag_v2.items():
                metadata_raw[_tag_name(tag_id)] = value
                metadata_raw[str(tag_id)] = value

        array = np.asarray(img)
        if array.ndim == 3:
            warnings.append(
                {
                    "code": "IMAGE_RGB_TO_GRAYSCALE",
                    "message": "Input image has RGB channels; converted to grayscale average.",
                    "field": "array",
                }
            )
            array = array.mean(axis=2)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D image but got shape {array.shape} for {path.name}.")

        metadata_norm, metadata_warnings = normalize_metadata(metadata_raw)
        warnings.extend(metadata_warnings)

        return ReadResult(
            array=array.astype(np.float32),
            metadata_raw=metadata_raw,
            metadata_norm=metadata_norm,
            warnings=warnings,
            provenance={
                "reader": "pil",
                "format": img.format,
                "path": str(path),
                "dtype_original": str(np.asarray(img).dtype),
            },
        )


def _read_dm(path: Path) -> ReadResult:
    try:
        from ncempy.io import dm  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Reading DM3/DM4 requires `ncempy`. Please install ncempy.") from exc

    payload = dm.dmReader(str(path))
    array = np.asarray(payload.get("data"))
    metadata_raw = dict(payload)
    metadata_raw.pop("data", None)

    metadata_norm, warnings = normalize_metadata(metadata_raw)
    return ReadResult(
        array=array.astype(np.float32),
        metadata_raw=metadata_raw,
        metadata_norm=metadata_norm,
        warnings=warnings,
        provenance={"reader": "ncempy.dmReader", "format": "DM", "path": str(path)},
    )


def _read_mrc(path: Path) -> ReadResult:
    try:
        import mrcfile  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Reading MRC requires `mrcfile`. Please install mrcfile.") from exc

    with mrcfile.open(str(path), permissive=True) as mrc:
        array = np.asarray(mrc.data)
        voxel = mrc.voxel_size
        metadata_raw = {
            "pixel_size_x_angstrom": float(voxel.x) if voxel.x else None,
            "pixel_size_y_angstrom": float(voxel.y) if voxel.y else None,
            "pixel_size_z_angstrom": float(voxel.z) if voxel.z else None,
        }
        if metadata_raw["pixel_size_x_angstrom"]:
            metadata_raw["pixel_size"] = f"{metadata_raw['pixel_size_x_angstrom']} angstrom"

    metadata_norm, warnings = normalize_metadata(metadata_raw)
    return ReadResult(
        array=array.astype(np.float32),
        metadata_raw=metadata_raw,
        metadata_norm=metadata_norm,
        warnings=warnings,
        provenance={"reader": "mrcfile", "format": "MRC", "path": str(path)},
    )


def _first_numeric_dataset(handle: Any) -> np.ndarray | None:
    for key in handle.keys():
        item = handle[key]
        if hasattr(item, "dtype") and np.issubdtype(item.dtype, np.number):
            return np.asarray(item)
        if hasattr(item, "keys"):
            nested = _first_numeric_dataset(item)
            if nested is not None:
                return nested
    return None


def _read_hdf5_emd(path: Path) -> ReadResult:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Reading HDF5/EMD requires `h5py`. Please install h5py.") from exc

    with h5py.File(path, "r") as h5f:
        array = _first_numeric_dataset(h5f)
        if array is None:
            raise ValueError(f"No numeric dataset found in {path.name}.")
        metadata_raw = dict(h5f.attrs)

    metadata_norm, warnings = normalize_metadata(metadata_raw)
    return ReadResult(
        array=np.asarray(array).astype(np.float32),
        metadata_raw=metadata_raw,
        metadata_norm=metadata_norm,
        warnings=warnings,
        provenance={"reader": "h5py", "format": "HDF5/EMD", "path": str(path)},
    )


def _read_ser(path: Path) -> ReadResult:
    try:
        from ncempy.io import ser  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Reading SER requires `ncempy`. Please install ncempy.") from exc

    with ser.fileSER(str(path)) as handle:
        array, metadata = handle.getDataset(0)
    metadata_raw = dict(metadata or {})
    metadata_norm, warnings = normalize_metadata(metadata_raw)

    return ReadResult(
        array=np.asarray(array).astype(np.float32),
        metadata_raw=metadata_raw,
        metadata_norm=metadata_norm,
        warnings=warnings,
        provenance={"reader": "ncempy.fileSER", "format": "SER", "path": str(path)},
    )


def build_default_registry() -> ReaderRegistry:
    """Build default registry with baseline and optional adapters."""
    registry = ReaderRegistry()
    registry.register([".tif", ".tiff", ".png", ".jpg", ".jpeg"], _read_standard_image)
    registry.register([".dm3", ".dm4"], _read_dm)
    registry.register(".mrc", _read_mrc)
    registry.register([".h5", ".hdf5", ".emd"], _read_hdf5_emd)
    registry.register([".ser", ".emi"], _read_ser)
    return registry


def read_image(path: str | Path, registry: ReaderRegistry | None = None) -> ReadResult:
    """Read an image file using the supplied/default registry."""
    registry = registry or build_default_registry()
    return registry.read(path)

