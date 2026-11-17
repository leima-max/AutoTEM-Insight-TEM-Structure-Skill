"""Multi-format reader registry and baseline image readers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, TiffTags

from autotem._paths import io_path

from .metadata import normalize_metadata

ReaderFn = Callable[[Path], "ReadResult"]
_IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def _mk_warning(
    code: str,
    message: str,
    field: str | None = None,
    severity: str = "warning",
) -> dict[str, Any]:
    warning: dict[str, Any] = {
        "code": code,
        "message": message,
        "severity": severity,
        "source": "reader",
    }
    if field:
        warning["field"] = field
    return warning


@dataclass
class ReadResult:
    """Normalized output for all readers."""

    array: np.ndarray
    metadata_raw: dict[str, Any]
    metadata_norm: dict[str, Any]
    warnings: list[dict[str, Any]]
    provenance: dict[str, Any]


@dataclass
class ReaderRegistration:
    """Metadata for one reader registration."""

    reader: ReaderFn | None
    requires: tuple[str, ...]
    description: str
    available: bool
    unavailable_reason: str | None = None


class ReaderRegistry:
    """Registry dispatching file readers by extension."""

    def __init__(self) -> None:
        self._entries: dict[str, ReaderRegistration] = {}

    def register(
        self,
        extensions: str | list[str] | tuple[str, ...],
        reader: ReaderFn,
        requires: tuple[str, ...] = (),
        description: str = "",
    ) -> None:
        if isinstance(extensions, str):
            extensions = [extensions]
        for ext in extensions:
            normalized = ext.lower().strip()
            if not normalized.startswith("."):
                normalized = "." + normalized
            self._entries[normalized] = ReaderRegistration(
                reader=reader,
                requires=requires,
                description=description,
                available=True,
            )

    def register_unavailable(
        self,
        extensions: str | list[str] | tuple[str, ...],
        requires: tuple[str, ...],
        description: str,
        unavailable_reason: str,
    ) -> None:
        if isinstance(extensions, str):
            extensions = [extensions]
        for ext in extensions:
            normalized = ext.lower().strip()
            if not normalized.startswith("."):
                normalized = "." + normalized
            self._entries[normalized] = ReaderRegistration(
                reader=None,
                requires=requires,
                description=description,
                available=False,
                unavailable_reason=unavailable_reason,
            )

    def get_reader(self, path: str | Path) -> ReaderFn:
        suffix = Path(path).suffix.lower()
        registration = self._entries.get(suffix)
        if registration is None:
            supported = ", ".join(self.list_supported_formats(only_available=True))
            raise ValueError(f"Unsupported file extension: {suffix}. Supported: {supported}")
        if not registration.available or registration.reader is None:
            if registration.requires:
                required = ", ".join(registration.requires)
            else:
                required = "optional dependency"
            reason = registration.unavailable_reason or "dependency unavailable"
            raise ValueError(
                f"Reader for {suffix} is unavailable ({reason}). Install: {required}."
            )
        return registration.reader

    def read(self, path: str | Path) -> ReadResult:
        reader = self.get_reader(path)
        return reader(Path(path))

    def list_supported_formats(self, only_available: bool = False) -> list[str]:
        formats: list[str] = []
        for ext, registration in sorted(self._entries.items()):
            if only_available and not registration.available:
                continue
            formats.append(ext)
        return formats

    def describe_formats(self) -> dict[str, dict[str, Any]]:
        description: dict[str, dict[str, Any]] = {}
        for ext, registration in sorted(self._entries.items()):
            description[ext] = {
                "available": registration.available,
                "requires": list(registration.requires),
                "description": registration.description,
                "unavailable_reason": registration.unavailable_reason,
            }
        return description


def _dependencies_available(requires: tuple[str, ...]) -> tuple[bool, str | None]:
    missing: list[str] = []
    for package in requires:
        if find_spec(package) is None:
            missing.append(package)
    if missing:
        return False, f"missing dependencies: {', '.join(missing)}"
    return True, None


def _tag_name(tag_id: int) -> str:
    return TiffTags.TAGS.get(tag_id, str(tag_id))


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def select_image_page(img: Image.Image, frame: int | None, channel: int | None) -> int:
    """Resolve page and band from metadata before decoding any pixel buffer."""
    from autotem.harness.contracts import HarnessError, NeedsInput

    count = getattr(img, "n_frames", 1)
    if count > 1 and frame is None:
        raise NeedsInput([{"name": "frame", "count": count}], "Select an image frame explicitly")
    page = frame if frame is not None else 0
    if not 0 <= page < count:
        raise HarnessError("INVALID_SELECTION", "Image frame is out of bounds")
    img.seek(page)
    bands = len(img.getbands())
    if bands > 1 and channel is None:
        raise NeedsInput(
            [{"name": "channel", "count": bands}], "Select an image channel explicitly"
        )
    if channel is not None and (bands == 1 or not 0 <= channel < bands):
        raise HarnessError("INVALID_SELECTION", "Channel index is invalid for this image")
    return page


def require_spatial_indices(shape: tuple[int, ...]) -> None:
    from autotem.harness.contracts import NeedsInput

    if len(shape) > 2:
        raise NeedsInput(
            [{"name": "selection_indices", "shape": list(shape), "count": len(shape) - 2}],
            "Select each leading dimension; the final two dimensions are spatial",
        )


def _read_standard_image(path: Path, *, require_plane: bool = False) -> ReadResult:
    with Image.open(io_path(path)) as img:
        if require_plane:
            select_image_page(img, frame=None, channel=None)
        metadata_raw: dict[str, Any] = _to_jsonable(dict(img.info))
        warnings: list[dict[str, Any]] = []

        if img.format == "TIFF" and hasattr(img, "tag_v2"):
            for tag_id, value in img.tag_v2.items():
                metadata_raw[_tag_name(tag_id)] = _to_jsonable(value)
                metadata_raw[str(tag_id)] = _to_jsonable(value)

        if not require_plane and getattr(img, "n_frames", 1) > 1:
            warnings.append(
                _mk_warning(
                    "IMAGE_MULTIFRAME_TIFF",
                    "Multi-frame TIFF detected. Only the first frame is read.",
                    field="array",
                )
            )

        array = np.asarray(img)
        original_dtype = str(array.dtype)
        if array.ndim == 3:
            warnings.append(
                _mk_warning(
                    "IMAGE_RGB_TO_GRAYSCALE",
                    "Input image has RGB channels; converted to grayscale average.",
                    field="array",
                )
            )
            array = array.mean(axis=2)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D image but got shape {array.shape} for {path.name}.")

        metadata_norm, metadata_warnings = normalize_metadata(
            metadata_raw,
            source_format=path.suffix,
        )
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
                "dtype_original": original_dtype,
            },
        )


def read_numeric_file(path: Path, *, mmap_mode: str | None = None) -> np.ndarray:
    """Read original NPY pixels or a whitespace- or comma-delimited numeric table."""
    path = io_path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
    if suffix == ".txt":
        return np.loadtxt(path)
    if suffix == ".csv":
        # Historical CSV files may contain whitespace-separated values.
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                fields = line.partition("#")[0].strip()
                if fields:
                    break
            else:
                raise ValueError(f"Numeric CSV contains no data: {path.name}")
            stream.seek(0)
            return np.loadtxt(stream, delimiter="," if "," in fields else None)
    raise ValueError(f"Unsupported numeric file: {suffix}")


def _read_numpy_or_text(path: Path, *, require_plane: bool = False) -> ReadResult:
    suffix = path.suffix.lower()
    array = read_numeric_file(path, mmap_mode="r" if require_plane else None)
    if require_plane and suffix == ".npy":
        require_spatial_indices(array.shape)

    if array.ndim != 2:
        raise ValueError(f"Expected 2D array but got shape {array.shape} for {path.name}.")

    metadata_raw: dict[str, Any] = {
        "source": "numpy_or_text",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    metadata_norm, warnings = normalize_metadata(metadata_raw, source_format=suffix)
    return ReadResult(
        array=np.asarray(array).astype(np.float32),
        metadata_raw=metadata_raw,
        metadata_norm=metadata_norm,
        warnings=warnings,
        provenance={"reader": "numpy", "format": suffix.lstrip("."), "path": str(path)},
    )


def _read_dm(path: Path) -> ReadResult:
    try:
        from ncempy.io import dm  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Reading DM3/DM4 requires `ncempy`. Please install ncempy.") from exc

    with dm.fileDM(str(io_path(path)), on_memory=True) as dm_file:
        payload = dm_file.getDataset(0)
        array = np.asarray(payload.get("data"))
        metadata_raw = _to_jsonable(dict(payload))
        metadata_raw.pop("data", None)

        dm_sections: dict[str, Any] = {}

        dataset_metadata = dm_file.getMetadata(0)
        if isinstance(dataset_metadata, dict):
            dataset_metadata_json = _to_jsonable(dataset_metadata)
            dm_sections["dataset_metadata"] = dataset_metadata_json
            if isinstance(dataset_metadata_json, dict):
                for key, value in dataset_metadata_json.items():
                    metadata_raw.setdefault(str(key), value)

        file_metadata = getattr(dm_file, "metadata", None)
        if isinstance(file_metadata, dict):
            file_metadata_json = _to_jsonable(file_metadata)
            dm_sections["file_metadata"] = file_metadata_json
            if isinstance(file_metadata_json, dict):
                for key, value in file_metadata_json.items():
                    metadata_raw.setdefault(str(key), value)

        all_tags = getattr(dm_file, "allTags", None)
        if isinstance(all_tags, dict):
            all_tags_json = _to_jsonable(all_tags)
            dm_sections["all_tags"] = all_tags_json
            if isinstance(all_tags_json, dict):
                for key, value in all_tags_json.items():
                    key_text = str(key)
                    metadata_raw.setdefault(key_text, value)
                    stripped = key_text.lstrip(".")
                    if stripped:
                        metadata_raw.setdefault(stripped, value)

        if dm_sections:
            metadata_raw["dm_sections"] = dm_sections

    metadata_norm, warnings = normalize_metadata(metadata_raw, source_format=path.suffix)
    return ReadResult(
        array=array.astype(np.float32),
        metadata_raw=metadata_raw,
        metadata_norm=metadata_norm,
        warnings=warnings,
        provenance={"reader": "ncempy.fileDM", "format": "DM", "path": str(path)},
    )


def _read_mrc(path: Path) -> ReadResult:
    try:
        import mrcfile  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Reading MRC requires `mrcfile`. Please install mrcfile.") from exc

    with mrcfile.open(str(io_path(path)), permissive=True) as mrc:
        array = np.asarray(mrc.data)
        voxel = mrc.voxel_size
        metadata_raw = {
            "pixel_size_x_angstrom": float(voxel.x) if voxel.x else None,
            "pixel_size_y_angstrom": float(voxel.y) if voxel.y else None,
            "pixel_size_z_angstrom": float(voxel.z) if voxel.z else None,
        }
        if metadata_raw["pixel_size_x_angstrom"]:
            metadata_raw["pixel_size"] = f"{metadata_raw['pixel_size_x_angstrom']} angstrom"

    metadata_norm, warnings = normalize_metadata(metadata_raw, source_format=path.suffix)
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

    with h5py.File(io_path(path), "r") as h5f:
        array = _first_numeric_dataset(h5f)
        if array is None:
            raise ValueError(f"No numeric dataset found in {path.name}.")
        metadata_raw = _to_jsonable(dict(h5f.attrs))

    metadata_norm, warnings = normalize_metadata(metadata_raw, source_format=path.suffix)
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

    with ser.fileSER(str(io_path(path))) as handle:
        array, metadata = handle.getDataset(0)
    metadata_raw = _to_jsonable(dict(metadata or {}))
    metadata_norm, warnings = normalize_metadata(metadata_raw, source_format=path.suffix)

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

    registry.register(
        [".npy", ".txt", ".csv"],
        _read_numpy_or_text,
        description="NumPy/text array reader.",
    )
    registry.register(
        _IMAGE_EXTENSIONS,
        _read_standard_image,
        description="Baseline PIL image reader for common image formats.",
    )

    optional_specs = [
        ([".dm3", ".dm4"], _read_dm, ("ncempy",), "DigitalMicrograph DM format reader."),
        ([".mrc"], _read_mrc, ("mrcfile",), "MRC format reader."),
        ([".h5", ".hdf5", ".emd"], _read_hdf5_emd, ("h5py",), "HDF5/EMD format reader."),
        ([".ser", ".emi"], _read_ser, ("ncempy",), "SER format reader."),
    ]

    for extensions, reader, requires, description in optional_specs:
        available, reason = _dependencies_available(requires)
        if available:
            registry.register(extensions, reader, requires=requires, description=description)
        else:
            registry.register_unavailable(
                extensions,
                requires=requires,
                description=description,
                unavailable_reason=reason or "dependency unavailable",
            )

    return registry


def read_image(path: str | Path, registry: ReaderRegistry | None = None) -> ReadResult:
    """Read an image file using the supplied/default registry."""
    registry = registry or build_default_registry()
    return registry.read(path)


def read_workflow_plane(path: Path) -> ReadResult | None:
    """Read a single image/NPY plane with legacy metadata, or defer to its registered reader."""
    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return _read_standard_image(path, require_plane=True)
    if suffix == ".npy":
        return _read_numpy_or_text(path, require_plane=True)
    return None


def _collect_series_paths(
    source: str | Path | Sequence[str | Path],
    registry: ReaderRegistry,
    recursive: bool,
) -> list[Path]:
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if source_path.is_dir():
            pattern = "**/*" if recursive else "*"
            supported = set(registry.list_supported_formats(only_available=True))
            return sorted(
                path
                for path in source_path.glob(pattern)
                if path.is_file() and path.suffix.lower() in supported
            )
        return [source_path]

    return sorted(Path(item) for item in source)


def read_series(
    source: str | Path | Sequence[str | Path],
    registry: ReaderRegistry | None = None,
    recursive: bool = False,
    skip_errors: bool = False,
) -> list[ReadResult]:
    """Read a directory or path sequence into a list of `ReadResult`."""
    registry = registry or build_default_registry()
    results: list[ReadResult] = []
    paths = _collect_series_paths(source, registry, recursive=recursive)

    for path in paths:
        try:
            results.append(registry.read(path))
        except Exception:
            if not skip_errors:
                raise
    return results


def read_series_as_temdataset(
    source: str | Path | Sequence[str | Path],
    registry: ReaderRegistry | None = None,
    recursive: bool = False,
    skip_errors: bool = False,
    dataset_cls: Callable[..., Any] | None = None,
) -> list[Any]:
    """Read a series and convert each item to `TEMDataset`."""
    from .converters import to_temdataset

    results = read_series(
        source,
        registry=registry,
        recursive=recursive,
        skip_errors=skip_errors,
    )
    return [to_temdataset(result, dataset_cls=dataset_cls) for result in results]
