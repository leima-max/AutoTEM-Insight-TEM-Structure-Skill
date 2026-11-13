"""Run-local adapters for real input selection, quality, calibration and preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
from pydantic import ConfigDict, Field, ValidationError, model_validator

from autotem.calibration.basic import calibrate_dataset
from autotem.calibration.pixel_calibration import infer_pixel_size_from_scalebar
from autotem.core import ArrayReference, Measurement, TEMDataset
from autotem.core.models import Modality
from autotem.harness.contracts import (
    HarnessError,
    NeedsInput,
    OperationContext,
    OperationDefinition,
    ServiceResult,
    StrictParameters,
)
from autotem.io.metadata import normalize_metadata
from autotem.io.readers import (
    _to_jsonable,
    read_numeric_file,
    require_spatial_indices,
    select_image_page,
)
from autotem.preprocessing.background import subtract_background
from autotem.preprocessing.basic import assess_quality, preprocess_dataset
from autotem.preprocessing.denoise import apply_denoise
from autotem.preprocessing.quality import load_dataset_array


class Parameters(StrictParameters):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class InputParameters(Parameters):
    modality: Modality = "unknown"
    dataset_path: str | None = None
    dataset_index: int | None = Field(default=None, ge=0)
    frame: int | None = Field(default=None, ge=0)
    channel: int | None = Field(default=None, ge=0)
    selection_indices: list[int] | None = None
    roi: list[int] | None = Field(
        default=None, min_length=4, max_length=4, description="[y0,y1,x0,x1] exclusive upper bounds"
    )


class QualityThresholds(Parameters):
    snr_min: float | None = Field(default=None, ge=-300, le=300)
    drift_px_max: float | None = Field(default=None, gt=0)
    blur_risk_max: float | None = Field(default=None, ge=0, le=1)
    contamination_risk_max: float | None = Field(default=None, ge=0, le=1)
    astigmatism_risk_max: float | None = Field(default=None, ge=0, le=1)
    undersampling_risk_max: float | None = Field(default=None, ge=0, le=1)
    overall_score_min: float | None = Field(default=None, ge=0, le=1)


class QualityWeights(Parameters):
    snr: float | None = Field(default=None, ge=0)
    drift: float | None = Field(default=None, ge=0)
    blur: float | None = Field(default=None, ge=0)
    contamination: float | None = Field(default=None, ge=0)
    astigmatism: float | None = Field(default=None, ge=0)
    undersampling: float | None = Field(default=None, ge=0)
    metadata: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def nonzero_total(self):
        values = self.model_dump(exclude_none=True)
        if len(values) == 7 and sum(values.values()) == 0:
            raise ValueError("At least one quality metric weight must be positive")
        return self


class QualityParameters(Parameters):
    thresholds: QualityThresholds = Field(default_factory=QualityThresholds)
    metric_weights: QualityWeights = Field(default_factory=QualityWeights)
    thresholds_by_modality: dict[Modality, QualityThresholds] = Field(default_factory=dict)
    metric_weights_by_modality: dict[Modality, QualityWeights] = Field(default_factory=dict)


class ReferencePair(Parameters):
    known_d_spacing: float = Field(gt=0)
    measured_radius_px: float = Field(gt=0)
    known_d_spacing_unit: Literal["nm", "pm", "um", "angstrom"] = "nm"
    known_d_uncertainty_nm: float | None = Field(default=None, ge=0)
    measured_radius_uncertainty_px: float | None = Field(default=None, ge=0)
    label: str = "reference"


class CalibrationParameters(Parameters):
    pixel_size_nm: float | None = Field(default=None, gt=0)
    pixel_size_uncertainty_nm: float | None = Field(default=None, ge=0)
    diffraction_scale_inv_nm_per_px: float | None = Field(default=None, gt=0)
    known_d_spacing: float | None = Field(default=None, gt=0)
    known_d_spacing_unit: Literal["nm", "pm", "um", "angstrom"] = "nm"
    measured_radius_px: float | None = Field(default=None, gt=0)
    known_d_uncertainty_nm: float | None = Field(default=None, ge=0)
    measured_radius_uncertainty_px: float | None = Field(default=None, ge=0)
    known_d_spacing_refs: list[ReferencePair] = Field(default_factory=list, max_length=128)
    multi_reference_outlier_sigma: float = Field(default=3.5, gt=0)
    multi_reference_min_inliers: int = Field(default=2, ge=1, le=128)
    voltage_kv: float | None = Field(default=None, gt=0)
    detector_pixel_size_um: float | None = Field(default=None, gt=0)
    reference_length: float | None = Field(default=None, gt=0)
    reference_pixels: float | None = Field(default=None, gt=0)
    reference_unit: Literal["nm", "pm", "um", "angstrom"] = "nm"
    require_physical: bool = False

    @model_validator(mode="after")
    def paired_references(self):
        for first, second in (
            ("known_d_spacing", "measured_radius_px"),
            ("reference_length", "reference_pixels"),
        ):
            if (getattr(self, first) is None) != (getattr(self, second) is None):
                raise ValueError(f"{first} and {second} must be supplied together")
        return self


class ScalebarParameters(Parameters):
    reference_length: float = Field(gt=0)
    reference_unit: Literal["nm", "pm", "um", "angstrom"] = "nm"
    roi: list[int] | None = Field(
        default=None, min_length=4, max_length=4, description="[y0,y1,x0,x1] exclusive upper bounds"
    )


class PreprocessingParameters(Parameters):
    enabled: bool = True
    recipe: Literal["auto", "manual"] = "auto"
    clip_percentile_low: float = Field(default=1.0, ge=0, le=100)
    clip_percentile_high: float = Field(default=99.0, ge=0, le=100)
    denoise_method: Literal["auto", "none", "gaussian", "median", "wiener", "nlm"] = "auto"
    denoise_sigma: float = Field(default=0.8, ge=0)
    median_size: int = Field(default=3, ge=1, le=101)
    wiener_window: int = Field(default=5, ge=1, le=101)
    wiener_noise: float | None = Field(default=None, ge=0)
    nlm_h: float | None = Field(default=None, gt=0)
    nlm_patch_size: int = Field(default=5, ge=1, le=31)
    nlm_patch_distance: int = Field(default=6, ge=1, le=64)
    background_method: Literal["none", "gaussian", "rolling_ball"] = "none"
    background_sigma: float = Field(default=12.0, gt=0)
    background_opening_size: int = Field(default=31, ge=1, le=301)
    background_clip_zero: bool = False

    @model_validator(mode="after")
    def ordered_percentiles(self):
        if self.clip_percentile_low >= self.clip_percentile_high:
            raise ValueError("clip_percentile_low must be less than clip_percentile_high")
        return self


def validated(cls: type[Parameters], params: dict) -> Parameters:
    try:
        return cls.model_validate(params)
    except ValidationError as exc:
        raise HarnessError(
            "INVALID_PARAMETERS", str(exc), exc.errors(include_context=False)
        ) from exc


def image_array(array: Any, role: str = "raw", *, nonnegative: bool = False) -> np.ndarray:
    result = np.asarray(array)
    if result.ndim != 2 or not result.size or min(result.shape) < 2:
        raise HarnessError("INVALID_IMAGE", f"{role} requires a nonempty 2D image of at least 2x2")
    if result.dtype.kind not in "uif" or not np.isfinite(result).all():
        raise HarnessError("INVALID_IMAGE", f"{role} requires finite real numeric pixels")
    if nonnegative and np.any(result < 0):
        raise HarnessError("INVALID_IMAGE", f"{role} requires nonnegative intensities")
    return result


def _selection(array: Any, p: InputParameters) -> np.ndarray:
    array = np.asarray(array)
    if p.channel is not None:
        raise HarnessError("INVALID_SELECTION", "channel applies only to channel images")
    if p.selection_indices is not None and p.frame is not None:
        raise HarnessError("INVALID_SELECTION", "Use either frame or selection_indices")
    indices = p.selection_indices
    if p.frame is not None:
        if array.ndim != 3:
            raise HarnessError("INVALID_SELECTION", "frame requires a three-dimensional stack")
        indices = [p.frame]
    if indices is None:
        require_spatial_indices(array.shape)
    else:
        if len(indices) != max(0, array.ndim - 2):
            raise HarnessError("INVALID_SELECTION", "Selection must leave exactly two dimensions")
        if any(index < 0 or index >= array.shape[axis] for axis, index in enumerate(indices)):
            raise HarnessError("INVALID_SELECTION", "Selection index is out of bounds")
        array = array[tuple(indices)]
    return image_array(array)


def _read_selected(path: Path, p: InputParameters) -> tuple[np.ndarray, dict, dict]:
    """Readers never choose a first dataset/frame/channel on the user's behalf."""
    suffix = path.suffix.lower()
    metadata: dict = {}
    selection: dict = {}
    if p.dataset_path == "":
        raise HarnessError("INVALID_SELECTION", "Choose a named dataset")
    if p.dataset_path is not None and suffix not in {".h5", ".hdf5", ".hdf", ".emd", ".npz"}:
        raise HarnessError("INVALID_SELECTION", "dataset_path applies only to HDF/EMD/NPZ")
    if p.dataset_index is not None and suffix not in {".dm3", ".dm4", ".ser"}:
        raise HarnessError("INVALID_SELECTION", "dataset_index applies only to DM or SER inputs")
    if suffix in {".h5", ".hdf5", ".hdf", ".emd"}:
        try:
            import h5py
        except ImportError as exc:
            raise HarnessError("DEPENDENCY_UNAVAILABLE", "HDF/EMD input requires h5py") from exc
        with h5py.File(path, "r") as handle:
            choices = []
            handle.visititems(
                lambda name, item: (
                    choices.append(name)
                    if isinstance(item, h5py.Dataset)
                    and item.ndim >= 2
                    and item.dtype.kind in "uif"
                    else None
                )
            )
            if not choices:
                raise HarnessError("INVALID_INPUT", "No numeric image datasets in HDF file")
            selected = p.dataset_path
            if selected is None and len(choices) > 1:
                raise NeedsInput(
                    [{"name": "dataset_path", "choices": choices}],
                    "Select an HDF dataset explicitly",
                )
            selected = selected or choices[0]
            if selected.lstrip("/") not in choices:
                raise HarnessError("INVALID_SELECTION", "HDF dataset_path is not a numeric image")
            item = handle[selected]
            selection = {"dataset_path": selected, "source_shape": list(item.shape)}
            metadata = {**_to_jsonable(dict(handle.attrs)), **_to_jsonable(dict(item.attrs))}
            # Index on disk before materializing large multidimensional data.
            leading = (
                p.selection_indices
                if p.selection_indices is not None
                else ([p.frame] if p.frame is not None else None)
            )
            if item.ndim > 2 and leading is None:
                raise NeedsInput(
                    [
                        {
                            "name": "selection_indices",
                            "shape": list(item.shape),
                            "count": item.ndim - 2,
                        }
                    ],
                    "Select HDF spatial frame explicitly",
                )
            if p.channel is not None or p.dataset_index is not None:
                raise HarnessError(
                    "INVALID_SELECTION", "Use dataset_path and selection_indices for HDF"
                )
            if p.frame is not None and (item.ndim != 3 or p.selection_indices is not None):
                raise HarnessError(
                    "INVALID_SELECTION", "frame requires a 3D dataset without indices"
                )
            if leading is not None:
                if len(leading) != item.ndim - 2 or any(
                    v < 0 or v >= item.shape[i] for i, v in enumerate(leading)
                ):
                    raise HarnessError("INVALID_SELECTION", "HDF indices must leave exactly 2D")
            return (
                image_array(item[tuple(leading or []) + (slice(None), slice(None))]),
                metadata,
                selection,
            )
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as handle:
            choices = list(handle.files)
            if p.dataset_path is None and len(choices) > 1:
                raise NeedsInput(
                    [{"name": "dataset_path", "choices": choices}],
                    "Select an NPZ dataset explicitly",
                )
            selected = p.dataset_path or (choices[0] if choices else "")
            if selected not in choices:
                raise HarnessError("INVALID_SELECTION", "Unknown NPZ dataset_path")
            if p.frame is None and p.selection_indices is None:
                # NpzFile.__getitem__ decompresses the entire member. Numeric NPY v1/v2
                # headers already contain the dimensions needed for the next question.
                with handle.zip.open(selected + ".npy") as member:
                    version = np.lib.format.read_magic(member)
                    reader = {
                        (1, 0): np.lib.format.read_array_header_1_0,
                        (2, 0): np.lib.format.read_array_header_2_0,
                    }.get(version)
                    if reader is not None:
                        shape, _, dtype = reader(member)
                        if dtype.kind not in "uif":
                            raise HarnessError("INVALID_IMAGE", "NPZ requires real numeric pixels")
                        require_spatial_indices(shape)
            array = handle[selected]
            selection["dataset_path"] = selected
    elif suffix in {".npy", ".csv", ".txt"}:
        array = read_numeric_file(path, mmap_mode="r")
    elif suffix in {".dm3", ".dm4"}:
        try:
            from ncempy.io import dm
        except ImportError as exc:
            raise HarnessError("DEPENDENCY_UNAVAILABLE", "DM input requires ncempy") from exc
        with dm.fileDM(str(path), on_memory=True) as handle:
            count = 1 if handle.numObjects == 1 else handle.numObjects - 1
            if p.dataset_index is None and count > 1:
                raise NeedsInput(
                    [{"name": "dataset_index", "count": count}],
                    "Select a DM dataset explicitly (thumbnail excluded)",
                )
            index = p.dataset_index or 0
            if index >= count:
                raise HarnessError("INVALID_SELECTION", "DM dataset index is out of bounds")
            payload = handle.getDataset(index)
            array = np.asarray(payload["data"])
            metadata = _to_jsonable({k: v for k, v in payload.items() if k != "data"})
            extra = handle.getMetadata(index)
            if isinstance(extra, dict):
                metadata.update(_to_jsonable(extra))
            selection["dataset_index"] = index
    elif suffix in {".mrc", ".mrcs"}:
        try:
            import mrcfile
        except ImportError as exc:
            raise HarnessError("DEPENDENCY_UNAVAILABLE", "MRC input requires mrcfile") from exc
        with mrcfile.mmap(path, permissive=False) as handle:
            selection["source_shape"] = list(handle.data.shape)
            metadata = {
                "pixel_size_x_angstrom": float(handle.voxel_size.x),
                "pixel_size_y_angstrom": float(handle.voxel_size.y),
            }
            return _selection(handle.data, p).copy(), metadata, selection
    elif suffix == ".ser":
        try:
            from ncempy.io import ser
        except ImportError as exc:
            raise HarnessError("DEPENDENCY_UNAVAILABLE", "SER input requires ncempy") from exc
        if p.dataset_path or p.channel is not None or p.selection_indices is not None:
            raise HarnessError("INVALID_SELECTION", "SER images use dataset_index or frame")
        if p.dataset_index is not None and p.frame is not None:
            raise HarnessError("INVALID_SELECTION", "Use either dataset_index or frame for SER")
        with ser.fileSER(str(path)) as handle:
            count = int(handle.head["ValidNumberElements"])
            index = p.dataset_index if p.dataset_index is not None else p.frame
            if index is None and count > 1:
                raise NeedsInput(
                    [{"name": "dataset_index", "count": count}], "Select a SER dataset explicitly"
                )
            index = index if index is not None else 0
            if index < 0 or index >= count:
                raise HarnessError("INVALID_SELECTION", "SER dataset index is out of bounds")
            array, metadata_raw = handle.getDataset(index)
            metadata = _to_jsonable(metadata_raw or {})
            selection = {"dataset_index": index, "source_shape": list(array.shape)}
            return image_array(array).copy(), metadata, selection
    elif suffix == ".emi":
        raise NeedsInput(
            [{"name": "input_path", "format": "SER"}],
            "EMI is acquisition metadata; select its associated SER image file",
        )
    elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}:
        with Image.open(path) as handle:
            frame = select_image_page(handle, p.frame, p.channel)
            if p.selection_indices is not None:
                raise HarnessError(
                    "INVALID_SELECTION", "Image input accepts frame/channel selection"
                )
            array = np.asarray(handle)
            metadata = _to_jsonable(dict(handle.info))
            if hasattr(handle, "tag_v2"):
                metadata.update(_to_jsonable(dict(handle.tag_v2)))
            selection = {"frame": frame, "source_shape": list(array.shape)}
            if p.channel is not None:
                array = array[..., p.channel]
            return image_array(array), metadata, selection
    else:
        raise HarnessError("UNSUPPORTED_FORMAT", f"Unsupported input extension: {suffix}")
    selection["source_shape"] = list(array.shape)
    return _selection(array, p), metadata, selection


def read_selected_input(
    path: Path, params: dict, *, dataset_id: str
) -> tuple[np.ndarray, TEMDataset]:
    """Read one explicit 2D selection. The caller assigns its persisted array reference."""
    p = validated(InputParameters, params)
    if not path.is_file():
        raise HarnessError("INPUT_NOT_FOUND", "Input snapshot is not a regular file")
    try:
        array, metadata, selection = _read_selected(path, p)
    except HarnessError:
        raise
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise HarnessError("INVALID_INPUT", f"Cannot read selected input: {exc}") from exc
    if p.modality in {"SAED", "NBED"}:
        array = image_array(array, nonnegative=True)
    selection["source_2d_shape"] = list(array.shape)
    y0 = x0 = 0
    if p.roi is not None:
        y0, y1, x0, x1 = p.roi
        if not (0 <= y0 < y1 <= array.shape[0] and 0 <= x0 < x1 <= array.shape[1]):
            raise HarnessError("INVALID_ROI", "Input ROI must lie within the selected 2D image")
        array = image_array(array[y0:y1, x0:x1], "raw_roi")
    selection.update(
        {
            "roi_origin_yx": [y0, x0],
            "selected_shape": list(array.shape),
            "coordinate_mapping": {
                "source_y": "selected_y + roi_origin_y",
                "source_x": "selected_x + roi_origin_x",
            },
        }
    )
    normalized, metadata_warnings = normalize_metadata(metadata, source_format=path.suffix)
    warnings = [str(item.get("message", item)) for item in metadata_warnings]
    # Keep multidimensional spatial calibration aligned with the two selected image axes.
    sizes, units = metadata.get("pixelSize"), metadata.get("pixelUnit")
    if isinstance(sizes, list) and len(sizes) >= 2 and isinstance(units, list):
        from autotem.calibration.pixel_calibration import coerce_length_to_nm

        spatial = [
            coerce_length_to_nm(value, unit)
            for value, unit in zip(sizes[-2:], units[-2:], strict=False)
        ]
        if len(spatial) == 2 and all(value is not None and value > 0 for value in spatial):
            if np.isclose(spatial[0], spatial[1], rtol=1e-6):
                normalized["pixel_size_nm"] = spatial[0]
            else:
                normalized["pixel_size_nm"] = None
                normalized["anisotropic_pixel_size_nm"] = spatial
                warnings.append(
                    "Anisotropic spatial calibration requires per-axis support; "
                    "scalar scale withheld."
                )
    mx, my = metadata.get("pixel_size_x_angstrom"), metadata.get("pixel_size_y_angstrom")
    if mx and my and not np.isclose(mx, my, rtol=1e-6):
        normalized["pixel_size_nm"] = None
        normalized["anisotropic_pixel_size_nm"] = [my * 0.1, mx * 0.1]
        warnings.append(
            "Anisotropic MRC calibration requires per-axis support; scalar scale withheld."
        )
    pixel_nm = normalized.get("pixel_size_nm")
    pixel = None
    if isinstance(pixel_nm, (int, float)) and np.isfinite(pixel_nm) and pixel_nm > 0:
        pixel = Measurement(value=float(pixel_nm), unit="nm/pixel", method="reader.metadata")
    dataset = TEMDataset(
        data=ArrayReference(shape=array.shape, dtype=str(array.dtype)),
        dataset_id=dataset_id,
        source_path=str(path),
        modality=p.modality,
        pixel_size=pixel,
        pixel_unit="nm" if pixel else "unknown",
        metadata={
            "raw": metadata,
            "normalized": normalized,
            "selection": {**selection, **p.model_dump(exclude_none=True)},
        },
        provenance={"method": "input.load", "source_path": str(path)},
        warnings=warnings,
    )
    return array, dataset


def load_input(ctx: OperationContext, params: dict) -> ServiceResult:
    supplied = {**ctx.request.get("selection", {}), **params}
    supplied.setdefault("modality", ctx.request.get("modality", "unknown"))
    path = Path(ctx.input_path)
    try:
        array, dataset = read_selected_input(path, supplied, dataset_id=f"{ctx.run_id}:selected")
    except NeedsInput as exc:
        fields = exc.details["fields"]
        evidence_id = ctx.emit_json(
            "input_selection_options",
            {
                "reason": str(exc),
                "fields": [dict(field) for field in fields],
                "source_format": path.suffix.lower(),
            },
            {"scientific_level": "diagnostic"},
        )
        for field in fields:
            field["evidence"] = {"selection_artifact_id": evidence_id}
        raise
    selection = dataset.metadata["selection"]
    raw = ctx.emit_array("raw", array, {"selection": selection, "source_path": str(path)})
    ctx.emit_image(
        "raw_preview",
        array,
        {
            "display_only": True,
            "selection": selection,
            "display_transform": "linear_minmax_to_uint8",
        },
    )
    dataset.data = raw
    dataset.provenance["method"] = "harness.input.load"
    outputs = {"raw": raw, "dataset": dataset}
    if dataset.modality in {"SAED", "NBED"}:
        outputs["reciprocal_image"] = ctx.emit_array("reciprocal_image", array)
        outputs["reciprocal_context"] = {"kind": "direct_diffraction", "shape": list(array.shape)}
    return ServiceResult(outputs=outputs, warnings=dataset.warnings)


def assess(ctx: OperationContext, params: dict) -> ServiceResult:
    p = validated(QualityParameters, params)
    image_array(ctx.array("raw"))
    dataset = ctx.model("dataset", TEMDataset)
    quality = assess_quality(dataset, parameters=p.model_dump(exclude_none=True))
    return ServiceResult(outputs={"quality": quality}, warnings=quality.warnings)


def scalebar_candidate(ctx: OperationContext, params: dict) -> ServiceResult:
    p = validated(ScalebarParameters, params)
    raw = image_array(ctx.array("raw"))
    crop = raw
    if p.roi is not None:
        y0, y1, x0, x1 = p.roi
        if not (0 <= x0 < x1 <= raw.shape[1] and 0 <= y0 < y1 <= raw.shape[0]):
            raise HarnessError(
                "INVALID_ROI", "Scale-bar ROI must lie within the selected raw image"
            )
        crop = image_array(raw[y0:y1, x0:x1], "scalebar_roi")
    reference = ctx.emit_array("scalebar_roi", crop, {"roi": p.roi})
    # Supplied text reaches the existing detector before OCR/filename fallback is considered.
    pixel, source, warnings = infer_pixel_size_from_scalebar(
        crop,
        metadata={"explicit_reference": f"{p.reference_length} {p.reference_unit}"},
    )
    if pixel is None or not np.isfinite(pixel) or pixel <= 0:
        raise NeedsInput(
            [{"name": "reference_pixels", "alternative": "roi"}],
            "Scale-bar detection was inconclusive; provide manual endpoints/length",
        )
    candidate = {
        "pixel_size_nm": pixel,
        "source": source,
        "roi": p.roi,
        "reference_length": p.reference_length,
        "reference_unit": p.reference_unit,
        "requires_review": True,
    }
    warnings.append(
        "Scale-bar length is a candidate; confirm endpoints before calibration.resolve."
    )
    return ServiceResult(
        outputs={"scalebar_candidate": candidate, "scalebar_roi": reference},
        warnings=warnings,
        scientific_level="candidate",
    )


def resolve_calibration(ctx: OperationContext, params: dict) -> ServiceResult:
    p = validated(CalibrationParameters, params)
    image_array(ctx.array("raw"))
    dataset = ctx.model("dataset", TEMDataset)
    values = p.model_dump(exclude_none=True, exclude={"require_physical"})
    reference = {
        key: values[key]
        for key in ("reference_length", "reference_pixels", "reference_unit")
        if key in values
    }
    # The legacy resolver may infer a scale bar from image pixels or a filename. The Harness
    # accepts supplied references/metadata only; unknown calibration remains unknown.
    calibration_input = dataset.model_copy(
        update={
            "data": ArrayReference(shape=dataset.data.shape, dtype=dataset.data.dtype),
            "source_path": "",
        },
        deep=True,
    )
    result = calibrate_dataset(calibration_input, reference=reference, parameters=values)
    if (
        result.transform.get("diffraction_source") == "pixel_size_fft_fallback"
        and len(dataset.data.shape) == 2
        and dataset.data.shape[0] != dataset.data.shape[1]
        and result.pixel_size is not None
    ):
        height, width = dataset.data.shape
        pixel = result.pixel_size.value
        result.diffraction_scale = None
        result.transform["fft_per_axis_inv_nm_per_px"] = {
            "x": 1 / (width * pixel),
            "y": 1 / (height * pixel),
        }
        result.warnings.append(
            "Rectangular FFT requires per-axis scales; scalar diffraction scale withheld."
        )
    # A direct diffraction pattern cannot use a real-space FFT pixel-size fallback.
    if (
        dataset.modality in {"SAED", "NBED"}
        and result.transform.get("diffraction_source") == "pixel_size_fft_fallback"
    ):
        result.diffraction_scale = None
        result.warnings.append(
            "Direct diffraction requires detector/reference calibration; FFT fallback withheld."
        )
    for measurement in (result.pixel_size, result.diffraction_scale, result.voltage):
        if measurement is not None and (
            not isinstance(measurement.value, (float, int))
            or not np.isfinite(measurement.value)
            or measurement.value <= 0
        ):
            raise HarnessError(
                "INVALID_CALIBRATION", "Calibration returned an invalid physical value"
            )
    physical = (
        result.diffraction_scale if dataset.modality in {"SAED", "NBED"} else result.pixel_size
    )
    if p.require_physical and physical is None:
        field = (
            "diffraction_scale_inv_nm_per_px"
            if dataset.modality in {"SAED", "NBED"}
            else "pixel_size_nm"
        )
        raise NeedsInput(
            [{"name": field, "exclusive_with": "known_d_spacing/reference_length"}],
            "Provide physical calibration before quantitative analysis",
        )
    return ServiceResult(
        outputs={"calibration": result},
        warnings=result.warnings,
        scientific_level="quantitative" if physical else "diagnostic",
    )


def preprocess(ctx: OperationContext, params: dict) -> ServiceResult:
    p = validated(PreprocessingParameters, params)
    raw = image_array(ctx.array("raw"))
    dataset = ctx.model("dataset", TEMDataset).model_copy(deep=True)
    values = p.model_dump(exclude={"recipe"}, exclude_none=True)
    result = preprocess_dataset(
        dataset, recipe=p.recipe, parameters=values, output_dir=ctx.output_dir / "intermediates"
    )
    if result.image is None:
        raise HarnessError("PREPROCESSING_FAILED", "Preprocessing did not produce an image")
    processed = load_dataset_array(dataset.model_copy(update={"data": result.image}))
    if processed is None:
        raise HarnessError("PREPROCESSING_FAILED", "Full processed array is unavailable")
    result.image = ctx.emit_array(
        "processed", image_array(processed, "processed"), {"parameters": result.parameters}
    )
    ctx.emit_image(
        "processed_preview",
        processed,
        {
            "display_only": True,
            "display_transform": "linear_minmax_to_uint8",
            "parameters": result.parameters,
        },
    )
    if result.background is not None:
        if result.background.inline is not None:
            background = np.asarray(result.background.inline)
        else:
            # The legacy wrapper omits large background pixels; reproduce using its saved values
            # and existing deterministic adapters, then register the full array.
            clipped = np.clip(
                raw, result.parameters["clip_value_low"], result.parameters["clip_value_high"]
            )
            span = float(np.ptp(clipped))
            normalized = (clipped - np.min(clipped)) / span if span > 1e-12 else np.zeros_like(raw)
            denoised, _, _, _ = apply_denoise(
                normalized.astype(np.float32),
                method=p.denoise_method,
                sigma=p.denoise_sigma,
                median_size=p.median_size,
                wiener_window=p.wiener_window,
                wiener_noise=p.wiener_noise,
                nlm_h=p.nlm_h,
                nlm_patch_size=p.nlm_patch_size,
                nlm_patch_distance=p.nlm_patch_distance,
            )
            _, background, _, _, _ = subtract_background(
                denoised,
                method=p.background_method,
                gaussian_sigma=p.background_sigma,
                opening_size=p.background_opening_size,
                clip_to_zero=p.background_clip_zero,
            )
        result.background = ctx.emit_array("background", image_array(background, "background"))
    return ServiceResult(
        outputs={"processed": result.image, "preprocessing": result}, warnings=result.warnings
    )


def definitions() -> list[OperationDefinition]:
    return [
        OperationDefinition(
            "input.load",
            "data-quality",
            load_input,
            InputParameters,
            output_roles=("raw", "dataset"),
            description="Read an explicitly selected real image into run-local storage",
        ),
        OperationDefinition(
            "quality.assess",
            "data-quality",
            assess,
            QualityParameters,
            input_roles=("raw", "dataset"),
            output_roles=("quality",),
        ),
        OperationDefinition(
            "calibration.scalebar_candidate",
            "data-quality",
            scalebar_candidate,
            ScalebarParameters,
            input_roles=("raw", "dataset"),
            output_roles=("scalebar_candidate", "scalebar_roi"),
            result_level="candidate",
            scientific_limitations=("Endpoint confirmation required",),
        ),
        OperationDefinition(
            "calibration.resolve",
            "data-quality",
            resolve_calibration,
            CalibrationParameters,
            input_roles=("raw", "dataset"),
            output_roles=("calibration",),
        ),
        OperationDefinition(
            "preprocessing.apply",
            "data-quality",
            preprocess,
            PreprocessingParameters,
            input_roles=("raw", "dataset"),
            output_roles=("processed", "preprocessing"),
        ),
    ]
