"""Run-local services for existing traditional atomic-structure algorithms."""

from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
from typing import Any, Literal

import numpy as np
from pydantic import ConfigDict, Field, ValidationError

from autotem.atomic.atom_detection import detect_atoms
from autotem.atomic.displacement import compute_atomic_displacement_from_basis
from autotem.atomic.distance_statistics import compute_distance_statistics
from autotem.atomic.gaussian_refinement import refine_atom_coordinates
from autotem.atomic.intensity_quantification import (
    AtomicColumnIntensityResult,
    cluster_atomic_column_intensities,
    group_atomic_column_intensities_by_lattice,
    quantify_atomic_column_intensity,
)
from autotem.atomic.lattice_constraint import apply_lattice_constraint
from autotem.atomic.plane_spacing import (
    compute_atomic_plane_spacing_segments,
    compute_atomic_plane_spacing_segments_by_angle,
    compute_atomic_plane_spacing_segments_from_basis,
    compute_atomic_plane_spacing_segments_from_lattice_angle,
    compute_atomic_plane_spacing_segments_from_picked_direction,
)
from autotem.atomic.sublattice import (
    SublatticeClassificationResult,
    apply_manual_sublattice_labels,
    classify_sublattices_auto,
    classify_sublattices_from_model,
)
from autotem.calibration.pixel_calibration import convert_length
from autotem.core.models import AtomLattice, CalibrationResult
from autotem.harness.contracts import (
    HarnessError,
    NeedsInput,
    OperationContext,
    OperationDefinition,
    ServiceResult,
    StrictParameters,
)


class CParameters(StrictParameters):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class CoordinatesParameters(CParameters):
    coordinate_role: Literal["atoms", "refined_atoms", "regularized_atoms"] = "refined_atoms"


class DetectParameters(CParameters):
    image_role: Literal["raw", "processed"] = "raw"
    mode: Literal["auto", "log", "dog"] = "auto"
    gaussian_sigma: float | None = Field(default=None, ge=0, le=100)
    min_distance: int = Field(default=4, ge=1, le=10000)
    threshold_rel: float = Field(default=0.25, gt=0, le=1)
    max_atoms: int = Field(default=5000, ge=1, le=50000)


class RefineParameters(CParameters):
    method: Literal["gaussian", "com"] = "gaussian"
    window_radius: int = Field(default=3, ge=1, le=100)
    max_shift: float = Field(default=1.5, gt=0, le=100)
    max_iter: int = Field(default=200, ge=1, le=10000)


class DistanceParameters(CoordinatesParameters):
    pixel_size_nm: float | None = Field(default=None, gt=0)
    output_unit: Literal["pixel", "nm", "pm", "angstrom", "um"] = "pixel"


class RegularizeParameters(CoordinatesParameters):
    enabled: bool = False
    k_neighbors: int = Field(default=6, ge=2, le=1000)
    strength: float = Field(default=0.2, ge=0, le=1)
    iterations: int = Field(default=1, ge=1, le=100)
    max_shift_per_iter: float = Field(default=0.6, gt=0, le=100)


class BasisParameters(CoordinatesParameters):
    base_index: int | None = Field(default=None, ge=0)
    a_index: int | None = Field(default=None, ge=0)
    b_index: int | None = Field(default=None, ge=0)


class DisplacementParameters(BasisParameters):
    reference_atom_indices: list[int] | None = None
    reference_region: list[int] | None = None
    lattice_index_tolerance: float = Field(default=0.40, ge=0.05, le=1)
    manual_lattice_indices: dict[str, list[int]] | None = None


class IndexedSpacingParameters(CoordinatesParameters):
    lattice_role: Literal["atomic_displacement", "sublattices"] = "atomic_displacement"
    directions: list[Literal["a", "b"]] = Field(default_factory=lambda: ["a", "b"])


class BasisSpacingParameters(BasisParameters):
    lattice_index_tolerance: float = Field(default=0.40, ge=0.05, le=1)
    directions: list[Literal["a", "b"]] = Field(default_factory=lambda: ["a", "b"])


class DirectionSpacingParameters(CoordinatesParameters):
    angle_tolerance_deg: float = Field(default=10.0, ge=0.1, le=45)
    min_spacing_px: float = Field(default=1.0, gt=0)
    max_spacing_px: float = Field(default=30.0, gt=0)


class LatticeAngleParameters(DirectionSpacingParameters):
    lattice_angle_deg: float | None = Field(default=None, ge=5, le=175)


class AngleParameters(DirectionSpacingParameters):
    angle_deg: float | None = None
    direction_label: str | None = None


class PickedParameters(DirectionSpacingParameters):
    start_index: int | None = Field(default=None, ge=0)
    end_index: int | None = Field(default=None, ge=0)


class SublatticeParameters(BasisParameters):
    n_sublattices: int = Field(default=2, ge=1, le=8)
    use_intensity: bool = True
    intensity_weight: float = Field(default=0.6, ge=0, le=10)
    manual_labels: dict[str, str] | None = None


class BasisSite(CParameters):
    u: float = Field(ge=0, le=1)
    v: float = Field(ge=0, le=1)
    label: str = Field(min_length=1, max_length=100)


class ModelSublatticeParameters(BasisParameters):
    basis_sites: list[BasisSite] | None = None
    match_tolerance: float = Field(default=0.20, ge=0.01, le=0.75)
    manual_labels: dict[str, str] | None = None


class ManualSublatticeParameters(CParameters):
    manual_labels: dict[str, str] = Field(min_length=1)


class IntensityParameters(CoordinatesParameters):
    integration_radius: float = Field(default=3.0, gt=0, le=1000)
    background_inner_radius: float = Field(default=5.0, gt=0, le=2000)
    background_outer_radius: float = Field(default=8.0, gt=0, le=3000)
    background_statistic: Literal["median", "mean", "percentile_10", "percentile_25"] = "median"
    normalization: Literal["none", "median", "mean"] = "median"
    saturation_threshold: float | None = None


class IntensityGroupParameters(CParameters):
    grouping: Literal["cluster", "lattice"] = "cluster"
    metric: Literal[
        "normalized_intensity",
        "background_subtracted_sum",
        "peak_intensity",
        "snr",
    ] = "normalized_intensity"
    n_groups: int = Field(default=2, ge=1, le=8)
    lattice_role: Literal["atomic_displacement", "sublattices"] = "atomic_displacement"
    mode: Literal["ij_parity", "i_parity", "j_parity", "sum_parity"] = "ij_parity"


def parameters(cls: type[CParameters], params: dict) -> CParameters:
    try:
        return cls.model_validate(params)
    except ValidationError as exc:
        raise HarnessError("INVALID_PARAMETERS", str(exc)) from exc


def image_array(ctx: OperationContext, role: str = "raw") -> np.ndarray:
    image = np.asarray(ctx.array(role))
    if image.ndim != 2 or min(image.shape) < 2 or image.dtype.kind not in "fiu":
        raise HarnessError("INVALID_ARRAY", "Expected a numeric two-dimensional image")
    if not np.isfinite(image).all():
        raise HarnessError("INVALID_ARRAY", "Image values must all be finite")
    return image


def coordinates(
    ctx: OperationContext, role: str = "refined_atoms"
) -> tuple[AtomLattice, np.ndarray]:
    lattice = ctx.model(role, AtomLattice)
    if lattice.coordinate_unit != "pixel":
        raise HarnessError("INVALID_UNIT", "Atom coordinates must be actual image pixels")
    coords = np.asarray([[a.y, a.x] for a in lattice.atoms], dtype=float).reshape(-1, 2)
    if not np.isfinite(coords).all():
        raise HarnessError("INVALID_COORDINATES", "Atom coordinates must be finite")
    if np.any(coords < 0):
        raise HarnessError("INVALID_COORDINATES", "Pixel coordinates must be nonnegative")
    if len(coords) and len(np.unique(coords, axis=0)) != len(coords):
        raise HarnessError("INVALID_COORDINATES", "Duplicate atom coordinates are not supported")
    return lattice, coords


def check_bounds(coords: np.ndarray, image: np.ndarray) -> None:
    if len(coords) and (
        np.any(coords < 0)
        or np.any(coords[:, 0] >= image.shape[0])
        or np.any(coords[:, 1] >= image.shape[1])
    ):
        raise HarnessError("INVALID_COORDINATES", "Atom coordinates lie outside the image")


def checked_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        raise HarnessError("INVALID_SCIENTIFIC_INPUT", str(exc)) from exc


def clean_json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def save_dataclass(
    ctx: OperationContext,
    role: str,
    result,
    *,
    inline_arrays: bool = True,
    metadata: dict | None = None,
    field_metadata: dict[str, dict] | None = None,
) -> dict:
    """Persist arrays; atom-sized inline copies keep explicit old revisions self-contained."""
    metadata = dict(metadata or {})
    for name in ("coordinates", "coordinates_yx", "coordinates_observed_yx", "coordinates_input"):
        if hasattr(result, name):
            coords = np.asarray(getattr(result, name), dtype="<f8")
            metadata["coordinates_sha256"] = sha256(coords.tobytes()).hexdigest()
            metadata["coordinate_convention"] = "yx/pixel"
            break
    payload: dict = {"array_references": {}, "metadata": metadata}
    for item in fields(result):
        value = getattr(result, item.name)
        if isinstance(value, np.ndarray):
            array = value.astype(str) if value.dtype.kind == "O" else value
            ref = ctx.emit_array(
                f"{role}.{item.name}",
                array,
                metadata={**metadata, **(field_metadata or {}).get(item.name, {})},
            )
            payload["array_references"][item.name] = ref.model_dump(mode="json")
            if inline_arrays:
                payload[item.name] = clean_json(array.tolist())
        else:
            payload[item.name] = clean_json(value)
    return payload


def restore_dataclass(payload: dict, cls, ctx: OperationContext):
    kwargs = {}
    for item in fields(cls):
        reference = payload.get("array_references", {}).get(item.name)
        if reference is not None:
            value = ctx.resolve_array(reference)
            kwargs[item.name] = value
            continue
        if item.name not in payload:
            raise HarnessError("INVALID_ARTIFACT", f"Missing atomic field: {item.name}")
        value = payload[item.name]
        kwargs[item.name] = value
    return cls(**kwargs)


def ask_atoms(ctx: OperationContext, coords: np.ndarray, requested: list[dict], message: str):
    rows = [
        {"atom_index": i, "y_px": float(y), "x_px": float(x)} for i, (y, x) in enumerate(coords)
    ]
    table = ctx.emit_table("atom_selection", rows, metadata={"coordinate_convention": "yx"})
    evidence = {"table_artifact_id": table}
    try:
        from PIL import Image, ImageDraw

        raw = image_array(ctx)
        low, high = np.percentile(raw, [1, 99])
        pixels = np.uint8(np.clip((raw - low) / max(float(high - low), 1e-12), 0, 1) * 255)
        canvas = Image.fromarray(pixels).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for i, (y, x) in enumerate(coords[:200]):
            draw.text((float(x) + 2, float(y) + 2), str(i), fill=(255, 80, 20))
        evidence["image_artifact_id"] = ctx.emit_image(
            "numbered_atoms", np.asarray(canvas), metadata={"labels_shown": min(len(coords), 200)}
        )
    except HarnessError:
        pass
    for field in requested:
        field["evidence"] = evidence
    raise NeedsInput(requested, message)


def basis(ctx: OperationContext, coords: np.ndarray, params: BasisParameters) -> dict:
    names = ("base_index", "a_index", "b_index")
    missing = [name for name in names if getattr(params, name) is None]
    if missing:
        ask_atoms(
            ctx,
            coords,
            [
                {"name": name, "type": "integer", "minimum": 0, "maximum": len(coords) - 1}
                for name in missing
            ],
            "Select three distinct, non-collinear atoms defining the projected basis",
        )
    result = {name: getattr(params, name) for name in names}
    if len(coords) < 3 or any(i >= len(coords) for i in result.values()):
        raise HarnessError("INVALID_COORDINATES", "Basis indices do not reference available atoms")
    points = coords[list(result.values())]
    if len(set(result.values())) != 3 or abs(np.linalg.det((points[1:] - points[0]).T)) < 1e-9:
        raise HarnessError("INVALID_COORDINATES", "Basis atoms must be distinct and non-collinear")
    return result


def region_mask(coords: np.ndarray, region: list[int] | None, shape=None) -> np.ndarray | None:
    if region is None:
        return None
    if len(region) != 4:
        raise HarnessError("INVALID_REFERENCE", "ROI order must be [y0,y1,x0,x1]")
    y0, y1, x0, x1 = region
    if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
        raise HarnessError("INVALID_REFERENCE", "ROI must be a nonempty positive image region")
    if shape is not None and (y1 > shape[0] or x1 > shape[1]):
        raise HarnessError("INVALID_REFERENCE", "ROI exceeds the actual image shape")
    return (coords[:, 0] >= y0) & (coords[:, 0] < y1) & (coords[:, 1] >= x0) & (coords[:, 1] < x1)


def index_mapping(mapping: dict | None, count: int, *, pairs: bool = False) -> dict | None:
    if mapping is None:
        return None
    result = {}
    for key, value in mapping.items():
        if not key.isdecimal() or not 0 <= int(key) < count:
            raise HarnessError("INVALID_COORDINATES", "Manual atom indices are out of bounds")
        if pairs and len(value) != 2:
            raise HarnessError("INVALID_PARAMETERS", "Manual lattice indices must be integer pairs")
        result[int(key)] = tuple(value) if pairs else value
    return result


def updated_lattice(lattice: AtomLattice, coords: np.ndarray, result, method: str) -> AtomLattice:
    atoms = [
        a.model_copy(update={"y": float(y), "x": float(x)})
        for a, (y, x) in zip(lattice.atoms, coords, strict=True)
    ]
    return lattice.model_copy(
        update={
            "atoms": atoms,
            "coordinate_unit": "pixel",
            "confidence": result.confidence,
            "warnings": [*lattice.warnings, *result.warnings],
            "provenance": {
                **lattice.provenance,
                method: result.provenance,
                "coordinates_sha256": sha256(np.asarray(coords, dtype="<f8").tobytes()).hexdigest(),
            },
        }
    )


def detect(ctx: OperationContext, params: dict) -> ServiceResult:
    p = parameters(DetectParameters, params)
    kwargs = p.model_dump(exclude={"image_role"})
    result = checked_call(
        detect_atoms,
        image_array(ctx, p.image_role),
        overlay_path=ctx.output_dir / "atom-candidates.png",
        **kwargs,
    )
    lattice = result.to_atom_lattice("pixel")
    lattice.overlay = None
    lattice.provenance["image_role"] = p.image_role
    lattice.provenance["overlay_artifact_id"] = ctx.emit_image(
        "atom_candidates_overlay",
        result.overlay_path,
        metadata={
            "image_role": p.image_role,
            "coordinates_sha256": sha256(
                np.asarray(result.coordinates, dtype="<f8").tobytes()
            ).hexdigest(),
            "display_only": True,
            "scientific_level": "candidate",
        },
    )
    ctx.emit_table(
        "atom_candidates",
        [{"atom_index": i, "y_px": a.y, "x_px": a.x} for i, a in enumerate(lattice.atoms)],
    )
    return ServiceResult({"atoms": lattice}, result.warnings, "candidate")


def refine(ctx: OperationContext, params: dict) -> ServiceResult:
    p = parameters(RefineParameters, params)
    raw = image_array(ctx)
    lattice, coords = coordinates(ctx, "atoms")
    check_bounds(coords, raw)
    result = checked_call(refine_atom_coordinates, raw, coords, **p.model_dump())
    payload = save_dataclass(
        ctx, "atomic_refinement", result, metadata={"unit": "pixel", "image_role": "raw"}
    )
    from autotem.services.atomic_overlay import save_refinement_overlay

    overlay_path = ctx.output_dir / "raw-refined-atoms.png"
    display = save_refinement_overlay(
        raw, result.coordinates_refined, overlay_path, result.provenance.get("point_status")
    )
    refined = updated_lattice(lattice, result.coordinates_refined, result, "raw_refinement")
    refined.overlay = None
    refined.provenance["overlay_artifact_id"] = ctx.emit_image(
        "refined_atoms_overlay",
        overlay_path,
        metadata={
            "image_role": "raw",
            "coordinates_sha256": refined.provenance["coordinates_sha256"],
            **display,
            "display_only": True,
        },
    )
    return ServiceResult(
        {
            "refined_atoms": refined,
            "atomic_refinement": payload,
        },
        result.warnings,
    )


def distances(ctx: OperationContext, params: dict) -> ServiceResult:
    p = parameters(DistanceParameters, params)
    _, coords = coordinates(ctx, p.coordinate_role)
    scale = None
    if p.output_unit != "pixel":
        pixel_nm = p.pixel_size_nm
        if pixel_nm is None:
            try:
                calibration = ctx.model("calibration", CalibrationResult)
                measurement = calibration.pixel_size
                if measurement is not None:
                    unit = measurement.unit.removesuffix("/pixel").removesuffix("/px")
                    pixel_nm = convert_length(measurement.value, unit, "nm")
            except HarnessError as exc:
                if exc.code != "MISSING_ARTIFACT":
                    raise
            except (ValueError, TypeError) as exc:
                raise HarnessError(
                    "INVALID_UNIT", "Calibration requires a real length/pixel unit"
                ) from exc
        if pixel_nm is None:
            raise NeedsInput(
                [{"name": "pixel_size_nm", "type": "number", "exclusiveMinimum": 0}],
                "Physical distances require a reliable pixel calibration",
            )
        if not np.isfinite(pixel_nm) or pixel_nm <= 0:
            raise HarnessError("INVALID_UNIT", "Pixel calibration must be finite and positive")
        scale = convert_length(pixel_nm, "nm", p.output_unit)
    elif p.pixel_size_nm is not None:
        raise HarnessError(
            "INVALID_PARAMETERS", "Select a physical output_unit to apply calibration"
        )
    result = checked_call(
        compute_distance_statistics,
        coords,
        pixel_size=scale,
        output_unit=p.output_unit if scale is not None else None,
    )
    payload = save_dataclass(
        ctx,
        "atom_distances",
        result,
        metadata={"coordinate_unit": "pixel", "distance_unit": result.unit},
    )
    ctx.emit_table(
        "atom_distance_table", clean_json(result.to_rows()), metadata={"unit": result.unit}
    )
    return ServiceResult(
        {"atom_distances": payload},
        result.warnings,
        "calibrated_measurement" if scale is not None else "diagnostic",
    )


def regularize(ctx: OperationContext, params: dict) -> ServiceResult:
    p = parameters(RegularizeParameters, params)
    lattice, coords = coordinates(ctx, p.coordinate_role)
    result = checked_call(
        apply_lattice_constraint, coords, **p.model_dump(exclude={"coordinate_role"})
    )
    return ServiceResult(
        {
            "regularized_atoms": updated_lattice(
                lattice, result.coordinates_regularized, result, "lattice_regularization"
            ),
            "atomic_regularization": save_dataclass(ctx, "atomic_regularization", result),
        },
        [
            *result.warnings,
            "Regularized coordinates are a separate branch; retain raw-refined positions",
        ],
    )


def displacement(ctx: OperationContext, params: dict) -> ServiceResult:
    p = parameters(DisplacementParameters, params)
    _, coords = coordinates(ctx, p.coordinate_role)
    selected = basis(ctx, coords, p)
    mask = region_mask(coords, p.reference_region)
    if p.reference_atom_indices is not None:
        if len(set(p.reference_atom_indices)) != len(p.reference_atom_indices) or any(
            i < 0 or i >= len(coords) for i in p.reference_atom_indices
        ):
            raise HarnessError(
                "INVALID_REFERENCE", "Reference indices must be distinct available atoms"
            )
        if mask is not None:
            raise HarnessError(
                "INVALID_REFERENCE", "Choose reference_region or reference_atom_indices"
            )
        mask = np.zeros(len(coords), bool)
        mask[p.reference_atom_indices] = True
    if mask is None or np.sum(mask) < 3:
        ask_atoms(
            ctx,
            coords,
            [{"name": "reference_atom_indices", "type": "array", "minItems": 3}],
            "Select at least three reference atoms; fallback is disabled",
        )
    result = checked_call(
        compute_atomic_displacement_from_basis,
        coords,
        **selected,
        reference_mask=mask,
        lattice_index_tolerance=p.lattice_index_tolerance,
        manual_lattice_indices=index_mapping(p.manual_lattice_indices, len(coords), pairs=True),
        allow_reference_fallback=False,
    )
    payload = save_dataclass(ctx, "atomic_displacement", result, metadata={"unit": "pixel"})
    return ServiceResult({"atomic_displacement": payload}, result.warnings)


def lattice_arrays(
    ctx: OperationContext, role: str, coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    payload = ctx.value(role)
    coordinate_field = (
        "coordinates_observed_yx" if "coordinates_observed_yx" in payload else "coordinates_yx"
    )

    def stored_array(name):
        ref = payload.get("array_references", {}).get(name)
        return ctx.resolve_array(ref) if ref is not None else np.asarray(payload.get(name))

    stored = stored_array(coordinate_field)
    source_hash = payload.get("metadata", {}).get("source_coordinates_sha256")
    same_revision = (
        source_hash == sha256(np.asarray(coords, dtype="<f8").tobytes()).hexdigest()
        if source_hash is not None
        else np.array_equal(np.asarray(stored), coords)
    )
    if np.asarray(stored).shape != coords.shape or not same_revision:
        raise HarnessError(
            "MIXED_COORDINATE_REVISION", "Lattice indices belong to different atom coordinates"
        )
    indices = stored_array("lattice_indices_ij")
    mask = np.asarray(stored_array("valid_atom_mask"), dtype=bool)
    if (
        indices.shape != coords.shape
        or indices.dtype.kind not in "iu"
        or mask.shape != (len(coords),)
    ):
        raise HarnessError("INVALID_ARTIFACT", "Lattice indices and valid mask must match atoms")
    return indices, mask


def spacing(ctx: OperationContext, params: dict, variant: str) -> ServiceResult:
    models = {
        "indexed": IndexedSpacingParameters,
        "basis": BasisSpacingParameters,
        "lattice_angle": LatticeAngleParameters,
        "angle": AngleParameters,
        "picked": PickedParameters,
    }
    p = parameters(models[variant], params)
    _, coords = coordinates(ctx, p.coordinate_role)
    kwargs = p.model_dump(exclude={"coordinate_role"})
    if variant == "indexed":
        indices, mask = lattice_arrays(ctx, kwargs.pop("lattice_role"), coords)
        result = checked_call(
            compute_atomic_plane_spacing_segments,
            coords,
            indices,
            valid_atom_mask=mask,
            directions=tuple(kwargs["directions"]),
        )
    elif variant == "basis":
        kwargs.update(basis(ctx, coords, p))
        kwargs["directions"] = tuple(kwargs["directions"])
        result = checked_call(compute_atomic_plane_spacing_segments_from_basis, coords, **kwargs)
    else:
        required = {
            "lattice_angle": ["lattice_angle_deg"],
            "angle": ["angle_deg"],
            "picked": ["start_index", "end_index"],
        }[variant]
        missing = [key for key in required if kwargs[key] is None]
        if missing:
            ask_atoms(
                ctx,
                coords,
                [{"name": key, "type": "number"} for key in missing],
                "Supply a lattice angle, direction, or selected atom pair",
            )
        if p.max_spacing_px <= p.min_spacing_px:
            raise HarnessError("INVALID_PARAMETERS", "max_spacing_px must exceed min_spacing_px")
        fn = {
            "lattice_angle": compute_atomic_plane_spacing_segments_from_lattice_angle,
            "angle": compute_atomic_plane_spacing_segments_by_angle,
            "picked": compute_atomic_plane_spacing_segments_from_picked_direction,
        }[variant]
        result = checked_call(fn, coords, **kwargs)
    payload = save_dataclass(ctx, "atomic_spacing", result, metadata={"unit": "pixel"})
    return ServiceResult(
        {"atomic_spacing": payload},
        [
            *result.warnings,
            "These are projected atom-to-atom segments; assigning hkl plane spacings "
            "requires structure evidence",
        ],
    )


def sublattice(ctx: OperationContext, params: dict, variant: str) -> ServiceResult:
    cls = {
        "auto": SublatticeParameters,
        "model": ModelSublatticeParameters,
        "manual": ManualSublatticeParameters,
    }[variant]
    p = parameters(cls, params)
    if variant == "manual":
        prior_payload = ctx.value("sublattices")
        previous = restore_dataclass(prior_payload, SublatticeClassificationResult, ctx)
        source_hash = prior_payload.get("metadata", {}).get("source_coordinates_sha256")
        labels = index_mapping(p.manual_labels, len(previous.coordinates_yx))
        result = checked_call(apply_manual_sublattice_labels, previous, labels)
    else:
        lattice, coords = coordinates(ctx, p.coordinate_role)
        source_hash = sha256(np.asarray(coords, dtype="<f8").tobytes()).hexdigest()
        selected = basis(ctx, coords, p)
        intensities = np.asarray(
            [a.intensity.value if a.intensity is not None else np.nan for a in lattice.atoms],
            dtype=float,
        )
        kwargs = p.model_dump(exclude={"coordinate_role", "base_index", "a_index", "b_index"})
        kwargs["manual_labels"] = index_mapping(p.manual_labels, len(coords))
        if variant == "model" and not p.basis_sites:
            raise NeedsInput(
                [
                    {
                        "name": "basis_sites",
                        "type": "array",
                        "description": "Projected fractional u/v sites with labels",
                    }
                ],
                "Projected structure sites are required",
            )
        fn = classify_sublattices_auto if variant == "auto" else classify_sublattices_from_model
        result = checked_call(fn, coords, **selected, intensities=intensities, **kwargs)
    payload = save_dataclass(
        ctx, "sublattices", result, metadata={"source_coordinates_sha256": source_hash}
    )
    return ServiceResult(
        {"sublattices": payload},
        [
            *result.warnings,
            "Sublattice labels are geometric/intensity hypotheses, "
            "not verified elemental identities",
        ],
        "candidate",
    )


def intensity(ctx: OperationContext, params: dict) -> ServiceResult:
    p = parameters(IntensityParameters, params)
    raw = image_array(ctx)
    _, coords = coordinates(ctx, p.coordinate_role)
    check_bounds(coords, raw)
    result = checked_call(
        quantify_atomic_column_intensity, raw, coords, **p.model_dump(exclude={"coordinate_role"})
    )
    payload = save_dataclass(
        ctx,
        "atomic_intensity",
        result,
        metadata={"image_role": "raw", "unit": "a.u."},
        field_metadata={
            "coordinates_yx": {"unit": "pixel"},
            "atom_area_px": {"unit": "pixel^2"},
            "background_area_px": {"unit": "pixel^2"},
            "nearest_neighbor_distance_px": {"unit": "pixel"},
            "normalized_intensity": {"unit": "dimensionless"},
            "snr": {"unit": "dimensionless"},
            **{
                name: {"unit": "boolean"}
                for name in ("valid_atom_mask", "edge_flag", "overlap_flag", "saturation_flag")
            },
        },
    )
    trusted = (
        result.valid_atom_mask & ~result.edge_flag & ~result.overlap_flag & ~result.saturation_flag
    )
    payload["trusted_atom_mask"] = trusted.tolist()
    ctx.emit_array("atomic_intensity.trusted_atom_mask", trusted)
    return ServiceResult(
        {"atomic_intensity": payload},
        [
            *result.warnings,
            "Relative column intensities do not establish quantitative Z or composition",
        ],
    )


def intensity_groups(ctx: OperationContext, params: dict) -> ServiceResult:
    p = parameters(IntensityGroupParameters, params)
    payload = ctx.value("atomic_intensity")
    result = restore_dataclass(payload, AtomicColumnIntensityResult, ctx)
    if "trusted_atom_mask" in payload:
        trusted = np.asarray(payload["trusted_atom_mask"], dtype=bool)
        if trusted.shape != result.valid_atom_mask.shape:
            raise HarnessError("INVALID_ARTIFACT", "Trusted atom mask shape mismatch")
        result.valid_atom_mask = result.valid_atom_mask & trusted
    if p.grouping == "cluster":
        grouped = checked_call(
            cluster_atomic_column_intensities, result, metric=p.metric, n_groups=p.n_groups
        )
    else:
        indices, mask = lattice_arrays(ctx, p.lattice_role, result.coordinates_yx)
        grouped = checked_call(
            group_atomic_column_intensities_by_lattice,
            result,
            indices,
            metric=p.metric,
            mode=p.mode,
            valid_lattice_mask=mask,
        )
    return ServiceResult(
        {"intensity_groups": save_dataclass(ctx, "intensity_groups", grouped)},
        [*grouped.warnings, "Intensity groups are not quantitative chemical composition"],
        "candidate",
    )


def definitions() -> list[OperationDefinition]:
    entries = [
        ("atomic.detect", detect, DetectParameters, ("raw",), ("atoms",)),
        ("atomic.refine", refine, RefineParameters, ("raw", "atoms"), ("refined_atoms",)),
        (
            "atomic.distances",
            distances,
            DistanceParameters,
            ("refined_atoms",),
            ("atom_distances",),
        ),
        (
            "atomic.regularize",
            regularize,
            RegularizeParameters,
            ("refined_atoms",),
            ("regularized_atoms",),
        ),
        (
            "atomic.displacement",
            displacement,
            DisplacementParameters,
            ("refined_atoms",),
            ("atomic_displacement",),
        ),
        (
            "atomic.intensity",
            intensity,
            IntensityParameters,
            ("raw", "refined_atoms"),
            ("atomic_intensity",),
        ),
        (
            "atomic.intensity_groups",
            intensity_groups,
            IntensityGroupParameters,
            ("atomic_intensity",),
            ("intensity_groups",),
        ),
    ]
    spacing_models = {
        "indexed": IndexedSpacingParameters,
        "basis": BasisSpacingParameters,
        "lattice_angle": LatticeAngleParameters,
        "angle": AngleParameters,
        "picked": PickedParameters,
    }
    for variant, model in spacing_models.items():

        def handler(ctx, params, variant=variant):
            return spacing(ctx, params, variant)

        entries.append(
            (f"atomic.spacing.{variant}", handler, model, ("refined_atoms",), ("atomic_spacing",))
        )
    for variant, model in {
        "auto": SublatticeParameters,
        "model": ModelSublatticeParameters,
        "manual": ManualSublatticeParameters,
    }.items():

        def handler(ctx, params, variant=variant):
            return sublattice(ctx, params, variant)

        inputs = ("sublattices",) if variant == "manual" else ("refined_atoms",)
        entries.append((f"atomic.sublattice.{variant}", handler, model, inputs, ("sublattices",)))
    return [
        OperationDefinition(
            operation_id=op,
            profile_id="atomic-strain",
            handler=handler,
            parameter_model=model,
            input_roles=inputs,
            output_roles=outputs,
            dependencies=("numpy", "scipy", "PIL", "matplotlib")
            if op == "atomic.detect"
            else ("numpy", "scipy", "PIL")
            if op == "atomic.refine"
            else ("numpy", "scipy"),
            description="Traditional atomic algorithms with pixel-coordinate and evidence checks",
            scientific_limitations=(
                "No new scientific accuracy admission; labels remain evidence-bounded",
            ),
            modality=("HRTEM", "STEM", "HAADF-STEM", "ABF-STEM", "TEM", "unknown"),
        )
        for op, handler, model, inputs, outputs in entries
    ]
