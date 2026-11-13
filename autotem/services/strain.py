"""Complete-array, diagnostic services for existing GPA and atom-strain algorithms."""

from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
from importlib.util import find_spec
from typing import Literal

import numpy as np
from pydantic import Field

from autotem.calibration.pixel_calibration import convert_length
from autotem.core.models import CalibrationResult, DiffractionResult
from autotem.harness.contracts import (
    HarnessError,
    NeedsInput,
    OperationContext,
    OperationDefinition,
    ServiceResult,
)
from autotem.services.atomic import (
    CoordinatesParameters,
    CParameters,
    ask_atoms,
    check_bounds,
    checked_call,
    clean_json,
    coordinates,
    image_array,
    parameters,
    region_mask,
)
from autotem.strain.atom_strain import compute_atom_strain_fields
from autotem.strain.gpa import (
    InsufficientGPAReference,
    _resolve_g_vectors,
    _validate_g_vector_pair,
    compute_gpa_fields,
    validate_gpa_references,
)
from autotem.strain.gpa_alignment import GPAAlignmentConfig, compare_gpa_configs

LIMITATION = (
    "Diagnostic strain only: existing numerical checks do not establish admission for "
    "quantitative scientific use; uncertainty is heuristic"
)


class GPAParameters(CParameters):
    image_role: Literal["raw", "processed"] = "raw"
    g_vectors: list[list[float]] | None = None
    g_vector_unit: Literal["1/pixel", "fft_pixel", "1/nm"] = "1/pixel"
    pixel_size_nm: float | None = Field(default=None, gt=0)
    reference_region: list[int] | None = None
    reference_regions: list[list[int]] | None = None
    absolute_reference: bool = False
    mask_sigma_px: float = Field(default=2.5, gt=0, le=1000)
    hann_window: bool = True
    refine_reference: bool = True
    min_quality_fraction: float = Field(default=0.08, ge=0, le=1)
    quality_percentile: float = Field(default=10.0, ge=0, le=100)
    apply_quality_mask: bool = True


class CompareParameters(CParameters):
    reference_config: GPAParameters
    candidate_config: GPAParameters
    sample_step: int = Field(default=1, ge=1, le=10000)


class AtomStrainParameters(CoordinatesParameters):
    reference_region: list[int] | None = None
    reference_atom_indices: list[int] | None = None
    neighbor_k: int = Field(default=10, ge=4, le=1000)
    direction_tolerance_deg: float = Field(default=18.0, gt=0, lt=90)
    interpolation_grid_step: int = Field(default=1, ge=1, le=1000)


def checked_region(region: list[int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    region_mask(np.empty((0, 2)), region, shape)
    y0, y1, x0, x1 = region
    if min(y1 - y0, x1 - x0) < 3:
        raise HarnessError("INVALID_REFERENCE", "GPA reference ROI must span at least 3x3 pixels")
    return y0, y1, x0, x1


def ask_g_vectors(ctx, image, image_role, reason):
    """Present measured FFT candidates for a human choice, without selecting a pair."""
    from PIL import Image, ImageDraw

    from autotem.diffraction.fft import compute_fft_analysis
    from autotem.diffraction.peak_detection import detect_bragg_peaks

    fft, _ = compute_fft_analysis(image, center_mode="geometric")
    height, width = image.shape
    # fftshift DC is at the integer midpoint, including for even, rectangular images.
    cx, cy = width // 2, height // 2
    peaks, _, warnings = detect_bragg_peaks(fft.log_magnitude, center=(cx, cy), top_k=16)
    intensity = fft.log_magnitude
    span = float(np.ptp(intensity))
    display = (
        np.uint8(255 * (intensity - intensity.min()) / span)
        if span
        else np.zeros_like(image, dtype=np.uint8)
    )
    canvas = Image.fromarray(display).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    rows = []
    for index, peak in enumerate(peaks):
        rows.append(
            {
                "peak_index": index,
                "x_px": peak.x,
                "y_px": peak.y,
                "gx_cycles_per_pixel": (peak.x - cx) / width,
                "gy_cycles_per_pixel": (peak.y - cy) / height,
            }
        )
        draw.text((peak.x + 1, peak.y + 1), str(index), fill=(255, 100, 40))
    meta = {
        "image_role": image_role,
        "dc_xy_px": [cx, cy],
        "coordinate_unit": "1/pixel",
        "scientific_level": "candidate",
        "method": "detect_bragg_peaks",
        "top_k": 16,
        "warnings": warnings,
    }
    evidence = {"image_artifact_id": ctx.emit_image("numbered_g_peaks", np.asarray(canvas), meta)}
    if rows:
        evidence["peak_table_artifact_id"] = ctx.emit_table("g_peak_selection", rows, meta)
    raise NeedsInput(
        [
            {
                "name": "g_vectors",
                "type": "array",
                "count": len(rows),
                "description": "Two reliable non-collinear (gx,gy) vectors with units, "
                "or two numbered FFT peaks when listed.",
                "evidence": evidence,
            }
        ],
        reason,
    )


def gpa_arguments(ctx: OperationContext, p: GPAParameters, image: np.ndarray) -> dict:
    if find_spec("skimage") is None:
        raise HarnessError(
            "DEPENDENCY_UNAVAILABLE", "GPA service requires masked 2D skimage unwrap"
        )
    arguments = p.model_dump(exclude={"image_role", "absolute_reference"})
    try:
        from skimage.restoration import unwrap_phase
    except Exception as exc:
        raise HarnessError("DEPENDENCY_BROKEN", "Masked 2D unwrap could not be imported") from exc
    if not callable(unwrap_phase):
        raise HarnessError("DEPENDENCY_BROKEN", "Masked 2D unwrap is unavailable")
    physical_g = p.g_vector_unit == "1/nm"
    if p.g_vectors is None:
        try:
            diffraction = ctx.model("diffraction", DiffractionResult)
        except HarnessError as exc:
            if exc.code != "MISSING_ARTIFACT":
                raise
            ask_g_vectors(ctx, image, p.image_role, "GPA needs a reliable non-collinear Bragg pair")
        arguments["diffraction_result"] = diffraction
        for vector in diffraction.g_vectors[:2]:
            if vector.gx.unit != vector.gy.unit:
                raise HarnessError("INVALID_UNIT", "GPA vector components must use the same unit")
            if isinstance(vector.gx.value, bool) or isinstance(vector.gy.value, bool):
                raise HarnessError(
                    "INVALID_UNIT", "GPA vector components must be numeric frequencies"
                )
            unit = vector.gx.unit.lower().replace(" ", "")
            physical_g |= unit in {"1/nm", "nm^-1", "cycles/nm"}
    elif np.asarray(p.g_vectors).shape != (2, 2):
        raise HarnessError("INVALID_PARAMETERS", "g_vectors must be exactly two (gx,gy) pairs")
    if physical_g and p.pixel_size_nm is None:
        try:
            calibration = ctx.model("calibration", CalibrationResult)
            measure = calibration.pixel_size
            if measure is not None:
                unit = measure.unit.removesuffix("/pixel").removesuffix("/px")
                pixel_nm = convert_length(measure.value, unit, "nm")
                if not np.isfinite(pixel_nm) or pixel_nm <= 0:
                    raise ValueError("Invalid physical calibration")
                arguments["pixel_size_nm"] = pixel_nm
        except HarnessError as exc:
            if exc.code != "MISSING_ARTIFACT":
                raise
        except (TypeError, ValueError) as exc:
            raise HarnessError(
                "INVALID_UNIT", "Physical GPA requires genuine pixel calibration"
            ) from exc
        if arguments["pixel_size_nm"] is None:
            raise NeedsInput(
                [{"name": "pixel_size_nm", "type": "number", "exclusiveMinimum": 0}],
                "Physical g vectors require actual pixel size to convert to image frequencies",
            )
    try:
        vectors, _, _ = _resolve_g_vectors(
            image.shape,
            diffraction_result=arguments.get("diffraction_result"),
            g_vectors=p.g_vectors,
            g_vector_unit=p.g_vector_unit,
            pixel_size_nm=arguments["pixel_size_nm"],
            warnings=[],
        )
        _validate_g_vector_pair(vectors)
    except (ValueError, TypeError) as exc:
        if p.g_vectors is None:
            ask_g_vectors(
                ctx, image, p.image_role, f"Automatic GPA vector evidence is insufficient: {exc}"
            )
        raise HarnessError("INVALID_SCIENTIFIC_INPUT", str(exc)) from exc
    if p.reference_region is not None and p.reference_regions is not None:
        raise HarnessError("INVALID_REFERENCE", "Choose one common ROI or two per-g ROIs")
    if p.reference_regions is not None:
        if len(p.reference_regions) != 2:
            raise HarnessError("INVALID_REFERENCE", "reference_regions requires two ROIs")
        arguments["reference_regions"] = tuple(
            checked_region(r, image.shape) for r in p.reference_regions
        )
    elif p.reference_region is not None:
        arguments["reference_region"] = checked_region(p.reference_region, image.shape)
    elif not p.absolute_reference:
        raise NeedsInput(
            [
                {
                    "name": "reference_region",
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "Clean reference [y0,y1,x0,x1]",
                }
            ],
            "Choose a reliable reference ROI or explicitly accept absolute g-vector reference",
        )
    return arguments


def persist_gpa(ctx: OperationContext, result, role: str = "gpa"):
    output = result.to_strain_result()
    unit_by_name = {
        "displacement_x": "pixel",
        "displacement_y": "pixel",
        "rotation_xy": "rad",
        "quality_mask": "boolean",
        "uncertainty": "heuristic",
    }
    refs = {}
    validity_hash = sha256(np.asarray(result.quality_mask, dtype=bool).tobytes()).hexdigest()
    for name in (
        "displacement_x",
        "displacement_y",
        "strain_xx",
        "strain_yy",
        "strain_xy",
        "rotation_xy",
        "uncertainty",
        "quality_mask",
    ):
        refs[name] = ctx.emit_array(
            f"{role}.{name}",
            getattr(result, name),
            metadata={
                "unit": unit_by_name.get(name, "dimensionless"),
                "scientific_level": "diagnostic",
                "validity_role": f"{role}.quality_mask",
                "validity_sha256": validity_hash,
            },
        )
        if name != "quality_mask":
            setattr(output, name, refs[name])
    phase_maps = {}
    for index, phase in enumerate(result.phase_maps, 1):
        payload = {}
        for item in fields(phase):
            value = getattr(phase, item.name)
            if isinstance(value, np.ndarray):
                ref = ctx.emit_array(
                    f"{role}.g{index}.{item.name}",
                    value,
                    metadata={
                        "unit": "rad"
                        if "phase" in item.name
                        else "dimensionless"
                        if item.name == "gaussian_mask"
                        else "a.u.",
                    },
                )
                payload[item.name] = ref.model_dump(mode="json")
            else:
                payload[item.name] = clean_json(value)
        phase_maps[f"g{index}"] = payload
    output.provenance.update(
        {
            "phase_maps": phase_maps,
            "quality_mask": refs["quality_mask"].model_dump(mode="json"),
            "scientific_limitation": LIMITATION,
            "complete_arrays": True,
            "units": {"displacement": "pixel", "strain": "dimensionless", "rotation": "rad"},
        }
    )
    output.warnings.append(LIMITATION)
    return output


def emit_strain_montage(ctx, role, result, *, quality_mask=None, coordinates_yx=None):
    """Render a display-only view; authoritative fields remain registered NPY arrays."""
    from matplotlib.figure import Figure

    is_gpa = quality_mask is not None
    figure = Figure(figsize=(10, 8), constrained_layout=True)
    axes = figure.subplots(2, 2)
    entries = (
        ("strain_xx" if is_gpa else "strain_xx_map", "exx", 100.0, "%"),
        ("strain_yy" if is_gpa else "strain_yy_map", "eyy", 100.0, "%"),
        ("strain_xy" if is_gpa else "strain_xy_map", "exy", 100.0, "%"),
        ("rotation_xy" if is_gpa else "rotation_xy_map", "rotation", 1.0, "rad"),
    )
    for axis, (name, label, scale, unit) in zip(axes.flat, entries, strict=True):
        axis.set_facecolor("#d0d0d0")
        values = np.asarray(getattr(result, name), dtype=float) * scale
        invalid = ~np.isfinite(values)
        if is_gpa:
            invalid |= ~quality_mask
        valid_values = np.abs(values[~invalid])
        limit = max(float(np.percentile(valid_values, 98)) if valid_values.size else 0.0, 1e-6)
        display = np.ma.array(values, mask=invalid)
        image = axis.imshow(display, cmap="coolwarm", vmin=-limit, vmax=limit)
        if coordinates_yx is not None:
            selected = coordinates_yx[result.valid_atom_mask]
            axis.scatter(selected[:, 1], selected[:, 0], s=7, facecolors="none", edgecolors="k")
        axis.set_title(f"{label} ({unit})")
        axis.set_xlabel("x (pixel)")
        axis.set_ylabel("y (pixel)")
        figure.colorbar(image, ax=axis, label=unit, shrink=0.85)
    semantics = "quality-masked diagnostic" if is_gpa else "display interpolation only"
    figure.suptitle(f"{role}: {semantics}")
    path = ctx.output_dir / f"{role}-maps.png"
    figure.savefig(path, dpi=140)
    validity = result.quality_mask if is_gpa else result.valid_atom_mask
    return ctx.emit_image(
        f"{role}_maps",
        path,
        metadata={
            "display_only": True,
            "semantics": semantics,
            "scientific_level": "diagnostic",
            "display_strain_unit": "%",
            "display_rotation_unit": "rad",
            "validity_sha256": sha256(np.asarray(validity, dtype=bool).tobytes()).hexdigest(),
            "invalid_display": "gray" if is_gpa else "only valid atoms outlined",
        },
    )


def ask_gpa_reference(ctx: OperationContext, error: InsufficientGPAReference):
    ctx.emit_array(
        "gpa_reference_quality", error.quality_mask, metadata={"scientific_level": "diagnostic"}
    )
    field = {"name": "reference_region", "type": "array"}
    if error.parameter_group:
        field["parameter_group"] = error.parameter_group
    raise NeedsInput([field], str(error)) from error


def gpa(ctx: OperationContext, params: dict) -> ServiceResult:
    p = parameters(GPAParameters, params)
    image = image_array(ctx, p.image_role)
    arguments = gpa_arguments(ctx, p, image)
    result = checked_call(compute_gpa_fields, image, **arguments)
    if not np.any(result.quality_mask):
        raise HarnessError("NO_VALID_STRAIN", "No image pixels support reliable Bragg phases")
    try:
        validate_gpa_references(result.quality_mask, result.reference_regions)
    except InsufficientGPAReference as exc:
        ask_gpa_reference(ctx, exc)
    output = persist_gpa(ctx, result)
    output.provenance["static_map_artifact_id"] = emit_strain_montage(
        ctx, "gpa", result, quality_mask=result.quality_mask
    )
    output.provenance["image_role"] = p.image_role
    output.provenance["absolute_reference_accepted"] = p.absolute_reference
    return ServiceResult({"gpa": output}, output.warnings)


def gpa_compare(ctx: OperationContext, params: dict) -> ServiceResult:
    p = parameters(CompareParameters, params)
    if p.reference_config.image_role != p.candidate_config.image_role:
        raise HarnessError("INVALID_PARAMETERS", "Compare configurations on the same image role")
    image = image_array(ctx, p.reference_config.image_role)
    configs = []
    for group in ("reference_config", "candidate_config"):
        config = getattr(p, group)
        try:
            arguments = gpa_arguments(ctx, config, image)
        except NeedsInput as exc:
            for field in exc.details["fields"]:
                field["parameter_group"] = group
            raise
        if config.g_vectors is None or config.reference_regions is not None:
            raise HarnessError(
                "INVALID_PARAMETERS", "Comparison requires explicit vectors and one common ROI"
            )
        arguments.pop("reference_regions", None)
        configs.append(GPAAlignmentConfig(**arguments))
    try:
        result = checked_call(compare_gpa_configs, image, *configs, sample_step=p.sample_step)
    except InsufficientGPAReference as exc:
        ask_gpa_reference(ctx, exc)
    result["scientific_limitation"] = (
        "Parameter comparison is not independent scientific validation"
    )
    warnings = [LIMITATION]
    if reason := result["diagnosis"].get("reason"):
        warnings.append(reason)
    return ServiceResult({"gpa_comparison": clean_json(result)}, warnings)


def atom_strain(ctx: OperationContext, params: dict) -> ServiceResult:
    p = parameters(AtomStrainParameters, params)
    raw = image_array(ctx)
    _, coords = coordinates(ctx, p.coordinate_role)
    check_bounds(coords, raw)
    if len(coords) < 6:
        raise HarnessError("INSUFFICIENT_ATOMS", "Atom strain requires at least six atom positions")
    mask = region_mask(coords, p.reference_region, raw.shape)
    image_mask = None
    if p.reference_atom_indices is not None:
        if mask is not None:
            raise HarnessError("INVALID_REFERENCE", "Choose ROI or reference atom indices")
        indices = p.reference_atom_indices
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(coords) for i in indices):
            raise HarnessError(
                "INVALID_REFERENCE", "Reference indices must identify distinct atoms"
            )
        mask = np.zeros(len(coords), bool)
        mask[indices] = True
        image_mask = np.zeros(raw.shape, bool)
        rounded = np.rint(coords[mask]).astype(int)
        if np.any(rounded[:, 0] >= raw.shape[0]) or np.any(rounded[:, 1] >= raw.shape[1]):
            raise HarnessError("INVALID_REFERENCE", "Rounded reference coordinates exceed image")
        image_mask[rounded[:, 0], rounded[:, 1]] = True
        effective = image_mask[
            np.rint(coords[:, 0]).astype(int).clip(0, raw.shape[0] - 1),
            np.rint(coords[:, 1]).astype(int).clip(0, raw.shape[1] - 1),
        ]
        if not np.array_equal(mask, effective):
            raise HarnessError(
                "INVALID_REFERENCE", "Selected reference atoms share image pixels with other atoms"
            )
    if mask is None or np.sum(mask) < 6:
        ask_atoms(
            ctx,
            coords,
            [
                {
                    "name": "reference_atom_indices",
                    "type": "array",
                    "minItems": 6,
                    "alternative": "reference_region",
                }
            ],
            "Choose at least six reference atoms; automatic whole-image fallback is disabled",
        )
    if np.linalg.matrix_rank(coords[mask] - np.mean(coords[mask], axis=0)) < 2:
        raise HarnessError(
            "INVALID_REFERENCE", "Reference atoms must span a two-dimensional lattice"
        )
    result = checked_call(
        compute_atom_strain_fields,
        coords,
        raw.shape,
        reference_region=tuple(p.reference_region) if p.reference_region else None,
        reference_mask=image_mask,
        neighbor_k=p.neighbor_k,
        direction_tolerance_deg=p.direction_tolerance_deg,
        interpolation_grid_step=p.interpolation_grid_step,
    )
    if not np.any(result.valid_atom_mask):
        raise HarnessError(
            "NO_VALID_STRAIN",
            "No atoms support a valid local affine fit",
            {"warnings": result.warnings},
        )
    output = result.to_strain_result()
    refs = {}
    coordinate_hash = sha256(np.asarray(result.coordinates_yx, dtype="<f8").tobytes()).hexdigest()
    validity_hash = sha256(np.asarray(result.valid_atom_mask, dtype=bool).tobytes()).hexdigest()
    map_names = {
        "displacement_x_map": "displacement_x",
        "displacement_y_map": "displacement_y",
        "strain_xx_map": "strain_xx",
        "strain_yy_map": "strain_yy",
        "strain_xy_map": "strain_xy",
        "rotation_xy_map": "rotation_xy",
        "uncertainty_map": "uncertainty",
    }
    for item in fields(result):
        value = getattr(result, item.name)
        if isinstance(value, np.ndarray):
            semantics = (
                "display_interpolation" if item.name in map_names else "per_atom_or_reference"
            )
            ref = ctx.emit_array(
                f"atom_strain.{item.name}",
                value,
                metadata={
                    "semantics": semantics,
                    "validity_role": "atom_strain.valid_atom_mask",
                    "scientific_level": "diagnostic",
                    "coordinates_sha256": coordinate_hash,
                    "coordinate_convention": "yx/pixel",
                    "validity_sha256": validity_hash,
                },
            )
            refs[item.name] = ref.model_dump(mode="json")
            if item.name in map_names:
                setattr(output, map_names[item.name], ref)
    ctx.emit_table(
        "atom_strain_values",
        [
            {
                "atom_index": i,
                "y_px": float(y),
                "x_px": float(x),
                "valid": bool(result.valid_atom_mask[i]),
                "exx": float(result.atom_strain_xx[i]) if result.valid_atom_mask[i] else None,
                "eyy": float(result.atom_strain_yy[i]) if result.valid_atom_mask[i] else None,
                "exy": float(result.atom_strain_xy[i]) if result.valid_atom_mask[i] else None,
            }
            for i, (y, x) in enumerate(coords)
        ],
    )
    output.provenance.update(
        {
            "array_references": refs,
            "complete_arrays": True,
            "map_semantics": "display_interpolation_including_nearest_extrapolation",
            "authoritative_values": "per-atom arrays filtered by valid_atom_mask",
            "scientific_limitation": LIMITATION,
            "coordinates_sha256": coordinate_hash,
            "validity_sha256": validity_hash,
            "units": {"displacement": "pixel", "strain": "dimensionless", "rotation": "rad"},
        }
    )
    output.warnings.extend(
        [LIMITATION, "Full image maps are display interpolation, not valid strain at every pixel"]
    )
    output.provenance["static_map_artifact_id"] = emit_strain_montage(
        ctx, "atom_strain", result, coordinates_yx=result.coordinates_yx
    )
    return ServiceResult({"atom_strain": output}, output.warnings)


def definitions() -> list[OperationDefinition]:
    entries = [
        ("strain.gpa", gpa, GPAParameters, ("raw",), ("gpa",)),
        ("strain.gpa_compare", gpa_compare, CompareParameters, ("raw",), ("gpa_comparison",)),
        (
            "strain.atom",
            atom_strain,
            AtomStrainParameters,
            ("raw", "refined_atoms"),
            ("atom_strain",),
        ),
    ]
    return [
        OperationDefinition(
            operation_id=op,
            profile_id="atomic-strain",
            handler=handler,
            parameter_model=model,
            input_roles=inputs,
            output_roles=outputs,
            dependencies=("numpy", "scipy", "matplotlib", "PIL")
            if op == "strain.atom"
            else ("numpy", "scipy", "skimage", "matplotlib", "PIL"),
            scientific_limitations=(LIMITATION,),
            description="Complete-array diagnostic strain service",
            modality=("HRTEM", "STEM", "HAADF-STEM", "ABF-STEM", "TEM", "unknown"),
        )
        for op, handler, model, inputs, outputs in entries
    ]
