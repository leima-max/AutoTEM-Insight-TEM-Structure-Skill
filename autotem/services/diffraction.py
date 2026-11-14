"""Independent, evidence-preserving adapters for implemented diffraction operations."""

from __future__ import annotations

import hashlib
import importlib
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from autotem.core import (
    CalibrationResult,
    DiffractionPeak,
    DiffractionResult,
    SimulationResult,
    TEMDataset,
)
from autotem.diffraction.b4_indexing import _phase_from_cif, run_b4_indexing
from autotem.diffraction.fft import compute_fft_analysis, locate_direct_beam
from autotem.diffraction.gvector import compute_gvectors_and_dspacing
from autotem.diffraction.peak_detection import detect_bragg_peaks, suggest_peak_parameter_defaults
from autotem.diffraction.phase_lookup import (
    lookup_phase_candidates,
    resolve_materials_project_api_key,
)
from autotem.diffraction.phase_matching import match_diffsims_patterns_to_diffraction
from autotem.harness.contracts import (
    HarnessError,
    NeedsInput,
    OperationContext,
    OperationDefinition,
    ServiceResult,
)
from autotem.services.data_quality import Parameters, image_array, validated
from autotem.simulation.basic import simulate_from_diffraction


class CenterParameters(Parameters):
    center_mode: Literal["auto", "geometric", "manual"] = "auto"
    center_x: float | None = None
    center_y: float | None = None
    search_radius_px: float | None = Field(default=None, gt=0)
    refine_window: int = Field(default=3, ge=1, le=51)

    @model_validator(mode="after")
    def manual_center(self):
        if self.center_mode == "manual" and (self.center_x is None or self.center_y is None):
            raise ValueError("manual center requires center_x and center_y")
        if self.center_mode != "manual" and (
            self.center_x is not None or self.center_y is not None
        ):
            raise ValueError("center_x/center_y require center_mode=manual")
        return self


class PeakParameters(Parameters):
    top_k: int = Field(default=32, ge=1, le=10000)
    min_distance: int = Field(default=3, ge=1, le=1000)
    center_exclusion_radius: float = Field(default=4.0, ge=0)
    max_peak_radius_px: float | None = Field(default=None, gt=0)
    radius_outlier_filter: bool = True
    radius_outlier_mad_factor: float = Field(default=4.0, gt=0)
    threshold_strategy: Literal["auto", "quantile", "std", "relative_max", "mad"] = "auto"
    threshold_value: float | None = None
    subpixel_window_radius: int = Field(default=2, ge=0, le=25)
    subpixel_max_shift: float = Field(default=1.5, ge=0)
    detector: Literal["local_max", "log", "dog"] = "local_max"
    detector_sigma: float = Field(default=1.2, gt=0)
    dog_sigma_ratio: float = Field(default=1.8, gt=1)

    @model_validator(mode="after")
    def threshold_range(self):
        value = self.threshold_value
        if value is not None and self.threshold_strategy in {"quantile", "relative_max"}:
            if not 0 <= value <= 1:
                raise ValueError("quantile/relative_max threshold must be in [0,1]")
        elif value is not None and value < 0:
            raise ValueError("threshold must be nonnegative")
        return self


class MetricParameters(Parameters):
    require_physical: bool = False
    center_uncertainty_px: float | None = Field(default=None, ge=0)


class ManualPeak(Parameters):
    x: float = Field(ge=0)
    y: float = Field(ge=0)


class ManualPeaksParameters(Parameters):
    peaks: list[ManualPeak] = Field(min_length=1, max_length=10000)


def _redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item, secret) for key, item in value.items()}
    return value


class AtomSiteParameters(Parameters):
    element: str = Field(min_length=1)
    abc: list[float] = Field(min_length=3, max_length=3)
    occupancy: float = Field(default=1.0, gt=0, le=1)
    source_index: int | None = Field(default=None, ge=0)


class PhaseParameters(Parameters):
    name: str = Field(min_length=1)
    a: float = Field(gt=0, description="Lattice length in nm")
    b: float = Field(gt=0, description="Lattice length in nm")
    c: float = Field(gt=0, description="Lattice length in nm")
    alpha: float = Field(default=90.0, gt=0, lt=180)
    beta: float = Field(default=90.0, gt=0, lt=180)
    gamma: float = Field(default=90.0, gt=0, lt=180)
    space_group: str = "P1"
    space_group_number: int | None = Field(default=None, ge=1, le=230)
    formula: str | None = None
    identifier: str | None = None
    source: str = "manual_local_parameters"
    atom_sites: list[AtomSiteParameters] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_cell(self):
        ca, cb, cg = np.cos(np.radians([self.alpha, self.beta, self.gamma]))
        if 1 + 2 * ca * cb * cg - ca**2 - cb**2 - cg**2 <= 1e-10:
            raise ValueError("Cell angles produce a singular or impossible lattice")
        return self


class IndexParameters(Parameters):
    candidate_phase_params: list[PhaseParameters] = Field(default_factory=list)
    cif_candidate_paths: list[str] = Field(default_factory=list, max_length=32)
    max_hkl: int = Field(default=4, ge=1, le=12)
    d_tolerance_rel: float = Field(default=0.05, gt=0, le=1)
    angle_tolerance_deg: float = Field(default=8.0, gt=0, le=90)
    strict_mode: bool = True
    top_n: int = Field(default=3, ge=1, le=100)


class LookupParameters(Parameters):
    source: Literal["builtin", "local", "cod", "optimade", "materials_project"] = "builtin"
    formula: str | None = Field(default=None, min_length=1, max_length=128)
    material_names: list[str] = Field(default_factory=list, max_length=16)
    elements: list[str] = Field(default_factory=list, max_length=16)
    materials_project_ids: list[str] = Field(default_factory=list, max_length=16)
    cif_candidate_paths: list[str] = Field(default_factory=list, max_length=32)
    allow_network: bool = False
    timeout_s: int = Field(default=15, ge=5, le=30)
    limit_per_source: int = Field(default=5, ge=1, le=32)


class SimulationParameters(Parameters):
    voltage_kv: float = Field(default=200.0, gt=0, le=1000)
    reciprocal_radius_inv_angstrom: float = Field(default=1.0, gt=0)
    max_candidates: int = Field(default=3, ge=1, le=32)
    max_simulated_spots: int = Field(default=256, ge=1, le=10000)
    diffsims_timeout_s: float = Field(default=30.0, ge=1, le=120)
    calibration: float = Field(default=0.01, gt=0)
    pattern_shape: list[int] = Field(default_factory=lambda: [128, 128], min_length=2, max_length=2)
    with_direct_beam: bool = False
    zone_axis_candidates: list[list[int]] = Field(default_factory=lambda: [[0, 0, 1]])

    @model_validator(mode="after")
    def bounded_pattern(self):
        if any(v < 2 or v > 4096 for v in self.pattern_shape):
            raise ValueError("pattern_shape dimensions must be in [2,4096]")
        if len(self.zone_axis_candidates) > 32 or not self.zone_axis_candidates:
            raise ValueError("Provide between 1 and 32 zone axes")
        if any(len(axis) != 3 or not any(axis) for axis in self.zone_axis_candidates):
            raise ValueError("Each zone axis must contain three nonzero-together integers")
        return self


class MatchParameters(Parameters):
    d_tolerance_rel: float = Field(default=0.10, gt=0, le=1)
    angle_tolerance_deg: float = Field(default=12.0, gt=0, le=90)
    min_matched_spots: int = Field(default=2, ge=1)
    top_n: int = Field(default=3, ge=1, le=100)
    previous_score_weight: float = Field(default=0.0, ge=0, le=1)
    max_rotation_candidates: int = Field(default=64, ge=1, le=256)


def fft(ctx: OperationContext, params: dict) -> ServiceResult:
    validated(Parameters, params)
    dataset = ctx.model("dataset", TEMDataset)
    if dataset.modality in {"SAED", "NBED"}:
        raise HarnessError("MODALITY_MISMATCH", "Direct diffraction intensities must bypass FFT")
    image = image_array(ctx.array("processed"), "processed")
    analysis, warnings = compute_fft_analysis(image, center_mode="geometric")
    if not np.isfinite(analysis.fft_complex).all():
        raise HarnessError("FFT_FAILED", "FFT returned nonfinite coefficients")
    magnitude = ctx.emit_array("reciprocal_image", analysis.magnitude)
    complex_ref = ctx.emit_array("fft_complex", analysis.fft_complex)
    ctx.emit_image("fft_preview", analysis.log_magnitude, {"display_transform": "log1p"})
    return ServiceResult(
        outputs={
            "reciprocal_image": magnitude,
            "fft_complex": complex_ref,
            "reciprocal_context": {
                "kind": "fft",
                "shape": list(image.shape),
                "zero_frequency": [image.shape[1] // 2, image.shape[0] // 2],
            },
        },
        warnings=warnings,
    )


def pattern(ctx: OperationContext, params: dict) -> ServiceResult:
    validated(Parameters, params)
    dataset = ctx.model("dataset", TEMDataset)
    if dataset.modality not in {"SAED", "NBED"}:
        raise HarnessError("MODALITY_MISMATCH", "Direct pattern requires SAED or NBED modality")
    array = image_array(ctx.array("raw"), nonnegative=True)
    return ServiceResult(
        outputs={
            "reciprocal_image": ctx.emit_array("reciprocal_image", array),
            "reciprocal_context": {"kind": "direct_diffraction", "shape": list(array.shape)},
        }
    )


def center(ctx: OperationContext, params: dict) -> ServiceResult:
    p = validated(CenterParameters, params)
    array = image_array(ctx.array("reciprocal_image"), "reciprocal_image", nonnegative=True)
    context = ctx.value("reciprocal_context")
    warnings = []
    if p.center_mode == "manual":
        point = (p.center_x, p.center_y)
        diagnostics = {"method": "manual"}
    elif context.get("kind") == "fft":
        # fftshift puts DC at floor(N/2), including odd image dimensions. The legacy
        # geometric detector uses N/2 and must not introduce a half-pixel bias here.
        point = (float(array.shape[1] // 2), float(array.shape[0] // 2))
        diagnostics = {"method": "fftshift_zero_frequency"}
    else:
        point, diagnostics, warnings = locate_direct_beam(
            array,
            center_mode=p.center_mode,
            search_radius_px=p.search_radius_px,
            refine_window=p.refine_window,
        )
    if not np.isfinite(point).all() or not (
        0 <= point[0] < array.shape[1] and 0 <= point[1] < array.shape[0]
    ):
        raise HarnessError("INVALID_CENTER", "Diffraction center lies outside the image")
    return ServiceResult(
        outputs={"center": {"x": point[0], "y": point[1], "diagnostics": diagnostics}},
        warnings=warnings,
    )


def _center(ctx: OperationContext) -> tuple[float, float]:
    value = ctx.value("center")
    try:
        point = (float(value["x"]), float(value["y"]))
    except (ValueError, KeyError, TypeError) as exc:
        raise HarnessError("INVALID_CENTER", "Expected finite center.x and center.y") from exc
    if not np.isfinite(point).all():
        raise HarnessError("INVALID_CENTER", "Expected finite center.x and center.y")
    shape = ctx.array("reciprocal_image").shape
    if not (0 <= point[0] < shape[1] and 0 <= point[1] < shape[0]):
        raise HarnessError("INVALID_CENTER", "Center must lie within the reciprocal image")
    return point


def peaks(ctx: OperationContext, params: dict) -> ServiceResult:
    p = validated(PeakParameters, params)
    array = image_array(ctx.array("reciprocal_image"), "reciprocal_image", nonnegative=True)
    found, diagnostics, warnings = detect_bragg_peaks(
        np.log1p(array), center=_center(ctx), **p.model_dump()
    )
    if not found:
        raise HarnessError("NO_PEAKS", "No Bragg peaks detected", diagnostics)
    rows = [peak.model_dump(mode="json") for peak in found]
    ctx.emit_table("peak_table", rows)
    return ServiceResult(
        outputs={"peaks": rows, "peak_diagnostics": diagnostics}, warnings=warnings
    )


def peak_suggestions(ctx: OperationContext, params: dict) -> ServiceResult:
    validated(Parameters, params)
    array = image_array(ctx.array("reciprocal_image"), "reciprocal_image", nonnegative=True)
    suggestions = suggest_peak_parameter_defaults(np.log1p(array))
    mapped = {
        "top_k": suggestions.pop("top_k_peaks"),
        "min_distance": suggestions.pop("min_peak_distance"),
        **suggestions,
    }
    return ServiceResult(outputs={"peak_parameter_suggestions": mapped})


def manual_peaks(ctx: OperationContext, params: dict) -> ServiceResult:
    p = validated(ManualPeaksParameters, params)
    array = image_array(ctx.array("reciprocal_image"), "reciprocal_image", nonnegative=True)
    rows = []
    for peak in p.peaks:
        if peak.x >= array.shape[1] or peak.y >= array.shape[0]:
            raise HarnessError("INVALID_PEAKS", "Manual peaks must lie inside the reciprocal image")
        row = DiffractionPeak(
            x=peak.x,
            y=peak.y,
            intensity=float(
                array[
                    min(int(round(peak.y)), array.shape[0] - 1),
                    min(int(round(peak.x)), array.shape[1] - 1),
                ]
            ),
            warnings=["Manual point; localization uncertainty is not estimated"],
        )
        rows.append(row.model_dump(mode="json"))
    ctx.emit_table("peak_table", rows)
    return ServiceResult(
        outputs={"peaks": rows, "peak_diagnostics": {"method": "manual_reviewed_points"}},
        warnings=["Manual coordinates require expert review"],
    )


def metrics(ctx: OperationContext, params: dict) -> ServiceResult:
    p = validated(MetricParameters, params)
    dataset = ctx.model("dataset", TEMDataset)
    calibration = ctx.model("calibration", CalibrationResult)
    array = image_array(ctx.array("reciprocal_image"), "reciprocal_image", nonnegative=True)
    context = ctx.value("reciprocal_context")
    point = _center(ctx)
    try:
        found = [DiffractionPeak.model_validate(row) for row in ctx.value("peaks")]
    except (ValueError, TypeError) as exc:
        raise HarnessError("INVALID_PEAKS", "Malformed peak evidence") from exc
    if not found or any(
        not np.isfinite([peak.x, peak.y]).all()
        or not (0 <= peak.x < array.shape[1] and 0 <= peak.y < array.shape[0])
        for peak in found
    ):
        raise HarnessError("INVALID_PEAKS", "Peaks must be finite, nonempty and inside the image")
    physical = False
    relative_scale_sigma = 0.0
    calibration_uncertainty_known = True
    if context.get("kind") == "fft":
        pixel = calibration.pixel_size
        if pixel is not None:
            if (
                pixel.unit != "nm/pixel"
                or not isinstance(pixel.value, (int, float))
                or (not np.isfinite(pixel.value) or pixel.value <= 0)
            ):
                raise HarnessError("INVALID_CALIBRATION", "FFT requires finite positive nm/pixel")
            # The Fourier grid spacing is per axis; rectangular/odd grids are fully supported.
            sx, sy = 1 / (array.shape[1] * pixel.value), 1 / (array.shape[0] * pixel.value)
            relative_scale_sigma = float(pixel.uncertainty or 0) / pixel.value
            calibration_uncertainty_known = pixel.uncertainty is not None
            physical = True
        else:
            sx, sy = 1 / array.shape[1], 1 / array.shape[0]
        method = "fft_frequency_per_axis"
    elif context.get("kind") == "direct_diffraction":
        scale = calibration.diffraction_scale
        if scale is not None:
            if (
                scale.unit != "1/nm/pixel"
                or not isinstance(scale.value, (int, float))
                or (not np.isfinite(scale.value) or scale.value <= 0)
            ):
                raise HarnessError(
                    "INVALID_CALIBRATION", "Pattern requires finite positive 1/nm/pixel"
                )
            sx = sy = scale.value
            relative_scale_sigma = float(scale.uncertainty or 0) / scale.value
            calibration_uncertainty_known = scale.uncertainty is not None
            physical = True
        else:
            # Peak offsets and their length are measured on the diffraction detector.
            sx = sy = 1.0
        method = "direct_pattern_detector_scale"
    else:
        raise HarnessError("INVALID_RECIPROCAL_CONTEXT", "Unknown reciprocal-space input kind")
    if p.require_physical and not physical:
        raise NeedsInput(
            [
                {
                    "name": "pixel_size_nm"
                    if context["kind"] == "fft"
                    else "diffraction_scale_inv_nm_per_px"
                }
            ],
            "Physical calibration required",
        )
    if not np.isfinite([sx, sy, relative_scale_sigma]).all() or relative_scale_sigma < 0:
        raise HarnessError(
            "INVALID_CALIBRATION", "Calibration uncertainty must be finite and nonnegative"
        )
    # Adapt coordinates and their errors into the existing isotropic metric helper. This
    # avoids treating a rectangular FFT's two distinct grid spacings as one scalar.
    transformed = []
    center_sigma = float(p.center_uncertainty_px or 0)
    for peak in found:
        dx, dy = peak.x - point[0], peak.y - point[1]
        sigma_x = float(peak.sigma_x or 0)
        sigma_y = float(peak.sigma_y or 0)
        if not np.isfinite([sigma_x, sigma_y]).all() or min(sigma_x, sigma_y) < 0:
            raise HarnessError("INVALID_PEAKS", "Peak uncertainties must be finite and nonnegative")
        transformed.append(
            peak.model_copy(
                update={
                    "x": dx * sx,
                    "y": dy * sy,
                    "sigma_x": float(sx * np.hypot(sigma_x, center_sigma)),
                    "sigma_y": float(sy * np.hypot(sigma_y, center_sigma)),
                }
            )
        )
    detector_offsets = context["kind"] == "direct_diffraction" and not physical
    vector_unit = "1/nm" if physical else "1/pixel"
    scale_record = {"scale_x": sx, "scale_y": sy, "physical": physical}
    if detector_offsets:
        method = "detector_pixel_offset"
        vector_unit = "pixel"
        scale_record["coordinate_space"] = "diffraction_detector"
    measured = compute_gvectors_and_dspacing(
        transformed,
        center=(0, 0),
        diffraction_scale_value=1.0,
        diffraction_scale_uncertainty=relative_scale_sigma,
        diffraction_scale_source=method,
        include_d_spacings=not detector_offsets,
    )
    warnings = list(measured.warnings)
    if detector_offsets:
        warnings.append(
            "Peak offsets are detector pixels. A reciprocal detector scale is required "
            "for d-spacing and indexing."
        )
    elif not physical:
        warnings.append(
            "Physical calibration is unknown; "
            "pixel diagnostics cannot support hkl/phase identification."
        )
    for vector in measured.g_vectors:
        for measurement in (vector.gx, vector.gy, vector.magnitude):
            if measurement is not None:
                measurement.unit = vector_unit
                measurement.method = method
                measurement.provenance = scale_record.copy()
        # A common scale error changes lengths, not angles. Use localization error only.
        local_peak = transformed[vector.source_peak_index]
        if vector.angle is not None and vector.magnitude.value > 1e-12:
            vector.angle.uncertainty = float(
                np.degrees(
                    np.hypot(
                        abs(vector.gy.value) * local_peak.sigma_x,
                        abs(vector.gx.value) * local_peak.sigma_y,
                    )
                    / vector.magnitude.value**2
                )
            )
    for measurement in measured.d_by_peak:
        if measurement is not None:
            measurement.unit = "nm" if physical else "pixel"
            measurement.method = method + ".inverse_g"
    unknown_uncertainty = False
    for idx, vector in enumerate(measured.g_vectors):
        peak = found[idx]
        unknown_common = p.center_uncertainty_px is None or not calibration_uncertainty_known
        unknown_x, unknown_y = (
            peak.sigma_x is None or unknown_common,
            peak.sigma_y is None or unknown_common,
        )
        if unknown_x:
            vector.gx.uncertainty = None
        if unknown_y:
            vector.gy.uncertainty = None
        if unknown_x or unknown_y:
            unknown_uncertainty = True
            vector.magnitude.uncertainty = None
            if vector.angle is not None:
                vector.angle.uncertainty = None
            if measured.d_by_peak[idx] is not None:
                measured.d_by_peak[idx].uncertainty = None
    if unknown_uncertainty:
        warnings.append(
            "Localization/center/calibration uncertainty is incomplete; "
            "total uncertainty is unknown."
        )
    quantitative = [
        measurement
        for vector in measured.g_vectors
        for measurement in (vector.gx, vector.gy, vector.magnitude, vector.angle)
    ]
    quantitative.extend(measured.d_by_peak)
    if any(
        measurement is not None
        and (
            not np.isfinite(measurement.value)
            or (measurement.uncertainty is not None and not np.isfinite(measurement.uncertainty))
        )
        for measurement in quantitative
    ):
        raise HarnessError("INVALID_METRICS", "Reciprocal metrics must be finite")
    ref = ctx.emit_array("diffraction_intensity", array)
    result = DiffractionResult(
        peaks=found,
        center=point,
        g_vectors=measured.g_vectors,
        d_spacings=[value for value in measured.d_by_peak if value is not None],
        fft_image=ref,
        provenance={
            "method": method,
            "modality": dataset.modality,
            **scale_record,
            "d_by_peak": [
                value.model_dump(mode="json") if value else None for value in measured.d_by_peak
            ],
        },
        warnings=warnings,
    )
    ctx.emit_table("g_vector_table", [v.model_dump(mode="json") for v in result.g_vectors])
    spacing_rows = [
        {
            "source_peak_index": i,
            "value": spacing.value,
            "unit": spacing.unit,
            "uncertainty": spacing.uncertainty,
            "method": spacing.method,
            "confidence": spacing.confidence,
        }
        for i, spacing in enumerate(measured.d_by_peak)
        if spacing is not None
    ]
    if spacing_rows:
        ctx.emit_table("d_spacing_table", spacing_rows, metadata=scale_record)
    return ServiceResult(
        outputs={"diffraction": result},
        warnings=warnings,
        scientific_level="quantitative" if physical else "diagnostic",
    )


def _physical(diffraction: DiffractionResult) -> None:
    if not diffraction.g_vectors or any(
        v.gx.unit != "1/nm"
        or v.gy.unit != "1/nm"
        or not isinstance(v.gx.value, (int, float))
        or not isinstance(v.gy.value, (int, float))
        or not np.isfinite([v.gx.value, v.gy.value]).all()
        for v in diffraction.g_vectors
    ):
        raise NeedsInput(
            [{"name": "physical_calibration"}],
            "Physical reciprocal-space calibration is required for this operation",
        )


def _snapshot_cifs(ctx, candidate_paths):
    paths, evidence = [], []
    for number, raw_path in enumerate(candidate_paths):
        path = Path(raw_path)
        if "://" in raw_path or str(path.drive).startswith("\\\\") or not path.is_file():
            raise HarnessError("INVALID_CANDIDATE", "CIF candidates must be existing local files")
        if path.stat().st_size > 8 * 1024 * 1024:
            raise HarnessError("CANDIDATE_TOO_LARGE", "A CIF candidate must not exceed 8 MiB")
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        snapshot = ctx.output_dir / "candidates" / f"{number:03d}" / path.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(content)
        artifact_id = ctx.emit_json(
            f"candidate_cif_{number}",
            {
                "source_path": str(path.resolve()),
                "sha256": digest,
                "text": content.decode("utf-8-sig"),
            },
        )
        paths.append(str(snapshot))
        evidence.append(
            {
                "source_path": str(path.resolve()),
                "snapshot": str(snapshot),
                "sha256": digest,
                "artifact_id": artifact_id,
            }
        )
    return paths, evidence


def lookup(ctx: OperationContext, params: dict) -> ServiceResult:
    p = validated(LookupParameters, params)
    if p.cif_candidate_paths:
        if p.source != "local":
            raise HarnessError("INVALID_PARAMETERS", "CIF paths require source=local")
        paths, snapshots = _snapshot_cifs(ctx, p.cif_candidate_paths)
        candidates = []
        for path in paths:
            phase = _phase_from_cif(Path(path))
            if phase is None:
                raise HarnessError("INVALID_CANDIDATE", "Local CIF could not be parsed")
            candidate = PhaseParameters.model_validate(
                {
                    "name": phase.name,
                    "a": phase.a,
                    "b": phase.b,
                    "c": phase.c,
                    "alpha": phase.alpha_deg,
                    "beta": phase.beta_deg,
                    "gamma": phase.gamma_deg,
                    "space_group": phase.space_group,
                    "atom_sites": phase.atom_sites,
                    "source": phase.source,
                    "identifier": phase.identifier,
                    "formula": phase.formula,
                    "space_group_number": phase.space_group_number,
                }
            )
            candidates.append(candidate.model_dump(exclude_none=True))
        return ServiceResult(
            outputs={
                "phase_candidates": candidates,
                "phase_lookup": {
                    "source": "local",
                    "network_used": False,
                    "candidate_snapshots": snapshots,
                },
            },
            scientific_level="candidate",
        )
    source = {
        "builtin": "local_builtin",
        "local": "local_builtin",
        "cod": "cod",
        "optimade": "materials_project_optimade",
        "materials_project": "materials_project_api",
    }[p.source]
    online = source != "local_builtin"
    if online and not (p.allow_network and ctx.request.get("network_enabled") is True):
        raise HarnessError(
            "NETWORK_DISABLED", "Online lookup requires plan.network_enabled and allow_network"
        )
    if not (p.formula or p.material_names or p.elements or p.materials_project_ids):
        raise NeedsInput(
            [{"name": "formula", "alternatives": ["material_names", "elements"]}],
            "Constrain candidate lookup with a formula/material query",
        )
    api_key = None
    credential_source = None
    if source == "materials_project_api":
        api_key, credential_source = resolve_materials_project_api_key()
        if not api_key:
            raise NeedsInput(
                [
                    {
                        "name": "materials_project_configuration",
                        "action": "configure-mp-api",
                        "transport": "env/config",
                        "secret_in_run_parameters": False,
                    }
                ],
                "Configure the Materials Project key outside the run request",
            )
        try:
            importlib.import_module("mp_api.client")
        except ModuleNotFoundError as exc:
            raise HarnessError(
                "DEPENDENCY_UNAVAILABLE"
                if (exc.name or "").startswith("mp_api")
                else "DEPENDENCY_BROKEN",
                "Materials Project lookup requires a working mp-api",
            ) from exc
        except Exception as exc:
            raise HarnessError(
                "DEPENDENCY_BROKEN",
                "Materials Project client import failed",
                {"type": type(exc).__name__},
            ) from exc
    try:
        result = lookup_phase_candidates(
            formula=p.formula,
            elements=p.elements,
            material_names=p.material_names,
            sources=[source],
            limit_per_source=p.limit_per_source,
            timeout_s=p.timeout_s,
            materials_project_ids=p.materials_project_ids,
            materials_project_api_key=api_key,
            materials_project_api_isolate=False,
        )
    except Exception as exc:
        raise HarnessError(
            "LOOKUP_FAILED",
            "Candidate provider failed",
            {"source": p.source, "type": type(exc).__name__},
        ) from exc
    warnings = [
        value.replace(api_key, "[REDACTED]") if api_key else value for value in result.warnings
    ]
    if not result.candidates:
        raise HarnessError(
            "LOOKUP_NO_RESULT",
            "No candidate matches the constrained query",
            {"source": p.source, "warnings": warnings},
        )
    fields = set(PhaseParameters.model_fields)
    candidates = []
    for row in result.candidates:
        try:
            candidate = PhaseParameters.model_validate(
                {k: v for k, v in row.items() if k in fields}
            )
        except ValueError:
            warnings.append("Candidate with invalid/missing lattice values was excluded")
            continue
        candidates.append(candidate.model_dump(exclude_none=True))
    if not candidates:
        raise HarnessError(
            "LOOKUP_NO_RESULT", "Provider returned no valid local lattice candidates"
        )
    evidence = _redact(
        {
            "source": p.source,
            "network_used": online,
            "provenance": result.provenance,
            "candidate_count": len(candidates),
            "credential_source": credential_source,
            "provider_candidates": result.candidates,
        },
        api_key,
    )
    return ServiceResult(
        outputs={"phase_candidates": candidates, "phase_lookup": evidence},
        warnings=warnings,
        scientific_level="candidate",
    )


def index(ctx: OperationContext, params: dict) -> ServiceResult:
    p = validated(IndexParameters, params)
    diffraction = ctx.model("diffraction", DiffractionResult).model_copy(deep=True)
    _physical(diffraction)
    if not p.candidate_phase_params and not p.cif_candidate_paths:
        try:
            candidates = ctx.value("phase_candidates")
        except (HarnessError, KeyError) as exc:
            raise NeedsInput(
                [{"name": "candidate_phase_params", "alternative": "phase.lookup/local_cif"}],
                "Supply constrained candidates before indexing",
            ) from exc
        p = validated(IndexParameters, {**p.model_dump(), "candidate_phase_params": candidates})
    paths, snapshot_evidence = _snapshot_cifs(ctx, p.cif_candidate_paths)
    from autotem.core import Measurement

    d_by_peak = [
        Measurement.model_validate(v) if v else None
        for v in diffraction.provenance.get("d_by_peak", [])
    ]
    if len(d_by_peak) != len(diffraction.peaks):
        # Do not infer which peak an abbreviated d-spacing list belongs to.
        raise HarnessError("INVALID_DIFFRACTION", "Per-peak d-spacing mapping is missing")
    result = run_b4_indexing(
        peaks=diffraction.peaks,
        g_vectors=diffraction.g_vectors,
        d_by_peak=d_by_peak,
        initial_indexed_spots=[],
        candidate_phase_params=[v.model_dump(exclude_none=True) for v in p.candidate_phase_params],
        allowed_phase_names=None,
        max_hkl=p.max_hkl,
        d_tolerance_rel=p.d_tolerance_rel,
        angle_tolerance_deg=p.angle_tolerance_deg,
        strict_mode=p.strict_mode,
        cif_candidate_paths=paths,
        top_n=p.top_n,
    )
    if not result.phase_candidates:
        raise HarnessError(
            "INDEXING_NO_RESULT",
            "No constrained candidate could be indexed",
            {"warnings": result.warnings, "provenance": result.provenance},
        )
    diffraction.indexed_spots = result.indexed_spots
    diffraction.phase_candidates = result.phase_candidates
    diffraction.provenance["indexing"] = {
        **result.provenance,
        "candidate_snapshots": snapshot_evidence,
    }
    diffraction.warnings.extend(result.warnings)
    return ServiceResult(
        outputs={"diffraction": diffraction}, warnings=result.warnings, scientific_level="candidate"
    )


def _dependencies() -> list[str]:
    missing = []
    for name in ("diffsims", "diffpy.structure", "orix"):
        try:
            present = find_spec(name) is not None
        except (ModuleNotFoundError, ValueError):
            present = False
        if not present:
            missing.append(name)
    return missing


def simulate(ctx: OperationContext, params: dict) -> ServiceResult:
    p = validated(SimulationParameters, params)
    diffraction = ctx.model("diffraction", DiffractionResult).model_copy(deep=True)
    if not diffraction.phase_candidates:
        raise NeedsInput([{"name": "phase_candidates"}], "Index constrained candidates first")
    if any(
        not candidate.lattice_parameters.get("atom_sites")
        for candidate in diffraction.phase_candidates
    ):
        raise NeedsInput(
            [{"name": "atom_sites", "alternative": "local_cif"}],
            "Simulation requires actual atom sites; single-atom fallback is disabled",
        )
    missing = _dependencies()
    if missing:
        raise HarnessError(
            "DEPENDENCY_UNAVAILABLE",
            "Diffraction simulation dependencies are unavailable",
            {"missing": missing},
        )
    values = {
        **p.model_dump(),
        "simulation_engine": "diffsims",
        "fallback_single_atom": False,
        "phase_match_enabled": False,
    }
    budgets = ctx.request.get("budgets", {})
    values["diffsims_timeout_s"] = min(
        p.diffsims_timeout_s, budgets.get("simulation_seconds", p.diffsims_timeout_s)
    )
    result = simulate_from_diffraction(diffraction, parameters=values)
    if not result.patterns:
        reason = result.provenance.get("adapter", {}).get("reason", "no_patterns")
        error_code = {
            "import_failed": "DEPENDENCY_BROKEN",
            "dependency_missing": "DEPENDENCY_UNAVAILABLE",
            "timeout": "SIMULATION_TIMEOUT",
        }.get(reason, "SIMULATION_FAILED")
        raise HarnessError(
            error_code,
            "Simulation did not produce real diffraction evidence",
            {"warnings": result.warnings, "provenance": result.provenance},
        )
    for number, item in enumerate(result.patterns):
        if not item.spots:
            raise HarnessError(
                "SIMULATION_FAILED", "Simulated pattern contains no reciprocal spots"
            )
        if item.data is not None:
            if item.data.inline is not None:
                array = np.asarray(item.data.inline)
            elif item.data.uri and Path(item.data.uri).is_file():
                array = np.load(item.data.uri, allow_pickle=False)
            else:
                raise HarnessError(
                    "SIMULATION_FAILED", "Full simulated pattern array is unavailable"
                )
            item.data = ctx.emit_array(
                f"simulated_pattern_{number}", image_array(array, "simulation")
            )
    return ServiceResult(
        outputs={"simulation": result}, warnings=result.warnings, scientific_level="candidate"
    )


def match(ctx: OperationContext, params: dict) -> ServiceResult:
    p = validated(MatchParameters, params)
    diffraction = ctx.model("diffraction", DiffractionResult).model_copy(deep=True)
    simulation = ctx.model("simulation", SimulationResult).model_copy(deep=True)
    _physical(diffraction)
    result = match_diffsims_patterns_to_diffraction(
        diffraction, simulation.patterns, **p.model_dump()
    )
    if not result.provenance.get("active"):
        raise HarnessError(
            "MATCHING_NOT_EVALUATED",
            "No valid measured/simulated evidence was compared",
            result.provenance,
        )
    diffraction.phase_candidates = result.candidates
    diffraction.provenance["phase_matching"] = result.provenance
    diffraction.warnings.extend(result.warnings)
    simulation.patterns = result.patterns
    simulation.matched_phase_candidates = result.candidates
    simulation.comparison_metrics = result.comparison_metrics
    simulation.provenance["phase_matching"] = result.provenance
    return ServiceResult(
        outputs={"diffraction": diffraction, "simulation": simulation},
        warnings=result.warnings,
        scientific_level="candidate",
    )


def definitions() -> list[OperationDefinition]:
    return [
        OperationDefinition(
            "diffraction.peaks.suggest",
            "diffraction",
            peak_suggestions,
            Parameters,
            input_roles=("reciprocal_image",),
            output_roles=("peak_parameter_suggestions",),
        ),
        OperationDefinition(
            "diffraction.peaks.manual",
            "diffraction",
            manual_peaks,
            ManualPeaksParameters,
            input_roles=("reciprocal_image",),
            output_roles=("peaks", "peak_diagnostics"),
            scientific_limitations=("Manual localization uncertainty is unknown",),
        ),
        OperationDefinition(
            "phase.lookup",
            "diffraction",
            lookup,
            LookupParameters,
            output_roles=("phase_candidates", "phase_lookup"),
            result_level="candidate",
            permission="local_read_or_explicit_network",
            timeout_seconds=120,
            scientific_limitations=(
                "Database candidates require calibrated experimental indexing",
            ),
        ),
        OperationDefinition(
            "diffraction.fft",
            "diffraction",
            fft,
            Parameters,
            input_roles=("processed", "dataset"),
            output_roles=("reciprocal_image", "fft_complex", "reciprocal_context"),
            scientific_limitations=("FFT artifacts are not direct SAED detector intensities",),
        ),
        OperationDefinition(
            "diffraction.pattern",
            "diffraction",
            pattern,
            Parameters,
            input_roles=("raw", "dataset"),
            output_roles=("reciprocal_image", "reciprocal_context"),
            modality=("SAED", "NBED"),
        ),
        OperationDefinition(
            "diffraction.center",
            "diffraction",
            center,
            CenterParameters,
            input_roles=("reciprocal_image", "reciprocal_context"),
            output_roles=("center",),
        ),
        OperationDefinition(
            "diffraction.peaks",
            "diffraction",
            peaks,
            PeakParameters,
            input_roles=("reciprocal_image", "center"),
            output_roles=("peaks", "peak_diagnostics"),
        ),
        OperationDefinition(
            "diffraction.metrics",
            "diffraction",
            metrics,
            MetricParameters,
            input_roles=(
                "dataset",
                "calibration",
                "reciprocal_image",
                "reciprocal_context",
                "center",
                "peaks",
            ),
            output_roles=("diffraction",),
        ),
        OperationDefinition(
            "diffraction.index",
            "diffraction",
            index,
            IndexParameters,
            input_roles=("diffraction",),
            output_roles=("diffraction",),
            result_level="candidate",
        ),
        OperationDefinition(
            "simulation.diffraction",
            "diffraction",
            simulate,
            SimulationParameters,
            input_roles=("diffraction",),
            output_roles=("simulation",),
            dependencies=("diffsims", "diffpy.structure", "orix"),
            timeout_seconds=120,
            result_level="candidate",
            scientific_limitations=("Kinematic simulation",),
        ),
        OperationDefinition(
            "diffraction.match",
            "diffraction",
            match,
            MatchParameters,
            input_roles=("diffraction", "simulation"),
            output_roles=("diffraction", "simulation"),
            result_level="candidate",
        ),
    ]
