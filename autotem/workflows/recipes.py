"""Workflow recipe loading, normalization, and persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from autotem.calibration.open_source_adapters import SUPPORTED_DIFFRACTION_CALIBRATION_ENGINES
from autotem.core.models import AutoTEMModel, WorkflowConfig, WorkflowStep

DEFAULT_WORKFLOW_NAME = "a2_default"
DEFAULT_WORKFLOW_DESCRIPTION = "A2 baseline load-calibrate-diffraction-simulation workflow."
WORKFLOW_PARAMETER_SCHEMA_VERSION = "2026-05-13.b1.1"
DEFAULT_STEP_SEQUENCE: tuple[str, ...] = (
    "load",
    "assess_quality",
    "calibrate",
    "preprocess",
    "analyze",
    "simulate",
    "report",
)
STEP_ALIASES: dict[str, tuple[str, ...]] = {
    "analyze": ("fft", "diffraction_peaks", "index_diffraction", "phase_match"),
}
_SUPPORTED_OUTPUT_FORMATS = {"json", "csv", "html", "svg", "pdf"}
_SUPPORTED_CALIBRATION_METHODS = {"auto", "metadata", "manual", "reference"}
_SUPPORTED_PREPROCESS_RECIPES = {"auto", "min_max", "b2_fft"}
_SUPPORTED_PEAK_THRESHOLD_STRATEGIES = {
    "auto",
    "quantile",
    "std",
    "relative_max",
    "mad",
    "radial_mad",
    "local_max",
    "log_blob",
    "hybrid",
}
_SUPPORTED_INDEXING_MODES = {"auto", "baseline", "lattice_fit", "b4_mvp", "b4_strict"}
_SUPPORTED_ACCEPTANCE_OPS = {">=", "<=", ">", "<", "=="}
_QUALITY_THRESHOLD_RANGE_01_KEYS = {
    "blur_risk_max",
    "contamination_risk_max",
    "astigmatism_risk_max",
    "undersampling_risk_max",
    "overall_score_min",
}
_QUALITY_THRESHOLD_NONNEG_KEYS = {"snr_min", "drift_px_max"}


class WorkflowOutputs(AutoTEMModel):
    """Output controls carried alongside a workflow recipe."""

    formats: list[str] = Field(default_factory=lambda: ["json"])
    save_intermediates: bool = True


class WorkflowRecipe(AutoTEMModel):
    """Normalized workflow recipe used by the A2 orchestration layer."""

    workflow: WorkflowConfig
    outputs: WorkflowOutputs = Field(default_factory=WorkflowOutputs)


def normalize_step_names(steps: list[str]) -> list[str]:
    """Expand aliases and de-duplicate while preserving order."""
    normalized: list[str] = []
    for raw_step in steps:
        step = str(raw_step)
        expanded = STEP_ALIASES.get(step, (step,))
        for candidate in expanded:
            if candidate not in normalized:
                normalized.append(candidate)
    return normalized


def default_workflow_recipe() -> WorkflowRecipe:
    """Return the built-in fallback recipe for A2 runs."""
    return WorkflowRecipe(
        workflow=WorkflowConfig(
            name=DEFAULT_WORKFLOW_NAME,
            description=DEFAULT_WORKFLOW_DESCRIPTION,
            steps=[
                WorkflowStep(name=step)
                for step in normalize_step_names(list(DEFAULT_STEP_SEQUENCE))
            ],
            output_formats=["json"],
            save_intermediates=True,
            provenance={
                "source": "built_in_default",
                "workflow_parameter_schema_version": WORKFLOW_PARAMETER_SCHEMA_VERSION,
            },
        ),
        outputs=WorkflowOutputs(formats=["json"], save_intermediates=True),
    )


def _as_mapping(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    msg = "Workflow recipe must be a mapping at YAML root."
    raise ValueError(msg)


def _as_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        msg = f"Workflow parameter '{field_name}' must be numeric, got boolean."
        raise ValueError(msg)
    if isinstance(value, (int, float)):
        return float(value)
    msg = f"Workflow parameter '{field_name}' must be numeric."
    raise ValueError(msg)


def _validate_output_formats(formats: list[str]) -> list[str]:
    normalized: list[str] = []
    for item in formats:
        name = str(item).strip().lower()
        if name not in _SUPPORTED_OUTPUT_FORMATS:
            msg = (
                f"Unsupported output format '{item}'. "
                f"Supported formats: {sorted(_SUPPORTED_OUTPUT_FORMATS)}."
            )
            raise ValueError(msg)
        if name not in normalized:
            normalized.append(name)
    return normalized


def _validate_workflow_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    params = dict(parameters)

    quality = params.get("quality")
    if quality is not None:
        if not isinstance(quality, dict):
            raise ValueError("workflow.parameters.quality must be a mapping.")
        quality_thresholds = quality.get("thresholds", quality)
        if not isinstance(quality_thresholds, dict):
            raise ValueError(
                "workflow.parameters.quality.thresholds must be a mapping when provided."
            )
        for key in _QUALITY_THRESHOLD_NONNEG_KEYS:
            if key in quality_thresholds:
                value = _as_float(
                    quality_thresholds[key],
                    field_name=f"quality.thresholds.{key}",
                )
                if value < 0:
                    raise ValueError(f"workflow.parameters.quality.thresholds.{key} must be >= 0.")
        for key in _QUALITY_THRESHOLD_RANGE_01_KEYS:
            if key in quality_thresholds:
                value = _as_float(
                    quality_thresholds[key],
                    field_name=f"quality.thresholds.{key}",
                )
                if not (0 <= value <= 1):
                    raise ValueError(
                        f"workflow.parameters.quality.thresholds.{key} must be within [0, 1]."
                    )

    calibration = params.get("calibration")
    if calibration is not None:
        if not isinstance(calibration, dict):
            raise ValueError("workflow.parameters.calibration must be a mapping.")
        method = calibration.get("method")
        if method is not None and str(method) not in _SUPPORTED_CALIBRATION_METHODS:
            raise ValueError(
                "workflow.parameters.calibration.method must be one of "
                f"{sorted(_SUPPORTED_CALIBRATION_METHODS)}."
            )
        if "pixel_size_nm" in calibration:
            pixel_size_nm = _as_float(
                calibration["pixel_size_nm"],
                field_name="calibration.pixel_size_nm",
            )
            if pixel_size_nm <= 0:
                raise ValueError("workflow.parameters.calibration.pixel_size_nm must be > 0.")
        if "voltage_kv" in calibration:
            voltage_kv = _as_float(calibration["voltage_kv"], field_name="calibration.voltage_kv")
            if voltage_kv <= 0:
                raise ValueError("workflow.parameters.calibration.voltage_kv must be > 0.")
        if "known_d_spacing" in calibration:
            known_d = _as_float(
                calibration["known_d_spacing"],
                field_name="calibration.known_d_spacing",
            )
            if known_d <= 0:
                raise ValueError("workflow.parameters.calibration.known_d_spacing must be > 0.")
        if "measured_radius_px" in calibration:
            measured_radius_px = _as_float(
                calibration["measured_radius_px"],
                field_name="calibration.measured_radius_px",
            )
            if measured_radius_px <= 0:
                raise ValueError("workflow.parameters.calibration.measured_radius_px must be > 0.")
        if "diffraction_scale_inv_nm_per_px" in calibration:
            diffraction_scale = _as_float(
                calibration["diffraction_scale_inv_nm_per_px"],
                field_name="calibration.diffraction_scale_inv_nm_per_px",
            )
            if diffraction_scale <= 0:
                raise ValueError(
                    "workflow.parameters.calibration.diffraction_scale_inv_nm_per_px must be > 0."
                )
        if "detector_pixel_size_um" in calibration:
            detector_pixel_size = _as_float(
                calibration["detector_pixel_size_um"],
                field_name="calibration.detector_pixel_size_um",
            )
            if detector_pixel_size <= 0:
                raise ValueError(
                    "workflow.parameters.calibration.detector_pixel_size_um must be > 0."
                )
        if "pixel_size_uncertainty_nm" in calibration:
            pixel_uncertainty_nm = _as_float(
                calibration["pixel_size_uncertainty_nm"],
                field_name="calibration.pixel_size_uncertainty_nm",
            )
            if pixel_uncertainty_nm < 0:
                raise ValueError(
                    "workflow.parameters.calibration.pixel_size_uncertainty_nm must be >= 0."
                )
        if "known_d_uncertainty_nm" in calibration:
            known_d_uncertainty_nm = _as_float(
                calibration["known_d_uncertainty_nm"],
                field_name="calibration.known_d_uncertainty_nm",
            )
            if known_d_uncertainty_nm < 0:
                raise ValueError(
                    "workflow.parameters.calibration.known_d_uncertainty_nm must be >= 0."
                )
        if "measured_radius_uncertainty_px" in calibration:
            measured_radius_uncertainty_px = _as_float(
                calibration["measured_radius_uncertainty_px"],
                field_name="calibration.measured_radius_uncertainty_px",
            )
            if measured_radius_uncertainty_px < 0:
                raise ValueError(
                    "workflow.parameters.calibration.measured_radius_uncertainty_px must be >= 0."
                )
        if "diffraction_calibration_engine" in calibration:
            calibration_engine = str(calibration["diffraction_calibration_engine"]).strip().lower()
            if calibration_engine not in SUPPORTED_DIFFRACTION_CALIBRATION_ENGINES:
                raise ValueError(
                    "workflow.parameters.calibration.diffraction_calibration_engine must be one of "
                    f"{sorted(SUPPORTED_DIFFRACTION_CALIBRATION_ENGINES)}."
                )
        if "known_d_spacing_unit" in calibration:
            if not isinstance(calibration["known_d_spacing_unit"], str) or not calibration[
                "known_d_spacing_unit"
            ].strip():
                raise ValueError(
                    "workflow.parameters.calibration.known_d_spacing_unit "
                    "must be a non-empty string."
                )
        reference = calibration.get("reference")
        if reference is not None and not isinstance(reference, dict):
            raise ValueError(
                "workflow.parameters.calibration.reference must be a mapping when provided."
            )
        if isinstance(reference, dict):
            if "known_d_spacing" in reference:
                ref_known_d = _as_float(
                    reference["known_d_spacing"],
                    field_name="calibration.reference.known_d_spacing",
                )
                if ref_known_d <= 0:
                    raise ValueError(
                        "workflow.parameters.calibration.reference.known_d_spacing must be > 0."
                    )
            if "measured_radius_px" in reference:
                ref_measured_radius_px = _as_float(
                    reference["measured_radius_px"],
                    field_name="calibration.reference.measured_radius_px",
                )
                if ref_measured_radius_px <= 0:
                    raise ValueError(
                        "workflow.parameters.calibration.reference.measured_radius_px must be > 0."
                    )
            if "detector_pixel_size_um" in reference:
                ref_detector_pixel_size = _as_float(
                    reference["detector_pixel_size_um"],
                    field_name="calibration.reference.detector_pixel_size_um",
                )
                if ref_detector_pixel_size <= 0:
                    raise ValueError(
                        "workflow.parameters.calibration.reference.detector_pixel_size_um "
                        "must be > 0."
                    )
            if "known_d_uncertainty_nm" in reference:
                ref_known_d_uncertainty = _as_float(
                    reference["known_d_uncertainty_nm"],
                    field_name="calibration.reference.known_d_uncertainty_nm",
                )
                if ref_known_d_uncertainty < 0:
                    raise ValueError(
                        "workflow.parameters.calibration.reference.known_d_uncertainty_nm "
                        "must be >= 0."
                    )
            if "measured_radius_uncertainty_px" in reference:
                ref_radius_uncertainty = _as_float(
                    reference["measured_radius_uncertainty_px"],
                    field_name="calibration.reference.measured_radius_uncertainty_px",
                )
                if ref_radius_uncertainty < 0:
                    raise ValueError(
                        "workflow.parameters.calibration.reference.measured_radius_uncertainty_px "
                        "must be >= 0."
                    )

    preprocessing = params.get("preprocessing")
    if preprocessing is not None:
        if not isinstance(preprocessing, dict):
            raise ValueError("workflow.parameters.preprocessing must be a mapping.")
        recipe = preprocessing.get("recipe")
        if recipe is not None and str(recipe) not in _SUPPORTED_PREPROCESS_RECIPES:
            raise ValueError(
                "workflow.parameters.preprocessing.recipe must be one of "
                f"{sorted(_SUPPORTED_PREPROCESS_RECIPES)}."
            )
        low = preprocessing.get("clip_percentile_low")
        high = preprocessing.get("clip_percentile_high")
        if low is not None or high is not None:
            low_value = _as_float(
                0.0 if low is None else low,
                field_name="preprocessing.clip_percentile_low",
            )
            high_value = _as_float(
                100.0 if high is None else high,
                field_name="preprocessing.clip_percentile_high",
            )
            if not (0.0 <= low_value < high_value <= 100.0):
                raise ValueError(
                    "workflow.parameters.preprocessing clip percentiles must satisfy "
                    "0 <= low < high <= 100."
                )
        if "denoise_sigma" in preprocessing:
            denoise_sigma = _as_float(
                preprocessing["denoise_sigma"],
                field_name="preprocessing.denoise_sigma",
            )
            if denoise_sigma < 0:
                raise ValueError("workflow.parameters.preprocessing.denoise_sigma must be >= 0.")
        if "background_sigma" in preprocessing:
            background_sigma = _as_float(
                preprocessing["background_sigma"],
                field_name="preprocessing.background_sigma",
            )
            if background_sigma < 0:
                raise ValueError(
                    "workflow.parameters.preprocessing.background_sigma must be >= 0."
                )

    diffraction = params.get("diffraction")
    if diffraction is not None:
        if not isinstance(diffraction, dict):
            raise ValueError("workflow.parameters.diffraction must be a mapping.")
        if "top_k_peaks" in diffraction:
            top_k = int(_as_float(diffraction["top_k_peaks"], field_name="diffraction.top_k_peaks"))
            if top_k < 1:
                raise ValueError("workflow.parameters.diffraction.top_k_peaks must be >= 1.")
        if "phase_top_n" in diffraction:
            phase_top_n = int(
                _as_float(
                    diffraction["phase_top_n"],
                    field_name="diffraction.phase_top_n",
                )
            )
            if phase_top_n < 1:
                raise ValueError("workflow.parameters.diffraction.phase_top_n must be >= 1.")
        if "min_peak_distance" in diffraction:
            min_distance = _as_float(
                diffraction["min_peak_distance"],
                field_name="diffraction.min_peak_distance",
            )
            if min_distance < 0:
                raise ValueError("workflow.parameters.diffraction.min_peak_distance must be >= 0.")
        if "max_peaks" in diffraction:
            max_peaks = int(_as_float(diffraction["max_peaks"], field_name="diffraction.max_peaks"))
            if max_peaks < 1:
                raise ValueError("workflow.parameters.diffraction.max_peaks must be >= 1.")
        if "min_snr" in diffraction:
            min_snr = _as_float(diffraction["min_snr"], field_name="diffraction.min_snr")
            if min_snr < 0:
                raise ValueError("workflow.parameters.diffraction.min_snr must be >= 0.")
        if "center_exclusion_radius" in diffraction:
            center_radius = _as_float(
                diffraction["center_exclusion_radius"],
                field_name="diffraction.center_exclusion_radius",
            )
            if center_radius < 0:
                raise ValueError(
                    "workflow.parameters.diffraction.center_exclusion_radius must be >= 0."
                )
        if "center_mask_radius" in diffraction:
            center_mask_radius = _as_float(
                diffraction["center_mask_radius"],
                field_name="diffraction.center_mask_radius",
            )
            if center_mask_radius < 0:
                raise ValueError(
                    "workflow.parameters.diffraction.center_mask_radius must be >= 0."
                )
        if "center_reject_margin" in diffraction:
            center_reject_margin = _as_float(
                diffraction["center_reject_margin"],
                field_name="diffraction.center_reject_margin",
            )
            if center_reject_margin < 0:
                raise ValueError(
                    "workflow.parameters.diffraction.center_reject_margin must be >= 0."
                )
        if "center_keep_snr" in diffraction:
            center_keep_snr = _as_float(
                diffraction["center_keep_snr"],
                field_name="diffraction.center_keep_snr",
            )
            if center_keep_snr < 0:
                raise ValueError(
                    "workflow.parameters.diffraction.center_keep_snr must be >= 0."
                )
        if "min_radius_px_hard" in diffraction:
            min_radius_px_hard = _as_float(
                diffraction["min_radius_px_hard"],
                field_name="diffraction.min_radius_px_hard",
            )
            if min_radius_px_hard < 0:
                raise ValueError(
                    "workflow.parameters.diffraction.min_radius_px_hard must be >= 0."
                )
        if "axis_streak_mask_width" in diffraction:
            axis_mask_width = _as_float(
                diffraction["axis_streak_mask_width"],
                field_name="diffraction.axis_streak_mask_width",
            )
            if axis_mask_width < 0:
                raise ValueError(
                    "workflow.parameters.diffraction.axis_streak_mask_width must be >= 0."
                )
        if "border_mask" in diffraction:
            border_mask = _as_float(
                diffraction["border_mask"],
                field_name="diffraction.border_mask",
            )
            if border_mask < 0:
                raise ValueError("workflow.parameters.diffraction.border_mask must be >= 0.")
        if "radial_bin_width" in diffraction:
            radial_bin_width = _as_float(
                diffraction["radial_bin_width"],
                field_name="diffraction.radial_bin_width",
            )
            if radial_bin_width <= 0:
                raise ValueError("workflow.parameters.diffraction.radial_bin_width must be > 0.")
        if "dedup_distance_px" in diffraction:
            dedup_distance_px = _as_float(
                diffraction["dedup_distance_px"],
                field_name="diffraction.dedup_distance_px",
            )
            if dedup_distance_px <= 0:
                raise ValueError(
                    "workflow.parameters.diffraction.dedup_distance_px must be > 0."
                )
        if "dedup_radial_tol_px" in diffraction:
            dedup_radial_tol_px = _as_float(
                diffraction["dedup_radial_tol_px"],
                field_name="diffraction.dedup_radial_tol_px",
            )
            if dedup_radial_tol_px <= 0:
                raise ValueError(
                    "workflow.parameters.diffraction.dedup_radial_tol_px must be > 0."
                )
        if "dedup_angle_tol_deg" in diffraction:
            dedup_angle_tol_deg = _as_float(
                diffraction["dedup_angle_tol_deg"],
                field_name="diffraction.dedup_angle_tol_deg",
            )
            if dedup_angle_tol_deg <= 0 or dedup_angle_tol_deg > 180:
                raise ValueError(
                    "workflow.parameters.diffraction.dedup_angle_tol_deg must satisfy "
                    "0 < value <= 180."
                )
        if "max_radius_ratio" in diffraction:
            max_radius_ratio = _as_float(
                diffraction["max_radius_ratio"],
                field_name="diffraction.max_radius_ratio",
            )
            if max_radius_ratio <= 0:
                raise ValueError("workflow.parameters.diffraction.max_radius_ratio must be > 0.")
        if "max_candidates_for_refine" in diffraction:
            max_candidates_for_refine = int(
                _as_float(
                    diffraction["max_candidates_for_refine"],
                    field_name="diffraction.max_candidates_for_refine",
                )
            )
            if max_candidates_for_refine < 1:
                raise ValueError(
                    "workflow.parameters.diffraction.max_candidates_for_refine must be >= 1."
                )
        if "max_gaussian_refine_count" in diffraction:
            max_gaussian_refine_count = int(
                _as_float(
                    diffraction["max_gaussian_refine_count"],
                    field_name="diffraction.max_gaussian_refine_count",
                )
            )
            if max_gaussian_refine_count < 0:
                raise ValueError(
                    "workflow.parameters.diffraction.max_gaussian_refine_count must be >= 0."
                )
        if "allow_relaxed_fallback" in diffraction:
            allow_relaxed_fallback = diffraction["allow_relaxed_fallback"]
            if not isinstance(allow_relaxed_fallback, bool):
                raise ValueError(
                    "workflow.parameters.diffraction.allow_relaxed_fallback must be bool."
                )
        if "enable_inner_pair_recovery" in diffraction:
            enable_inner_pair_recovery = diffraction["enable_inner_pair_recovery"]
            if not isinstance(enable_inner_pair_recovery, bool):
                raise ValueError(
                    "workflow.parameters.diffraction.enable_inner_pair_recovery must be bool."
                )
        if "inner_pair_min_snr" in diffraction:
            inner_pair_min_snr = _as_float(
                diffraction["inner_pair_min_snr"],
                field_name="diffraction.inner_pair_min_snr",
            )
            if inner_pair_min_snr < 0:
                raise ValueError(
                    "workflow.parameters.diffraction.inner_pair_min_snr must be >= 0."
                )
        if "inner_pair_recover_radius_px" in diffraction:
            inner_pair_recover_radius_px = _as_float(
                diffraction["inner_pair_recover_radius_px"],
                field_name="diffraction.inner_pair_recover_radius_px",
            )
            if inner_pair_recover_radius_px <= 0:
                raise ValueError(
                    "workflow.parameters.diffraction.inner_pair_recover_radius_px must be > 0."
                )
        if "inner_pair_max_pairs" in diffraction:
            inner_pair_max_pairs = int(
                _as_float(
                    diffraction["inner_pair_max_pairs"],
                    field_name="diffraction.inner_pair_max_pairs",
                )
            )
            if inner_pair_max_pairs < 0:
                raise ValueError(
                    "workflow.parameters.diffraction.inner_pair_max_pairs must be >= 0."
                )
        if "ensure_multiring_coverage" in diffraction:
            ensure_multiring_coverage = diffraction["ensure_multiring_coverage"]
            if not isinstance(ensure_multiring_coverage, bool):
                raise ValueError(
                    "workflow.parameters.diffraction.ensure_multiring_coverage must be bool."
                )
        if "inner_ring_recover_pairs" in diffraction:
            inner_ring_recover_pairs = int(
                _as_float(
                    diffraction["inner_ring_recover_pairs"],
                    field_name="diffraction.inner_ring_recover_pairs",
                )
            )
            if inner_ring_recover_pairs < 0:
                raise ValueError(
                    "workflow.parameters.diffraction.inner_ring_recover_pairs must be >= 0."
                )
        if "inner_ring_recover_min_snr" in diffraction:
            inner_ring_recover_min_snr = _as_float(
                diffraction["inner_ring_recover_min_snr"],
                field_name="diffraction.inner_ring_recover_min_snr",
            )
            if inner_ring_recover_min_snr < 0:
                raise ValueError(
                    "workflow.parameters.diffraction.inner_ring_recover_min_snr must be >= 0."
                )
        if "rescue_on_empty" in diffraction:
            rescue_on_empty = diffraction["rescue_on_empty"]
            if not isinstance(rescue_on_empty, bool):
                raise ValueError("workflow.parameters.diffraction.rescue_on_empty must be bool.")
        if "rescue_min_snr" in diffraction:
            rescue_min_snr = _as_float(
                diffraction["rescue_min_snr"],
                field_name="diffraction.rescue_min_snr",
            )
            if rescue_min_snr < 0:
                raise ValueError("workflow.parameters.diffraction.rescue_min_snr must be >= 0.")
        if "enable_lattice_consistency_filter" in diffraction:
            enable_lattice_consistency_filter = diffraction["enable_lattice_consistency_filter"]
            if not isinstance(enable_lattice_consistency_filter, bool):
                raise ValueError(
                    "workflow.parameters.diffraction.enable_lattice_consistency_filter "
                    "must be bool."
                )
        if "auto_center_cluster_filter" in diffraction:
            auto_center_cluster_filter = diffraction["auto_center_cluster_filter"]
            if not isinstance(auto_center_cluster_filter, bool):
                raise ValueError(
                    "workflow.parameters.diffraction.auto_center_cluster_filter must be bool."
                )
        if "lattice_coeff_tol" in diffraction:
            lattice_coeff_tol = _as_float(
                diffraction["lattice_coeff_tol"],
                field_name="diffraction.lattice_coeff_tol",
            )
            if lattice_coeff_tol <= 0:
                raise ValueError("workflow.parameters.diffraction.lattice_coeff_tol must be > 0.")
        if "lattice_recon_tol_px" in diffraction:
            lattice_recon_tol_px = _as_float(
                diffraction["lattice_recon_tol_px"],
                field_name="diffraction.lattice_recon_tol_px",
            )
            if lattice_recon_tol_px <= 0:
                raise ValueError(
                    "workflow.parameters.diffraction.lattice_recon_tol_px must be > 0."
                )
        if "lattice_filter_min_keep" in diffraction:
            lattice_filter_min_keep = int(
                _as_float(
                    diffraction["lattice_filter_min_keep"],
                    field_name="diffraction.lattice_filter_min_keep",
                )
            )
            if lattice_filter_min_keep < 2:
                raise ValueError(
                    "workflow.parameters.diffraction.lattice_filter_min_keep must be >= 2."
                )
        if "origin_g_norm_threshold" in diffraction:
            origin_g_norm_threshold = _as_float(
                diffraction["origin_g_norm_threshold"],
                field_name="diffraction.origin_g_norm_threshold",
            )
            if origin_g_norm_threshold < 0:
                raise ValueError(
                    "workflow.parameters.diffraction.origin_g_norm_threshold must be >= 0."
                )
        if "origin_hkl_g_max" in diffraction:
            origin_hkl_g_max = _as_float(
                diffraction["origin_hkl_g_max"],
                field_name="diffraction.origin_hkl_g_max",
            )
            if origin_hkl_g_max < 0:
                raise ValueError("workflow.parameters.diffraction.origin_hkl_g_max must be >= 0.")
        if "origin_filter_max_radius_px" in diffraction:
            origin_filter_max_radius_px = _as_float(
                diffraction["origin_filter_max_radius_px"],
                field_name="diffraction.origin_filter_max_radius_px",
            )
            if origin_filter_max_radius_px < 0:
                raise ValueError(
                    "workflow.parameters.diffraction.origin_filter_max_radius_px must be >= 0."
                )
        if "candidate_phases" in diffraction:
            candidate_phases = diffraction["candidate_phases"]
            if not isinstance(candidate_phases, list) or any(
                not isinstance(item, str) or not item.strip() for item in candidate_phases
            ):
                raise ValueError(
                    "workflow.parameters.diffraction.candidate_phases must be a non-empty "
                    "string list."
                )
        if "indexing_mode" in diffraction:
            indexing_mode = str(diffraction["indexing_mode"]).strip().lower()
            if indexing_mode not in _SUPPORTED_INDEXING_MODES:
                raise ValueError(
                    "workflow.parameters.diffraction.indexing_mode must be one of "
                    f"{sorted(_SUPPORTED_INDEXING_MODES)}."
                )
        if "peak_threshold_strategy" in diffraction:
            strategy = str(diffraction["peak_threshold_strategy"]).strip().lower()
            if strategy not in _SUPPORTED_PEAK_THRESHOLD_STRATEGIES:
                raise ValueError(
                    "workflow.parameters.diffraction.peak_threshold_strategy must be one of "
                    f"{sorted(_SUPPORTED_PEAK_THRESHOLD_STRATEGIES)}."
                )
        if "peak_threshold_value" in diffraction:
            _as_float(
                diffraction["peak_threshold_value"],
                field_name="diffraction.peak_threshold_value",
            )
        if "fft_window" in diffraction:
            fft_window = str(diffraction["fft_window"]).strip().lower()
            if fft_window not in {"none", "hann", "tukey"}:
                raise ValueError(
                    "workflow.parameters.diffraction.fft_window must be one of "
                    "['none', 'hann', 'tukey']."
                )
        if "fft_window_tukey_alpha" in diffraction:
            alpha = _as_float(
                diffraction["fft_window_tukey_alpha"],
                field_name="diffraction.fft_window_tukey_alpha",
            )
            if alpha < 0 or alpha > 1:
                raise ValueError(
                    "workflow.parameters.diffraction.fft_window_tukey_alpha "
                    "must satisfy 0 <= value <= 1."
                )
        if "fft_compression" in diffraction:
            comp = str(diffraction["fft_compression"]).strip().lower()
            if comp not in {"log1p", "asinh"}:
                raise ValueError(
                    "workflow.parameters.diffraction.fft_compression must be one of "
                    "['log1p', 'asinh']."
                )
        if "background_subtraction_method" in diffraction:
            method = str(diffraction["background_subtraction_method"]).strip().lower()
            if method not in {"radial_mad", "gaussian", "hybrid"}:
                raise ValueError(
                    "workflow.parameters.diffraction.background_subtraction_method "
                    "must be one of ['radial_mad', 'gaussian', 'hybrid']."
                )
        if "gaussian_background_sigma" in diffraction:
            sigma = _as_float(
                diffraction["gaussian_background_sigma"],
                field_name="diffraction.gaussian_background_sigma",
            )
            if sigma <= 0:
                raise ValueError(
                    "workflow.parameters.diffraction.gaussian_background_sigma must be > 0."
                )
        if "clip_low_percentile" in diffraction:
            clip_low = _as_float(
                diffraction["clip_low_percentile"],
                field_name="diffraction.clip_low_percentile",
            )
            if clip_low < 0 or clip_low >= 100:
                raise ValueError(
                    "workflow.parameters.diffraction.clip_low_percentile must satisfy "
                    "0 <= value < 100."
                )
        if "clip_high_percentile" in diffraction:
            clip_high = _as_float(
                diffraction["clip_high_percentile"],
                field_name="diffraction.clip_high_percentile",
            )
            if clip_high <= 0 or clip_high > 100:
                raise ValueError(
                    "workflow.parameters.diffraction.clip_high_percentile must satisfy "
                    "0 < value <= 100."
                )
        if (
            "clip_low_percentile" in diffraction
            and "clip_high_percentile" in diffraction
            and float(diffraction["clip_high_percentile"])
            <= float(diffraction["clip_low_percentile"])
        ):
            raise ValueError(
                "workflow.parameters.diffraction.clip_high_percentile must be greater "
                "than clip_low_percentile."
            )
        if "subpixel_window_radius" in diffraction:
            radius = int(
                _as_float(
                    diffraction["subpixel_window_radius"],
                    field_name="diffraction.subpixel_window_radius",
                )
            )
            if radius < 1:
                raise ValueError(
                    "workflow.parameters.diffraction.subpixel_window_radius must be >= 1."
                )
        if "subpixel_max_shift" in diffraction:
            max_shift = _as_float(
                diffraction["subpixel_max_shift"],
                field_name="diffraction.subpixel_max_shift",
            )
            if max_shift < 0:
                raise ValueError(
                    "workflow.parameters.diffraction.subpixel_max_shift must be >= 0."
                )
        if "max_hkl" in diffraction:
            max_hkl = int(_as_float(diffraction["max_hkl"], field_name="diffraction.max_hkl"))
            if max_hkl < 1:
                raise ValueError("workflow.parameters.diffraction.max_hkl must be >= 1.")
        if "d_spacing_tolerance_rel" in diffraction:
            d_tol = _as_float(
                diffraction["d_spacing_tolerance_rel"],
                field_name="diffraction.d_spacing_tolerance_rel",
            )
            if d_tol <= 0:
                raise ValueError(
                    "workflow.parameters.diffraction.d_spacing_tolerance_rel must be > 0."
                )
        if "angle_tolerance_deg" in diffraction:
            angle_tol = _as_float(
                diffraction["angle_tolerance_deg"],
                field_name="diffraction.angle_tolerance_deg",
            )
            if angle_tol <= 0 or angle_tol > 180:
                raise ValueError(
                    "workflow.parameters.diffraction.angle_tolerance_deg "
                    "must satisfy 0 < value <= 180."
                )
        if "candidate_phase_params" in diffraction:
            phase_params = diffraction["candidate_phase_params"]
            if not isinstance(phase_params, list):
                raise ValueError(
                    "workflow.parameters.diffraction.candidate_phase_params must be a list."
                )
            for idx, row in enumerate(phase_params):
                if not isinstance(row, dict):
                    raise ValueError(
                        "workflow.parameters.diffraction.candidate_phase_params"
                        f"[{idx}] must be a mapping."
                    )

    def _validate_acceptance_criteria(
        value: Any,
        *,
        field_name: str,
    ) -> None:
        if value is None:
            return
        if not isinstance(value, list):
            raise ValueError(f"{field_name} must be a list when provided.")
        for idx, row in enumerate(value):
            if not isinstance(row, dict):
                raise ValueError(f"{field_name}[{idx}] must be a mapping.")
            metric = row.get("metric")
            threshold = row.get("threshold")
            op = row.get("op")
            if not isinstance(metric, str) or not metric.strip():
                raise ValueError(f"{field_name}[{idx}].metric must be a non-empty string.")
            if not isinstance(threshold, (int, float)):
                raise ValueError(f"{field_name}[{idx}].threshold must be numeric.")
            if not isinstance(op, str) or op not in _SUPPORTED_ACCEPTANCE_OPS:
                raise ValueError(
                    f"{field_name}[{idx}].op must be one of {sorted(_SUPPORTED_ACCEPTANCE_OPS)}."
                )

    validation = params.get("validation")
    if validation is not None:
        if not isinstance(validation, dict):
            raise ValueError("workflow.parameters.validation must be a mapping.")
        _validate_acceptance_criteria(
            validation.get("acceptance_criteria"),
            field_name="workflow.parameters.validation.acceptance_criteria",
        )

    benchmark = params.get("benchmark")
    if benchmark is not None:
        if not isinstance(benchmark, dict):
            raise ValueError("workflow.parameters.benchmark must be a mapping.")
        case_id = benchmark.get("case_id")
        if case_id is not None and (not isinstance(case_id, str) or not case_id.strip()):
            raise ValueError("workflow.parameters.benchmark.case_id must be a non-empty string.")
        reference = benchmark.get("reference")
        if reference is not None and not isinstance(reference, dict):
            raise ValueError("workflow.parameters.benchmark.reference must be a mapping.")
        _validate_acceptance_criteria(
            benchmark.get("acceptance_criteria"),
            field_name="workflow.parameters.benchmark.acceptance_criteria",
        )

    return params


def load_workflow_recipe(workflow_path: str | Path | None) -> WorkflowRecipe:
    """Load and normalize a YAML workflow recipe; fallback to defaults when omitted."""
    if workflow_path is None:
        return default_workflow_recipe()

    path = Path(workflow_path)
    payload = _as_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))
    workflow_payload = _as_mapping(payload.get("workflow"))
    outputs_payload = _as_mapping(payload.get("outputs"))

    raw_steps = workflow_payload.get("steps", list(DEFAULT_STEP_SEQUENCE))
    normalized_steps = normalize_step_names([str(step) for step in raw_steps])
    output_formats = _validate_output_formats(
        [str(item) for item in outputs_payload.get("formats", ["json"])]
    )
    save_intermediates = bool(outputs_payload.get("save_intermediates", True))
    workflow_parameters = _validate_workflow_parameters(
        _as_mapping(workflow_payload.get("parameters"))
    )

    workflow = WorkflowConfig(
        name=str(workflow_payload.get("name", path.stem)),
        description=str(workflow_payload.get("description", "")),
        steps=[WorkflowStep(name=step) for step in normalized_steps],
        parameters=workflow_parameters,
        output_formats=output_formats,
        save_intermediates=save_intermediates,
        provenance={
            "source": "yaml",
            "workflow_path": str(path),
            "workflow_parameter_schema_version": WORKFLOW_PARAMETER_SCHEMA_VERSION,
        },
    )

    return WorkflowRecipe(
        workflow=workflow,
        outputs=WorkflowOutputs(
            formats=output_formats,
            save_intermediates=save_intermediates,
        ),
    )


def save_effective_workflow(recipe: WorkflowRecipe, output_dir: str | Path) -> Path:
    """Persist normalized workflow config to the run output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "workflow_used.yaml"
    payload = {
        "workflow": {
            "name": recipe.workflow.name,
            "description": recipe.workflow.description,
            "workflow_parameter_schema_version": recipe.workflow.provenance.get(
                "workflow_parameter_schema_version",
                WORKFLOW_PARAMETER_SCHEMA_VERSION,
            ),
            "steps": [step.name for step in recipe.workflow.steps],
            "parameters": recipe.workflow.parameters,
            "output_formats": recipe.workflow.output_formats,
            "save_intermediates": recipe.workflow.save_intermediates,
            "provenance": recipe.workflow.provenance,
        },
        "outputs": {
            "formats": recipe.outputs.formats,
            "save_intermediates": recipe.outputs.save_intermediates,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
