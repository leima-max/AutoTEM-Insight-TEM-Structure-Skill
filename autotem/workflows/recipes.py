"""Workflow recipe loading, normalization, and persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from autotem.calibration.open_source_adapters import SUPPORTED_DIFFRACTION_CALIBRATION_ENGINES
from autotem.core.models import AutoTEMModel, WorkflowConfig, WorkflowStep
from autotem.diffraction.open_source_adapters import SUPPORTED_PEAK_ADAPTER_ENGINES
from autotem.diffraction.phase_lookup import supported_phase_lookup_sources

DEFAULT_WORKFLOW_NAME = "a2_default"
DEFAULT_WORKFLOW_DESCRIPTION = "A2 baseline load-calibrate-diffraction-simulation workflow."
WORKFLOW_PARAMETER_SCHEMA_VERSION = "2026-05-29.b5.3"
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
_SUPPORTED_PREPROCESS_RECIPES = {"auto", "min_max"}
_SUPPORTED_DENOISE_METHODS = {"auto", "none", "gaussian", "median", "wiener", "nlm"}
_SUPPORTED_BACKGROUND_METHODS = {"none", "gaussian", "rolling_ball"}
_SUPPORTED_PEAK_THRESHOLD_STRATEGIES = {"auto", "quantile", "std", "relative_max", "mad"}
_SUPPORTED_PEAK_DETECTORS = {"local_max", "log", "dog"}
_SUPPORTED_INDEXING_MODES = {"auto", "baseline", "lattice_fit"}
_SUPPORTED_FFT_CENTER_MODES = {"auto", "geometric"}
_SUPPORTED_PHASE_LOOKUP_SOURCES = supported_phase_lookup_sources()
_SUPPORTED_MATERIALS_PROJECT_SORT_MODES = {
    "stability",
    "energy_above_hull",
    "material_id",
    "none",
}
_SUPPORTED_SIMULATION_ENGINES = {"baseline", "diffsims"}
_SUPPORTED_ACCEPTANCE_OPS = {">=", "<=", ">", "<", "=="}
_QUALITY_THRESHOLD_RANGE_01_KEYS = {
    "blur_risk_max",
    "contamination_risk_max",
    "astigmatism_risk_max",
    "undersampling_risk_max",
    "overall_score_min",
}
_QUALITY_THRESHOLD_NONNEG_KEYS = {"snr_min", "drift_px_max"}
_QUALITY_WEIGHT_KEYS = {
    "snr",
    "drift",
    "blur",
    "contamination",
    "astigmatism",
    "undersampling",
    "metadata",
}


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


def _as_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
            return token in {"true", "1", "yes", "on"}
    raise ValueError(f"Workflow parameter '{field_name}' must be boolean.")


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


def _validate_candidate_phase_params(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise ValueError(
            "workflow.parameters.diffraction.candidate_phase_params must be a list."
        )

    def _phase_numeric(raw: Any, *, field_name: str) -> float:
        if isinstance(raw, bool):
            raise ValueError(f"{field_name} must be numeric.")
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str) and raw.strip():
            token = raw.strip()
            keep: list[str] = []
            seen_dot = False
            seen_exp = False
            for idx, ch in enumerate(token):
                if ch in "+-" and (idx == 0 or (seen_exp and keep[-1] in {"e", "E"})):
                    keep.append(ch)
                    continue
                if ch.isdigit():
                    keep.append(ch)
                    continue
                if ch == "." and not seen_dot and not seen_exp:
                    seen_dot = True
                    keep.append(ch)
                    continue
                if ch in "eE" and not seen_exp and keep:
                    seen_exp = True
                    keep.append(ch)
                    continue
                break
            try:
                return float("".join(keep))
            except ValueError as exc:
                raise ValueError(f"{field_name} must be numeric.") from exc
        raise ValueError(f"{field_name} must be numeric.")

    for idx, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(
                f"workflow.parameters.diffraction.candidate_phase_params[{idx}] "
                "must be a mapping."
            )
        name = row.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError(
                f"workflow.parameters.diffraction.candidate_phase_params[{idx}].name "
                "must be a non-empty string when provided."
            )
        for axis in ("a", "b", "c"):
            if axis not in row:
                raise ValueError(
                    "workflow.parameters.diffraction."
                    f"candidate_phase_params[{idx}].{axis} is required."
                )
            length = _phase_numeric(
                row[axis],
                field_name=(
                    "workflow.parameters.diffraction."
                    f"candidate_phase_params[{idx}].{axis}"
                ),
            )
            if length <= 0:
                raise ValueError(
                    "workflow.parameters.diffraction."
                    f"candidate_phase_params[{idx}].{axis} must be > 0."
                )
        for angle_name in ("alpha", "beta", "gamma"):
            if angle_name not in row:
                continue
            angle = _phase_numeric(
                row[angle_name],
                field_name=(
                    "workflow.parameters.diffraction."
                    f"candidate_phase_params[{idx}].{angle_name}"
                ),
            )
            if not (0 < angle < 180):
                raise ValueError(
                    "workflow.parameters.diffraction."
                    f"candidate_phase_params[{idx}].{angle_name} must be within (0, 180)."
                )


def _validate_non_empty_string_list(value: Any, *, field_name: str) -> None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(
            f"workflow.parameters.diffraction.{field_name} must be a non-empty string list."
        )


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
        thresholds_by_modality = quality.get("thresholds_by_modality")
        if thresholds_by_modality is not None:
            if not isinstance(thresholds_by_modality, dict):
                raise ValueError(
                    "workflow.parameters.quality.thresholds_by_modality must be a mapping."
                )
            for modality_key, row in thresholds_by_modality.items():
                if not isinstance(modality_key, str) or not modality_key.strip():
                    raise ValueError(
                        "workflow.parameters.quality.thresholds_by_modality keys "
                        "must be non-empty strings."
                    )
                if not isinstance(row, dict):
                    raise ValueError(
                        "workflow.parameters.quality.thresholds_by_modality entries "
                        "must be mappings."
                    )
                for key in _QUALITY_THRESHOLD_NONNEG_KEYS:
                    if key in row:
                        value = _as_float(
                            row[key],
                            field_name=(
                                "quality.thresholds_by_modality."
                                f"{modality_key}.{key}"
                            ),
                        )
                        if value < 0:
                            raise ValueError(
                                "workflow.parameters.quality.thresholds_by_modality."
                                f"{modality_key}.{key} must be >= 0."
                            )
                for key in _QUALITY_THRESHOLD_RANGE_01_KEYS:
                    if key in row:
                        value = _as_float(
                            row[key],
                            field_name=(
                                "quality.thresholds_by_modality."
                                f"{modality_key}.{key}"
                            ),
                        )
                        if not (0 <= value <= 1):
                            raise ValueError(
                                "workflow.parameters.quality.thresholds_by_modality."
                                f"{modality_key}.{key} must be within [0, 1]."
                            )
        metric_weights = quality.get("metric_weights")
        if metric_weights is not None:
            if not isinstance(metric_weights, dict):
                raise ValueError("workflow.parameters.quality.metric_weights must be a mapping.")
            for key, raw_value in metric_weights.items():
                if key not in _QUALITY_WEIGHT_KEYS:
                    continue
                value = _as_float(raw_value, field_name=f"quality.metric_weights.{key}")
                if value <= 0:
                    raise ValueError(
                        f"workflow.parameters.quality.metric_weights.{key} must be > 0."
                    )
        metric_weights_by_modality = quality.get("metric_weights_by_modality")
        if metric_weights_by_modality is not None:
            if not isinstance(metric_weights_by_modality, dict):
                raise ValueError(
                    "workflow.parameters.quality.metric_weights_by_modality must be a mapping."
                )
            for modality_key, row in metric_weights_by_modality.items():
                if not isinstance(modality_key, str) or not modality_key.strip():
                    raise ValueError(
                        "workflow.parameters.quality.metric_weights_by_modality keys "
                        "must be non-empty strings."
                    )
                if not isinstance(row, dict):
                    raise ValueError(
                        "workflow.parameters.quality.metric_weights_by_modality entries "
                        "must be mappings."
                    )
                for key, raw_value in row.items():
                    if key not in _QUALITY_WEIGHT_KEYS:
                        continue
                    value = _as_float(
                        raw_value,
                        field_name=(
                            "quality.metric_weights_by_modality."
                            f"{modality_key}.{key}"
                        ),
                    )
                    if value <= 0:
                        raise ValueError(
                            "workflow.parameters.quality.metric_weights_by_modality."
                            f"{modality_key}.{key} must be > 0."
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
        if "multi_reference_outlier_sigma" in calibration:
            outlier_sigma = _as_float(
                calibration["multi_reference_outlier_sigma"],
                field_name="calibration.multi_reference_outlier_sigma",
            )
            if outlier_sigma < 0:
                raise ValueError(
                    "workflow.parameters.calibration.multi_reference_outlier_sigma must be >= 0."
                )
        if "multi_reference_min_inliers" in calibration:
            min_inliers = int(
                _as_float(
                    calibration["multi_reference_min_inliers"],
                    field_name="calibration.multi_reference_min_inliers",
                )
            )
            if min_inliers < 1:
                raise ValueError(
                    "workflow.parameters.calibration.multi_reference_min_inliers must be >= 1."
                )
        if "known_d_spacing_refs" in calibration:
            refs = calibration["known_d_spacing_refs"]
            if not isinstance(refs, list):
                raise ValueError(
                    "workflow.parameters.calibration.known_d_spacing_refs must be a list."
                )
            for idx, row in enumerate(refs):
                if isinstance(row, dict):
                    if "known_d_spacing" not in row or "measured_radius_px" not in row:
                        raise ValueError(
                            "workflow.parameters.calibration.known_d_spacing_refs."
                            f"[{idx}] must include known_d_spacing and measured_radius_px."
                        )
                    known_d = _as_float(
                        row["known_d_spacing"],
                        field_name=f"calibration.known_d_spacing_refs[{idx}].known_d_spacing",
                    )
                    radius = _as_float(
                        row["measured_radius_px"],
                        field_name=f"calibration.known_d_spacing_refs[{idx}].measured_radius_px",
                    )
                    if known_d <= 0 or radius <= 0:
                        raise ValueError(
                            "workflow.parameters.calibration.known_d_spacing_refs."
                            f"[{idx}] known_d_spacing and measured_radius_px must be > 0."
                        )
                elif isinstance(row, (list, tuple)) and len(row) >= 2:
                    known_d = _as_float(
                        row[0],
                        field_name=f"calibration.known_d_spacing_refs[{idx}][0]",
                    )
                    radius = _as_float(
                        row[1],
                        field_name=f"calibration.known_d_spacing_refs[{idx}][1]",
                    )
                    if known_d <= 0 or radius <= 0:
                        raise ValueError(
                            "workflow.parameters.calibration.known_d_spacing_refs."
                            f"[{idx}] values must be > 0."
                        )
                else:
                    raise ValueError(
                        "workflow.parameters.calibration.known_d_spacing_refs."
                        f"[{idx}] must be a mapping or [known_d_spacing, measured_radius_px]."
                    )
        known_d_spacings = calibration.get("known_d_spacings")
        measured_radii_px_list = calibration.get("measured_radii_px")
        if known_d_spacings is not None or measured_radii_px_list is not None:
            if (
                not isinstance(known_d_spacings, list)
                or not isinstance(measured_radii_px_list, list)
            ):
                raise ValueError(
                    "workflow.parameters.calibration.known_d_spacings and measured_radii_px "
                    "must be lists when either is provided."
                )
            if not known_d_spacings or not measured_radii_px_list:
                raise ValueError(
                    "workflow.parameters.calibration.known_d_spacings and measured_radii_px "
                    "must be non-empty lists."
                )
            for idx, value in enumerate(known_d_spacings):
                known_d = _as_float(
                    value,
                    field_name=f"calibration.known_d_spacings[{idx}]",
                )
                if known_d <= 0:
                    raise ValueError(
                        f"workflow.parameters.calibration.known_d_spacings[{idx}] must be > 0."
                    )
            for idx, value in enumerate(measured_radii_px_list):
                radius = _as_float(
                    value,
                    field_name=f"calibration.measured_radii_px[{idx}]",
                )
                if radius <= 0:
                    raise ValueError(
                        f"workflow.parameters.calibration.measured_radii_px[{idx}] must be > 0."
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
        if "denoise_method" in preprocessing:
            denoise_method = str(preprocessing["denoise_method"]).strip().lower()
            if denoise_method not in _SUPPORTED_DENOISE_METHODS:
                raise ValueError(
                    "workflow.parameters.preprocessing.denoise_method must be one of "
                    f"{sorted(_SUPPORTED_DENOISE_METHODS)}."
                )
        if "median_size" in preprocessing:
            median_size = int(
                _as_float(
                    preprocessing["median_size"],
                    field_name="preprocessing.median_size",
                )
            )
            if median_size < 1:
                raise ValueError("workflow.parameters.preprocessing.median_size must be >= 1.")
        if "wiener_window" in preprocessing:
            wiener_window = int(
                _as_float(
                    preprocessing["wiener_window"],
                    field_name="preprocessing.wiener_window",
                )
            )
            if wiener_window < 3:
                raise ValueError("workflow.parameters.preprocessing.wiener_window must be >= 3.")
        if "wiener_noise" in preprocessing:
            wiener_noise = _as_float(
                preprocessing["wiener_noise"],
                field_name="preprocessing.wiener_noise",
            )
            if wiener_noise < 0:
                raise ValueError("workflow.parameters.preprocessing.wiener_noise must be >= 0.")
        if "nlm_h" in preprocessing:
            nlm_h = _as_float(
                preprocessing["nlm_h"],
                field_name="preprocessing.nlm_h",
            )
            if nlm_h <= 0:
                raise ValueError("workflow.parameters.preprocessing.nlm_h must be > 0.")
        if "nlm_patch_size" in preprocessing:
            patch_size = int(
                _as_float(
                    preprocessing["nlm_patch_size"],
                    field_name="preprocessing.nlm_patch_size",
                )
            )
            if patch_size < 3:
                raise ValueError(
                    "workflow.parameters.preprocessing.nlm_patch_size must be >= 3."
                )
        if "nlm_patch_distance" in preprocessing:
            patch_distance = int(
                _as_float(
                    preprocessing["nlm_patch_distance"],
                    field_name="preprocessing.nlm_patch_distance",
                )
            )
            if patch_distance < 1:
                raise ValueError(
                    "workflow.parameters.preprocessing.nlm_patch_distance must be >= 1."
                )
        if "background_sigma" in preprocessing:
            background_sigma = _as_float(
                preprocessing["background_sigma"],
                field_name="preprocessing.background_sigma",
            )
            if background_sigma < 0:
                raise ValueError(
                    "workflow.parameters.preprocessing.background_sigma must be >= 0."
                )
        if "background_method" in preprocessing:
            background_method = str(preprocessing["background_method"]).strip().lower()
            if background_method not in _SUPPORTED_BACKGROUND_METHODS:
                raise ValueError(
                    "workflow.parameters.preprocessing.background_method must be one of "
                    f"{sorted(_SUPPORTED_BACKGROUND_METHODS)}."
                )
        if "background_opening_size" in preprocessing:
            opening_size = int(
                _as_float(
                    preprocessing["background_opening_size"],
                    field_name="preprocessing.background_opening_size",
                )
            )
            if opening_size < 3:
                raise ValueError(
                    "workflow.parameters.preprocessing.background_opening_size must be >= 3."
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
        if "auto_peak_tuning" in diffraction:
            _as_bool(
                diffraction["auto_peak_tuning"],
                field_name="diffraction.auto_peak_tuning",
            )
        if "min_peak_distance" in diffraction:
            min_distance = _as_float(
                diffraction["min_peak_distance"],
                field_name="diffraction.min_peak_distance",
            )
            if min_distance < 0:
                raise ValueError("workflow.parameters.diffraction.min_peak_distance must be >= 0.")
        if "center_exclusion_radius" in diffraction:
            center_radius = _as_float(
                diffraction["center_exclusion_radius"],
                field_name="diffraction.center_exclusion_radius",
            )
            if center_radius < 0:
                raise ValueError(
                    "workflow.parameters.diffraction.center_exclusion_radius must be >= 0."
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
        _validate_candidate_phase_params(diffraction.get("candidate_phase_params"))
        if "cif_candidate_paths" in diffraction:
            _validate_non_empty_string_list(
                diffraction["cif_candidate_paths"],
                field_name="cif_candidate_paths",
            )
        if "indexing_mode" in diffraction:
            indexing_mode = str(diffraction["indexing_mode"]).strip().lower()
            if indexing_mode not in _SUPPORTED_INDEXING_MODES:
                raise ValueError(
                    "workflow.parameters.diffraction.indexing_mode must be one of "
                    f"{sorted(_SUPPORTED_INDEXING_MODES)}."
                )
        if "max_hkl" in diffraction:
            max_hkl = int(_as_float(diffraction["max_hkl"], field_name="diffraction.max_hkl"))
            if max_hkl < 1:
                raise ValueError("workflow.parameters.diffraction.max_hkl must be >= 1.")
        if "d_tolerance_rel" in diffraction:
            d_tolerance_rel = _as_float(
                diffraction["d_tolerance_rel"],
                field_name="diffraction.d_tolerance_rel",
            )
            if d_tolerance_rel <= 0:
                raise ValueError("workflow.parameters.diffraction.d_tolerance_rel must be > 0.")
        if "angle_tolerance_deg" in diffraction:
            angle_tolerance = _as_float(
                diffraction["angle_tolerance_deg"],
                field_name="diffraction.angle_tolerance_deg",
            )
            if angle_tolerance <= 0:
                raise ValueError(
                    "workflow.parameters.diffraction.angle_tolerance_deg must be > 0."
                )
        if "b4_strict_mode" in diffraction:
            _as_bool(diffraction["b4_strict_mode"], field_name="diffraction.b4_strict_mode")
        if "cif_candidates_dir" in diffraction:
            cif_candidates_dir = diffraction["cif_candidates_dir"]
            if not isinstance(cif_candidates_dir, str) or not cif_candidates_dir.strip():
                raise ValueError(
                    "workflow.parameters.diffraction.cif_candidates_dir must be a "
                    "non-empty string."
                )
        if "phase_lookup_enabled" in diffraction:
            _as_bool(
                diffraction["phase_lookup_enabled"],
                field_name="diffraction.phase_lookup_enabled",
            )
        if "phase_lookup_formula" in diffraction:
            phase_lookup_formula = diffraction["phase_lookup_formula"]
            if not isinstance(phase_lookup_formula, str) or not phase_lookup_formula.strip():
                raise ValueError(
                    "workflow.parameters.diffraction.phase_lookup_formula must be a "
                    "non-empty string."
                )
        if "phase_lookup_materials" in diffraction:
            _validate_non_empty_string_list(
                diffraction["phase_lookup_materials"],
                field_name="phase_lookup_materials",
            )
        if "phase_lookup_elements" in diffraction:
            _validate_non_empty_string_list(
                diffraction["phase_lookup_elements"],
                field_name="phase_lookup_elements",
            )
        if "phase_lookup_sources" in diffraction:
            _validate_non_empty_string_list(
                diffraction["phase_lookup_sources"],
                field_name="phase_lookup_sources",
            )
            invalid_sources = [
                item
                for item in diffraction["phase_lookup_sources"]
                if str(item).strip() not in _SUPPORTED_PHASE_LOOKUP_SOURCES
            ]
            if invalid_sources:
                raise ValueError(
                    "workflow.parameters.diffraction.phase_lookup_sources must contain "
                    f"only {sorted(_SUPPORTED_PHASE_LOOKUP_SOURCES)}."
                )
        if "materials_project_ids" in diffraction:
            _validate_non_empty_string_list(
                diffraction["materials_project_ids"],
                field_name="materials_project_ids",
            )
        if "materials_project_config_path" in diffraction:
            mp_config_path = diffraction["materials_project_config_path"]
            if not isinstance(mp_config_path, str) or not mp_config_path.strip():
                raise ValueError(
                    "workflow.parameters.diffraction.materials_project_config_path "
                    "must be a non-empty string."
                )
        if "materials_project_api_key" in diffraction:
            mp_api_key = diffraction["materials_project_api_key"]
            if not isinstance(mp_api_key, str) or not mp_api_key.strip():
                raise ValueError(
                    "workflow.parameters.diffraction.materials_project_api_key "
                    "must be a non-empty string."
                )
        if "materials_project_sort_by" in diffraction:
            mp_sort_by = str(diffraction["materials_project_sort_by"]).strip().lower()
            if mp_sort_by not in _SUPPORTED_MATERIALS_PROJECT_SORT_MODES:
                raise ValueError(
                    "workflow.parameters.diffraction.materials_project_sort_by "
                    f"must be one of {sorted(_SUPPORTED_MATERIALS_PROJECT_SORT_MODES)}."
                )
        if "materials_project_include_deprecated" in diffraction:
            _as_bool(
                diffraction["materials_project_include_deprecated"],
                field_name="diffraction.materials_project_include_deprecated",
            )
        if "materials_project_filter_stable" in diffraction:
            _as_bool(
                diffraction["materials_project_filter_stable"],
                field_name="diffraction.materials_project_filter_stable",
            )
        if "materials_project_prefer_experimental" in diffraction:
            _as_bool(
                diffraction["materials_project_prefer_experimental"],
                field_name="diffraction.materials_project_prefer_experimental",
            )
        if "materials_project_max_energy_above_hull" in diffraction:
            mp_max_hull = _as_float(
                diffraction["materials_project_max_energy_above_hull"],
                field_name="diffraction.materials_project_max_energy_above_hull",
            )
            if mp_max_hull < 0:
                raise ValueError(
                    "workflow.parameters.diffraction."
                    "materials_project_max_energy_above_hull must be >= 0."
                )
        if "materials_project_candidate_pool_size" in diffraction:
            mp_pool_size = int(
                _as_float(
                    diffraction["materials_project_candidate_pool_size"],
                    field_name="diffraction.materials_project_candidate_pool_size",
                )
            )
            if mp_pool_size < 1:
                raise ValueError(
                    "workflow.parameters.diffraction."
                    "materials_project_candidate_pool_size must be >= 1."
                )
        if "phase_lookup_limit_per_source" in diffraction:
            lookup_limit = int(
                _as_float(
                    diffraction["phase_lookup_limit_per_source"],
                    field_name="diffraction.phase_lookup_limit_per_source",
                )
            )
            if lookup_limit < 1:
                raise ValueError(
                    "workflow.parameters.diffraction.phase_lookup_limit_per_source "
                    "must be >= 1."
                )
        if "phase_lookup_timeout_s" in diffraction:
            lookup_timeout = _as_float(
                diffraction["phase_lookup_timeout_s"],
                field_name="diffraction.phase_lookup_timeout_s",
            )
            if lookup_timeout < 1:
                raise ValueError(
                    "workflow.parameters.diffraction.phase_lookup_timeout_s must be >= 1."
                )
        if "peak_threshold_strategy" in diffraction:
            strategy = str(diffraction["peak_threshold_strategy"]).strip().lower()
            if strategy not in _SUPPORTED_PEAK_THRESHOLD_STRATEGIES:
                raise ValueError(
                    "workflow.parameters.diffraction.peak_threshold_strategy must be one of "
                    f"{sorted(_SUPPORTED_PEAK_THRESHOLD_STRATEGIES)}."
                )
        if "peak_detector" in diffraction:
            detector = str(diffraction["peak_detector"]).strip().lower()
            if detector not in _SUPPORTED_PEAK_DETECTORS:
                raise ValueError(
                    "workflow.parameters.diffraction.peak_detector must be one of "
                    f"{sorted(_SUPPORTED_PEAK_DETECTORS)}."
                )
        if "peak_adapter_engine" in diffraction:
            peak_adapter_engine = str(diffraction["peak_adapter_engine"]).strip().lower()
            if peak_adapter_engine not in SUPPORTED_PEAK_ADAPTER_ENGINES:
                raise ValueError(
                    "workflow.parameters.diffraction.peak_adapter_engine must be one of "
                    f"{sorted(SUPPORTED_PEAK_ADAPTER_ENGINES)}."
                )
        if "peak_threshold_value" in diffraction:
            _as_float(
                diffraction["peak_threshold_value"],
                field_name="diffraction.peak_threshold_value",
            )
        if "detector_sigma" in diffraction:
            detector_sigma = _as_float(
                diffraction["detector_sigma"],
                field_name="diffraction.detector_sigma",
            )
            if detector_sigma <= 0:
                raise ValueError("workflow.parameters.diffraction.detector_sigma must be > 0.")
        if "dog_sigma_ratio" in diffraction:
            sigma_ratio = _as_float(
                diffraction["dog_sigma_ratio"],
                field_name="diffraction.dog_sigma_ratio",
            )
            if sigma_ratio <= 1:
                raise ValueError(
                    "workflow.parameters.diffraction.dog_sigma_ratio must be > 1."
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
        if "fft_center_mode" in diffraction:
            fft_center_mode = str(diffraction["fft_center_mode"]).strip().lower()
            if fft_center_mode not in _SUPPORTED_FFT_CENTER_MODES:
                raise ValueError(
                    "workflow.parameters.diffraction.fft_center_mode must be one of "
                    f"{sorted(_SUPPORTED_FFT_CENTER_MODES)}."
                )
        if "direct_beam_search_radius" in diffraction:
            direct_beam_search_radius = _as_float(
                diffraction["direct_beam_search_radius"],
                field_name="diffraction.direct_beam_search_radius",
            )
            if direct_beam_search_radius < 0:
                raise ValueError(
                    "workflow.parameters.diffraction.direct_beam_search_radius must be >= 0."
                )
        if "direct_beam_refine_window" in diffraction:
            direct_beam_refine_window = int(
                _as_float(
                    diffraction["direct_beam_refine_window"],
                    field_name="diffraction.direct_beam_refine_window",
                )
            )
            if direct_beam_refine_window < 1:
                raise ValueError(
                    "workflow.parameters.diffraction.direct_beam_refine_window must be >= 1."
                )
        if "direct_beam_center_uncertainty_px" in diffraction:
            center_uncertainty_px = _as_float(
                diffraction["direct_beam_center_uncertainty_px"],
                field_name="diffraction.direct_beam_center_uncertainty_px",
            )
            if center_uncertainty_px < 0:
                raise ValueError(
                    "workflow.parameters.diffraction.direct_beam_center_uncertainty_px "
                    "must be >= 0."
                )

    reporting = params.get("reporting")
    if reporting is not None:
        if not isinstance(reporting, dict):
            raise ValueError("workflow.parameters.reporting must be a mapping.")
        diffraction_gates = reporting.get("diffraction_gates")
        if diffraction_gates is not None:
            if not isinstance(diffraction_gates, dict):
                raise ValueError(
                    "workflow.parameters.reporting.diffraction_gates must be a mapping."
                )
            if "min_peak_count" in diffraction_gates:
                min_peak_count = int(
                    _as_float(
                        diffraction_gates["min_peak_count"],
                        field_name="reporting.diffraction_gates.min_peak_count",
                    )
                )
                if min_peak_count < 1:
                    raise ValueError(
                        "workflow.parameters.reporting.diffraction_gates.min_peak_count "
                        "must be >= 1."
                    )
            if "max_center_offset_px" in diffraction_gates:
                max_center_offset = _as_float(
                    diffraction_gates["max_center_offset_px"],
                    field_name="reporting.diffraction_gates.max_center_offset_px",
                )
                if max_center_offset < 0:
                    raise ValueError(
                        "workflow.parameters.reporting.diffraction_gates.max_center_offset_px "
                        "must be >= 0."
                    )
            if "min_uncertainty_coverage" in diffraction_gates:
                min_uncertainty_coverage = _as_float(
                    diffraction_gates["min_uncertainty_coverage"],
                    field_name="reporting.diffraction_gates.min_uncertainty_coverage",
                )
                if not (0 <= min_uncertainty_coverage <= 1):
                    raise ValueError(
                        "workflow.parameters.reporting.diffraction_gates."
                        "min_uncertainty_coverage must be within [0, 1]."
                    )
            if "max_mean_relative_uncertainty" in diffraction_gates:
                max_mean_relative_uncertainty = _as_float(
                    diffraction_gates["max_mean_relative_uncertainty"],
                    field_name="reporting.diffraction_gates.max_mean_relative_uncertainty",
                )
                if max_mean_relative_uncertainty < 0:
                    raise ValueError(
                        "workflow.parameters.reporting.diffraction_gates."
                        "max_mean_relative_uncertainty must be >= 0."
                    )
            if "min_indexed_nonzero_fraction" in diffraction_gates:
                min_indexed_fraction = _as_float(
                    diffraction_gates["min_indexed_nonzero_fraction"],
                    field_name="reporting.diffraction_gates.min_indexed_nonzero_fraction",
                )
                if not (0 <= min_indexed_fraction <= 1):
                    raise ValueError(
                        "workflow.parameters.reporting.diffraction_gates."
                        "min_indexed_nonzero_fraction must be within [0, 1]."
                    )

    simulation = params.get("simulation")
    if simulation is not None:
        if not isinstance(simulation, dict):
            raise ValueError("workflow.parameters.simulation must be a mapping.")
        if "simulation_engine" in simulation:
            engine = str(simulation["simulation_engine"]).strip().lower()
            if engine not in _SUPPORTED_SIMULATION_ENGINES:
                raise ValueError(
                    "workflow.parameters.simulation.simulation_engine must be one of "
                    f"{sorted(_SUPPORTED_SIMULATION_ENGINES)}."
                )
        if "voltage_kv" in simulation:
            voltage = _as_float(simulation["voltage_kv"], field_name="simulation.voltage_kv")
            if voltage <= 0:
                raise ValueError("workflow.parameters.simulation.voltage_kv must be > 0.")
        if "reciprocal_radius_inv_angstrom" in simulation:
            radius = _as_float(
                simulation["reciprocal_radius_inv_angstrom"],
                field_name="simulation.reciprocal_radius_inv_angstrom",
            )
            if radius <= 0:
                raise ValueError(
                    "workflow.parameters.simulation.reciprocal_radius_inv_angstrom must be > 0."
                )
        if "max_candidates" in simulation:
            max_candidates = int(
                _as_float(simulation["max_candidates"], field_name="simulation.max_candidates")
            )
            if max_candidates < 1:
                raise ValueError("workflow.parameters.simulation.max_candidates must be >= 1.")
        if "calibration" in simulation:
            calibration = _as_float(simulation["calibration"], field_name="simulation.calibration")
            if calibration <= 0:
                raise ValueError("workflow.parameters.simulation.calibration must be > 0.")
        if "pattern_shape" in simulation:
            shape = simulation["pattern_shape"]
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(not isinstance(item, (int, float)) or int(item) < 1 for item in shape)
            ):
                raise ValueError(
                    "workflow.parameters.simulation.pattern_shape must be a two-item "
                    "positive integer list."
                )
        if "with_direct_beam" in simulation:
            _as_bool(
                simulation["with_direct_beam"],
                field_name="simulation.with_direct_beam",
            )
        if "fallback_single_atom" in simulation:
            _as_bool(
                simulation["fallback_single_atom"],
                field_name="simulation.fallback_single_atom",
            )
        if "max_excitation_error" in simulation:
            max_excitation_error = _as_float(
                simulation["max_excitation_error"],
                field_name="simulation.max_excitation_error",
            )
            if max_excitation_error <= 0:
                raise ValueError(
                    "workflow.parameters.simulation.max_excitation_error must be > 0."
                )
        if "shape_factor_width" in simulation:
            shape_factor_width = _as_float(
                simulation["shape_factor_width"],
                field_name="simulation.shape_factor_width",
            )
            if shape_factor_width <= 0:
                raise ValueError(
                    "workflow.parameters.simulation.shape_factor_width must be > 0."
                )
        if "zone_axis_candidates" in simulation:
            zone_axes = simulation["zone_axis_candidates"]

            def _valid_zone_axis(raw_axis: Any) -> bool:
                return (
                    isinstance(raw_axis, list)
                    and len(raw_axis) == 3
                    and all(isinstance(item, (int, float)) for item in raw_axis)
                    and sum(float(item) * float(item) for item in raw_axis) > 0
                )

            if _valid_zone_axis(zone_axes):
                pass
            elif (
                not isinstance(zone_axes, list)
                or not zone_axes
                or any(not _valid_zone_axis(axis) for axis in zone_axes)
            ):
                raise ValueError(
                    "workflow.parameters.simulation.zone_axis_candidates must be a "
                    "three-item numeric list or a non-empty list of three-item numeric lists."
                )
        if "rotation_axis" in simulation:
            axis = simulation["rotation_axis"]
            if (
                not isinstance(axis, list)
                or len(axis) != 3
                or any(not isinstance(item, (int, float)) for item in axis)
            ):
                raise ValueError(
                    "workflow.parameters.simulation.rotation_axis must be a three-item "
                    "numeric list."
                )
            if sum(float(item) * float(item) for item in axis) <= 0:
                raise ValueError(
                    "workflow.parameters.simulation.rotation_axis must be non-zero."
                )
        if "rotation_angles_deg" in simulation:
            angles = simulation["rotation_angles_deg"]
            if (
                not isinstance(angles, list)
                or not angles
                or any(not isinstance(item, (int, float)) for item in angles)
            ):
                raise ValueError(
                    "workflow.parameters.simulation.rotation_angles_deg must be a "
                    "non-empty numeric list."
                )
        if "pattern_sigma" in simulation:
            pattern_sigma = _as_float(
                simulation["pattern_sigma"],
                field_name="simulation.pattern_sigma",
            )
            if pattern_sigma <= 0:
                raise ValueError("workflow.parameters.simulation.pattern_sigma must be > 0.")
        if "pattern_in_plane_angle_deg" in simulation:
            _as_float(
                simulation["pattern_in_plane_angle_deg"],
                field_name="simulation.pattern_in_plane_angle_deg",
            )
        if "pattern_mirrored" in simulation:
            _as_bool(
                simulation["pattern_mirrored"],
                field_name="simulation.pattern_mirrored",
            )
        if "pattern_normalize" in simulation:
            _as_bool(
                simulation["pattern_normalize"],
                field_name="simulation.pattern_normalize",
            )
        if "pattern_clip_threshold" in simulation:
            clip_threshold = _as_float(
                simulation["pattern_clip_threshold"],
                field_name="simulation.pattern_clip_threshold",
            )
            if clip_threshold < 0:
                raise ValueError(
                    "workflow.parameters.simulation.pattern_clip_threshold must be >= 0."
                )
        if "max_simulated_spots" in simulation:
            max_simulated_spots = int(
                _as_float(
                    simulation["max_simulated_spots"],
                    field_name="simulation.max_simulated_spots",
                )
            )
            if max_simulated_spots < 1:
                raise ValueError(
                    "workflow.parameters.simulation.max_simulated_spots must be >= 1."
                )
        if "minimum_simulated_g_inv_nm" in simulation:
            minimum_simulated_g = _as_float(
                simulation["minimum_simulated_g_inv_nm"],
                field_name="simulation.minimum_simulated_g_inv_nm",
            )
            if minimum_simulated_g < 0:
                raise ValueError(
                    "workflow.parameters.simulation.minimum_simulated_g_inv_nm must be >= 0."
                )
        if "phase_match_enabled" in simulation:
            _as_bool(
                simulation["phase_match_enabled"],
                field_name="simulation.phase_match_enabled",
            )
        if "phase_match_d_tolerance_rel" in simulation:
            d_tolerance = _as_float(
                simulation["phase_match_d_tolerance_rel"],
                field_name="simulation.phase_match_d_tolerance_rel",
            )
            if d_tolerance <= 0:
                raise ValueError(
                    "workflow.parameters.simulation.phase_match_d_tolerance_rel must be > 0."
                )
        if "phase_match_angle_tolerance_deg" in simulation:
            angle_tolerance = _as_float(
                simulation["phase_match_angle_tolerance_deg"],
                field_name="simulation.phase_match_angle_tolerance_deg",
            )
            if angle_tolerance <= 0:
                raise ValueError(
                    "workflow.parameters.simulation.phase_match_angle_tolerance_deg "
                    "must be > 0."
                )
        if "phase_match_min_spots" in simulation:
            min_spots = int(
                _as_float(
                    simulation["phase_match_min_spots"],
                    field_name="simulation.phase_match_min_spots",
                )
            )
            if min_spots < 1:
                raise ValueError(
                    "workflow.parameters.simulation.phase_match_min_spots must be >= 1."
                )
        if "phase_match_top_n" in simulation:
            top_n = int(
                _as_float(
                    simulation["phase_match_top_n"],
                    field_name="simulation.phase_match_top_n",
                )
            )
            if top_n < 1:
                raise ValueError(
                    "workflow.parameters.simulation.phase_match_top_n must be >= 1."
                )
        if "phase_match_previous_score_weight" in simulation:
            previous_score_weight = _as_float(
                simulation["phase_match_previous_score_weight"],
                field_name="simulation.phase_match_previous_score_weight",
            )
            if not (0 <= previous_score_weight <= 1):
                raise ValueError(
                    "workflow.parameters.simulation.phase_match_previous_score_weight "
                    "must be within [0, 1]."
                )
        if "phase_match_max_rotation_candidates" in simulation:
            max_rotations = int(
                _as_float(
                    simulation["phase_match_max_rotation_candidates"],
                    field_name="simulation.phase_match_max_rotation_candidates",
                )
            )
            if max_rotations < 1:
                raise ValueError(
                    "workflow.parameters.simulation.phase_match_max_rotation_candidates "
                    "must be >= 1."
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
