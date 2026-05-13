"""Workflow recipe loading, normalization, and persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from autotem.core.models import AutoTEMModel, WorkflowConfig, WorkflowStep

DEFAULT_WORKFLOW_NAME = "a2_default"
DEFAULT_WORKFLOW_DESCRIPTION = "A2 baseline load-calibrate-diffraction-simulation workflow."
WORKFLOW_PARAMETER_SCHEMA_VERSION = "2026-05-13.a2"
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
_SUPPORTED_ACCEPTANCE_OPS = {">=", "<=", ">", "<", "=="}


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
        reference = calibration.get("reference")
        if reference is not None and not isinstance(reference, dict):
            raise ValueError(
                "workflow.parameters.calibration.reference must be a mapping when provided."
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
