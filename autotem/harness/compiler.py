"""Explicit recipes and validated domain dependency graphs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .contracts import HarnessError, HarnessPlan


def finite_tree(value: Any):
    if isinstance(value, float) and not math.isfinite(value):
        raise HarnessError("INVALID_PARAMETERS", "Non-finite parameters are forbidden")
    if isinstance(value, dict):
        if {str(key).lower() for key in value} & {
            "api_key",
            "apikey",
            "access_token",
            "secret",
            "password",
            "authorization",
        }:
            raise HarnessError(
                "CREDENTIAL_PARAMETER_FORBIDDEN",
                "Credentials must be resolved outside analysis parameters",
            )
        for item in value.values():
            finite_tree(item)
    if isinstance(value, list):
        for item in value:
            finite_tree(item)


def compile_plan(source: dict | str | Path) -> HarnessPlan:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise HarnessError("RECIPE_NOT_FOUND", str(path))
        source = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    finite_tree(source)
    if not isinstance(source, dict):
        raise HarnessError("INVALID_PLAN", "Plan must be an object")
    try:
        return HarnessPlan.model_validate(source)
    except ValidationError as exc:
        raise HarnessError(
            "INVALID_PLAN",
            "Invalid plan schema",
            exc.errors(include_url=False, include_context=False, include_input=False),
        ) from exc


def required_roles(operation, parameters):
    required = set(operation.input_roles)
    if "refined_atoms" in required and "coordinate_role" in parameters:
        required.discard("refined_atoms")
        required.add(parameters["coordinate_role"])
    if "image_role" in parameters:
        required.add(parameters["image_role"])
    return required


def validate_graph(plan: HarnessPlan, registry, store=None, seed_roles=None, seed_steps=None):
    ids = [step.id for step in plan.steps]
    if len(set(ids)) != len(ids):
        raise HarnessError("DUPLICATE_STEP", "Step IDs must be unique")
    seen: set[str] = set(seed_steps or ())
    if seen.intersection(ids):
        raise HarnessError("IMMUTABLE_STEP", "Use new step IDs or create a child run")
    roles: set[str] = set(seed_roles or ())
    warnings = []
    profile_counts: dict[str, int] = {}
    profile_limits = {"data-quality": 8, "diffraction": 16, "atomic-strain": 24, "delivery": 8}
    for step in plan.steps:
        if not set(step.depends_on) <= seen:
            raise HarnessError(
                "INVALID_DEPENDENCY",
                "Dependencies must precede the node",
                {"step": step.id, "depends_on": step.depends_on},
            )
        operation = registry.get(step.operation_id)
        profile = operation.profile_id
        profile_counts[profile] = profile_counts.get(profile, 0) + 1
        if profile_counts[profile] > profile_limits.get(profile, 24):
            raise HarnessError(
                "PROFILE_STEP_LIMIT",
                "Split the work into bounded domain packages",
                {"profile_id": profile, "limit": profile_limits.get(profile, 24)},
            )
        if step.operation_version is not None and step.operation_version != operation.version:
            raise HarnessError("OPERATION_VERSION_MISMATCH", step.operation_id)
        step.operation_version = operation.version
        if operation.modality and plan.modality not in operation.modality:
            raise HarnessError("UNSUPPORTED_INPUT", f"{step.operation_id}: {plan.modality}")
        step.parameters = registry.validate_parameters(operation, step.parameters)
        if not plan.network_enabled and any(
            step.parameters.get(key) for key in ("network_enabled", "allow_network", "online")
        ):
            raise HarnessError("NETWORK_DISABLED", "Enable network explicitly in the plan")
        available = registry.availability(operation)
        if available["state"] != "available":
            if not step.optional:
                raise HarnessError(available["state"].upper(), step.operation_id, available)
            warnings.append({"step": step.id, "availability": available})
        supplied = set(step.inputs)
        for artifact_id in step.inputs.values():
            if store is None:
                raise HarnessError("ARTIFACT_STORE_REQUIRED", "Cannot validate artifact inputs")
            artifact = store.artifact(artifact_id)
            if not artifact["execution_usable"]:
                raise HarnessError("ARTIFACT_NOT_EXECUTABLE", "Generating step did not complete")
        required = required_roles(operation, step.parameters)
        missing = required - roles - supplied
        if missing:
            raise HarnessError("MISSING_DEPENDENCY", step.id, {"roles": sorted(missing)})
        roles.update(operation.output_roles)
        seen.add(step.id)
    return warnings


def compile_legacy_recipe(
    input_path: str | Path, workflow_path: str | Path, report_formats: list[str] | None = None,
) -> dict:
    """Translate explicitly requested legacy steps and account for every parameter."""
    from autotem.workflows.recipes import load_workflow_recipe

    from .registry import default_registry

    raw = yaml.safe_load(Path(workflow_path).read_text(encoding="utf-8-sig"))
    settings = raw.get("workflow") if isinstance(raw, dict) else None
    if not isinstance(settings, dict) or not isinstance(settings.get("steps"), list):
        raise HarnessError("INVALID_PLAN", "Specify the intended traditional workflow steps")
    output = raw.get("outputs", {})
    if not isinstance(output, dict) or not isinstance(settings.get("parameters", {}), dict):
        raise HarnessError("INVALID_PLAN", "Legacy outputs and parameters must be objects")
    unknown = (
        (set(raw) - {"workflow", "outputs"})
        | (set(settings) - {"name", "description", "steps", "parameters"})
        | (set(output) - {"formats", "save_intermediates"})
    )
    if unknown:
        raise HarnessError(
            "LEGACY_PARAMETER_REQUIRES_MIGRATION", ", ".join(sorted(map(str, unknown)))
        )
    mapping = {
        "load": "input.load",
        "assess_quality": "quality.assess",
        "calibrate": "calibration.resolve",
        "preprocess": "preprocessing.apply",
        "fft": "diffraction.fft",
        "diffraction_peaks": "diffraction.peaks",
        "index_diffraction": "diffraction.index",
        "phase_match": "diffraction.index",
        "detect_atoms": "atomic.detect",
        "run_gpa": "strain.gpa",
        "atom_strain": "strain.atom",
        "classify_sublattices": "atomic.sublattice.auto",
        "simulate": "simulation.diffraction",
    }
    for name in settings["steps"]:
        if not isinstance(name, str) or name not in {*mapping, "analyze", "report"}:
            raise HarnessError("LEGACY_STEP_REQUIRES_MIGRATION", str(name))
    recipe = load_workflow_recipe(workflow_path)
    if not recipe.outputs.save_intermediates:
        raise HarnessError("LEGACY_OUTPUT_REQUIRES_MIGRATION", "Harness retains full evidence")
    formats = recipe.outputs.formats if report_formats is None else report_formats
    unsupported = set(formats) - {"json", "html", "pdf"}
    if unsupported:
        raise HarnessError(
            "LEGACY_OUTPUT_REQUIRES_MIGRATION", "Choose JSON, HTML or PDF reports",
            {"unsupported_formats": sorted(unsupported)},
        )
    operations = []
    prerequisites = {
        "diffraction.peaks": ["diffraction.center"],
        "diffraction.index": ["diffraction.metrics"],
        "strain.atom": ["atomic.detect", "atomic.refine"],
        "atomic.sublattice.auto": ["atomic.detect", "atomic.refine"],
    }
    for legacy in recipe.workflow.steps:
        if legacy.name == "report":
            continue
        operation = mapping[legacy.name]
        for item in [*prerequisites.get(operation, []), operation]:
            if item not in operations:
                operations.append(item)
    if not operations:
        raise HarnessError("INVALID_PLAN", "Legacy recipe must actually load or analyze input")
    groups = {name: [op] for name, op in mapping.items()}
    for alias, original in {
        "quality": "assess_quality", "calibration": "calibrate", "preprocessing": "preprocess",
        "simulation": "simulate", "atom_detection": "detect_atoms", "gpa": "run_gpa",
    }.items():
        groups[alias] = groups[original]
    diffraction = [op for op in operations if op.startswith("diffraction.")]
    groups.update(diffraction=diffraction, analyze=diffraction, fft=diffraction)
    aliases = {
        "top_k_peaks": "top_k", "min_peak_distance": "min_distance",
        "fft_center_mode": "center_mode", "direct_beam_refine_window": "refine_window",
        "peak_threshold_strategy": "threshold_strategy", "phase_top_n": "top_n",
    }
    registry = default_registry()
    parameters = {op: {} for op in operations}
    for group, values in recipe.workflow.parameters.items():
        candidates = [op for op in groups.get(group, []) if op in parameters]
        if not candidates or not isinstance(values, dict):
            raise HarnessError("LEGACY_PARAMETER_REQUIRES_MIGRATION", group)
        for key, value in values.items():
            if candidates == ["calibration.resolve"] and key == "method" and value == "auto":
                continue
            is_diffraction = all(op.startswith("diffraction.") for op in candidates)
            mapped = aliases.get(key, key) if is_diffraction else key
            matches = [
                op for op in candidates if mapped in registry.get(op).parameter_model.model_fields
            ]
            if len(matches) != 1 or mapped in parameters[matches[0]]:
                raise HarnessError("LEGACY_PARAMETER_REQUIRES_MIGRATION", f"{group}.{key}")
            parameters[matches[0]][mapped] = value
    steps = []
    for operation in operations:
        params = registry.validate_parameters(registry.get(operation), parameters[operation])
        steps.append({
            "id": operation.replace(".", "_"), "operation_id": operation, "parameters": params,
            "depends_on": [steps[-1]["id"]] if steps else [],
        })
    return {
        "input_path": str(input_path),
        "modality": parameters.get("input.load", {}).get("modality", "unknown"),
        "steps": steps,
        "user_input": {"legacy_recipe": {"name": recipe.workflow.name, "steps": settings["steps"],
                                         "report_formats": formats}},
    }
