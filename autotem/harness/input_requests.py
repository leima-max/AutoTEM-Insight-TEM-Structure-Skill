"""Versioned scientific input requests and their binding to a child plan."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field, TypeAdapter, ValidationError

from .contracts import HarnessError, StrictParameters

RECORD_VERSION = "1.0.0"
LENGTH_UNITS = ["nm", "pm", "angstrom", "um"]
ANSWER_UNITS = {
    "pixel_size_nm": [f"{unit}/pixel" for unit in LENGTH_UNITS],
    "diffraction_scale_inv_nm_per_px": ["1/nm/pixel"],
    "reference_length": LENGTH_UNITS,
    "known_d_spacing": LENGTH_UNITS,
    "reference_pixels": ["pixel"],
    "measured_radius_px": ["pixel"],
    "g_vectors": ["1/pixel", "fft_pixel", "1/nm"],
    "reference_region": ["pixel"],
    "reference_regions": ["pixel"],
    "roi": ["pixel"],
    "angle_deg": ["deg"],
    "lattice_angle_deg": ["deg"],
}


class Answer(StrictParameters):
    value: Any
    unit: str | None = Field(default=None, min_length=1, max_length=32)


class InputResponse(StrictParameters):
    schema_version: Literal["1.0.0"] = RECORD_VERSION
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source: Literal["user", "metadata", "artifact"]
    basis: str = Field(min_length=1, max_length=1000)
    answers: dict[str, Answer] = Field(min_length=1, max_length=32)
    evidence_artifact_ids: list[str] = Field(default_factory=list, max_length=64)
    raw_reply: str | None = Field(default=None, max_length=1000)


def reject_credentials(value: Any):
    """Free text provenance is bounded and must not contain a pasted credential."""
    if isinstance(value, dict):
        for key, item in value.items():
            if re.fullmatch(
                r"(?i)(?:.*[_-])?(?:api[_-]?key|token|secret|password|authorization)", str(key)
            ):
                raise HarnessError(
                    "CREDENTIAL_PARAMETER_FORBIDDEN", "Keep credentials outside the run"
                )
            reject_credentials(item)
    elif isinstance(value, list):
        for item in value:
            reject_credentials(item)
    elif isinstance(value, str) and re.search(
        r"(?i)\b(?:bearer\s+\S+|(?:api[_-]?key|token|secret|password|authorization)\s*[=:]\s*\S+)",
        value,
    ):
        raise HarnessError("CREDENTIAL_PARAMETER_FORBIDDEN", "Keep credentials outside the run")


def invalid(message, field=None, **expected):
    return HarnessError("INVALID_INPUT_RESPONSE", message, {"field": field, **expected})


def parse_response(user_input: dict) -> InputResponse | None:
    reject_credentials(user_input)
    if "response" not in user_input:
        return None
    if set(user_input) != {"response"}:
        raise invalid("Use only the response envelope for linked input")
    try:
        response = InputResponse.model_validate(user_input["response"])
    except ValidationError as exc:
        raise HarnessError(
            "INVALID_INPUT_RESPONSE",
            "Supply the request ID, source, basis and requested values with their specified units",
            exc.errors(include_url=False, include_context=False, include_input=False),
        ) from exc
    if not response.basis.strip():
        raise invalid("Describe the source of the measurement or selection", "basis")
    if response.source == "artifact" and not response.evidence_artifact_ids:
        raise invalid("Cite the parent evidence used for the answer", "evidence_artifact_ids")
    return response


def rejected_response(user_input, record, error):
    """Retain only bounded, typed answers to recognized fields in the parent question."""
    if error.code == "CREDENTIAL_PARAMETER_FORBIDDEN":
        return None
    try:
        response = parse_response(user_input)
    except HarnessError:
        return None
    schemas = {
        name: spec for field in record["fields"] for name, spec in field["answer_schema"].items()
    }
    if response is None or not response.answers.keys() <= schemas.keys():
        return None
    # Values that failed validation might be arbitrary pasted text or nested objects.
    # An audit of the rejection only needs the source and bounded numeric selections.
    safe = {}
    for name, answer in response.answers.items():
        value = answer.value
        if type(value) in (int, float, bool) or (
            isinstance(value, list)
            and len(value) <= 32
            and all(type(v) in (int, float) for v in value)
        ):
            safe[name] = {"value": value}
            if answer.unit in schemas[name].get("units", []):
                safe[name]["unit"] = answer.unit
    return {"request_id": response.request_id, "source": response.source, "answers": safe}


def _answer_options(field, operation_id, modality):
    name = field["name"]
    if name == "input_path":
        return []  # Changing files starts a new analysis, without inheriting this snapshot.
    if name == "physical_calibration":
        name = (
            "diffraction_scale_inv_nm_per_px" if modality in {"SAED", "NBED"} else "pixel_size_nm"
        )
    if name in {"atom_sites", "phase_candidates"}:
        return [["candidate_phase_params"], ["cif_candidate_paths"]]
    if name == "reference_pixels" and operation_id == "calibration.scalebar_candidate":
        return [["roi"]]
    options = [[name]]
    if (
        name == "g_vectors"
        and field.get("count", 0) >= 2
        and "peak_table_artifact_id" in field.get("evidence", {})
    ):
        options.append(["g_peak_indices"])
    if name == "pixel_size_nm" and operation_id == "calibration.resolve":
        options.append(["reference_length", "reference_pixels"])
    if name == "diffraction_scale_inv_nm_per_px" and operation_id == "calibration.resolve":
        options.extend([["known_d_spacing", "measured_radius_px"], ["known_d_spacing_refs"]])
    if name == "reference_region" and operation_id.startswith("strain.gpa"):
        options.append(["absolute_reference"])
        if operation_id == "strain.gpa":
            options.append(["reference_regions"])
    if name == "reference_atom_indices":
        options.append(["reference_region"])
    if name == "selection_indices" and field.get("count") == 1:
        options.append(["frame"])
    if name == "candidate_phase_params":
        options.append(["cif_candidate_paths"])
    if name == "formula":
        options += [[key] for key in ("elements", "material_names", "materials_project_ids")]
    return options


def _target_operation(name, operation, model):
    if name in model.model_fields:
        return operation
    if name in {"pixel_size_nm", "diffraction_scale_inv_nm_per_px"}:
        return "calibration.resolve"
    if name in {"candidate_phase_params", "cif_candidate_paths"}:
        return "diffraction.index"
    return operation


def describe_fields(fields, operation, modality, registry):
    """Persist both the operation's typed parameter schema and image-specific constraints."""
    definition = registry.get(operation)
    if operation == "strain.atom" and {f["name"] for f in fields} == {
        "reference_region",
        "reference_atom_indices",
    }:
        # Older workers described the two alternative reference forms as separate fields.
        fields = [f for f in fields if f["name"] == "reference_atom_indices"]
    records = []
    for raw in fields:
        field = dict(raw)
        model = definition.parameter_model
        if field.get("parameter_group"):
            model = model.model_fields[field["parameter_group"]].annotation
        options = _answer_options(field, operation, modality)
        schemas = {}
        for name in dict.fromkeys(key for group in options for key in group):
            target = _target_operation(name, operation, model)
            target_model = model if target == operation else registry.get(target).parameter_model
            if name == "materials_project_configuration":
                schema = {"const": True, "description": "Confirm external configuration only"}
            elif name == "g_peak_indices":
                schema = {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "uniqueItems": True,
                    "items": {"type": "integer", "minimum": 0, "maximum": field["count"] - 1},
                }
            elif name in target_model.model_fields:
                info = target_model.model_fields[name]
                schema = TypeAdapter(info.rebuild_annotation()).json_schema()
                # A request value must be supplied, even for optional operation parameters.
                non_null = [s for s in schema.get("anyOf", []) if s.get("type") != "null"]
                if len(non_null) == 1:
                    schema = {**{k: v for k, v in schema.items() if k != "anyOf"}, **non_null[0]}
            else:
                schema = {}
            if name == field["name"]:
                for key in ("minimum", "maximum", "exclusiveMinimum", "minItems", "maxItems"):
                    if key in field:
                        schema[key] = field[key]
                if "choices" in field:
                    schema["enum"] = field["choices"]
            if name in {"frame", "channel", "dataset_index"} and "count" in field:
                count = (
                    field["shape"][0] if field["name"] == "selection_indices" else field["count"]
                )
                schema.update(minimum=0, maximum=count - 1)
            if name == "selection_indices" and "shape" in field:
                sizes = field["shape"][:-2]
                schema.update(
                    minItems=len(sizes),
                    maxItems=len(sizes),
                    prefixItems=[
                        {"type": "integer", "minimum": 0, "maximum": size - 1} for size in sizes
                    ],
                )
            schemas[name] = {"value": schema, "target_operation": target}
            if name in ANSWER_UNITS:
                schemas[name]["units"] = ANSWER_UNITS[name]
        records.append({**field, "answer_options": options, "answer_schema": schemas})
    return records


def _check_value(name, answer, spec):
    value = answer.value
    schema = spec["value"]
    units = spec.get("units", [])
    if (units and answer.unit not in units) or (not units and answer.unit is not None):
        raise invalid("Use the unit specified by the request", name, allowed_units=units)
    if value is None or (isinstance(value, (list, str)) and not value):
        raise invalid("Supply a nonempty value", name, schema=schema)
    if "const" in schema and value is not schema["const"]:
        raise invalid("Confirm the external action explicitly", name, schema=schema)
    if "enum" in schema and value not in schema["enum"]:
        raise invalid("Choose a listed value", name, schema=schema)
    numeric = type(value) in (int, float)
    if schema.get("type") in {"integer", "number"} and not numeric:
        raise invalid("Supply a numeric value", name, schema=schema)
    if schema.get("type") == "integer" and type(value) is not int:
        raise invalid("Supply an integer", name, schema=schema)
    if numeric and (
        ("minimum" in schema and value < schema["minimum"])
        or ("maximum" in schema and value > schema["maximum"])
        or ("exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"])
    ):
        raise invalid("Value is outside the requested range", name, schema=schema)
    if schema.get("type") == "array":
        if not isinstance(value, list) or not schema.get("minItems", 0) <= len(value) <= schema.get(
            "maxItems", len(value)
        ):
            raise invalid("Supply the requested array length", name, schema=schema)
        for item, axis in zip(value, schema.get("prefixItems", []), strict=False):
            if type(item) is not int or not axis["minimum"] <= item <= axis["maximum"]:
                raise invalid("Selection is outside an available axis", name, schema=schema)
    if name == "absolute_reference" and value is not True:
        raise invalid("Accept absolute g-vector reference explicitly", name)
    if name == "pixel_size_nm":
        from autotem.calibration.pixel_calibration import convert_length

        return convert_length(value, answer.unit.removesuffix("/pixel"), "nm")
    if "coordinates" in spec:
        from .input_geometry import source_region

        if name == "reference_regions":
            if len(value) != 2:
                raise invalid("Supply one reference ROI per g vector", name)
            return [source_region(region, spec["coordinates"], name) for region in value]
        return source_region(value, spec["coordinates"], name)
    return value


def _step_for_response(plan, request):
    matches = [step for step in plan.steps if step.operation_id == request["operation_id"]]
    exact = [step for step in matches if step.id == request["step_id"]]
    if len(exact or matches) != 1:
        raise invalid(
            "Include the requesting step in the child plan",
            "steps",
            step_id=request["step_id"],
            operation_id=request["operation_id"],
        )
    step = (exact or matches)[0]
    if step.operation_version and step.operation_version != request["operation_version"]:
        raise invalid("The requesting operation version changed", "operation_version")
    # Pin any omitted version to the version that produced the question.
    step.operation_version = request["operation_version"]
    return step


def apply_response(plan, reply, record, store, parent):
    """Bind every accepted answer to an actual operation before any run is allocated."""
    if plan.input_path != parent["request"]["input_path"]:
        raise HarnessError("CHILD_INPUT_MISMATCH", "A child run must use the same source")
    for artifact_id in reply.evidence_artifact_ids:
        if store.artifact(artifact_id)["run_id"] != record["run_id"]:
            raise HarnessError("CROSS_RUN_ARTIFACT", "Cite evidence from the requesting run")
        store._check(artifact_id)
    requesting_step = _step_for_response(plan, record)
    adopted = {}
    used = set()
    for field in record["fields"]:
        options = [group for group in field["answer_options"] if set(group) <= reply.answers.keys()]
        if len(options) != 1:
            raise invalid(
                "Choose exactly one listed way to answer the field",
                field["name"],
                answer_options=field["answer_options"],
            )
        for name in options[0]:
            spec = field["answer_schema"][name]
            answer = reply.answers[name]
            value = _check_value(name, answer, spec)
            target = requesting_step
            if spec["target_operation"] != target.operation_id:
                candidates = [
                    step
                    for step in plan.steps[: plan.steps.index(requesting_step)]
                    if step.operation_id == spec["target_operation"]
                ]
                if not candidates:
                    raise invalid(
                        "Include the upstream operation receiving the answer",
                        name,
                        target_operation=spec["target_operation"],
                    )
                target = candidates[-1]
            group = field.get("parameter_group")
            params = target.parameters.setdefault(group, {}) if group else target.parameters
            if not isinstance(params, dict):
                raise invalid("Expected a parameter object for this response", group)
            bindings = {name: value}
            if name == "g_peak_indices":
                from .input_geometry import selected_g_vectors

                bindings = {
                    "g_vectors": selected_g_vectors(store, field, value),
                    "g_vector_unit": "1/pixel",
                }
            if name == "g_vectors":
                bindings["g_vector_unit"] = answer.unit
            if name == "reference_length":
                bindings["reference_unit"] = answer.unit
            if name == "known_d_spacing":
                bindings["known_d_spacing_unit"] = answer.unit
            if name == "materials_project_configuration":
                from autotem.diffraction.phase_lookup import resolve_materials_project_api_key

                key, _ = resolve_materials_project_api_key()
                if not key:
                    raise invalid(
                        "Configure Materials Project credentials outside the run first", name
                    )
                if key in str(reply.model_dump()):
                    raise HarnessError(
                        "CREDENTIAL_PARAMETER_FORBIDDEN", "Keep credentials outside the run"
                    )
                bindings = {}
            before = next(
                (
                    s["parameters"]
                    for s in parent["request"]["steps"]
                    if s["id"] == target.id and s["operation_id"] == target.operation_id
                ),
                {},
            )
            before = before.get(group, {}) if group else before
            # A replacement reference supersedes its previous mutually exclusive form.
            alternatives = {
                "reference_region": (
                    "reference_atom_indices",
                    "reference_regions",
                    "absolute_reference",
                ),
                "reference_regions": ("reference_region", "absolute_reference"),
                "reference_atom_indices": ("reference_region",),
                "absolute_reference": ("reference_region", "reference_regions"),
            }
            for other in alternatives.get(name, ()):
                if params.get(other):
                    if params[other] != before.get(other):
                        raise invalid("Choose one reference form", other)
                    params.pop(other)
            for key, bound in bindings.items():
                existing = params.get(key)
                if existing is not None and existing != bound and existing != before.get(key):
                    raise invalid("Explicit plan value conflicts with the supplied answer", key)
                params[key] = bound
                if target.operation_id == "input.load":
                    selection = plan.selection.get(key)
                    prior_selection = parent["request"]["selection"].get(key)
                    if (
                        selection is not None
                        and selection != bound
                        and selection != prior_selection
                    ):
                        raise invalid("Selection conflicts with the supplied answer", key)
                    plan.selection[key] = bound
            adopted[name] = {
                "step_id": target.id,
                "operation_id": target.operation_id,
                "parameters": bindings,
            }
            if group:
                adopted[name]["parameter_group"] = group
            used.add(name)
    if used != reply.answers.keys():
        raise invalid(
            "Reply contains fields outside this request", "answers", allowed=list(sorted(used))
        )
    return adopted
