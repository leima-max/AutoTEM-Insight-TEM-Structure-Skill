"""Single source of truth for controlled domain operation handlers."""

from __future__ import annotations

import importlib.metadata
import json
import re
import subprocess
import sys
from functools import lru_cache

from pydantic import BaseModel, ValidationError

from .contracts import HarnessError, OperationDefinition


def role_schema(role: str) -> dict:
    from autotem.core import models

    names = {
        "dataset": "TEMDataset",
        "quality": "QualityAssessment",
        "calibration": "CalibrationResult",
        "preprocessing": "PreprocessingResult",
        "diffraction": "DiffractionResult",
        "atoms": "AtomLattice",
        "refined_atoms": "AtomLattice",
        "regularized_atoms": "AtomLattice",
        "gpa": "StrainResult",
        "atom_strain": "StrainResult",
        "simulation": "SimulationResult",
    }
    if role in names:
        return getattr(models, names[role]).model_json_schema()
    if role in {"raw", "processed", "reciprocal_image", "fft_complex", "background"}:
        return {
            "type": "object",
            "required": ["artifact_id", "sha256", "metadata"],
            "properties": {
                "artifact_id": {"type": "string"},
                "sha256": {"type": "string"},
                "metadata": {
                    "type": "object",
                    "required": ["shape", "dtype"],
                    "properties": {
                        "shape": {"type": "array", "items": {"type": "integer"}},
                        "dtype": {"type": "string"},
                    },
                },
            },
            "description": "Full NPY array artifact; read preview never replaces this array",
        }
    return {
        "type": "object",
        "description": f"Registered JSON evidence role {role}",
        "additionalProperties": True,
    }


def probe_dependency(name: str) -> dict:
    try:
        return _cached_dependency(name)
    except subprocess.TimeoutExpired:
        # A timeout describes this attempt. The next request must be able to probe again.
        return {"state": "dependency_broken", "dependency": name, "reason": "import_timeout"}


@lru_cache(maxsize=128)
def _cached_dependency(name: str) -> dict:
    # Import health is isolated: a broken optional extension must not crash the host.
    code = (
        "import importlib,importlib.util,json,sys; n=sys.argv[1]; "
        "s=importlib.util.find_spec(n); "
        "print(json.dumps({'found':s is not None})); "
        "importlib.import_module(n) if s else None"
    )
    options = ["-I"] if sys.flags.isolated else []
    result = subprocess.run(
        [sys.executable, *options, "-c", code, name], capture_output=True, text=True, timeout=8
    )
    if result.returncode:
        missing = "ModuleNotFoundError" in result.stderr and "No module named" in result.stderr
        # A missing transitive dependency is a broken installation, not an absent backend.
        primary_missing = missing and f"'{name.split('.')[0]}'" in result.stderr
        return {
            "state": "dependency_unavailable" if primary_missing else "dependency_broken",
            "dependency": name,
            "reason": result.stderr[-1500:],
        }
    try:
        found = json.loads(result.stdout.splitlines()[0])["found"]
    except (ValueError, IndexError, KeyError):
        return {"state": "dependency_broken", "dependency": name, "reason": "invalid_probe"}
    return {"state": "available" if found else "dependency_unavailable", "dependency": name}


class OperationRegistry:
    def __init__(self):
        self.operations: dict[str, OperationDefinition] = {}
        self.plugin_sources: dict[str, dict] = {}

    def register(self, definition: OperationDefinition):
        if not isinstance(definition, OperationDefinition):
            raise HarnessError("INVALID_PLUGIN_CONTRACT", "Expected OperationDefinition")
        if not re.fullmatch(r"[a-z][a-z0-9_.-]{1,100}", definition.operation_id):
            raise HarnessError("INVALID_OPERATION_ID", definition.operation_id)
        if definition.operation_id in self.operations:
            raise HarnessError("DUPLICATE_OPERATION", definition.operation_id)
        if definition.implementation_status not in {"implemented", "reserved", "deprecated"}:
            raise HarnessError("INVALID_OPERATION_CONTRACT", "Invalid implementation status")
        if not issubclass(definition.parameter_model, BaseModel):
            raise HarnessError("INVALID_OPERATION_CONTRACT", "Parameters must be a Pydantic model")
        if definition.parameter_model.model_config.get("extra") != "forbid":
            raise HarnessError("INVALID_OPERATION_CONTRACT", "Parameters must forbid extra fields")
        definition.parameter_model.model_json_schema()
        if definition.implementation_status == "implemented" and not callable(definition.handler):
            raise HarnessError("INVALID_OPERATION_CONTRACT", "Implemented handler must be callable")
        if definition.implementation_status == "reserved" and definition.handler is not None:
            raise HarnessError("INVALID_OPERATION_CONTRACT", "Reserved handlers must be empty")
        if definition.timeout_seconds <= 0:
            raise HarnessError("INVALID_OPERATION_CONTRACT", "Budget must be positive")
        self.operations[definition.operation_id] = definition

    def get(self, operation_id: str) -> OperationDefinition:
        try:
            return self.operations[operation_id]
        except KeyError as exc:
            raise HarnessError("UNKNOWN_OPERATION", operation_id) from exc

    def availability(self, definition: OperationDefinition, *, probe=None) -> dict:
        if definition.implementation_status == "reserved":
            return {"state": "unsupported_input", "code": "DEFERRED_THIS_PHASE"}
        if probe is None:
            probe = probe_dependency
        for name in definition.dependencies:
            result = probe(name)
            if result["state"] != "available":
                return result
        return {"state": "available"}

    def validate_parameters(self, definition, parameters) -> dict:
        if definition.implementation_status == "reserved":
            raise HarnessError("DEFERRED_THIS_PHASE", definition.description)
        try:
            result = definition.parameter_model.model_validate(parameters)
        except ValidationError as exc:
            raise HarnessError(
                "INVALID_PARAMETERS",
                definition.operation_id,
                exc.errors(include_url=False, include_context=False, include_input=False),
            ) from exc
        return result.model_dump(mode="json")

    def capabilities(self) -> list[dict]:
        result = []
        # Shared dependencies should consume only one import budget per inventory request.
        probe = lru_cache(maxsize=None)(probe_dependency)
        for definition in self.operations.values():
            handler = definition.handler
            result.append(
                {
                    "operation_id": definition.operation_id,
                    "version": definition.version,
                    "profile_id": definition.profile_id,
                    "description": definition.description,
                    "implementation_status": definition.implementation_status,
                    "availability": self.availability(definition, probe=probe),
                    "parameter_schema": definition.parameter_model.model_json_schema(),
                    "input_schema": {role: role_schema(role) for role in definition.input_roles},
                    "output_schema": {role: role_schema(role) for role in definition.output_roles},
                    "input_roles": definition.input_roles,
                    "output_roles": definition.output_roles,
                    "modality": definition.modality,
                    "permission": definition.permission,
                    "budget_seconds": definition.timeout_seconds,
                    "dependencies": definition.dependencies,
                    "implementation_source": f"{handler.__module__}:{handler.__name__}"
                    if handler
                    else None,
                    "scientific_limitations": definition.scientific_limitations,
                    "result_level": definition.result_level,
                    "scientific_accuracy": "not_evaluated",
                    "coordinate_contract": "array axes yx; atom coordinates yx/pixel; "
                    "physical values "
                    "require calibration; see result artifact unit metadata",
                    "errors": [
                        "INVALID_PARAMETERS",
                        "MISSING_ARTIFACT",
                        "NEEDS_INPUT",
                        "DEPENDENCY_UNAVAILABLE",
                        "DEPENDENCY_BROKEN",
                        "NUMERICAL_FAILURE",
                        "TIMEOUT",
                        "CANCELLED",
                    ],
                    "plugin_source": self.plugin_sources.get(definition.operation_id),
                }
            )
        return result

    def enable_plugins(self, names: list[str]):
        entries = importlib.metadata.entry_points(group="autotem.operations")
        for name in names:
            matches = [entry for entry in entries if entry.name == name]
            if len(matches) != 1:
                raise HarnessError("PLUGIN_NOT_FOUND", f"Expected one installed plugin: {name}")
            entry = matches[0]
            definitions = entry.load()()
            if not isinstance(definitions, (list, tuple)) or not 1 <= len(definitions) <= 100:
                raise HarnessError(
                    "INVALID_PLUGIN_CONTRACT", "Plugin must return 1..100 definitions"
                )
            # Validate all before publishing any; plugins cannot partially overwrite the registry.
            temporary = OperationRegistry()
            temporary.operations = self.operations.copy()
            for definition in definitions:
                previous = temporary.operations.get(definition.operation_id)
                if previous and previous.implementation_status == "reserved":
                    if (
                        previous.input_roles != definition.input_roles
                        or previous.output_roles != definition.output_roles
                    ):
                        raise HarnessError(
                            "INCOMPATIBLE_PLUGIN_CONTRACT",
                            "Reserved role contracts must be preserved",
                        )
                    if definition.version == previous.version:
                        raise HarnessError(
                            "PLUGIN_VERSION_REQUIRED",
                            "Publish a new version when implementing a reserved slot",
                        )
                    del temporary.operations[definition.operation_id]
                temporary.register(definition)
            self.operations = temporary.operations
            for definition in definitions:
                self.plugin_sources[definition.operation_id] = {
                    "name": name,
                    "entry_point": entry.value,
                    "distribution": entry.dist.name if entry.dist else None,
                    "version": entry.dist.version if entry.dist else None,
                    "explicitly_enabled": True,
                }


def default_registry(enabled_plugins: list[str] | None = None) -> OperationRegistry:
    from autotem.services import atomic, data_quality, delivery, diffraction, strain

    registry = OperationRegistry()
    for module in (data_quality, diffraction, atomic, strain, delivery):
        for definition in module.definitions():
            registry.register(definition)
    reserved = {
        "ai.denoise": "Deep-learning denoising",
        "ai.diffraction_spots": "Deep-learning diffraction spot detection",
        "ai.atom_columns": "Deep-learning atomic column detection",
        "microstructure.grains": "Grain segmentation",
        "microstructure.twins": "Twin detection",
        "microstructure.defects": "Defect segmentation",
        "fourdstem.analyze": "4D-STEM workflow",
        "spectroscopy.eds": "EDS composition",
        "spectroscopy.eels": "EELS analysis",
        "simulation.stem": "Multislice STEM simulation",
        "workbench.launch": "Unified visualization workbench",
        "revision.cas": "Interactive revision and CAS",
    }
    for operation_id, description in reserved.items():
        inputs = ("raw",)
        outputs = {
            "ai.denoise": ("processed",),
            "ai.diffraction_spots": ("peaks",),
            "ai.atom_columns": ("atoms",),
            "microstructure.grains": ("grain_masks",),
            "microstructure.twins": ("twins",),
            "microstructure.defects": ("defects",),
            "fourdstem.analyze": ("fourdstem",),
            "spectroscopy.eds": ("eds",),
            "spectroscopy.eels": ("eels",),
            "simulation.stem": ("simulated_stem",),
            "workbench.launch": ("workbench_session",),
            "revision.cas": ("revision",),
        }[operation_id]
        registry.register(
            OperationDefinition(
                operation_id,
                "reserved",
                None,
                input_roles=inputs,
                output_roles=outputs,
                description=f"{description}: deferred in this headless classical phase",
                implementation_status="reserved",
                result_level="unavailable",
            )
        )
    registry.enable_plugins(enabled_plugins or [])
    return registry
