"""Versioned domain-operation contracts, independent of the legacy result models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class StrictParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class HarnessError(Exception):
    def __init__(self, code: str, message: str, details: Any = None):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


class NeedsInput(HarnessError):
    def __init__(self, fields: list[dict], message: str = "Additional input is required"):
        super().__init__("NEEDS_INPUT", message, {"fields": fields})


@dataclass
class ServiceResult:
    outputs: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    scientific_level: str = "diagnostic"


@dataclass(frozen=True)
class OperationDefinition:
    operation_id: str
    profile_id: str
    handler: Callable | None
    parameter_model: type[BaseModel] = StrictParameters
    input_roles: tuple[str, ...] = ()
    output_roles: tuple[str, ...] = ()
    description: str = ""
    version: str = "1.0.0"
    implementation_status: str = "implemented"
    dependencies: tuple[str, ...] = ()
    scientific_limitations: tuple[str, ...] = ()
    timeout_seconds: float = 120
    modality: tuple[str, ...] = ()
    result_level: str = "diagnostic"
    permission: str = "local_read"


class Step(StrictParameters):
    id: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,80}$")
    operation_id: str
    operation_version: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    inputs: dict[str, str] = Field(default_factory=dict)
    optional: bool = False


class Budgets(StrictParameters):
    operation_seconds: float = Field(default=120, gt=0, le=3600)
    simulation_seconds: float = Field(default=30, gt=0, le=3600)
    run_seconds: float = Field(default=900, gt=0, le=86400)
    max_snapshot_bytes: int = Field(default=2_147_483_648, ge=1)


class HarnessPlan(StrictParameters):
    envelope_version: Literal["1.0.0"] = "1.0.0"
    input_path: str
    modality: str = "unknown"
    selection: dict[str, Any] = Field(default_factory=dict)
    steps: list[Step] = Field(min_length=1, max_length=100)
    budgets: Budgets = Field(default_factory=Budgets)
    network_enabled: bool = False
    input_mode: Literal["snapshot", "reference"] = "snapshot"
    enabled_plugins: list[str] = Field(default_factory=list, max_length=16)
    user_input: dict[str, Any] = Field(default_factory=dict)


class OperationContext:
    """Only registered artifacts and a run-local output directory reach handlers."""

    def __init__(
        self, store, run_id: str, step_id: str, request: dict, inputs: dict[str, str] | None = None
    ):
        self.store, self.run_id, self.step_id, self.request = store, run_id, step_id, request
        self.inputs = inputs or {}
        self.consumed_artifacts: dict[str, str] = {}
        self.output_dir = store.run_dir(run_id) / "operations" / step_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_path = Path(store.get_run(run_id)["input_path"])

    def _artifact(self, role: str) -> dict:
        artifact_id = self.inputs.get(role) or self.store.role(self.run_id, role)
        if not artifact_id:
            raise HarnessError("MISSING_ARTIFACT", f"Required artifact role: {role}")
        artifact = self.store.artifact(artifact_id)
        if artifact["run_id"] != self.run_id:
            raise HarnessError("CROSS_RUN_ARTIFACT", "Artifacts must belong to this run")
        if not artifact["execution_usable"]:
            raise HarnessError("ARTIFACT_NOT_EXECUTABLE", "Generating operation did not complete")
        self.consumed_artifacts[role] = artifact["artifact_id"]
        return artifact

    def array(self, role: str) -> np.ndarray:
        return self.store.load_array(self._artifact(role)["artifact_id"])

    def value(self, role: str) -> Any:
        return self.store.load_json(self._artifact(role)["artifact_id"])

    def model(self, role: str, cls: type[BaseModel]) -> BaseModel:
        return cls.model_validate(self.value(role))

    def emit_array(self, role: str, array: np.ndarray, metadata: dict | None = None):
        from autotem.core.models import ArrayReference

        artifact = self.store.write_array(self.run_id, self.step_id, role, array, metadata)
        return ArrayReference(
            shape=tuple(array.shape), dtype=str(array.dtype), uri=artifact["path"], description=role
        )

    def emit_json(self, role: str, value: Any, metadata: dict | None = None) -> str:
        return self.store.write_json(self.run_id, self.step_id, role, value, metadata)[
            "artifact_id"
        ]

    def emit_table(self, role: str, rows: list[dict], metadata: dict | None = None) -> str:
        return self.store.write_table(self.run_id, self.step_id, role, rows, metadata)[
            "artifact_id"
        ]

    def emit_image(self, role: str, value: Any, metadata: dict | None = None) -> str:
        return self.store.write_image(self.run_id, self.step_id, role, value, metadata)[
            "artifact_id"
        ]

    def resolve_array(self, reference) -> np.ndarray:
        uri = reference.uri if hasattr(reference, "uri") else reference.get("uri")
        matches = [
            a
            for a in self.store.artifacts(self.run_id)
            if a["path"] == uri and a["kind"] == "array"
        ]
        if len(matches) != 1:
            raise HarnessError(
                "UNREGISTERED_ARRAY", "Array must be a registered artifact in this run"
            )
        artifact = matches[0]
        if not artifact["execution_usable"] and not (
            artifact["step_id"] == self.step_id and artifact["operation_status"] == "running"
        ):
            raise HarnessError("ARTIFACT_NOT_EXECUTABLE", "Generating operation did not complete")
        self.consumed_artifacts[f"reference:{artifact['role']}"] = artifact["artifact_id"]
        return self.store.load_array(matches[0]["artifact_id"])
