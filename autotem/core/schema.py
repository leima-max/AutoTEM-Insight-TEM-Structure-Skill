"""Schema and JSON helpers for AutoTEM-Insight data models."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from autotem.core.models import (
    CURRENT_SCHEMA_VERSION,
    AIInferenceResult,
    AnalysisReport,
    AtomLattice,
    BenchmarkResult,
    CalibrationResult,
    DiffractionResult,
    FourDSTEMResult,
    Measurement,
    MicrostructureResult,
    PreprocessingResult,
    QualityAssessment,
    SimulationResult,
    SpectroscopyResult,
    StrainResult,
    TEMDataset,
    WorkflowConfig,
    ZContrastResult,
)

ModelT = TypeVar("ModelT", bound=BaseModel)

CORE_MODELS: tuple[type[BaseModel], ...] = (
    Measurement,
    WorkflowConfig,
    TEMDataset,
    QualityAssessment,
    CalibrationResult,
    PreprocessingResult,
    DiffractionResult,
    AtomLattice,
    StrainResult,
    MicrostructureResult,
    ZContrastResult,
    SpectroscopyResult,
    FourDSTEMResult,
    SimulationResult,
    AIInferenceResult,
    BenchmarkResult,
    AnalysisReport,
)


def core_json_schemas() -> dict[str, dict]:
    """Return JSON schemas for the stable cross-module model set."""
    return {model.__name__: model.model_json_schema() for model in CORE_MODELS}


def schema_manifest() -> dict[str, object]:
    """Return a compact manifest for schema export and compatibility checks."""
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "models": [model.__name__ for model in CORE_MODELS],
    }


def to_json(model: BaseModel, *, indent: int = 2) -> str:
    """Serialize a model to JSON with deterministic field ordering from the model definition."""
    return model.model_dump_json(indent=indent)


def from_json(model_type: type[ModelT], payload: str) -> ModelT:
    """Deserialize a model from a JSON string."""
    return model_type.model_validate_json(payload)


def write_json(path: str | Path, model: BaseModel, *, indent: int = 2) -> None:
    """Write a model as UTF-8 JSON."""
    Path(path).write_text(to_json(model, indent=indent), encoding="utf-8")


def read_json(path: str | Path, model_type: type[ModelT]) -> ModelT:
    """Read a model from UTF-8 JSON."""
    return from_json(model_type, Path(path).read_text(encoding="utf-8"))


def write_core_json_schemas(directory: str | Path) -> None:
    """Write one JSON schema file per core model plus a schema manifest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, schema in core_json_schemas().items():
        (directory / f"{name}.schema.json").write_text(
            to_json_text(schema),
            encoding="utf-8",
        )
    (directory / "manifest.json").write_text(
        to_json_text(schema_manifest()),
        encoding="utf-8",
    )


def to_json_text(payload: object, *, indent: int = 2) -> str:
    """Serialize plain JSON-compatible payloads consistently."""
    import json

    return json.dumps(payload, indent=indent, sort_keys=True)
