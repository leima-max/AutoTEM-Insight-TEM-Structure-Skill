"""Benchmark case/suite schemas and loaders for A4 multi-case runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from autotem.core.models import AutoTEMModel

BENCHMARK_CASE_SCHEMA_VERSION = "2026-05-13.a4.case-schema.v1"
BenchmarkOp = Literal[">=", "<=", ">", "<", "=="]


class AcceptanceCriterion(AutoTEMModel):
    """Single metric threshold criterion used by validate/benchmark checks."""

    metric: str
    op: BenchmarkOp
    threshold: float
    unit: str = ""


class BenchmarkCaseSpec(AutoTEMModel):
    """Benchmark case descriptor for one dataset/workflow/reference contract."""

    case_id: str
    description: str = ""
    modality: str = "unknown"
    owner: str = "A"
    tags: list[str] = Field(default_factory=list)
    enabled: bool = True
    input_path: str | None = None
    workflow_path: str | None = None
    report_path: str | None = None
    output_subdir: str = ""
    reference: dict[str, Any] = Field(default_factory=dict)
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    regression_weights: RegressionAlertWeightOverrides = Field(
        default_factory=lambda: RegressionAlertWeightOverrides()
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegressionAlertWeights(AutoTEMModel):
    """Configurable severity weights for case-level regression alerts."""

    status_flip_pass_to_fail: int = 5
    acceptance_failed_increase: int = 2
    warning_count_increase: int = 1
    acceptance_pass_rate_drop: int = 2


class RegressionAlertWeightOverrides(AutoTEMModel):
    """Optional case-level overrides for regression alert weights."""

    status_flip_pass_to_fail: int | None = None
    acceptance_failed_increase: int | None = None
    warning_count_increase: int | None = None
    acceptance_pass_rate_drop: int | None = None


class RegressionWeightTemplate(AutoTEMModel):
    """Template weights for a modality or case-class pattern."""

    modality: str
    weights: RegressionAlertWeightOverrides = Field(
        default_factory=lambda: RegressionAlertWeightOverrides()
    )
    enabled: bool = True


class BenchmarkSuiteSpec(AutoTEMModel):
    """Collection of benchmark cases with shared output and metadata."""

    schema_version: str = BENCHMARK_CASE_SCHEMA_VERSION
    suite_id: str = "default_suite"
    description: str = ""
    default_output_dir: str = "outputs/benchmark-suite"
    regression_weights: RegressionAlertWeights = Field(
        default_factory=RegressionAlertWeights
    )
    regression_weight_templates: list[RegressionWeightTemplate] = Field(
        default_factory=list
    )
    cases: list[BenchmarkCaseSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _as_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise ValueError(f"{field_name} must be a mapping.")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    payload = yaml.safe_load(path_obj.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Benchmark YAML root must be a mapping: {path_obj}")


def load_benchmark_suite(path: str | Path) -> BenchmarkSuiteSpec:
    """Load benchmark suite YAML (suite + cases, or case list shortcut)."""
    payload = _load_yaml(path)
    suite_payload = _as_mapping(payload.get("suite"), field_name="suite")
    cases_payload = payload.get("cases")

    if cases_payload is None and "case_id" in payload:
        cases_payload = [payload]
    if not isinstance(cases_payload, list):
        raise ValueError("Benchmark suite must provide `cases` as a list.")

    cases = [BenchmarkCaseSpec.model_validate(item) for item in cases_payload]
    suite = BenchmarkSuiteSpec(
        schema_version=str(
            suite_payload.get("schema_version", BENCHMARK_CASE_SCHEMA_VERSION)
        ),
        suite_id=str(suite_payload.get("suite_id", Path(path).stem)),
        description=str(suite_payload.get("description", "")),
        default_output_dir=str(
            suite_payload.get("default_output_dir", "outputs/benchmark-suite")
        ),
        regression_weights=RegressionAlertWeights.model_validate(
            _as_mapping(
                suite_payload.get("regression_weights"),
                field_name="suite.regression_weights",
            )
        ),
        regression_weight_templates=[
            RegressionWeightTemplate.model_validate(item)
            for item in (
                suite_payload.get("regression_weight_templates")
                if isinstance(suite_payload.get("regression_weight_templates"), list)
                else []
            )
        ],
        cases=cases,
        metadata=_as_mapping(suite_payload.get("metadata"), field_name="suite.metadata"),
    )
    return suite


def load_benchmark_case(path: str | Path) -> BenchmarkCaseSpec:
    """Load one benchmark case YAML."""
    payload = _load_yaml(path)
    if "case" in payload and isinstance(payload["case"], dict):
        return BenchmarkCaseSpec.model_validate(payload["case"])
    return BenchmarkCaseSpec.model_validate(payload)


def write_benchmark_case_schemas(directory: str | Path) -> None:
    """Write benchmark case/suite JSON schemas for external contract checks."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    case_schema_path = directory / "BenchmarkCaseSpec.schema.json"
    suite_schema_path = directory / "BenchmarkSuiteSpec.schema.json"
    manifest_path = directory / "benchmark_case_schema_manifest.json"

    case_schema_path.write_text(
        json.dumps(BenchmarkCaseSpec.model_json_schema(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    suite_schema_path.write_text(
        json.dumps(BenchmarkSuiteSpec.model_json_schema(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest_path.write_text(
        (
            "{\n"
            f'  "schema_version": "{BENCHMARK_CASE_SCHEMA_VERSION}",\n'
            '  "models": ["BenchmarkCaseSpec", "BenchmarkSuiteSpec"]\n'
            "}\n"
        ),
        encoding="utf-8",
    )
