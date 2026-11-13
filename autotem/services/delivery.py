"""Existing A4 metric APIs consume registered results, with missing metrics fail-closed."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from autotem.core.models import (
    AnalysisReport,
    AtomLattice,
    CalibrationResult,
    DiffractionResult,
    PreprocessingResult,
    QualityAssessment,
    SimulationResult,
    StrainResult,
    TEMDataset,
)
from autotem.harness.contracts import OperationDefinition, ServiceResult, StrictParameters
from autotem.validation.benchmarks import run_benchmark_summary, run_validation_summary


class Criterion(StrictParameters):
    metric: str = Field(min_length=1)
    op: Literal[">=", "<=", ">", "<", "=="] = ">="
    threshold: float
    unit: str = ""


class ValidationParameters(StrictParameters):
    acceptance_criteria: list[Criterion] | None = None
    strain_role: Literal["gpa", "atom_strain"] = "gpa"


def compose_report(ctx, strain_role):
    report = AnalysisReport(
        summary="Registered headless results; scientific accuracy not_evaluated"
    )
    for role, field_name, cls in (
        ("dataset", "dataset", TEMDataset),
        ("quality", "quality", QualityAssessment),
        ("calibration", "calibration", CalibrationResult),
        ("preprocessing", "preprocessing", PreprocessingResult),
        ("diffraction", "diffraction", DiffractionResult),
        ("refined_atoms", "atom_lattice", AtomLattice),
        (strain_role, "strain", StrainResult),
        ("simulation", "simulation", SimulationResult),
    ):
        if ctx.store.role(ctx.run_id, role) or role in ctx.inputs:
            setattr(report, field_name, ctx.model(role, cls))
    report.provenance = {
        "run_id": ctx.run_id,
        "primary_strain": strain_role,
        "scientific_accuracy": "not_evaluated",
    }
    return report


def validate(ctx, params):
    p = ValidationParameters.model_validate(params)
    values = (
        {}
        if p.acceptance_criteria is None
        else {"acceptance_criteria": [c.model_dump() for c in p.acceptance_criteria]}
    )
    _, payload, warnings = run_validation_summary(compose_report(ctx, p.strain_role), values)
    return ServiceResult({"validation": payload}, warnings, "engineering_metrics")


def benchmark(ctx, params):
    p = ValidationParameters.model_validate(params)
    values = (
        {}
        if p.acceptance_criteria is None
        else {"acceptance_criteria": [c.model_dump() for c in p.acceptance_criteria]}
    )
    _, payload, warnings = run_benchmark_summary(compose_report(ctx, p.strain_role), values)
    return ServiceResult({"benchmark": payload}, warnings, "engineering_metrics")


def dependencies(ctx, params):
    from autotem.harness.registry import default_registry

    capabilities = default_registry(ctx.request.get("enabled_plugins", [])).capabilities()
    return ServiceResult({"dependency_diagnostics": capabilities})


def definitions():
    return [
        OperationDefinition(
            "validation.evaluate",
            "delivery",
            validate,
            ValidationParameters,
            input_roles=("dataset",),
            output_roles=("validation",),
            scientific_limitations=("Runtime metrics do not establish real scientific accuracy",),
        ),
        OperationDefinition(
            "benchmark.evaluate",
            "delivery",
            benchmark,
            ValidationParameters,
            input_roles=("dataset",),
            output_roles=("benchmark",),
            scientific_limitations=("No ground truth is synthesized or inferred",),
        ),
        OperationDefinition(
            "maintenance.dependencies",
            "delivery",
            dependencies,
            output_roles=("dependency_diagnostics",),
        ),
    ]
