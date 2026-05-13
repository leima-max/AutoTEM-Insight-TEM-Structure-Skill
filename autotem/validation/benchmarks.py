"""A4 validation and benchmark orchestration utilities."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

from autotem.core import (
    AnalysisReport,
    BenchmarkCase,
    BenchmarkResult,
    Measurement,
    ValidationMetric,
)
from autotem.validation.metrics import collect_validation_metrics
from autotem.validation.uncertainty import coordinate_rmse, mean_relative_error

_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
}


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Expected numeric value, got: {type(value).__name__}.")


def _metric_index(metrics: list[ValidationMetric]) -> dict[str, ValidationMetric]:
    return {metric.name: metric for metric in metrics}


def _default_acceptance_criteria(report: AnalysisReport) -> list[dict[str, object]]:
    criteria: list[dict[str, object]] = [
        {"metric": "workflow_no_failed_steps", "op": ">=", "threshold": 1.0, "unit": "flag"},
        {"metric": "quality_gates_passed_flag", "op": ">=", "threshold": 1.0, "unit": "flag"},
    ]
    if report.workflow is not None:
        step_names = {step.name for step in report.workflow.steps}
    else:
        step_names = set()
    if step_names.intersection({"fft", "diffraction_peaks", "index_diffraction", "phase_match"}):
        criteria.extend(
            [
                {"metric": "diffraction_peak_count", "op": ">=", "threshold": 1.0, "unit": "count"},
                {"metric": "d_spacing_count", "op": ">=", "threshold": 1.0, "unit": "count"},
            ]
        )
    if "detect_atoms" in step_names:
        criteria.append(
            {"metric": "atom_coordinate_count", "op": ">=", "threshold": 1.0, "unit": "count"}
        )
    return criteria


def _apply_acceptance_criteria(
    metrics: list[ValidationMetric],
    criteria: list[dict[str, object]],
) -> tuple[list[ValidationMetric], list[str], dict[str, object]]:
    warnings: list[str] = []
    index = _metric_index(metrics)
    evaluated = 0
    passed = 0

    for row in criteria:
        metric_name = str(row.get("metric", "")).strip()
        op_name = str(row.get("op", ">=")).strip()
        threshold = row.get("threshold")
        if not metric_name:
            warnings.append("Benchmark criterion skipped: missing metric name.")
            continue
        if op_name not in _OPS:
            warnings.append(f"Benchmark criterion skipped: unsupported op '{op_name}'.")
            continue
        if threshold is None:
            warnings.append(f"Benchmark criterion skipped: missing threshold for '{metric_name}'.")
            continue
        metric = index.get(metric_name)
        if metric is None:
            warnings.append(
                f"Benchmark criterion skipped: metric '{metric_name}' was not produced."
            )
            continue

        evaluated += 1
        threshold_value = _to_float(threshold)
        observed_value = _to_float(metric.value.value)
        is_passed = bool(_OPS[op_name](observed_value, threshold_value))
        if is_passed:
            passed += 1
        metric.target = Measurement(value=threshold_value, unit=str(row.get("unit", "")))
        metric.passed = is_passed
        metric.details = {
            **metric.details,
            "acceptance": {"op": op_name, "threshold": threshold_value},
        }

    payload = {
        "evaluated": evaluated,
        "passed": passed,
        "failed": max(evaluated - passed, 0),
        "all_passed": (passed == evaluated) if evaluated > 0 else True,
    }
    return metrics, warnings, payload


def _append_reference_metrics(
    report: AnalysisReport,
    metrics: list[ValidationMetric],
    benchmark_parameters: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    reference = benchmark_parameters.get("reference")
    if not isinstance(reference, dict):
        return warnings

    d_spacing_ref = reference.get("d_spacings_nm")
    if isinstance(d_spacing_ref, list):
        observed = (
            [float(item.value) for item in report.diffraction.d_spacings]
            if report.diffraction is not None
            else []
        )
        reference_values = [float(item) for item in d_spacing_ref if isinstance(item, (int, float))]
        pair_count = min(len(observed), len(reference_values))
        if pair_count > 0:
            rel_err = mean_relative_error(observed[:pair_count], reference_values[:pair_count])
            metrics.append(
                ValidationMetric(
                    name="d_spacing_mean_relative_error",
                    value=Measurement(value=rel_err, unit="fraction", method="reference_compare"),
                    direction="lower_is_better",
                    details={"pair_count": pair_count},
                )
            )
        else:
            warnings.append(
                "Benchmark reference hook skipped d-spacing metric: "
                "no overlapping observed/reference values."
            )

    atom_ref = reference.get("atom_coordinates")
    if isinstance(atom_ref, list):
        observed_coords = (
            [(float(atom.x), float(atom.y)) for atom in report.atom_lattice.atoms]
            if report.atom_lattice is not None
            else []
        )
        reference_coords: list[tuple[float, float]] = []
        for row in atom_ref:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                x_raw, y_raw = row[0], row[1]
                if isinstance(x_raw, (int, float)) and isinstance(y_raw, (int, float)):
                    reference_coords.append((float(x_raw), float(y_raw)))
        if observed_coords and reference_coords:
            rmse = coordinate_rmse(observed_coords, reference_coords)
            metrics.append(
                ValidationMetric(
                    name="atom_coordinate_rmse",
                    value=Measurement(value=rmse, unit="pixel", method="reference_compare"),
                    direction="lower_is_better",
                    details={
                        "observed_count": len(observed_coords),
                        "reference_count": len(reference_coords),
                    },
                )
            )
        else:
            warnings.append(
                "Benchmark reference hook skipped atom RMSE metric: "
                "observed/reference coordinates unavailable."
            )

    return warnings


def run_validation_summary(
    report: AnalysisReport,
    validate_parameters: dict[str, Any] | None = None,
) -> tuple[list[ValidationMetric], dict[str, object], list[str]]:
    """Return validation metrics and summary payload for `validate` step outputs."""
    parameters = dict(validate_parameters or {})
    metrics, warnings = collect_validation_metrics(report)
    criteria = parameters.get("acceptance_criteria")
    if isinstance(criteria, list):
        metrics, crit_warnings, acceptance = _apply_acceptance_criteria(metrics, criteria)
    else:
        acceptance = {"evaluated": 0, "passed": 0, "failed": 0, "all_passed": True}
        crit_warnings = []
    warnings.extend(crit_warnings)

    payload = {
        "schema_version": report.schema_version,
        "metric_count": len(metrics),
        "acceptance": acceptance,
        "metrics": [metric.model_dump(mode="json") for metric in metrics],
    }
    return metrics, payload, warnings


def run_benchmark_summary(
    report: AnalysisReport,
    benchmark_parameters: dict[str, Any] | None = None,
    *,
    baseline_metrics: list[ValidationMetric] | None = None,
) -> tuple[BenchmarkResult, dict[str, object], list[str]]:
    """Build A4 benchmark summary from report outputs plus optional references/criteria."""
    parameters = dict(benchmark_parameters or {})
    warnings: list[str] = []

    if baseline_metrics is not None:
        metrics = list(baseline_metrics)
    else:
        metrics = collect_validation_metrics(report)[0]
    warnings.extend(_append_reference_metrics(report, metrics, parameters))

    criteria_raw = parameters.get("acceptance_criteria")
    if isinstance(criteria_raw, list):
        criteria = criteria_raw
    else:
        criteria = _default_acceptance_criteria(report)
    metrics, crit_warnings, acceptance = _apply_acceptance_criteria(metrics, criteria)
    warnings.extend(crit_warnings)

    case_id = str(parameters.get("case_id", "default_case"))
    benchmark = BenchmarkResult(
        cases=[
            BenchmarkCase(
                case_id=case_id,
                dataset=report.dataset,
                workflow=report.workflow,
                reference=parameters.get("reference", {})
                if isinstance(parameters.get("reference"), dict)
                else {},
            )
        ],
        metrics=metrics,
        summary=(
            f"{acceptance['passed']}/{acceptance['evaluated']} acceptance criteria passed "
            f"for benchmark case '{case_id}'."
        ),
        passed=bool(acceptance.get("all_passed", True)),
        provenance={
            "acceptance_criteria": criteria,
            "acceptance_summary": acceptance,
        },
        warnings=warnings,
    )

    payload = {
        "schema_version": report.schema_version,
        "case_id": case_id,
        "summary": benchmark.summary,
        "passed": benchmark.passed,
        "acceptance": acceptance,
        "metric_count": len(metrics),
        "metrics": [metric.model_dump(mode="json") for metric in metrics],
    }
    return benchmark, payload, warnings
