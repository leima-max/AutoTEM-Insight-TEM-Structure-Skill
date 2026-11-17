"""A4 validation and benchmark orchestration utilities."""

from __future__ import annotations

import json
import math
import operator
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from autotem.atomic.distance_statistics import compute_distance_statistics_from_lattice
from autotem.core import (
    AnalysisReport,
    ArrayReference,
    BenchmarkCase,
    BenchmarkResult,
    IndexedSpot,
    Measurement,
    PhaseCandidate,
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
_REFERENCE_ERROR_PREFIX = "Benchmark reference error: "


def _as_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _normalize_hkl(value: Any) -> tuple[int, int, int] | None:
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
        if len(parts) != 3:
            return None
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        h, k, ell = value[0], value[1], value[2]
        if isinstance(h, (int, float)) and isinstance(k, (int, float)) and isinstance(
            ell, (int, float)
        ):
            return int(h), int(k), int(ell)
    return None


def _indexed_hkl_topk(indexed_spots: list[IndexedSpot], *, top_k: int) -> set[tuple[int, int, int]]:
    if top_k <= 0:
        return set()
    result: set[tuple[int, int, int]] = set()
    for spot in indexed_spots:
        result.add((int(spot.h), int(spot.k), int(spot.l)))
        if len(result) >= top_k:
            break
    return result


def _phase_names_topk(candidates: list[PhaseCandidate], *, top_k: int) -> set[str]:
    if top_k <= 0:
        return set()
    return {str(item.name).strip().lower() for item in candidates[:top_k] if str(item.name).strip()}


def _array_from_reference(
    array_ref: ArrayReference | None,
    *,
    report_base_dir: Path | None = None,
) -> np.ndarray | None:
    if array_ref is None:
        return None
    if array_ref.inline is not None:
        return np.asarray(array_ref.inline, dtype=np.float64)
    if array_ref.uri is None or not array_ref.uri:
        return None

    raw_path = Path(array_ref.uri)
    if not raw_path.is_absolute():
        if report_base_dir is None:
            return None
        raw_path = report_base_dir / raw_path
    if not raw_path.exists():
        return None
    if raw_path.suffix.lower() != ".npy":
        return None
    loaded = np.load(raw_path, allow_pickle=False)
    return np.asarray(loaded, dtype=np.float64)


def _masked_mae_rmse(
    observed: np.ndarray,
    reference: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> tuple[float, float, int]:
    if observed.shape != reference.shape:
        raise ValueError("Observed and reference arrays must have identical shape.")
    if mask is None:
        valid = np.ones(observed.shape, dtype=bool)
    else:
        if mask.shape != observed.shape:
            raise ValueError("Mask shape must match observed/reference shape.")
        valid = mask.astype(bool)
    diffs = observed - reference
    finite = np.isfinite(diffs)
    valid = valid & finite
    values = diffs[valid]
    if values.size == 0:
        raise ValueError("No valid values available after mask/finite filtering.")
    mae = float(np.mean(np.abs(values)))
    rmse = float(np.sqrt(np.mean(np.square(values))))
    return mae, rmse, int(values.size)


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not numeric acceptance evidence.")
    if isinstance(value, (int, float)):
        result = float(value)
        if math.isfinite(result):
            return result
        raise ValueError("Acceptance evidence must be finite (NaN/Inf are invalid).")
    raise ValueError(f"Expected numeric value, got: {type(value).__name__}.")


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
    if report.simulation is not None:
        criteria.extend(
            [
                {
                    "metric": "b5_simulation_pattern_count",
                    "op": ">=",
                    "threshold": 1.0,
                    "unit": "count",
                },
                {
                    "metric": "b5_simulated_spot_count",
                    "op": ">=",
                    "threshold": 1.0,
                    "unit": "count",
                },
                {
                    "metric": "b5_matched_phase_candidate_count",
                    "op": ">=",
                    "threshold": 1.0,
                    "unit": "count",
                },
            ]
        )
    return criteria


def _apply_acceptance_criteria(
    metrics: list[ValidationMetric],
    criteria: Any,
) -> tuple[list[ValidationMetric], list[str], dict[str, object]]:
    """Evaluate every requested check; missing or invalid evidence cannot pass.

    ``failed`` retains its comparison-count meaning. ``unevaluated`` and ``errors``
    separately identify checks that could not be compared; either prevents passing.
    An omitted unit adopts the metric unit, while an explicit unit must match.
    """
    metrics = [metric.model_copy(deep=True) for metric in metrics]
    warnings: list[str] = []
    index: dict[str, list[ValidationMetric]] = {}
    for metric in metrics:
        index.setdefault(metric.name, []).append(metric)
    errors: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    evaluated = 0
    passed = 0
    requested = len(criteria) if isinstance(criteria, list) else 0

    def record_error(
        code: str,
        message: str,
        *,
        criterion_index: int | None = None,
        metric_name: str = "",
    ) -> None:
        evidence: dict[str, object] = {"code": code, "message": message}
        if criterion_index is not None:
            evidence["criterion_index"] = criterion_index
        if metric_name:
            evidence["metric"] = metric_name
        errors.append(evidence)
        warnings.append(f"Benchmark acceptance error: {message}")
        if criterion_index is not None:
            results.append({**evidence, "status": "unevaluated", "passed": False})

    if not isinstance(criteria, list):
        record_error("invalid_criteria", "acceptance_criteria must be a list.")
        criteria = []
    elif not criteria:
        record_error("empty_criteria", "Explicit acceptance_criteria must not be empty.")

    metric_decisions: dict[str, bool] = {}
    for criterion_index, row in enumerate(criteria):
        if not isinstance(row, dict):
            record_error(
                "invalid_criterion", "Each acceptance criterion must be a mapping.",
                criterion_index=criterion_index,
            )
            continue
        raw_name = row.get("metric")
        metric_name = raw_name.strip() if isinstance(raw_name, str) else ""
        op_name = row.get("op")
        unknown_fields = set(row).difference({"metric", "op", "threshold", "unit"})
        if not metric_name or unknown_fields:
            record_error(
                "invalid_criterion", "Criterion requires a metric name and known fields.",
                criterion_index=criterion_index, metric_name=metric_name,
            )
            continue
        if not isinstance(op_name, str) or op_name not in _OPS:
            record_error(
                "invalid_operator", f"Unsupported or missing operator for '{metric_name}'.",
                criterion_index=criterion_index, metric_name=metric_name,
            )
            continue
        try:
            threshold_value = _to_float(row.get("threshold"))
        except (ValueError, OverflowError) as exc:
            record_error(
                "invalid_threshold", f"Invalid threshold for '{metric_name}': {exc}",
                criterion_index=criterion_index, metric_name=metric_name,
            )
            continue
        matching_metrics = index.get(metric_name, [])
        if len(matching_metrics) != 1:
            record_error(
                "missing_metric" if not matching_metrics else "ambiguous_metric",
                f"Metric '{metric_name}' must be produced exactly once.",
                criterion_index=criterion_index, metric_name=metric_name,
            )
            continue
        metric = matching_metrics[0]
        raw_unit = row.get("unit", "")
        if not isinstance(raw_unit, str):
            record_error(
                "invalid_unit", f"Criterion unit for '{metric_name}' must be a string.",
                criterion_index=criterion_index, metric_name=metric_name,
            )
            metric_decisions[metric_name] = False
            metric.passed = False
            continue
        unit = raw_unit.strip() or metric.value.unit
        if unit != metric.value.unit:
            record_error(
                "unit_mismatch",
                f"Unit mismatch for '{metric_name}': metric '{metric.value.unit}', "
                f"criterion '{unit}'; convert explicitly before comparison.",
                criterion_index=criterion_index, metric_name=metric_name,
            )
            metric_decisions[metric_name] = False
            metric.passed = False
            continue
        try:
            observed_value = _to_float(metric.value.value)
        except (ValueError, OverflowError) as exc:
            record_error(
                "invalid_metric", f"Invalid observed value for '{metric_name}': {exc}",
                criterion_index=criterion_index, metric_name=metric_name,
            )
            metric_decisions[metric_name] = False
            metric.passed = False
            continue
        evaluated += 1
        is_passed = bool(_OPS[op_name](observed_value, threshold_value))
        if is_passed:
            passed += 1
        metric.target = Measurement(value=threshold_value, unit=unit)
        metric_decisions[metric_name] = metric_decisions.get(metric_name, True) and is_passed
        metric.passed = metric_decisions[metric_name]
        metric.details = {
            **metric.details,
            "acceptance": {"op": op_name, "threshold": threshold_value, "unit": unit},
        }
        results.append({
            "criterion_index": criterion_index,
            "metric": metric_name,
            "op": op_name,
            "threshold": threshold_value,
            "observed": observed_value,
            "unit": unit,
            "status": "passed" if is_passed else "failed",
            "passed": is_passed,
        })

    all_passed = requested > 0 and evaluated == requested and passed == requested and not errors
    payload = {
        "status": "passed" if all_passed else "failed",
        "requested": requested,
        "evaluated": evaluated,
        "unevaluated": requested - evaluated,
        "passed": passed,
        "failed": max(evaluated - passed, 0),
        "error_count": len(errors),
        "errors": errors,
        "results": results,
        "all_passed": all_passed,
    }
    return metrics, warnings, payload


def _append_reference_metrics(
    report: AnalysisReport,
    metrics: list[ValidationMetric],
    benchmark_parameters: dict[str, Any],
    *,
    report_base_dir: Path | None = None,
) -> list[str]:
    warnings: list[str] = []
    reference = benchmark_parameters.get("reference")
    if not isinstance(reference, dict):
        return warnings

    def unit_error(metric_name: str, expected_unit: str, observed_units: list[str]) -> None:
        warnings.append(_REFERENCE_ERROR_PREFIX + json.dumps({
            "code": "reference_unit_mismatch",
            "metric": metric_name,
            "expected_unit": expected_unit,
            "observed_units": observed_units,
            "action": "metric_not_generated",
        }, sort_keys=True))

    d_spacing_ref = reference.get("d_spacings_nm")
    if isinstance(d_spacing_ref, list):
        observed_measurements = (
            report.diffraction.d_spacings if report.diffraction is not None else []
        )
        invalid_units = sorted({item.unit for item in observed_measurements if item.unit != "nm"})
        if invalid_units:
            unit_error("d_spacing_mean_relative_error", "nm", invalid_units)
        observed = (
            [float(item.value) for item in observed_measurements]
            if not invalid_units
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
        elif not invalid_units:
            warnings.append(
                "Benchmark reference hook skipped d-spacing metric: "
                "no overlapping observed/reference values."
            )

    hkl_truth_raw = reference.get("hkl_truth")
    if isinstance(hkl_truth_raw, list):
        truth = [
            item for item in (_normalize_hkl(row) for row in hkl_truth_raw) if item is not None
        ]
        top_k = _as_int(reference.get("hkl_top_k"), default=3)
        predicted = (
            _indexed_hkl_topk(report.diffraction.indexed_spots, top_k=top_k)
            if report.diffraction is not None
            else set()
        )
        if truth:
            matched = sum(1 for row in truth if row in predicted)
            accuracy = matched / len(truth)
            metrics.append(
                ValidationMetric(
                    name="hkl_topk_accuracy",
                    value=Measurement(value=accuracy, unit="fraction", method="reference_compare"),
                    direction="higher_is_better",
                    details={
                        "truth_count": len(truth),
                        "matched_count": matched,
                        "top_k": top_k,
                    },
                )
            )
        else:
            warnings.append("Benchmark reference hook skipped hkl metric: no valid `hkl_truth`.")

    phase_truth_raw = reference.get("phase_truth")
    if isinstance(phase_truth_raw, list):
        truth_phase = {str(item).strip().lower() for item in phase_truth_raw if str(item).strip()}
        phase_top_k = _as_int(reference.get("phase_top_k"), default=3)
        predicted_phase = (
            _phase_names_topk(report.diffraction.phase_candidates, top_k=phase_top_k)
            if report.diffraction is not None
            else set()
        )
        if truth_phase:
            matched_phase = len(truth_phase.intersection(predicted_phase))
            phase_accuracy = matched_phase / len(truth_phase)
            metrics.append(
                ValidationMetric(
                    name="phase_topk_accuracy",
                    value=Measurement(
                        value=phase_accuracy,
                        unit="fraction",
                        method="reference_compare",
                    ),
                    direction="higher_is_better",
                    details={
                        "truth_count": len(truth_phase),
                        "matched_count": matched_phase,
                        "top_k": phase_top_k,
                    },
                )
            )
        else:
            warnings.append(
                "Benchmark reference hook skipped phase metric: no valid `phase_truth`."
            )

    b5_phase_truth_raw = reference.get("b5_phase_truth")
    if isinstance(b5_phase_truth_raw, list):
        truth_phase = {
            str(item).strip().lower() for item in b5_phase_truth_raw if str(item).strip()
        }
        phase_top_k = _as_int(reference.get("b5_phase_top_k"), default=3)
        predicted_phase = (
            _phase_names_topk(
                report.simulation.matched_phase_candidates,
                top_k=phase_top_k,
            )
            if report.simulation is not None
            else set()
        )
        if truth_phase:
            matched_phase = len(truth_phase.intersection(predicted_phase))
            phase_accuracy = matched_phase / len(truth_phase)
            metrics.append(
                ValidationMetric(
                    name="b5_phase_topk_accuracy",
                    value=Measurement(
                        value=phase_accuracy,
                        unit="fraction",
                        method="reference_compare",
                    ),
                    direction="higher_is_better",
                    details={
                        "truth_count": len(truth_phase),
                        "matched_count": matched_phase,
                        "top_k": phase_top_k,
                    },
                )
            )
        else:
            warnings.append(
                "Benchmark reference hook skipped B5 phase metric: no valid `b5_phase_truth`."
            )

    atom_ref = reference.get("atom_coordinates")
    if isinstance(atom_ref, list):
        atom_unit = report.atom_lattice.coordinate_unit if report.atom_lattice is not None else None
        if atom_unit is not None and atom_unit != "pixel":
            unit_error("atom_coordinate_rmse", "pixel", [atom_unit])
        observed_coords = (
            [(float(atom.x), float(atom.y)) for atom in report.atom_lattice.atoms]
            if report.atom_lattice is not None and atom_unit == "pixel"
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
        elif atom_unit in {None, "pixel"}:
            warnings.append(
                "Benchmark reference hook skipped atom RMSE metric: "
                "observed/reference coordinates unavailable."
            )

    atom_distance_ref = reference.get("atom_nearest_distances")
    if isinstance(atom_distance_ref, list):
        if report.atom_lattice is None or not report.atom_lattice.atoms:
            warnings.append(
                "Benchmark reference hook skipped atom distance metric: AtomLattice missing."
            )
        elif report.atom_lattice.coordinate_unit != "pixel":
            unit_error(
                "atom_nearest_distance_mean_relative_error", "pixel",
                [report.atom_lattice.coordinate_unit],
            )
        else:
            stats = compute_distance_statistics_from_lattice(report.atom_lattice)
            observed_nn = [float(value) for value in stats.nearest_distances if np.isfinite(value)]
            reference_nn = [
                float(item)
                for item in atom_distance_ref
                if isinstance(item, (int, float))
            ]
            pair_count = min(len(observed_nn), len(reference_nn))
            if pair_count > 0:
                distance_rel_err = mean_relative_error(
                    observed_nn[:pair_count],
                    reference_nn[:pair_count],
                )
                metrics.append(
                    ValidationMetric(
                        name="atom_nearest_distance_mean_relative_error",
                        value=Measurement(
                            value=distance_rel_err,
                            unit="fraction",
                            method="reference_compare",
                        ),
                        direction="lower_is_better",
                        details={"pair_count": pair_count},
                    )
                )
            else:
                warnings.append(
                    "Benchmark reference hook skipped atom distance metric: "
                    "no overlapping observed/reference nearest distances."
                )

    strain_map_refs: dict[str, Any] = {}
    if isinstance(reference.get("strain_maps"), dict):
        strain_map_refs = dict(reference.get("strain_maps", {}))
    else:
        for component in ("strain_xx", "strain_yy", "strain_xy", "rotation_xy"):
            if component in reference:
                strain_map_refs[component] = {
                    "values": reference[component],
                    "mask": reference.get(f"{component}_mask"),
                }

    if strain_map_refs:
        if report.strain is None:
            warnings.append(
                "Benchmark reference hook skipped strain metrics: StrainResult missing."
            )
        else:
            for component, payload in strain_map_refs.items():
                if not isinstance(payload, dict):
                    continue
                values_raw = payload.get("values")
                if not isinstance(values_raw, list):
                    continue
                reference_values = np.asarray(values_raw, dtype=np.float64)
                mask_raw = payload.get("mask")
                mask_values = None
                if isinstance(mask_raw, list):
                    mask_values = np.asarray(mask_raw, dtype=bool)

                observed_ref = getattr(report.strain, component, None)
                if not isinstance(observed_ref, ArrayReference):
                    warnings.append(
                        f"Benchmark strain metric skipped for '{component}': "
                        "component not available in StrainResult."
                    )
                    continue
                observed_values = _array_from_reference(
                    observed_ref,
                    report_base_dir=report_base_dir,
                )
                if observed_values is None:
                    warnings.append(
                        f"Benchmark strain metric skipped for '{component}': "
                        "unable to load observed array (inline or .npy URI required)."
                    )
                    continue
                try:
                    mae, rmse, valid_count = _masked_mae_rmse(
                        observed_values,
                        reference_values,
                        mask=mask_values,
                    )
                except ValueError as exc:
                    warnings.append(
                        f"Benchmark strain metric skipped for '{component}': {exc}"
                    )
                    continue
                metrics.append(
                    ValidationMetric(
                        name=f"{component}_mae",
                        value=Measurement(value=mae, unit="strain", method="reference_compare"),
                        direction="lower_is_better",
                        details={"valid_count": valid_count},
                    )
                )
                metrics.append(
                    ValidationMetric(
                        name=f"{component}_rmse",
                        value=Measurement(value=rmse, unit="strain", method="reference_compare"),
                        direction="lower_is_better",
                        details={"valid_count": valid_count},
                    )
                )

    return warnings


def run_validation_summary(
    report: AnalysisReport,
    validate_parameters: dict[str, Any] | None = None,
) -> tuple[list[ValidationMetric], dict[str, object], list[str]]:
    """Return validation metrics and summary payload for `validate` step outputs."""
    parameters = dict(validate_parameters or {})
    metrics, warnings = collect_validation_metrics(report)
    if "acceptance_criteria" in parameters:
        criteria = parameters["acceptance_criteria"]
        metrics, crit_warnings, acceptance = _apply_acceptance_criteria(metrics, criteria)
    else:
        acceptance = {
            "status": "not_evaluated", "requested": 0, "evaluated": 0, "unevaluated": 0,
            "passed": 0, "failed": 0, "error_count": 0, "errors": [], "results": [],
            "all_passed": None,
        }
        crit_warnings = []
    warnings.extend(crit_warnings)

    payload = {
        "schema_version": report.schema_version,
        "metric_count": len(metrics),
        "acceptance": acceptance,
        "metrics": [json.loads(metric.model_dump_json()) for metric in metrics],
    }
    return metrics, payload, warnings


def run_benchmark_summary(
    report: AnalysisReport,
    benchmark_parameters: dict[str, Any] | None = None,
    *,
    baseline_metrics: list[ValidationMetric] | None = None,
    report_base_dir: Path | None = None,
) -> tuple[BenchmarkResult, dict[str, object], list[str]]:
    """Build A4 benchmark summary from report outputs plus optional references/criteria."""
    parameters = dict(benchmark_parameters or {})
    warnings: list[str] = []

    if baseline_metrics is not None:
        metrics = list(baseline_metrics)
    else:
        metrics = collect_validation_metrics(report)[0]
    warnings.extend(
        _append_reference_metrics(
            report,
            metrics,
            parameters,
            report_base_dir=report_base_dir,
        )
    )

    criteria = (
        parameters["acceptance_criteria"]
        if "acceptance_criteria" in parameters
        else _default_acceptance_criteria(report)
    )
    metrics, crit_warnings, acceptance = _apply_acceptance_criteria(metrics, criteria)
    warnings.extend(crit_warnings)
    reference_errors = [
        json.loads(warning.removeprefix(_REFERENCE_ERROR_PREFIX))
        for warning in warnings if warning.startswith(_REFERENCE_ERROR_PREFIX)
    ]

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
            f"{acceptance['passed']}/{acceptance['requested']} "
            "requested acceptance criteria passed "
            f"for benchmark case '{case_id}'."
        ),
        passed=acceptance.get("all_passed") is True,
        provenance={
            "acceptance_criteria": criteria,
            "acceptance_summary": acceptance,
            "reference_errors": reference_errors,
        },
        warnings=warnings,
    )

    payload = {
        "schema_version": report.schema_version,
        "case_id": case_id,
        "summary": benchmark.summary,
        "passed": benchmark.passed,
        "acceptance": acceptance,
        "reference_errors": reference_errors,
        "metric_count": len(metrics),
        "metrics": [json.loads(metric.model_dump_json()) for metric in metrics],
    }
    return benchmark, payload, warnings
