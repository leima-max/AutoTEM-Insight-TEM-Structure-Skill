import json
from pathlib import Path

import numpy as np
import pytest

from autotem import TEMAnalysisSkill
from autotem.core import (
    AnalysisReport,
    ArrayReference,
    AtomLattice,
    AtomSite,
    DiffractionResult,
    Measurement,
    TEMDataset,
    ValidationMetric,
    WorkflowConfig,
    WorkflowStep,
    read_json,
)
from autotem.validation import run_benchmark_summary, run_validation_summary

pytestmark = pytest.mark.smoke


def test_a4_validate_and_benchmark_outputs(tmp_path: Path) -> None:
    image = np.zeros((48, 48), dtype=np.float32)
    image[24, 16] = 7.0
    image[24, 32] = 6.0
    input_path = tmp_path / "a4_input.npy"
    np.save(input_path, image)

    workflow_path = tmp_path / "workflow_a4.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: a4_validation_benchmark",
                "  steps:",
                "    - load",
                "    - assess_quality",
                "    - calibrate",
                "    - preprocess",
                "    - analyze",
                "    - validate",
                "    - benchmark",
                "    - report",
                "  parameters:",
                "    validation:",
                "      acceptance_criteria:",
                "        - metric: workflow_no_failed_steps",
                "          op: \">=\"",
                "          threshold: 1",
                "        - metric: d_spacing_count",
                "          op: \">=\"",
                "          threshold: 1",
                "    benchmark:",
                "      case_id: hrtem_mvp",
                "      acceptance_criteria:",
                "        - metric: workflow_no_failed_steps",
                "          op: \">=\"",
                "          threshold: 1",
                "        - metric: diffraction_peak_count",
                "          op: \">=\"",
                "          threshold: 1",
                "outputs:",
                "  formats: [json]",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "outputs"
    result = TEMAnalysisSkill().run(
        input_path=input_path,
        workflow_path=workflow_path,
        output_dir=output_dir,
    )
    report = read_json(Path(result["report_path"]), AnalysisReport)

    validation_path = output_dir / "json" / "validation_summary.json"
    benchmark_path = output_dir / "json" / "benchmark_report.json"
    assert validation_path.exists()
    assert benchmark_path.exists()
    assert report.benchmark is not None
    metric_names = {metric.name for metric in report.benchmark.metrics}
    assert "workflow_no_failed_steps" in metric_names
    assert "diffraction_peak_count" in metric_names

    assert report.workflow is not None
    step_status = {step.name: step.status for step in report.workflow.steps}
    assert step_status["validate"] in {"done", "warning"}
    assert step_status["benchmark"] in {"done", "warning"}


def test_a4_benchmark_reference_hook_metrics() -> None:
    report = AnalysisReport(
        workflow=WorkflowConfig(
            name="a4_reference_test",
            steps=[
                WorkflowStep(name="load", status="done"),
                WorkflowStep(name="fft", status="done"),
                WorkflowStep(name="validate", status="done"),
                WorkflowStep(name="benchmark", status="done"),
            ],
        ),
        dataset=TEMDataset(data=ArrayReference(shape=(16, 16), dtype="float32", uri="sample.npy")),
        diffraction=DiffractionResult(
            d_spacings=[
                Measurement(value=0.2, unit="nm"),
                Measurement(value=0.4, unit="nm"),
            ]
        ),
        atom_lattice=AtomLattice(
            coordinate_unit="pixel",
            atoms=[
                AtomSite(x=0.0, y=0.0),
                AtomSite(x=2.0, y=2.0),
            ]
        ),
        provenance={"quality_gates": {"passed": True, "blocking_warning_count": 0}},
    )

    benchmark, payload, warnings = run_benchmark_summary(
        report,
        benchmark_parameters={
            "case_id": "synthetic_case",
            "reference": {
                "d_spacings_nm": [0.2, 0.5],
                "atom_coordinates": [[0.0, 0.0], [1.0, 1.0]],
            },
            "acceptance_criteria": [
                {"metric": "d_spacing_mean_relative_error", "op": "<=", "threshold": 0.3},
                {"metric": "atom_coordinate_rmse", "op": "<=", "threshold": 2.0},
            ],
        },
    )

    assert not warnings
    assert payload["case_id"] == "synthetic_case"
    assert benchmark.passed is True
    metric_names = {item.name for item in benchmark.metrics}
    assert "d_spacing_mean_relative_error" in metric_names
    assert "atom_coordinate_rmse" in metric_names
    assert "d_spacing_uncertainty_coverage" in metric_names
    assert "d_spacing_uncertainty_mean_relative" in metric_names
    assert "d_spacing_uncertainty_plausible_fraction" in metric_names


def test_a4_missing_metric_blocks_partial_acceptance() -> None:
    benchmark, payload, warnings = run_benchmark_summary(
        AnalysisReport(),
        baseline_metrics=[
            ValidationMetric(name="known", value=Measurement(value=1, unit="flag")),
        ],
        benchmark_parameters={"acceptance_criteria": [
            {"metric": "known", "op": ">=", "threshold": 1},
            {"metric": "typo_or_missing", "op": "<=", "threshold": 0},
        ]},
    )
    assert benchmark.passed is False
    assert payload["acceptance"]["requested"] == 2
    assert payload["acceptance"]["evaluated"] == 1
    assert payload["acceptance"]["unevaluated"] == 1
    assert payload["acceptance"]["passed"] == 1
    assert payload["acceptance"]["all_passed"] is False
    assert payload["acceptance"]["errors"][0]["code"] == "missing_metric"
    assert warnings


@pytest.mark.parametrize("criteria,error_code", [
    ([], "empty_criteria"),
    (None, "invalid_criteria"),
    ({"metric": "known"}, "invalid_criteria"),
    ([None], "invalid_criterion"),
    ([{"op": ">=", "threshold": 1}], "invalid_criterion"),
    ([{"metric": "known", "op": "bad", "threshold": 1}], "invalid_operator"),
    ([{"metric": "known", "threshold": 1}], "invalid_operator"),
    ([{"metric": "known", "op": ">="}], "invalid_threshold"),
    ([{"metric": "known", "op": ">=", "threshold": True}], "invalid_threshold"),
    ([{"metric": "known", "op": ">=", "threshold": "1"}], "invalid_threshold"),
    ([{"metric": "known", "op": ">=", "threshold": float("nan")}], "invalid_threshold"),
    ([{"metric": "known", "op": ">=", "threshold": float("inf")}], "invalid_threshold"),
    ([{"metric": "known", "op": ">=", "threshold": 1, "typo": 1}], "invalid_criterion"),
    ([{"metric": "known", "op": ">=", "threshold": 1, "unit": 1}], "invalid_unit"),
    ([{"metric": "known", "op": ">=", "threshold": 1, "unit": "pm"}], "unit_mismatch"),
])
def test_a4_invalid_criterion_cannot_pass(criteria, error_code: str) -> None:
    metric = ValidationMetric(name="known", value=Measurement(value=1, unit="nm"))
    benchmark, payload, warnings = run_benchmark_summary(
        AnalysisReport(),
        baseline_metrics=[metric],
        benchmark_parameters={"acceptance_criteria": criteria},
    )
    acceptance = payload["acceptance"]
    assert benchmark.passed is False
    assert acceptance["all_passed"] is False
    assert acceptance["evaluated"] == 0
    assert acceptance["error_count"] == 1
    assert acceptance["errors"][0]["code"] == error_code
    assert warnings
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("value", [None, "bad", True, float("nan"), float("inf"), -float("inf")])
def test_a4_invalid_observed_evidence_cannot_pass(value) -> None:
    benchmark, payload, warnings = run_benchmark_summary(
        AnalysisReport(),
        baseline_metrics=[
            ValidationMetric(name="known", value=Measurement(value=value, unit="nm")),
        ],
        benchmark_parameters={"acceptance_criteria": [
            {"metric": "known", "op": ">=", "threshold": 0, "unit": "nm"},
        ]},
    )
    assert benchmark.passed is False
    assert payload["acceptance"]["unevaluated"] == 1
    assert payload["acceptance"]["errors"][0]["code"] == "invalid_metric"
    assert warnings
    json.dumps(payload, allow_nan=False)


def test_a4_metric_unit_is_preserved_and_repeated_checks_do_not_hide_failure() -> None:
    metric = ValidationMetric(name="known", value=Measurement(value=2, unit="nm"))
    benchmark, payload, warnings = run_benchmark_summary(
        AnalysisReport(),
        baseline_metrics=[metric],
        benchmark_parameters={"acceptance_criteria": [
            {"metric": "known", "op": "<=", "threshold": 1},
            {"metric": "known", "op": ">=", "threshold": 1, "unit": "nm"},
        ]},
    )
    assert not warnings
    assert benchmark.passed is False
    assert payload["acceptance"]["evaluated"] == 2
    assert payload["acceptance"]["failed"] == 1
    assert benchmark.metrics[0].passed is False
    assert benchmark.metrics[0].target.unit == "nm"
    assert metric.passed is None
    assert metric.target is None


def test_a4_duplicate_metric_is_ambiguous_evidence() -> None:
    metric = ValidationMetric(name="known", value=Measurement(value=1, unit="nm"))
    benchmark, payload, _ = run_benchmark_summary(
        AnalysisReport(),
        baseline_metrics=[metric, metric.model_copy(deep=True)],
        benchmark_parameters={"acceptance_criteria": [
            {"metric": "known", "op": ">=", "threshold": 1},
        ]},
    )
    assert benchmark.passed is False
    assert payload["acceptance"]["errors"][0]["code"] == "ambiguous_metric"


def test_a4_validate_without_criteria_is_not_evaluated() -> None:
    _, payload, _ = run_validation_summary(AnalysisReport())
    assert payload["acceptance"]["status"] == "not_evaluated"
    assert payload["acceptance"]["all_passed"] is None
    assert payload["acceptance"]["requested"] == 0
    _, explicit_empty, _ = run_validation_summary(
        AnalysisReport(), {"acceptance_criteria": []},
    )
    assert explicit_empty["acceptance"]["status"] == "failed"
    assert explicit_empty["acceptance"]["all_passed"] is False


@pytest.mark.parametrize("unit", ["pixel", "angstrom", "unknown", ""])
def test_a4_d_spacing_reference_rejects_unconverted_units(unit: str) -> None:
    report = AnalysisReport(diffraction=DiffractionResult(
        d_spacings=[Measurement(value=0.2, unit=unit)],
    ))
    benchmark, payload, warnings = run_benchmark_summary(report, {
        "reference": {"d_spacings_nm": [0.2]},
        "acceptance_criteria": [
            {"metric": "d_spacing_mean_relative_error", "op": "<=", "threshold": 0.1},
        ],
    })
    assert benchmark.passed is False
    assert "d_spacing_mean_relative_error" not in {metric.name for metric in benchmark.metrics}
    assert payload["acceptance"]["unevaluated"] == 1
    assert payload["reference_errors"] == [{
        "code": "reference_unit_mismatch",
        "metric": "d_spacing_mean_relative_error",
        "expected_unit": "nm",
        "observed_units": [unit],
        "action": "metric_not_generated",
    }]
    assert any("reference_unit_mismatch" in warning for warning in warnings)


@pytest.mark.parametrize("unit", ["nm", "pm", "unknown"])
def test_a4_atom_references_require_explicit_pixel_coordinates(unit: str) -> None:
    report = AnalysisReport(atom_lattice=AtomLattice(
        coordinate_unit=unit,
        atoms=[AtomSite(x=0.0, y=0.0), AtomSite(x=2.0, y=0.0)],
    ))
    benchmark, payload, _ = run_benchmark_summary(report, {
        "reference": {
            "atom_coordinates": [[0.0, 0.0], [2.0, 0.0]],
            "atom_nearest_distances": [2.0, 2.0],
        },
        "acceptance_criteria": [
            {"metric": "atom_coordinate_rmse", "op": "<=", "threshold": 0.1},
            {
                "metric": "atom_nearest_distance_mean_relative_error",
                "op": "<=", "threshold": 0.1,
            },
        ],
    })
    assert benchmark.passed is False
    metric_names = {metric.name for metric in benchmark.metrics}
    assert "atom_coordinate_rmse" not in metric_names
    assert "atom_nearest_distance_mean_relative_error" not in metric_names
    assert payload["acceptance"]["unevaluated"] == 2
    assert len(payload["reference_errors"]) == 2
