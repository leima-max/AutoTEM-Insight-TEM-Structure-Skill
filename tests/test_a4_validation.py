from pathlib import Path

import numpy as np

from autotem import TEMAnalysisSkill
from autotem.core import (
    AnalysisReport,
    ArrayReference,
    AtomLattice,
    AtomSite,
    DiffractionResult,
    Measurement,
    TEMDataset,
    WorkflowConfig,
    WorkflowStep,
    read_json,
)
from autotem.validation import run_benchmark_summary


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
