"""Regression checks for report finalization and honest CLI outcomes."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import yaml

from autotem import TEMAnalysisSkill
from autotem.cli import EXIT_RUNTIME_ERROR, main
from autotem.core import (
    AnalysisReport,
    DiffractionPeak,
    DiffractionResult,
    Measurement,
    OutputArtifact,
    SimulationResult,
    WorkflowConfig,
    WorkflowStep,
    read_json,
)
from autotem.workflows.pipeline import _evaluate_quality_gates


def test_exports_include_final_gates_acceptance_and_step_status(tmp_path, monkeypatch):
    input_path = tmp_path / "input.npy"
    np.save(input_path, np.zeros((16, 16), dtype=np.float32))
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        "workflow:\n"
        "  name: finalization_regression\n"
        "  steps: [assess_quality, validate, benchmark, report]\n"
        "  parameters:\n"
        "    validation:\n"
        "      acceptance_criteria:\n"
        "        - {metric: workflow_no_failed_steps, op: '>=', threshold: 1, unit: flag}\n"
        "    benchmark:\n"
        "      acceptance_criteria:\n"
        "        - {metric: workflow_no_failed_steps, op: '>=', threshold: 1, unit: flag}\n"
        "outputs:\n"
        "  formats: [json, csv, html, pdf]\n",
        encoding="utf-8",
    )
    pdf_evidence = {}

    def capture_pdf(report, output_dir, *, options):
        pdf_evidence.update(report.model_dump(mode="json"))
        return OutputArtifact(kind="other", uri=str(output_dir / "report.pdf"))

    monkeypatch.setattr("autotem.workflows.pipeline.export_pdf_report", capture_pdf)
    output_dir = tmp_path / "run"
    result = TEMAnalysisSkill().run(input_path, workflow_path, output_dir)
    report = read_json(Path(result["report_path"]), AnalysisReport)
    bundle = json.loads((output_dir / "json/report_bundle_summary.json").read_text("utf-8"))
    workflow = json.loads((output_dir / "json/workflow.json").read_text("utf-8"))
    benchmark = json.loads((output_dir / "json/benchmark.json").read_text("utf-8"))
    assert bundle["warnings"] == report.warnings
    assert bundle["warning_count"] == len(report.warnings)
    assert bundle["table_count"] == len(report.tables)
    assert bundle["provenance"]["quality_gates"]["passed"] is False
    assert bundle["provenance"]["validation_summary"] == report.provenance["validation_summary"]
    assert workflow == report.workflow.model_dump(mode="json")
    assert benchmark["passed"] is False
    html = (output_dir / "report.html").read_text("utf-8")
    for warning in report.warnings:
        if warning.startswith("BLOCKING:"):
            assert warning in html
    with (output_dir / "tables/warnings.csv").open(encoding="utf-8", newline="") as handle:
        assert [row["warning"] for row in csv.DictReader(handle)] == report.warnings
    assert pdf_evidence["warnings"] == report.warnings
    assert pdf_evidence["benchmark"]["passed"] is False
    assert result["execution_status"] == "failed"
    assert result["quality_gates_passed"] is False
    assert result["benchmark_passed"] is False


def test_cli_serializes_real_atom_lattice(tmp_path, capsys):
    yy, xx = np.indices((48, 48))
    image = np.exp(-((yy - 24) ** 2 + (xx - 24) ** 2) / 8).astype(np.float32)
    input_path = tmp_path / "atoms.tif"
    tifffile.imwrite(input_path, image)
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text("workflow:\n  steps: [load, detect_atoms, report]\n", "utf-8")
    exit_code = main([
        "analyze", str(input_path), "--workflow", str(workflow_path),
        "--output", str(tmp_path / "run"),
    ])
    output = capsys.readouterr().out
    payload = json.loads(output[output.index("{"):])
    assert exit_code == 0
    assert payload["atom_count"] > 0
    assert len(payload["atom_lattice"]["atoms"]) == payload["atom_count"]


@pytest.mark.parametrize("steps", ["[assess_quality, report]", "[load, ai_inference, report]"])
def test_cli_fail_on_blocking_preserves_diagnostics(tmp_path, capsys, steps):
    input_path = tmp_path / "input.npy"
    np.save(input_path, np.zeros((8, 8), dtype=np.float32))
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(f"workflow:\n  steps: {steps}\n", "utf-8")
    output_dir = tmp_path / "run"
    exit_code = main([
        "analyze", str(input_path), "--workflow", str(workflow_path),
        "--output", str(output_dir), "--fail-on-blocking",
    ])
    capsys.readouterr()
    assert exit_code == EXIT_RUNTIME_ERROR
    report = read_json(output_dir / "analysis_report.json", AnalysisReport)
    assert report.provenance["run_outcome"]["execution_status"] in {"failed", "partial"}


def test_cli_fail_on_blocking_accepts_completed_engineering_run(tmp_path, capsys):
    input_path = tmp_path / "input.npy"
    np.save(input_path, np.zeros((8, 8), dtype=np.float32))
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text("workflow:\n  steps: [load, report]\n", "utf-8")
    exit_code = main([
        "analyze", str(input_path), "--workflow", str(workflow_path),
        "--output", str(tmp_path / "run"), "--fail-on-blocking",
    ])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"validation_status": "not_evaluated"' in output


@pytest.mark.parametrize("configured_budget", [None, 5.0])
def test_pipeline_simulation_has_recorded_budget_and_honest_fallback(
    tmp_path, monkeypatch, configured_budget,
):
    input_path = tmp_path / "input.npy"
    np.save(input_path, np.zeros((16, 16), dtype=np.float32))
    workflow_path = tmp_path / "workflow.yaml"
    recipe = {
        "workflow": {
            "steps": ["load", "calibrate", "preprocess", "fft", "simulate", "report"],
            "parameters": {"simulation": {"diffsims_timeout_s": configured_budget}}
            if configured_budget is not None else {},
        },
    }
    workflow_path.write_text(yaml.safe_dump(recipe), "utf-8")
    observed = {}

    def simulate(self, diffraction, *, parameters):
        observed.update(parameters)
        return SimulationResult(
            warnings=["simulation worker timed out"],
            provenance={"adapter": {"active": False, "reason": "timeout"}},
        )

    monkeypatch.setattr(TEMAnalysisSkill, "simulate_from_diffraction", simulate)
    result = TEMAnalysisSkill().run(input_path, workflow_path, tmp_path / "run")
    effective = yaml.safe_load(Path(result["workflow_used_path"]).read_text("utf-8"))
    expected = configured_budget if configured_budget is not None else 30.0
    assert observed["diffsims_timeout_s"] == expected
    assert effective["workflow"]["parameters"]["simulation"]["diffsims_timeout_s"] == expected
    assert result["execution_status"] == "partial"
    assert "simulate" in result["incomplete_steps"]


@pytest.mark.parametrize("missing_evidence", ["center", "indexed_fraction", "physical_units"])
def test_required_diffraction_evidence_cannot_default_to_pass(missing_evidence):
    provenance = {
        "fft": {"center": {"offset_px": 0.0}, "image_shape": [32, 32]},
        "b3_b4_consistency": {"indexed_nonzero_fraction": 1.0},
    }
    if missing_evidence == "center":
        provenance["fft"]["center"] = {}
    if missing_evidence == "indexed_fraction":
        provenance["b3_b4_consistency"] = {}
    workflow = WorkflowConfig(steps=[WorkflowStep(name="index_diffraction", status="done")])
    report = AnalysisReport(
        workflow=workflow,
        diffraction=DiffractionResult(
            peaks=[DiffractionPeak(x=float(i), y=4.0) for i in range(4)],
            d_spacings=[Measurement(
                value=0.2, uncertainty=0.001,
                unit="pixel" if missing_evidence == "physical_units" else "nm",
            ) for _ in range(4)],
            provenance=provenance,
        ),
    )
    warnings, summary = _evaluate_quality_gates(workflow, report)
    assert summary["passed"] is False
    assert any(warning.startswith("BLOCKING:") for warning in warnings)
