from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from autotem import TEMAnalysisSkill
from autotem.core import AnalysisReport, RunLogEntry, RunSummary, from_json, read_json
from autotem.diffraction.basic import run_diffraction_analysis
from autotem.workflows.recipes import load_workflow_recipe


def test_a2_workflow_generates_report_with_diffraction(tmp_path: Path) -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[32, 40] = 10.0
    image[32, 24] = 10.0
    input_path = tmp_path / "sample.npy"
    np.save(input_path, image)

    output_dir = tmp_path / "outputs"
    result = TEMAnalysisSkill().run(input_path=input_path, output_dir=output_dir)

    assert result["status"] == "A2 baseline workflow completed."
    report_path = Path(result["report_path"])
    assert report_path.exists()
    assert Path(result["workflow_used_path"]).exists()
    assert Path(result["run_summary_path"]).exists()
    assert Path(result["log_path"]).exists()

    report = read_json(report_path, AnalysisReport)
    assert report.dataset is not None
    assert report.dataset.data.shape == (64, 64)
    assert report.calibration is not None
    assert report.calibration.pixel_size is not None
    assert report.diffraction is not None
    assert len(report.diffraction.peaks) >= 1
    assert report.simulation is not None
    assert report.provenance.get("runtime")
    assert report.provenance.get("input_fingerprint")
    assert report.provenance.get("workflow_fingerprint")
    summary = read_json(Path(result["run_summary_path"]), RunSummary)
    assert summary.report_path == str(report_path)
    assert summary.warning_count >= 0
    assert summary.workflow_parameter_schema_version
    assert isinstance(summary.bc_interface_summary, dict)
    assert "pass_rate" in summary.bc_interface_summary
    workflow_used = yaml.safe_load(Path(result["workflow_used_path"]).read_text(encoding="utf-8"))
    assert (
        workflow_used["workflow"]["workflow_parameter_schema_version"]
        == summary.workflow_parameter_schema_version
    )
    log_lines = Path(result["log_path"]).read_text(encoding="utf-8").splitlines()
    assert log_lines
    for line in log_lines:
        parsed = from_json(RunLogEntry, line)
        assert parsed.level in {"INFO", "WARNING", "ERROR", "DEBUG"}


def test_a2_workflow_supports_default_config_alias(tmp_path: Path) -> None:
    image = np.ones((16, 16), dtype=np.float32)
    input_path = tmp_path / "constant.npy"
    np.save(input_path, image)

    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: alias_test",
                "  description: alias expansion test",
                "  steps:",
                "    - load",
                "    - analyze",
                "    - report",
            ]
        ),
        encoding="utf-8",
    )

    result = TEMAnalysisSkill().run(
        input_path=input_path,
        workflow_path=workflow_path,
        output_dir=tmp_path / "outputs",
    )

    report = read_json(Path(result["report_path"]), AnalysisReport)
    assert report.workflow is not None
    step_names = [step.name for step in report.workflow.steps]
    assert "fft" in step_names
    assert "diffraction_peaks" in step_names
    assert "phase_match" in step_names


def test_a2_workflow_can_disable_simulation_step(tmp_path: Path) -> None:
    image = np.ones((24, 24), dtype=np.float32)
    input_path = tmp_path / "no_sim.npy"
    np.save(input_path, image)

    workflow_path = tmp_path / "workflow_no_sim.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: no_sim",
                "  steps:",
                "    - load",
                "    - assess_quality",
                "    - calibrate",
                "    - preprocess",
                "    - analyze",
                "    - report",
                "outputs:",
                "  formats: [json]",
                "  save_intermediates: false",
            ]
        ),
        encoding="utf-8",
    )

    result = TEMAnalysisSkill().run(
        input_path=input_path,
        workflow_path=workflow_path,
        output_dir=tmp_path / "outputs",
    )
    report = read_json(Path(result["report_path"]), AnalysisReport)
    assert report.workflow is not None
    step_names = [step.name for step in report.workflow.steps]
    assert "simulate" not in step_names
    assert report.simulation is None


def test_a2_workflow_applies_workflow_parameters(tmp_path: Path) -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[20, 20] = 10.0
    image[20, 44] = 8.0
    image[44, 20] = 9.0
    input_path = tmp_path / "param_image.npy"
    np.save(input_path, image)

    workflow_path = tmp_path / "workflow_params.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: param_test",
                "  steps:",
                "    - load",
                "    - assess_quality",
                "    - calibrate",
                "    - preprocess",
                "    - analyze",
                "    - report",
                "  parameters:",
                "    calibration:",
                "      pixel_size_nm: 0.05",
                "      voltage_kv: 300",
                "    preprocessing:",
                "      recipe: min_max",
                "      clip_percentile_low: 5",
                "      clip_percentile_high: 95",
                "    diffraction:",
                "      top_k_peaks: 2",
                "      phase_top_n: 1",
                "      candidate_phases: [Gold]",
            ]
        ),
        encoding="utf-8",
    )

    result = TEMAnalysisSkill().run(
        input_path=input_path,
        workflow_path=workflow_path,
        output_dir=tmp_path / "outputs",
    )
    report = read_json(Path(result["report_path"]), AnalysisReport)

    assert report.calibration is not None
    assert report.calibration.pixel_size is not None
    assert report.calibration.pixel_size.value == 0.05
    assert report.calibration.voltage is not None
    assert report.calibration.voltage.value == 300

    assert report.preprocessing is not None
    assert any(op.startswith("clip_percentile(") for op in report.preprocessing.operations)

    assert report.diffraction is not None
    assert len(report.diffraction.peaks) <= 2
    assert len(report.diffraction.phase_candidates) <= 1
    if report.diffraction.phase_candidates:
        assert report.diffraction.phase_candidates[0].name == "Gold"


def test_a2_workflow_b2_preprocess_recipe_applies_denoise_and_background(tmp_path: Path) -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[30:34, 30:34] = 10.0
    image += np.random.default_rng(7).normal(0.0, 0.2, size=image.shape).astype(np.float32)
    input_path = tmp_path / "b2_preprocess.npy"
    np.save(input_path, image)

    workflow_path = tmp_path / "workflow_b2_preprocess.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: b2_preprocess_test",
                "  steps:",
                "    - load",
                "    - assess_quality",
                "    - calibrate",
                "    - preprocess",
                "    - analyze",
                "    - report",
                "  parameters:",
                "    preprocessing:",
                "      recipe: b2_fft",
                "      denoise_sigma: 1.2",
                "      background_sigma: 10.0",
            ]
        ),
        encoding="utf-8",
    )

    result = TEMAnalysisSkill().run(
        input_path=input_path,
        workflow_path=workflow_path,
        output_dir=tmp_path / "outputs",
    )
    report = read_json(Path(result["report_path"]), AnalysisReport)

    assert report.preprocessing is not None
    ops = report.preprocessing.operations
    assert any(op.startswith("gaussian_denoise(") for op in ops)
    assert any(op.startswith("background_subtract_gaussian(") for op in ops)
    assert "min_max_normalize" in ops


def test_a2_workflow_exports_csv_and_html_when_requested(tmp_path: Path) -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[10, 16] = 6.0
    image[22, 16] = 6.0
    input_path = tmp_path / "export_input.npy"
    np.save(input_path, image)

    workflow_path = tmp_path / "workflow_export.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: export_test",
                "  steps:",
                "    - load",
                "    - assess_quality",
                "    - calibrate",
                "    - preprocess",
                "    - analyze",
                "    - report",
                "outputs:",
                "  formats: [json, csv, html]",
                "  save_intermediates: true",
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

    assert (output_dir / "tables" / "diffraction_peaks.csv").exists()
    assert (output_dir / "tables" / "d_spacings.csv").exists()
    assert (output_dir / "report.html").exists()
    artifact_uris = {artifact.uri for artifact in report.artifacts}
    assert str(output_dir / "tables" / "diffraction_peaks.csv") in artifact_uris
    assert str(output_dir / "report.html") in artifact_uris


def test_a2_workflow_reuses_diffraction_across_substeps(tmp_path: Path) -> None:
    image = np.zeros((48, 48), dtype=np.float32)
    image[24, 30] = 7.0
    image[24, 18] = 7.0
    input_path = tmp_path / "reuse.npy"
    np.save(input_path, image)

    workflow_path = tmp_path / "workflow_reuse.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: reuse_diffraction",
                "  steps:",
                "    - load",
                "    - assess_quality",
                "    - calibrate",
                "    - preprocess",
                "    - fft",
                "    - diffraction_peaks",
                "    - index_diffraction",
                "    - phase_match",
                "    - report",
            ]
        ),
        encoding="utf-8",
    )

    skill = TEMAnalysisSkill()
    with patch(
        "autotem.workflows.pipeline.run_diffraction_analysis",
        wraps=run_diffraction_analysis,
    ) as spy:
        result = skill.run(
            input_path=input_path,
            workflow_path=workflow_path,
            output_dir=tmp_path / "outputs",
        )
        assert spy.call_count == 1

    report = read_json(Path(result["report_path"]), AnalysisReport)
    assert report.workflow is not None
    reused = [
        step
        for step in report.workflow.steps
        if step.name in {"diffraction_peaks", "index_diffraction", "phase_match"}
    ]
    assert reused


def test_workflow_recipe_rejects_invalid_parameters(tmp_path: Path) -> None:
    workflow_path = tmp_path / "invalid_params.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: invalid_params",
                "  steps: [load, assess_quality, calibrate, preprocess, analyze, report]",
                "  parameters:",
                "    preprocessing:",
                "      clip_percentile_low: 95",
                "      clip_percentile_high: 5",
                "outputs:",
                "  formats: [json]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_workflow_recipe(workflow_path)


def test_workflow_recipe_rejects_negative_preprocess_sigmas(tmp_path: Path) -> None:
    workflow_path = tmp_path / "invalid_preprocess_sigma.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: invalid_preprocess_sigma",
                "  steps: [load, assess_quality, calibrate, preprocess, analyze, report]",
                "  parameters:",
                "    preprocessing:",
                "      recipe: b2_fft",
                "      denoise_sigma: -1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_workflow_recipe(workflow_path)


def test_workflow_recipe_rejects_invalid_peak_threshold_strategy(tmp_path: Path) -> None:
    workflow_path = tmp_path / "invalid_peak_threshold_strategy.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: invalid_peak_threshold_strategy",
                "  steps: [load, assess_quality, calibrate, preprocess, analyze, report]",
                "  parameters:",
                "    diffraction:",
                "      peak_threshold_strategy: unknown_strategy",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_workflow_recipe(workflow_path)


def test_workflow_recipe_rejects_invalid_indexing_mode(tmp_path: Path) -> None:
    workflow_path = tmp_path / "invalid_indexing_mode.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: invalid_indexing_mode",
                "  steps: [load, assess_quality, calibrate, preprocess, analyze, report]",
                "  parameters:",
                "    diffraction:",
                "      indexing_mode: impossible_mode",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_workflow_recipe(workflow_path)


def test_workflow_recipe_rejects_invalid_b4_tolerances(tmp_path: Path) -> None:
    workflow_path = tmp_path / "invalid_b4_tolerance.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: invalid_b4_tolerance",
                "  steps: [load, assess_quality, calibrate, preprocess, analyze, report]",
                "  parameters:",
                "    diffraction:",
                "      indexing_mode: b4_mvp",
                "      max_hkl: 0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_workflow_recipe(workflow_path)


def test_workflow_recipe_rejects_non_list_candidate_phase_params(tmp_path: Path) -> None:
    workflow_path = tmp_path / "invalid_b4_phase_params.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: invalid_b4_phase_params",
                "  steps: [load, assess_quality, calibrate, preprocess, analyze, report]",
                "  parameters:",
                "    diffraction:",
                "      indexing_mode: b4_mvp",
                "      candidate_phase_params: bad_payload",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_workflow_recipe(workflow_path)


def test_workflow_recipe_accepts_b1_calibration_parameters(tmp_path: Path) -> None:
    workflow_path = tmp_path / "b1_calibration_params.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: b1_calibration_params",
                "  steps: [load, assess_quality, calibrate, preprocess, fft, report]",
                "  parameters:",
                "    calibration:",
                "      method: reference",
                "      voltage_kv: 200",
                "      known_d_spacing: 0.213",
                "      known_d_spacing_unit: nm",
                "      measured_radius_px: 46.5",
                "      detector_pixel_size_um: 15.0",
                "outputs:",
                "  formats: [json]",
            ]
        ),
        encoding="utf-8",
    )

    recipe = load_workflow_recipe(workflow_path)
    calibration = recipe.workflow.parameters.get("calibration")
    assert isinstance(calibration, dict)
    assert calibration["known_d_spacing"] == 0.213
    assert calibration["measured_radius_px"] == 46.5


def test_workflow_recipe_rejects_invalid_b1_calibration_radius(tmp_path: Path) -> None:
    workflow_path = tmp_path / "invalid_b1_radius.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: invalid_b1_radius",
                "  parameters:",
                "    calibration:",
                "      known_d_spacing: 0.213",
                "      measured_radius_px: 0",
                "outputs:",
                "  formats: [json]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_workflow_recipe(workflow_path)


def test_workflow_recipe_accepts_quality_threshold_parameters(tmp_path: Path) -> None:
    workflow_path = tmp_path / "quality_thresholds.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: quality_thresholds",
                "  steps: [load, assess_quality, calibrate, preprocess, analyze, report]",
                "  parameters:",
                "    quality:",
                "      thresholds:",
                "        snr_min: 3.5",
                "        blur_risk_max: 0.7",
                "        overall_score_min: 0.45",
                "outputs:",
                "  formats: [json]",
            ]
        ),
        encoding="utf-8",
    )

    recipe = load_workflow_recipe(workflow_path)
    quality = recipe.workflow.parameters.get("quality")
    assert isinstance(quality, dict)
    thresholds = quality.get("thresholds")
    assert isinstance(thresholds, dict)
    assert thresholds["snr_min"] == 3.5


def test_workflow_recipe_rejects_invalid_quality_threshold_value(tmp_path: Path) -> None:
    workflow_path = tmp_path / "invalid_quality_threshold.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: invalid_quality_threshold",
                "  parameters:",
                "    quality:",
                "      thresholds:",
                "        blur_risk_max: 1.2",
                "outputs:",
                "  formats: [json]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_workflow_recipe(workflow_path)


def test_workflow_recipe_accepts_optional_calibration_adapter_engine(tmp_path: Path) -> None:
    workflow_path = tmp_path / "adapter_engine.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: adapter_engine",
                "  parameters:",
                "    calibration:",
                "      known_d_spacing: 0.213",
                "      measured_radius_px: 46.5",
                "      diffraction_calibration_engine: py4dstem",
                "outputs:",
                "  formats: [json]",
            ]
        ),
        encoding="utf-8",
    )
    recipe = load_workflow_recipe(workflow_path)
    calibration = recipe.workflow.parameters.get("calibration")
    assert isinstance(calibration, dict)
    assert calibration["diffraction_calibration_engine"] == "py4dstem"


def test_workflow_recipe_rejects_invalid_calibration_adapter_engine(tmp_path: Path) -> None:
    workflow_path = tmp_path / "bad_adapter_engine.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: bad_adapter_engine",
                "  parameters:",
                "    calibration:",
                "      known_d_spacing: 0.213",
                "      measured_radius_px: 46.5",
                "      diffraction_calibration_engine: invalid_engine",
                "outputs:",
                "  formats: [json]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_workflow_recipe(workflow_path)


def test_workflow_recipe_rejects_negative_calibration_uncertainty(tmp_path: Path) -> None:
    workflow_path = tmp_path / "bad_calibration_uncertainty.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: bad_calibration_uncertainty",
                "  parameters:",
                "    calibration:",
                "      known_d_spacing: 0.213",
                "      measured_radius_px: 46.5",
                "      known_d_uncertainty_nm: -0.001",
                "outputs:",
                "  formats: [json]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_workflow_recipe(workflow_path)


def test_a2_workflow_applies_quality_threshold_warnings(tmp_path: Path) -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[32, 32] = 1.0
    input_path = tmp_path / "quality_threshold_input.npy"
    np.save(input_path, image)

    workflow_path = tmp_path / "quality_threshold_workflow.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: quality_threshold_apply",
                "  steps:",
                "    - load",
                "    - assess_quality",
                "    - calibrate",
                "    - preprocess",
                "    - analyze",
                "    - report",
                "  parameters:",
                "    quality:",
                "      thresholds:",
                "        snr_min: 5.0",
                "        overall_score_min: 0.8",
            ]
        ),
        encoding="utf-8",
    )

    result = TEMAnalysisSkill().run(
        input_path=input_path,
        workflow_path=workflow_path,
        output_dir=tmp_path / "outputs",
    )
    report = read_json(Path(result["report_path"]), AnalysisReport)
    assert report.workflow is not None
    quality_step = next(step for step in report.workflow.steps if step.name == "assess_quality")
    assert quality_step.status == "warning"
    assert any("Quality threshold not met:" in warning for warning in quality_step.warnings)
    assert report.quality is not None
    assert report.quality.provenance.get("thresholds")


def test_workflow_recipe_rejects_unsupported_output_format(tmp_path: Path) -> None:
    workflow_path = tmp_path / "invalid_format.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: invalid_format",
                "outputs:",
                "  formats: [json, xml]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_workflow_recipe(workflow_path)


def test_workflow_recipe_accepts_svg_and_pdf_output_formats(tmp_path: Path) -> None:
    workflow_path = tmp_path / "svg_pdf_format.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: svg_pdf_format",
                "outputs:",
                "  formats: [json, html, svg, pdf]",
            ]
        ),
        encoding="utf-8",
    )

    recipe = load_workflow_recipe(workflow_path)
    assert set(recipe.outputs.formats) == {"json", "html", "svg", "pdf"}


def test_reserved_step_contracts_are_emitted(tmp_path: Path) -> None:
    image = np.ones((24, 24), dtype=np.float32)
    input_path = tmp_path / "reserved.npy"
    np.save(input_path, image)

    workflow_path = tmp_path / "reserved_steps.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: reserved_contracts",
                "  steps:",
                "    - load",
                "    - assess_quality",
                "    - calibrate",
                "    - preprocess",
                "    - detect_atoms",
                "    - run_gpa",
                "    - report",
            ]
        ),
        encoding="utf-8",
    )

    result = TEMAnalysisSkill().run(
        input_path=input_path,
        workflow_path=workflow_path,
        output_dir=tmp_path / "outputs",
    )
    report = read_json(Path(result["report_path"]), AnalysisReport)
    assert report.workflow is not None
    detect_atoms_step = next(step for step in report.workflow.steps if step.name == "detect_atoms")
    assert detect_atoms_step.status == "warning"
    assert detect_atoms_step.inputs
    assert detect_atoms_step.outputs
    assert any("Reserved ownership: Member C." in warning for warning in detect_atoms_step.warnings)
