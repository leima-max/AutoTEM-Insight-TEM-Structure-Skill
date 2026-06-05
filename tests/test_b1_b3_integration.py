from __future__ import annotations

from pathlib import Path

import numpy as np

from autotem import TEMAnalysisSkill
from autotem.core import AnalysisReport, read_json


def _write_b1_b3_workflow(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: b1_b3_integration",
                "  steps:",
                "    - load",
                "    - assess_quality",
                "    - calibrate",
                "    - preprocess",
                "    - analyze",
                "    - report",
                "  parameters:",
                "    calibration:",
                "      method: reference",
                "      known_d_spacing: 0.2",
                "      known_d_spacing_unit: nm",
                "      measured_radius_px: 10",
                "      known_d_uncertainty_nm: 0.002",
                "      measured_radius_uncertainty_px: 0.2",
                "    preprocessing:",
                "      denoise_method: gaussian",
                "      denoise_sigma: 0.6",
                "      background_method: none",
                "    diffraction:",
                "      peak_detector: local_max",
                "      top_k_peaks: 24",
                "      min_peak_distance: 3",
                "      center_exclusion_radius: 4",
                "      peak_threshold_strategy: relative_max",
                "      peak_threshold_value: 0.12",
                "      fft_center_mode: auto",
                "      direct_beam_refine_window: 3",
                "outputs:",
                "  formats: [json, csv]",
                "  save_intermediates: true",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_b1_b3_raw_workflow(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: b1_b3_raw_input",
                "  steps: [load, assess_quality, calibrate, preprocess, analyze, report]",
                "  parameters:",
                "    calibration:",
                "      method: reference",
                "      known_d_spacing: 0.2",
                "      known_d_spacing_unit: nm",
                "      measured_radius_px: 10",
                "    preprocessing:",
                "      enabled: false",
                "      denoise_method: gaussian",
                "      background_method: gaussian",
                "    diffraction:",
                "      peak_detector: local_max",
                "      top_k_peaks: 8",
                "      min_peak_distance: 3",
                "      center_exclusion_radius: 4",
                "outputs:",
                "  formats: [json]",
                "  save_intermediates: true",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _synthetic_image(size: int = 80) -> np.ndarray:
    spectrum = np.zeros((size, size), dtype=np.complex128)
    cy = size // 2
    cx = size // 2
    spectrum[cy, cx] = 2200.0
    for dy, dx, amp in [
        (0, 10, 320.0),
        (0, -10, 320.0),
        (10, 0, 300.0),
        (-10, 0, 300.0),
        (7, 7, 180.0),
        (-7, -7, 180.0),
    ]:
        spectrum[cy + dy, cx + dx] = amp
    image = np.fft.ifft2(np.fft.ifftshift(spectrum)).real
    image = image - float(image.min())
    scale = float(image.max()) if float(image.max()) > 0 else 1.0
    image = image / scale
    return image.astype(np.float32)


def test_b1_b3_end_to_end_pipeline_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "b1_b3_input.npy"
    np.save(input_path, _synthetic_image())
    workflow_path = _write_b1_b3_workflow(tmp_path / "b1_b3_workflow.yaml")
    output_dir = tmp_path / "outputs"

    run_result = TEMAnalysisSkill().run(
        input_path=input_path,
        workflow_path=workflow_path,
        output_dir=output_dir,
    )
    report = read_json(Path(run_result["report_path"]), AnalysisReport)

    assert report.diffraction is not None
    assert report.calibration is not None
    assert report.calibration.diffraction_scale is not None
    assert report.diffraction.peaks
    assert report.diffraction.g_vectors
    assert report.diffraction.d_spacings
    assert any(item.unit == "nm" for item in report.diffraction.d_spacings)
    assert any(
        item.uncertainty is not None and item.uncertainty > 0
        for item in report.diffraction.d_spacings
    )
    assert report.diffraction.fft_image is not None
    assert report.diffraction.fft_image.uri is not None
    assert Path(report.diffraction.fft_image.uri).exists()
    assert (output_dir / "tables" / "diffraction_peaks.csv").exists()
    assert (output_dir / "tables" / "d_spacings.csv").exists()
    assert (output_dir / "tables" / "indexed_spots.csv").exists()


def test_b1_b3_workflow_can_analyze_raw_input_without_b2_denoise(tmp_path: Path) -> None:
    input_path = tmp_path / "b1_b3_raw_input.npy"
    np.save(input_path, _synthetic_image())
    workflow_path = _write_b1_b3_raw_workflow(tmp_path / "b1_b3_raw_workflow.yaml")
    output_dir = tmp_path / "raw_outputs"

    run_result = TEMAnalysisSkill().run(
        input_path=input_path,
        workflow_path=workflow_path,
        output_dir=output_dir,
    )
    report = read_json(Path(run_result["report_path"]), AnalysisReport)

    assert report.preprocessing is not None
    assert report.preprocessing.operations == ["raw_passthrough"]
    assert report.preprocessing.provenance["mode"] == "raw_passthrough"
    assert report.preprocessing.image is not None
    assert report.preprocessing.image.uri == str(input_path)
    assert report.diffraction is not None
    assert report.diffraction.provenance["method"] == "run_diffraction_analysis"
