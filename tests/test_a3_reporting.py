import json
from pathlib import Path

import numpy as np
import pytest

from autotem import TEMAnalysisSkill
from autotem.core import AnalysisReport, read_json


def test_a3_exports_figures_and_json_bundle(tmp_path: Path) -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[32, 40] = 12.0
    image[32, 24] = 9.0
    image[24, 32] = 6.0
    input_path = tmp_path / "a3_input.npy"
    np.save(input_path, image)

    output_dir = tmp_path / "outputs"
    result = TEMAnalysisSkill().run(input_path=input_path, output_dir=output_dir)
    report = read_json(Path(result["report_path"]), AnalysisReport)

    assert (output_dir / "figures" / "input_image.png").exists()
    assert (output_dir / "figures" / "fft_magnitude.png").exists()
    assert (output_dir / "figures" / "fft_peaks_overlay.png").exists()
    assert (output_dir / "json" / "report_bundle_summary.json").exists()
    assert (output_dir / "json" / "output_manifest.json").exists()
    assert report.figures
    assert any("fft" in figure.figure_type for figure in report.figures)
    manifest_text = (output_dir / "json" / "output_manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert "directories" in manifest
    assert set(manifest["directories"]) == {"figures", "tables", "json", "intermediates"}


def test_a3_html_report_contains_sections(tmp_path: Path) -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[12, 12] = 5.0
    input_path = tmp_path / "a3_html.npy"
    np.save(input_path, image)
    output_dir = tmp_path / "outputs"
    workflow_path = tmp_path / "workflow_html.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: html_report",
                "  steps:",
                "    - load",
                "    - assess_quality",
                "    - calibrate",
                "    - preprocess",
                "    - analyze",
                "    - report",
                "outputs:",
                "  formats: [json, html]",
            ]
        ),
        encoding="utf-8",
    )

    TEMAnalysisSkill().run(
        input_path=input_path,
        workflow_path=workflow_path,
        output_dir=output_dir,
    )
    html_text = (output_dir / "report.html").read_text(encoding="utf-8")

    assert "AutoTEM A3 Analysis Report" in html_text
    assert "Workflow Steps" in html_text
    assert "Provenance" in html_text
    assert "Workflow Step Details" in html_text
    assert "Interface Completeness (B/C)" in html_text
    assert "class=\"step-detail\"" in html_text


def test_a3_html_report_section_override(tmp_path: Path) -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[10, 20] = 5.0
    input_path = tmp_path / "a3_section_override.npy"
    np.save(input_path, image)
    output_dir = tmp_path / "outputs"

    TEMAnalysisSkill().run(
        input_path=input_path,
        output_dir=output_dir,
        output_formats_override=["json", "html"],
        report_sections_override=["overview", "interface_checks"],
    )
    html_text = (output_dir / "report.html").read_text(encoding="utf-8")

    assert "Overview" in html_text
    assert "Interface Completeness (B/C)" in html_text
    assert "Workflow Steps" not in html_text
    assert "Workflow Step Details" not in html_text


def test_a3_output_format_override_without_workflow(tmp_path: Path) -> None:
    image = np.zeros((24, 24), dtype=np.float32)
    image[8, 16] = 4.0
    input_path = tmp_path / "a3_override.npy"
    np.save(input_path, image)
    output_dir = tmp_path / "outputs"

    TEMAnalysisSkill().run(
        input_path=input_path,
        output_dir=output_dir,
        output_formats_override=["json", "csv", "html"],
        report_title="A3 Stage2 Custom Report",
    )

    assert (output_dir / "report.html").exists()
    assert (output_dir / "tables" / "diffraction_peaks.csv").exists()
    assert (output_dir / "tables" / "interface_completeness_bc.csv").exists()
    assert (output_dir / "json" / "interface_completeness_bc.json").exists()
    report = read_json(output_dir / "analysis_report.json", AnalysisReport)
    assert report.summary == "A3 Stage2 Custom Report"
    assert any(table.table_type == "interface_completeness_bc" for table in report.tables)


def test_a3_svg_pdf_and_intermediates_exports(tmp_path: Path) -> None:
    image = np.zeros((40, 40), dtype=np.float32)
    image[20, 10] = 5.0
    image[20, 30] = 5.0
    input_path = tmp_path / "a3_svg_pdf.npy"
    np.save(input_path, image)
    output_dir = tmp_path / "outputs"

    TEMAnalysisSkill().run(
        input_path=input_path,
        output_dir=output_dir,
        output_formats_override=["json", "svg", "pdf", "html"],
    )

    assert (output_dir / "report.pdf").exists()
    assert (output_dir / "figures" / "input_image.svg").exists()
    assert (output_dir / "figures" / "fft_magnitude.svg").exists()
    assert (output_dir / "intermediates" / "input_image.npy").exists()
    assert (output_dir / "intermediates" / "preprocessed_image.npy").exists()
    assert (output_dir / "intermediates" / "fft_magnitude.npy").exists()


def test_a3_save_intermediates_false_skips_intermediate_files(tmp_path: Path) -> None:
    image = np.zeros((20, 20), dtype=np.float32)
    image[8, 11] = 3.0
    input_path = tmp_path / "a3_no_intermediate.npy"
    np.save(input_path, image)
    output_dir = tmp_path / "outputs"
    workflow_path = tmp_path / "workflow_no_intermediate.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: no_intermediate",
                "  steps: [load, assess_quality, calibrate, preprocess, analyze, report]",
                "outputs:",
                "  formats: [json, html]",
                "  save_intermediates: false",
            ]
        ),
        encoding="utf-8",
    )

    TEMAnalysisSkill().run(
        input_path=input_path,
        workflow_path=workflow_path,
        output_dir=output_dir,
    )

    intermediates_dir = output_dir / "intermediates"
    assert intermediates_dir.exists()
    assert not list(intermediates_dir.glob("*"))


def test_a3_rejects_unsupported_output_override(tmp_path: Path) -> None:
    image = np.zeros((16, 16), dtype=np.float32)
    input_path = tmp_path / "a3_invalid.npy"
    np.save(input_path, image)

    with pytest.raises(ValueError):
        TEMAnalysisSkill().run(
            input_path=input_path,
            output_dir=tmp_path / "outputs",
            output_formats_override=["json", "xml"],
        )


def test_a3_manifest_includes_runtime_contract_artifacts(tmp_path: Path) -> None:
    image = np.zeros((28, 28), dtype=np.float32)
    image[14, 18] = 3.0
    input_path = tmp_path / "a3_manifest_runtime.npy"
    np.save(input_path, image)
    output_dir = tmp_path / "outputs"

    TEMAnalysisSkill().run(
        input_path=input_path,
        output_dir=output_dir,
        output_formats_override=["json", "html"],
    )

    manifest = json.loads((output_dir / "json" / "output_manifest.json").read_text(encoding="utf-8"))
    artifact_uris = [Path(item["uri"]).name for item in manifest["artifacts"]]

    assert "analysis_report.json" in artifact_uris
    assert "workflow_used.yaml" in artifact_uris
    assert "run_summary.json" in artifact_uris
    assert "run.log" in artifact_uris
    assert "interface_completeness_bc.json" in artifact_uris
    assert manifest["artifact_kind_counts"].get("log", 0) >= 1


def test_a3_interface_contract_json_summary_fields(tmp_path: Path) -> None:
    image = np.zeros((24, 24), dtype=np.float32)
    image[12, 16] = 5.0
    input_path = tmp_path / "a3_interface_contract.npy"
    np.save(input_path, image)
    output_dir = tmp_path / "outputs"

    TEMAnalysisSkill().run(
        input_path=input_path,
        output_dir=output_dir,
        output_formats_override=["json", "csv"],
    )

    contract_payload = json.loads(
        (output_dir / "json" / "interface_completeness_bc.json").read_text(encoding="utf-8")
    )
    assert contract_payload["contract"] == "A3_BC_interface_completeness"
    assert contract_payload["summary"]["total"] >= 1
    assert "pass_rate" in contract_payload["summary"]
