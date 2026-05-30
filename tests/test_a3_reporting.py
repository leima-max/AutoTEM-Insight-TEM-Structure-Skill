import json
from pathlib import Path

import numpy as np
import pytest

from autotem import TEMAnalysisSkill
from autotem.core import (
    AnalysisReport,
    DiffractionPeak,
    DiffractionResult,
    GVector,
    Measurement,
    PhaseCandidate,
    SimulatedPattern,
    SimulatedSpot,
    SimulationResult,
    read_json,
)
from autotem.reporting import export_csv_tables_with_models, export_html_report
from autotem.visualization import build_core_overlays


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


def test_b5_exports_measured_simulated_diffraction_overlay(tmp_path: Path) -> None:
    report = AnalysisReport(
        diffraction=DiffractionResult(
            peaks=[
                DiffractionPeak(x=1.0, y=0.0, intensity=10.0),
                DiffractionPeak(x=0.0, y=1.0, intensity=8.0),
            ],
            g_vectors=[
                GVector(
                    gx=Measurement(value=2.5, unit="1/nm"),
                    gy=Measurement(value=0.0, unit="1/nm"),
                    magnitude=Measurement(value=2.5, unit="1/nm"),
                    source_peak_index=0,
                ),
                GVector(
                    gx=Measurement(value=0.0, unit="1/nm"),
                    gy=Measurement(value=2.5, unit="1/nm"),
                    magnitude=Measurement(value=2.5, unit="1/nm"),
                    source_peak_index=1,
                ),
            ],
        ),
        simulation=SimulationResult(
            patterns=[
                SimulatedPattern(
                    kind="diffraction",
                    structure_id="fake",
                    zone_axis=(1, 1, 0),
                    simulation_parameters={
                        "engine": "diffsims",
                        "phase_match": {
                            "rotation_deg": 0.0,
                            "matched_pairs": [
                                {"measured_g_index": 0, "simulated_spot_index": 0},
                                {"measured_g_index": 1, "simulated_spot_index": 1},
                            ],
                        },
                    },
                    spots=[
                        SimulatedSpot(
                            gx=Measurement(value=2.5, unit="1/nm"),
                            gy=Measurement(value=0.0, unit="1/nm"),
                        ),
                        SimulatedSpot(
                            gx=Measurement(value=0.0, unit="1/nm"),
                            gy=Measurement(value=2.5, unit="1/nm"),
                        ),
                    ],
                )
            ],
            matched_phase_candidates=[PhaseCandidate(name="fake", zone_axis=(1, 1, 0))],
            provenance={
                "phase_matching": {
                    "active": True,
                    "selected_phase": "fake",
                    "per_phase": [{"phase_name": "fake", "best_pattern_index": 0}],
                }
            },
        ),
    )

    artifacts, figures, warnings = build_core_overlays(report, tmp_path)

    overlay_path = tmp_path / "figures" / "simulation_diffraction_overlay.png"
    assert overlay_path.exists()
    assert any(figure.figure_type == "simulation_diffraction_overlay" for figure in figures)
    assert any(artifact.label == "Simulation diffraction overlay" for artifact in artifacts)
    assert report.simulation is not None
    assert report.simulation.overlays
    assert report.simulation.overlays[0].provenance["zone_axis"] == [1, 1, 0]
    assert not any("B5 simulation overlay skipped" in warning for warning in warnings)


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
    assert "Simulation Phase Matching" in html_text
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

    manifest = json.loads(
        (output_dir / "json" / "output_manifest.json").read_text(encoding="utf-8")
    )
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


def test_a3_reporting_surfaces_diffsims_closed_loop_summary(tmp_path: Path) -> None:
    candidate = PhaseCandidate(
        name="mp-149",
        formula="Si",
        source="materials_project_api",
        identifier="mp-149",
        score=0.99,
        matched_spot_count=4,
        mean_d_error=Measurement(
            value=0.001,
            unit="relative",
            method="diffsims_phase_match",
            provenance={"rotation_deg": 0.0, "raw_diffsims_score": 0.99},
        ),
        mean_angle_error=Measurement(value=0.2, unit="deg", method="diffsims_phase_match"),
        lattice_parameters={
            "materials_project": {
                "rank": 0,
                "energy_above_hull": 0.0,
                "is_stable": True,
                "theoretical": False,
                "deprecated": False,
                "site_count": 2,
            }
        },
    )
    alternate = PhaseCandidate(
        name="mp-alt",
        formula="Si",
        source="materials_project_api",
        identifier="mp-alt",
        score=0.35,
        matched_spot_count=1,
        lattice_parameters={
            "materials_project": {
                "rank": 1,
                "energy_above_hull": 0.12,
                "is_stable": False,
                "theoretical": True,
                "deprecated": False,
                "site_count": 8,
            }
        },
    )
    report = AnalysisReport(
        diffraction=DiffractionResult(phase_candidates=[candidate, alternate]),
        simulation=SimulationResult(
            patterns=[
                SimulatedPattern(
                    kind="diffraction",
                    structure_id="mp-149",
                    simulation_parameters={
                        "engine": "diffsims",
                        "simulated_spot_count": 12,
                        "rotation_angle_deg": 0.0,
                        "phase_match": {
                            "matched_spot_count": 4,
                            "rotation_deg": 0.0,
                        },
                    },
                    spots=[
                        SimulatedSpot(
                            gx=Measurement(value=2.5, unit="1/nm"),
                            gy=Measurement(value=0.0, unit="1/nm"),
                        )
                    ],
                )
            ],
            matched_phase_candidates=[candidate],
            provenance={
                "method": "diffsims",
                "adapter": {"active": True},
                "phase_matching": {
                    "active": True,
                    "method": "diffsims_template_spot_match",
                    "selected_phase": "mp-149",
                    "measured_spot_count": 4,
                    "simulated_pattern_count": 1,
                    "d_tolerance_rel": 0.03,
                    "angle_tolerance_deg": 3.0,
                    "per_phase": [
                        {
                            "phase_name": "mp-149",
                            "best_pattern_index": 0,
                            "simulated_spot_count": 12,
                            "matched_spot_count": 4,
                            "rotation_deg": 0.0,
                        }
                    ],
                },
            },
        )
    )

    export_csv_tables_with_models(report, tmp_path)
    export_html_report(report, tmp_path)

    matches_csv = (tmp_path / "tables" / "simulation_phase_matches.csv").read_text(
        encoding="utf-8"
    )
    assert "source,identifier,engine,adapter_active,phase_matching_active" in matches_csv
    assert "diffsims_template_spot_match" in matches_csv
    assert "mp-149" in matches_csv
    html_text = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Simulation Phase Matching" in html_text
    assert "Real adapter active" in html_text
    assert "Candidate Comparison" in html_text
    assert "MP E above hull" in html_text
    assert "materials_project_api" in html_text
    assert "mp-149" in html_text
    assert "mp-alt" in html_text
    assert "0.12" in html_text
