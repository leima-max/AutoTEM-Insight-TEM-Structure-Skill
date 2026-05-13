from pathlib import Path

import numpy as np

from autotem import TEMAnalysisSkill
from autotem.core import (
    AnalysisReport,
    ArrayReference,
    AtomLattice,
    AtomSite,
    DiffractionPeak,
    DiffractionResult,
    Measurement,
    PhaseCandidate,
    ReportFigure,
    StrainResult,
    TEMDataset,
    read_json,
)
from autotem.reporting import evaluate_bc_interface_completeness


def test_reporting_provenance_contains_hashes_and_dependency_versions(tmp_path: Path) -> None:
    image = np.zeros((48, 48), dtype=np.float32)
    image[24, 18] = 6.0
    image[24, 30] = 6.0
    input_path = tmp_path / "prov.npy"
    np.save(input_path, image)

    result = TEMAnalysisSkill().run(
        input_path=input_path,
        output_dir=tmp_path / "outputs",
        output_formats_override=["json", "html"],
        report_title="Provenance Test",
    )
    report = read_json(Path(result["report_path"]), AnalysisReport)
    runtime = report.provenance.get("runtime", {})
    assert isinstance(runtime, dict)
    dependency_versions = runtime.get("dependency_versions", {})
    assert isinstance(dependency_versions, dict)
    assert "numpy" in dependency_versions
    assert "workflow_parameters_hash_sha256" in report.provenance
    assert "report_options_hash_sha256" in report.provenance
    assert "requested_formats_hash_sha256" in report.provenance


def test_quality_gate_adds_blocking_warnings_for_failed_workflow_steps(
    tmp_path: Path,
) -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    input_path = tmp_path / "failed_step_input.npy"
    np.save(input_path, image)
    workflow_path = tmp_path / "failed_workflow.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: failed_step_gate",
                "  steps:",
                "    - assess_quality",
                "    - report",
                "outputs:",
                "  formats: [json]",
            ]
        ),
        encoding="utf-8",
    )

    result = TEMAnalysisSkill().run(
        input_path=input_path,
        workflow_path=workflow_path,
        output_dir=tmp_path / "outputs",
        output_formats_override=["json"],
    )
    report = read_json(Path(result["report_path"]), AnalysisReport)
    blocking = [item for item in report.warnings if item.startswith("BLOCKING:")]
    assert blocking
    quality_gates = report.provenance.get("quality_gates", {})
    assert isinstance(quality_gates, dict)
    assert quality_gates.get("passed") is False


def test_quality_gate_interface_required_items_can_be_configured(tmp_path: Path) -> None:
    image = np.zeros((48, 48), dtype=np.float32)
    image[24, 18] = 4.0
    image[24, 30] = 4.0
    input_path = tmp_path / "interface_gate.npy"
    np.save(input_path, image)
    workflow_path = tmp_path / "interface_gate.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: interface_gate_required",
                "  steps: [load, assess_quality, calibrate, preprocess, analyze, report]",
                "  parameters:",
                "    reporting:",
                "      interface_gates:",
                "        enabled: true",
                "        required_items: [B.DiffractionResult, C.AtomLattice]",
                "outputs:",
                "  formats: [json]",
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
    assert any(
        warning.startswith("BLOCKING: B/C interface completeness gate failed")
        for warning in report.warnings
    )
    quality_gates = report.provenance.get("quality_gates", {})
    assert isinstance(quality_gates, dict)
    gates = quality_gates.get("gates", {})
    assert isinstance(gates, dict)
    interface_gate = gates.get("bc_interface_required_items_present", {})
    assert isinstance(interface_gate, dict)
    assert interface_gate.get("passed") is False
    assert "C.AtomLattice" in interface_gate.get("missing_items", [])


def test_large_array_layering_metadata_present(tmp_path: Path) -> None:
    image = np.zeros((2048, 2048), dtype=np.float32)
    image[1024, 1000] = 8.0
    image[1024, 1048] = 8.0
    input_path = tmp_path / "large.npy"
    np.save(input_path, image)

    TEMAnalysisSkill().run(
        input_path=input_path,
        output_dir=tmp_path / "outputs",
        output_formats_override=["json"],
    )
    intermediates = tmp_path / "outputs" / "intermediates"
    metadata_path = intermediates / "input_image_metadata.json"
    assert metadata_path.exists()
    metadata = metadata_path.read_text(encoding="utf-8")
    assert "storage_tier" in metadata
    assert "threshold_bytes" in metadata


def test_bc_interface_completeness_passes_with_complete_synthetic_outputs() -> None:
    diffraction = DiffractionResult(
        peaks=[DiffractionPeak(x=12.0, y=16.0, intensity=100.0)],
        d_spacings=[Measurement(value=0.2, unit="nm")],
        phase_candidates=[PhaseCandidate(name="Gold", score=0.8)],
    )
    dataset = TEMDataset(data=ArrayReference(shape=(16, 16), dtype="float32", uri="sample.npy"))
    atom_lattice = AtomLattice(
        atoms=[AtomSite(x=10.0, y=10.0)],
        overlay=ArrayReference(shape=tuple(), dtype="uint8", uri="atom_overlay.png"),
    )
    strain = StrainResult(
        strain_xx=ArrayReference(shape=(8, 8), dtype="float32", uri="strain_xx.npy"),
    )
    report = AnalysisReport(
        dataset=dataset,
        diffraction=diffraction,
        atom_lattice=atom_lattice,
        strain=strain,
        figures=[],
        tables=[],
        artifacts=[],
    )
    report.figures = [
        ReportFigure(
            label="FFT peaks overlay",
            uri="fft_peaks_overlay.png",
            caption="overlay",
            figure_type="diffraction_overlay",
        ),
        ReportFigure(
            label="Atom overlay",
            uri="atom_overlay.png",
            caption="overlay",
            figure_type="atom_overlay",
        ),
    ]

    checks = evaluate_bc_interface_completeness(report)
    assert checks
    assert all(item["status"] == "pass" for item in checks)
