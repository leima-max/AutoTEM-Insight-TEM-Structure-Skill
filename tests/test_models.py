from pathlib import Path

import pytest
from pydantic import ValidationError

from autotem.core import (
    AIInferenceResult,
    AIModelInfo,
    AnalysisReport,
    ArrayReference,
    AtomLattice,
    AtomSite,
    BenchmarkCase,
    BenchmarkResult,
    CalibrationResult,
    DefectCandidate,
    DiffractionPeak,
    DiffractionResult,
    ElementMap,
    FourDSTEMResult,
    GVector,
    Measurement,
    MicrostructureFeature,
    MicrostructureResult,
    OutputArtifact,
    PreprocessingResult,
    QualityAssessment,
    RunSummary,
    RunSummaryStep,
    SimulatedPattern,
    SimulatedSpot,
    SimulationResult,
    SpectroscopyResult,
    SpectroscopySignal,
    StrainResult,
    TEMDataset,
    ValidationMetric,
    WorkflowConfig,
    WorkflowStep,
    ZContrastResult,
)
from autotem.core.schema import (
    core_json_schemas,
    from_json,
    read_json,
    schema_manifest,
    to_json,
    write_core_json_schemas,
    write_json,
)


def test_measurement_json_round_trip() -> None:
    measurement = Measurement(
        value=0.204,
        unit="nm",
        uncertainty=0.002,
        method="fft_peak_fit",
        confidence=0.91,
        warnings=["calibration estimated"],
    )

    restored = from_json(Measurement, to_json(measurement))

    assert restored == measurement
    assert restored.confidence == 0.91


def test_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Measurement(value=1.0, confidence=1.5)


def test_dataset_rejects_negative_shape() -> None:
    with pytest.raises(ValidationError):
        ArrayReference(shape=(-1, 128), dtype="float32")


def test_top_level_models_include_schema_version() -> None:
    dataset = TEMDataset(data=ArrayReference(shape=(32, 32)))
    diffraction = DiffractionResult()
    atom_lattice = AtomLattice()
    strain = StrainResult()
    report = AnalysisReport()

    assert WorkflowConfig().schema_version == "0.1.0"
    assert dataset.schema_version == "0.1.0"
    assert QualityAssessment().schema_version == "0.1.0"
    assert CalibrationResult().schema_version == "0.1.0"
    assert PreprocessingResult().schema_version == "0.1.0"
    assert diffraction.schema_version == "0.1.0"
    assert atom_lattice.schema_version == "0.1.0"
    assert strain.schema_version == "0.1.0"
    assert MicrostructureResult().schema_version == "0.1.0"
    assert ZContrastResult().schema_version == "0.1.0"
    assert SpectroscopyResult().schema_version == "0.1.0"
    assert FourDSTEMResult().schema_version == "0.1.0"
    assert SimulationResult().schema_version == "0.1.0"
    assert AIInferenceResult(model=AIModelInfo(name="baseline")).schema_version == "0.1.0"
    assert BenchmarkResult().schema_version == "0.1.0"
    assert report.schema_version == "0.1.0"


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Measurement(value=1.0, unexpected=True)


def test_core_report_round_trip() -> None:
    dataset = TEMDataset(
        data=ArrayReference(shape=(256, 256), dtype="float32", uri="data/sample.zarr"),
        dataset_id="sample-001",
        source_path="inputs/sample.tif",
        modality="HAADF-STEM",
        pixel_size=Measurement(value=0.02, unit="nm", method="metadata", confidence=0.8),
        pixel_unit="nm",
        metadata={"voltage_kv": 200},
        provenance={"reader": "fixture"},
    )
    workflow = WorkflowConfig(
        name="haadf_atomic",
        steps=[
            WorkflowStep(name="load", status="done", outputs=["dataset"]),
            WorkflowStep(name="detect_atoms", status="warning", warnings=["low SNR"]),
        ],
        output_formats=["json", "html"],
    )
    quality = QualityAssessment(
        overall_score=0.72,
        snr=Measurement(value=12.0, unit="dB", method="baseline"),
        metadata_completeness=0.6,
        warnings=["pixel calibration estimated"],
    )
    calibration = CalibrationResult(
        pixel_size=Measurement(value=0.02, unit="nm", method="metadata"),
        pixel_unit="nm",
        confidence=0.8,
    )
    preprocessing = PreprocessingResult(
        image=ArrayReference(shape=(256, 256), uri="preprocessed/sample.npy"),
        operations=["normalize", "median_filter"],
        quality_after=quality,
    )
    diffraction = DiffractionResult(
        peaks=[DiffractionPeak(x=128.0, y=96.5, intensity=1200.0, confidence=0.95)],
        center=(128.0, 128.0),
        g_vectors=[
            GVector(
                gx=Measurement(value=3.1, unit="1/nm"),
                gy=Measurement(value=0.0, unit="1/nm"),
                magnitude=Measurement(value=3.1, unit="1/nm"),
            )
        ],
        d_spacings=[Measurement(value=0.322, unit="nm", method="fft")],
        confidence=0.86,
    )
    atom_lattice = AtomLattice(
        atoms=[
            AtomSite(
                x=12.4,
                y=15.8,
                intensity=Measurement(value=2400.0, unit="a.u."),
                confidence=0.9,
            )
        ],
        coordinate_unit="pixel",
    )
    strain = StrainResult(strain_xx=ArrayReference(shape=(64, 64), uri="strain/exx.npy"))
    microstructure = MicrostructureResult(
        features=[
            MicrostructureFeature(
                feature_id="grain-1",
                feature_type="grain",
                area=Measurement(value=120.0, unit="nm^2"),
            )
        ],
        defects=[DefectCandidate(defect_id="d-1", defect_type="twin", confidence=0.7)],
    )
    z_contrast = ZContrastResult(
        atom_intensities=[Measurement(value=3200.0, unit="a.u.")],
        relative_z=[Measurement(value=1.2, unit="relative")],
    )
    spectroscopy = SpectroscopyResult(
        signals=[SpectroscopySignal(signal_id="eds-1", kind="EDS")],
        element_maps=[
            ElementMap(element="Ti", map=ArrayReference(shape=(32, 32), uri="eds/ti.npy"))
        ],
    )
    fourdstem = FourDSTEMResult(
        datacube=ArrayReference(shape=(8, 8, 64, 64), uri="4dstem/cube.zarr"),
        virtual_bf=ArrayReference(shape=(8, 8), uri="4dstem/vbf.npy"),
    )
    simulation = SimulationResult(
        patterns=[
            SimulatedPattern(
                kind="diffraction",
                structure_id="cif-001",
                zone_axis=(0, 0, 1),
                spots=[
                    SimulatedSpot(
                        gx=Measurement(value=2.5, unit="1/nm"),
                        gy=Measurement(value=0.0, unit="1/nm"),
                    )
                ],
            )
        ],
        comparison_metrics=[Measurement(value=0.03, unit="nm", method="mean_d_error")],
    )
    ai_inference = AIInferenceResult(
        model=AIModelInfo(name="atom-detector", version="0.1"),
        labels=["atom"],
        confidence=0.75,
    )
    benchmark = BenchmarkResult(
        cases=[BenchmarkCase(case_id="std-001", dataset=dataset, workflow=workflow)],
        metrics=[
            ValidationMetric(
                name="d_spacing_relative_error",
                value=Measurement(value=0.01, unit="fraction"),
                direction="lower_is_better",
                passed=True,
            )
        ],
        passed=True,
    )
    report = AnalysisReport(
        summary="Scaffold report",
        workflow=workflow,
        dataset=dataset,
        quality=quality,
        calibration=calibration,
        preprocessing=preprocessing,
        diffraction=diffraction,
        atom_lattice=atom_lattice,
        strain=strain,
        microstructure=microstructure,
        z_contrast=z_contrast,
        spectroscopy=spectroscopy,
        fourdstem=fourdstem,
        simulation=simulation,
        ai_inference=ai_inference,
        benchmark=benchmark,
        artifacts=[
            OutputArtifact(
                kind="html",
                uri="reports/sample.html",
                label="HTML report",
                mime_type="text/html",
            )
        ],
    )

    restored = from_json(AnalysisReport, to_json(report))

    assert restored.dataset is not None
    assert restored.dataset.dataset_id == "sample-001"
    assert restored.dataset.modality == "HAADF-STEM"
    assert restored.workflow is not None
    assert restored.workflow.steps[1].status == "warning"
    assert restored.quality is not None
    assert restored.quality.overall_score == 0.72
    assert restored.calibration is not None
    assert restored.calibration.pixel_size.value == 0.02
    assert restored.preprocessing is not None
    assert restored.preprocessing.operations == ["normalize", "median_filter"]
    assert restored.diffraction is not None
    assert restored.diffraction.peaks[0].x == 128.0
    assert restored.atom_lattice is not None
    assert restored.atom_lattice.atoms[0].intensity.value == 2400.0
    assert restored.strain is not None
    assert restored.strain.strain_xx.shape == (64, 64)
    assert restored.microstructure is not None
    assert restored.microstructure.defects[0].defect_type == "twin"
    assert restored.z_contrast is not None
    assert restored.z_contrast.relative_z[0].value == 1.2
    assert restored.spectroscopy is not None
    assert restored.spectroscopy.element_maps[0].element == "Ti"
    assert restored.fourdstem is not None
    assert restored.fourdstem.datacube.shape == (8, 8, 64, 64)
    assert restored.simulation is not None
    assert restored.simulation.patterns[0].kind == "diffraction"
    assert restored.ai_inference is not None
    assert restored.ai_inference.model.name == "atom-detector"
    assert restored.benchmark is not None
    assert restored.benchmark.metrics[0].passed is True
    assert restored.artifacts[0].kind == "html"


def test_schema_exports_core_models() -> None:
    schemas = core_json_schemas()

    assert set(schemas) == {
        "Measurement",
        "WorkflowConfig",
        "TEMDataset",
        "QualityAssessment",
        "CalibrationResult",
        "PreprocessingResult",
        "DiffractionResult",
        "AtomLattice",
        "StrainResult",
        "MicrostructureResult",
        "ZContrastResult",
        "SpectroscopyResult",
        "FourDSTEMResult",
        "SimulationResult",
        "AIInferenceResult",
        "BenchmarkResult",
        "AnalysisReport",
        "RunSummary",
    }
    assert schemas["Measurement"]["properties"]["confidence"]["anyOf"]


def test_schema_manifest_tracks_core_models() -> None:
    manifest = schema_manifest()

    assert manifest == {
        "schema_version": "0.1.0",
        "models": [
            "Measurement",
            "WorkflowConfig",
            "TEMDataset",
            "QualityAssessment",
            "CalibrationResult",
            "PreprocessingResult",
            "DiffractionResult",
            "AtomLattice",
            "StrainResult",
            "MicrostructureResult",
            "ZContrastResult",
            "SpectroscopyResult",
            "FourDSTEMResult",
            "SimulationResult",
            "AIInferenceResult",
            "BenchmarkResult",
            "AnalysisReport",
            "RunSummary",
        ],
    }


def test_json_file_helpers(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    dataset = TEMDataset(data=ArrayReference(shape=(8, 8), inline=[[1, 2], [3, 4]]))

    write_json(path, dataset)
    restored = read_json(path, TEMDataset)

    assert restored == dataset


def test_workflow_rejects_unknown_task() -> None:
    with pytest.raises(ValidationError):
        WorkflowStep(name="unknown_task")


def test_output_artifact_round_trip() -> None:
    artifact = OutputArtifact(kind="table", uri="tables/peaks.csv", label="Peak table")

    restored = from_json(OutputArtifact, to_json(artifact))

    assert restored == artifact


def test_quality_bounds_are_validated() -> None:
    with pytest.raises(ValidationError):
        QualityAssessment(overall_score=-0.1)


def test_future_module_models_round_trip() -> None:
    benchmark = BenchmarkResult(
        metrics=[
            ValidationMetric(
                name="atom_coordinate_rmse",
                value=Measurement(value=4.5, unit="pm"),
                target=Measurement(value=10.0, unit="pm"),
                direction="lower_is_better",
                passed=True,
            )
        ]
    )

    restored = from_json(BenchmarkResult, to_json(benchmark))

    assert restored.metrics[0].name == "atom_coordinate_rmse"
    assert restored.metrics[0].passed is True


def test_write_core_json_schemas(tmp_path: Path) -> None:
    write_core_json_schemas(tmp_path)

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "TEMDataset.schema.json").exists()
    assert (tmp_path / "RunSummary.schema.json").exists()
    assert (tmp_path / "CalibrationResult.schema.json").exists()
    assert (tmp_path / "BenchmarkResult.schema.json").exists()
    assert '"schema_version": "0.1.0"' in (tmp_path / "manifest.json").read_text(
        encoding="utf-8"
    )


def test_run_summary_model_round_trip() -> None:
    summary = RunSummary(
        status="A2 baseline workflow completed.",
        workflow_name="default",
        input_path="inputs/sample.npy",
        workflow_path="configs/default.yaml",
        output_dir="outputs/sample",
        report_path="outputs/sample/analysis_report.json",
        log_path="outputs/sample/run.log",
        workflow_used_path="outputs/sample/workflow_used.yaml",
        steps=[RunSummaryStep(name="load", status="done", warning_count=0)],
        warning_count=0,
    )

    restored = from_json(RunSummary, to_json(summary))

    assert restored.workflow_name == "default"
    assert restored.steps[0].name == "load"
