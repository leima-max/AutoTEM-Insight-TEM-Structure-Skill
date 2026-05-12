"""Shared data models for AutoTEM-Insight module boundaries."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

JsonScalar = str | int | float | bool | None
JsonDict = dict[str, Any]
CURRENT_SCHEMA_VERSION = "0.1.0"
PixelUnit = Literal["pm", "nm", "um", "angstrom", "pixel", "unknown"]
ArtifactKind = Literal["array", "image", "table", "json", "html", "log", "model", "other"]
WorkflowTask = Literal[
    "load",
    "assess_quality",
    "calibrate",
    "preprocess",
    "fft",
    "diffraction_peaks",
    "index_diffraction",
    "phase_match",
    "detect_atoms",
    "run_gpa",
    "atom_strain",
    "classify_sublattices",
    "segment_microstructure",
    "detect_defects",
    "quantify_z_contrast",
    "analyze_spectroscopy",
    "analyze_4dstem",
    "simulate",
    "ai_inference",
    "validate",
    "benchmark",
    "report",
]
StepStatus = Literal["pending", "running", "done", "warning", "failed", "skipped"]
DefectType = Literal[
    "grain_boundary",
    "twin",
    "dislocation",
    "stacking_fault",
    "vacancy",
    "interstitial",
    "phase_boundary",
    "unknown",
]
SpectroscopyKind = Literal["EDS", "EELS", "HAADF-Z", "unknown"]
SimulationKind = Literal["diffraction", "stem_image", "tem_image", "multislice", "other"]
MetricDirection = Literal["higher_is_better", "lower_is_better", "target_range", "informational"]
Modality = Literal[
    "TEM",
    "HRTEM",
    "STEM",
    "HAADF-STEM",
    "ABF-STEM",
    "SAED",
    "NBED",
    "4D-STEM",
    "unknown",
]


class AutoTEMModel(BaseModel):
    """Base model with strict fields and stable JSON behavior."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
    )


class VersionedAutoTEMModel(AutoTEMModel):
    """Base model for cross-module contract objects."""

    schema_version: Literal["0.1.0"] = CURRENT_SCHEMA_VERSION


class ArrayReference(AutoTEMModel):
    """Reference to array data stored inline for tiny arrays or externally for large data."""

    shape: tuple[int, ...] = Field(default_factory=tuple)
    dtype: str = "float32"
    uri: str | None = None
    inline: list[Any] | None = None
    description: str = ""

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(axis < 0 for axis in value):
            msg = "Array shape dimensions must be non-negative."
            raise ValueError(msg)
        return value


class Measurement(AutoTEMModel):
    """A measured or derived quantity with uncertainty and provenance."""

    value: JsonScalar
    unit: str = ""
    uncertainty: float | None = None
    method: str = ""
    parameters: JsonDict = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class OutputArtifact(AutoTEMModel):
    """Reference to a saved output produced by a workflow step."""

    kind: ArtifactKind
    uri: str
    label: str = ""
    mime_type: str = ""
    description: str = ""
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class WorkflowStep(AutoTEMModel):
    """Single workflow step declaration or execution record."""

    name: WorkflowTask
    status: StepStatus = "pending"
    parameters: JsonDict = Field(default_factory=dict)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class WorkflowConfig(VersionedAutoTEMModel):
    """Serializable workflow recipe used to reproduce an analysis run."""

    name: str = "default"
    description: str = ""
    steps: list[WorkflowStep] = Field(default_factory=list)
    parameters: JsonDict = Field(default_factory=dict)
    output_formats: list[str] = Field(default_factory=lambda: ["json"])
    save_intermediates: bool = True
    provenance: JsonDict = Field(default_factory=dict)


class TEMDataset(VersionedAutoTEMModel):
    """Normalized microscopy dataset passed from readers into workflows."""

    data: ArrayReference
    dataset_id: str = ""
    source_path: str = ""
    modality: Modality = "unknown"
    pixel_size: Measurement | None = None
    pixel_unit: PixelUnit = "unknown"
    metadata: JsonDict = Field(default_factory=dict)
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class QualityAssessment(VersionedAutoTEMModel):
    """Image and metadata quality metrics used to gate downstream analysis."""

    overall_score: float | None = Field(default=None, ge=0.0, le=1.0)
    snr: Measurement | None = None
    drift: Measurement | None = None
    blur: Measurement | None = None
    contamination: Measurement | None = None
    astigmatism: Measurement | None = None
    undersampling_risk: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata_completeness: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class CalibrationResult(VersionedAutoTEMModel):
    """Pixel and diffraction calibration outputs shared by analysis modules."""

    pixel_size: Measurement | None = None
    pixel_unit: PixelUnit = "unknown"
    diffraction_scale: Measurement | None = None
    camera_length: Measurement | None = None
    voltage: Measurement | None = None
    reference_measurements: list[Measurement] = Field(default_factory=list)
    transform: JsonDict = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class PreprocessingResult(VersionedAutoTEMModel):
    """Preprocessed image references and methods applied before analysis."""

    image: ArrayReference | None = None
    mask: ArrayReference | None = None
    background: ArrayReference | None = None
    operations: list[str] = Field(default_factory=list)
    parameters: JsonDict = Field(default_factory=dict)
    quality_after: QualityAssessment | None = None
    artifacts: list[OutputArtifact] = Field(default_factory=list)
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class DiffractionPeak(AutoTEMModel):
    """Detected diffraction peak position and intensity."""

    x: float
    y: float
    intensity: float | None = None
    sigma_x: float | None = None
    sigma_y: float | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class GVector(AutoTEMModel):
    """Reciprocal-space vector measurement."""

    gx: Measurement
    gy: Measurement
    magnitude: Measurement | None = None
    angle: Measurement | None = None
    source_peak_index: int | None = None


class IndexedSpot(AutoTEMModel):
    """Candidate hkl assignment for a diffraction spot."""

    peak_index: int
    h: int
    k: int
    l: int  # noqa: E741 - conventional hkl index name.
    d_spacing: Measurement | None = None
    error: Measurement | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class PhaseCandidate(AutoTEMModel):
    """Candidate phase matched against measured diffraction evidence."""

    name: str
    formula: str | None = None
    source: str | None = None
    identifier: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    matched_spot_count: int | None = Field(default=None, ge=0)
    mean_d_error: Measurement | None = None
    mean_angle_error: Measurement | None = None
    zone_axis: tuple[int, int, int] | None = None
    warnings: list[str] = Field(default_factory=list)


class DiffractionResult(VersionedAutoTEMModel):
    """FFT or diffraction analysis result shared by diffraction modules."""

    peaks: list[DiffractionPeak] = Field(default_factory=list)
    center: tuple[float, float] | None = None
    g_vectors: list[GVector] = Field(default_factory=list)
    d_spacings: list[Measurement] = Field(default_factory=list)
    indexed_spots: list[IndexedSpot] = Field(default_factory=list)
    phase_candidates: list[PhaseCandidate] = Field(default_factory=list)
    fft_image: ArrayReference | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class AtomSite(AutoTEMModel):
    """Single atom-column coordinate and optional refined attributes."""

    x: float
    y: float
    z: float | None = None
    intensity: Measurement | None = None
    sublattice_label: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class AtomLattice(VersionedAutoTEMModel):
    """Atom-column coordinates and projected lattice metadata."""

    atoms: list[AtomSite] = Field(default_factory=list)
    lattice_vectors: list[tuple[Measurement, Measurement]] = Field(default_factory=list)
    zone_axis: tuple[int, int, int] | None = None
    coordinate_unit: PixelUnit = "unknown"
    overlay: ArrayReference | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class StrainResult(VersionedAutoTEMModel):
    """Displacement, strain, rotation, and uncertainty maps."""

    displacement_x: ArrayReference | None = None
    displacement_y: ArrayReference | None = None
    strain_xx: ArrayReference | None = None
    strain_yy: ArrayReference | None = None
    strain_xy: ArrayReference | None = None
    rotation_xy: ArrayReference | None = None
    uncertainty: ArrayReference | None = None
    reference_region: tuple[int, int, int, int] | None = None
    g_vectors: list[GVector] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class MicrostructureFeature(AutoTEMModel):
    """Region, line, or point feature used for grains, phases, and interfaces."""

    feature_id: str = ""
    feature_type: str = ""
    label: str = ""
    mask: ArrayReference | None = None
    boundary: ArrayReference | None = None
    centroid: tuple[float, float] | None = None
    area: Measurement | None = None
    orientation: Measurement | None = None
    phase_label: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    properties: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class DefectCandidate(AutoTEMModel):
    """Detected defect candidate with location, geometry, and confidence."""

    defect_id: str = ""
    defect_type: DefectType = "unknown"
    location: tuple[float, float] | None = None
    mask: ArrayReference | None = None
    line: ArrayReference | None = None
    burgers_vector: tuple[Measurement, Measurement] | None = None
    plane: tuple[int, int, int] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[OutputArtifact] = Field(default_factory=list)
    properties: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class MicrostructureResult(VersionedAutoTEMModel):
    """Grains, phases, boundaries, twins, and defect statistics."""

    features: list[MicrostructureFeature] = Field(default_factory=list)
    defects: list[DefectCandidate] = Field(default_factory=list)
    phase_masks: list[MicrostructureFeature] = Field(default_factory=list)
    orientation_map: ArrayReference | None = None
    grain_map: ArrayReference | None = None
    statistics: list[Measurement] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ZContrastResult(VersionedAutoTEMModel):
    """Relative or calibrated Z-contrast output linked to atom columns."""

    atom_intensities: list[Measurement] = Field(default_factory=list)
    normalized_intensities: list[Measurement] = Field(default_factory=list)
    relative_z: list[Measurement] = Field(default_factory=list)
    sublattice_statistics: list[Measurement] = Field(default_factory=list)
    intensity_map: ArrayReference | None = None
    background: ArrayReference | None = None
    calibration_curve: JsonDict = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class SpectroscopySignal(AutoTEMModel):
    """Reference to an EDS/EELS spectrum or spectral image."""

    signal_id: str = ""
    kind: SpectroscopyKind = "unknown"
    data: ArrayReference | None = None
    energy_axis: ArrayReference | None = None
    energy_unit: str = "eV"
    metadata: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ElementMap(AutoTEMModel):
    """Elemental or chemical-state map derived from spectroscopy."""

    element: str
    edge: str | None = None
    map: ArrayReference
    quantification: Measurement | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class SpectroscopyResult(VersionedAutoTEMModel):
    """EDS/EELS maps, spectra, quantification, and structure-linked chemistry."""

    signals: list[SpectroscopySignal] = Field(default_factory=list)
    element_maps: list[ElementMap] = Field(default_factory=list)
    line_profiles: list[OutputArtifact] = Field(default_factory=list)
    quantification_table: list[JsonDict] = Field(default_factory=list)
    atom_links: list[JsonDict] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class FourDSTEMResult(VersionedAutoTEMModel):
    """4D-STEM outputs including Bragg disks, orientation, strain, and virtual images."""

    datacube: ArrayReference | None = None
    bragg_disk_table: list[JsonDict] = Field(default_factory=list)
    virtual_bf: ArrayReference | None = None
    virtual_df: ArrayReference | None = None
    orientation_map: ArrayReference | None = None
    strain: StrainResult | None = None
    phase_map: ArrayReference | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class SimulatedPattern(AutoTEMModel):
    """Single simulated diffraction pattern or image."""

    kind: SimulationKind
    data: ArrayReference | None = None
    structure_id: str = ""
    zone_axis: tuple[int, int, int] | None = None
    microscope_parameters: JsonDict = Field(default_factory=dict)
    simulation_parameters: JsonDict = Field(default_factory=dict)
    score: Measurement | None = None
    warnings: list[str] = Field(default_factory=list)


class SimulationResult(VersionedAutoTEMModel):
    """Simulation outputs and comparisons against experimental results."""

    patterns: list[SimulatedPattern] = Field(default_factory=list)
    matched_phase_candidates: list[PhaseCandidate] = Field(default_factory=list)
    comparison_metrics: list[Measurement] = Field(default_factory=list)
    overlays: list[OutputArtifact] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class AIModelInfo(AutoTEMModel):
    """Model registry metadata for AI-assisted analysis."""

    name: str
    version: str = ""
    task: str = ""
    uri: str | None = None
    training_data: str = ""
    metrics: list[Measurement] = Field(default_factory=list)
    license: str = ""
    warnings: list[str] = Field(default_factory=list)


class AIInferenceResult(VersionedAutoTEMModel):
    """AI model predictions, uncertainty, and low-confidence regions."""

    model: AIModelInfo
    predictions: list[OutputArtifact] = Field(default_factory=list)
    masks: list[ArrayReference] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    uncertainty: ArrayReference | None = None
    low_confidence_regions: list[ArrayReference] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ValidationMetric(AutoTEMModel):
    """Single validation or benchmark metric."""

    name: str
    value: Measurement
    target: Measurement | None = None
    direction: MetricDirection = "informational"
    passed: bool | None = None
    details: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class BenchmarkCase(AutoTEMModel):
    """Benchmark input, reference outputs, and produced artifacts."""

    case_id: str
    dataset: TEMDataset | None = None
    workflow: WorkflowConfig | None = None
    reference: JsonDict = Field(default_factory=dict)
    artifacts: list[OutputArtifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BenchmarkResult(VersionedAutoTEMModel):
    """Validation and benchmark summary for one or more cases."""

    cases: list[BenchmarkCase] = Field(default_factory=list)
    metrics: list[ValidationMetric] = Field(default_factory=list)
    summary: str = ""
    passed: bool | None = None
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ReportFigure(AutoTEMModel):
    """A generated figure included in an analysis report."""

    label: str
    uri: str
    caption: str = ""
    figure_type: str = ""
    provenance: JsonDict = Field(default_factory=dict)


class ReportTable(AutoTEMModel):
    """A generated table included in an analysis report."""

    label: str
    uri: str | None = None
    rows: list[JsonDict] = Field(default_factory=list)
    table_type: str = ""
    provenance: JsonDict = Field(default_factory=dict)


class AnalysisReport(VersionedAutoTEMModel):
    """Top-level report envelope for saved analysis outputs."""

    summary: str = ""
    workflow: WorkflowConfig | None = None
    dataset: TEMDataset | None = None
    quality: QualityAssessment | None = None
    calibration: CalibrationResult | None = None
    preprocessing: PreprocessingResult | None = None
    diffraction: DiffractionResult | None = None
    atom_lattice: AtomLattice | None = None
    strain: StrainResult | None = None
    microstructure: MicrostructureResult | None = None
    z_contrast: ZContrastResult | None = None
    spectroscopy: SpectroscopyResult | None = None
    fourdstem: FourDSTEMResult | None = None
    simulation: SimulationResult | None = None
    ai_inference: AIInferenceResult | None = None
    benchmark: BenchmarkResult | None = None
    figures: list[ReportFigure] = Field(default_factory=list)
    tables: list[ReportTable] = Field(default_factory=list)
    artifacts: list[OutputArtifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: JsonDict = Field(default_factory=dict)
