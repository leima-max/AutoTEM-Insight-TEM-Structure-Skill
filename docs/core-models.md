# Core Model Contracts

The A1 model layer defines the stable data contracts between the platform owner and the
algorithm modules.

## Versioning

Top-level cross-module objects include `schema_version`. The current version is `0.1.0`.

Versioned objects:

- `WorkflowConfig`
- `TEMDataset`
- `QualityAssessment`
- `CalibrationResult`
- `PreprocessingResult`
- `DiffractionResult`
- `AtomLattice`
- `StrainResult`
- `MicrostructureResult`
- `ZContrastResult`
- `SpectroscopyResult`
- `FourDSTEMResult`
- `SimulationResult`
- `AIInferenceResult`
- `BenchmarkResult`
- `AnalysisReport`

`Measurement`, `ArrayReference`, workflow steps, output artifacts, peaks, atom sites, tables,
and figures are reusable nested objects. They are validated by Pydantic and reject unknown fields.

## Array Data

Large microscopy arrays should not be embedded directly in JSON. Use `ArrayReference`:

- `shape`: array shape
- `dtype`: array dtype
- `uri`: external path such as `.npy`, `.zarr`, `.h5`, `.png`, or `.tif`
- `inline`: only for tiny fixtures and tests
- `description`: short human-readable purpose

## Member B Contract

Member B should return:

- `TEMDataset` from readers and calibration
- `QualityAssessment` for image quality and metadata completeness
- `CalibrationResult` for pixel and diffraction calibration
- `PreprocessingResult` for filtered images and preprocessing provenance
- `DiffractionResult` from FFT, peak detection, g-vector, d-spacing, indexing, and phase matching

## Member C Contract

Member C should return:

- `AtomLattice` from atom detection and refinement
- `StrainResult` from GPA or atom-based strain
- `MicrostructureResult` for grains, twins, boundaries, phases, and defects
- `ZContrastResult` for relative atom-column intensity and Z-contrast outputs
- `SpectroscopyResult` when atom or phase outputs are linked to EDS/EELS

## Extended Contracts

Later modules should use these reserved top-level contracts:

- `FourDSTEMResult`: 4D-STEM datacubes, Bragg disks, virtual images, orientation maps, and strain.
- `SimulationResult`: simulated diffraction/STEM/TEM outputs and comparison metrics.
- `AIInferenceResult`: model registry metadata, predictions, masks, uncertainty, and low-confidence regions.
- `BenchmarkResult`: validation cases, metrics, pass/fail summaries, and benchmark artifacts.

## Reporting Contract

Member A integrates module outputs into `AnalysisReport`, plus figure and table references.
Every reported physical quantity should use `Measurement` so uncertainty, method, confidence,
warnings, and provenance remain attached to the value.

Workflow recipes should be represented as `WorkflowConfig`, while saved files should be listed
as `OutputArtifact` records so JSON, CSV, HTML, figures, arrays, and logs can be discovered by
downstream tools.
