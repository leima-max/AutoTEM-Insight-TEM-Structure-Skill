# A2 Reserved Step Contracts (B/C Integration)

These contracts define expected I/O for A2 steps that are intentionally reserved and not yet implemented in the A2 baseline runtime.

The pipeline marks these steps with `status=warning` and includes expected inputs/outputs in step records to support integration planning.

| Step | Owner | Expected Inputs | Expected Outputs |
|---|---|---|---|
| `detect_atoms` | Member C | `TEMDataset`, `PreprocessingResult`, `CalibrationResult` | `AtomLattice`, `atom_coordinates.csv`, `atom_overlay.png` |
| `run_gpa` | Member C | `TEMDataset`, `PreprocessingResult`, `DiffractionResult` | `StrainResult`, `strain_maps`, `gpa_profiles.csv` |
| `atom_strain` | Member C | `AtomLattice`, `TEMDataset` | `StrainResult`, `atomic_displacement.csv` |
| `classify_sublattices` | Member C | `AtomLattice`, `DiffractionResult` | `AtomLattice`, `sublattice_table.csv` |
| `segment_microstructure` | Member C | `TEMDataset`, `PreprocessingResult`, `DiffractionResult` | `MicrostructureResult`, `grain_masks` |
| `detect_defects` | Member C | `TEMDataset`, `MicrostructureResult`, `AtomLattice` | `MicrostructureResult`, `defect_table.csv` |
| `quantify_z_contrast` | Member C | `TEMDataset`, `AtomLattice`, `CalibrationResult` | `ZContrastResult`, `z_contrast_table.csv` |
| `analyze_spectroscopy` | Member C | `TEMDataset`, `CalibrationResult` | `SpectroscopyResult`, `element_maps` |
| `analyze_4dstem` | Member B | `TEMDataset`, `CalibrationResult` | `FourDSTEMResult`, `bragg_disk_table.csv` |
| `ai_inference` | Member C | `TEMDataset`, `PreprocessingResult` | `AIInferenceResult`, `ai_masks` |
| `validate` | Member A | `AnalysisReport` | `ValidationMetric[]`, `validation_summary.json` |
| `benchmark` | Member A | `AnalysisReport`, `benchmark_config` | `BenchmarkResult`, `benchmark_report.json` |

Notes:

- Ownership follows the team mapping rule (`A* -> A`, `B* -> B`, `C* -> C`), and this table is the explicit runtime contract for A2 placeholders.
- Future implementation should keep these names stable or provide migration notes in docs and release notes.
