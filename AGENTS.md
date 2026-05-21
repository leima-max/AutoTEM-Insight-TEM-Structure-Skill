# AutoTEM-Insight / TEM-Structure Skill Agent Guide

## Project Mission

Build a reusable, extensible TEM data-analysis skill that automates the workflow from raw TEM/HRTEM/STEM/4D-STEM inputs to structure parameters, phase candidates, lattice metrics, strain maps, defect statistics, atom coordinates, Z-contrast results, simulations, and reproducible reports.

Treat this as a scientific software ecosystem, not a single script:

- Use reliable in-house algorithms where precision or workflow control matters.
- Wrap mature open-source microscopy and crystallography libraries instead of reimplementing everything.
- Expose modular Python APIs, CLI workflows, YAML recipes, notebook/Web interaction, and agent-callable tools.
- Preserve provenance, parameters, uncertainty, confidence, intermediate outputs, and warnings for every analysis.

## Current Codex Operating Context

The current user is **Member A**:

- GitHub: `leima-max`
- Repository role: owner / admin
- Project role: platform architecture, data flow, and delivery-loop owner
- Default responsibility group: **A. 平台架构、数据流与交付闭环**

When the user refers to "my role", "my tasks", "Member A", or asks what they should work on, treat it as the `leima-max` / Member A scope unless they explicitly say otherwise.

Additional persistent rule from user (2026-05-13):

- Task ownership is determined by the leading letter of the task ID: `A*` -> Member A, `B*` -> Member B, `C*` -> Member C.
- If task ownership or module scope is unclear, check `configs/three-person-module-distribution.md` first before implementation.

Member A owns:

- project skeleton and package structure
- core data models and schemas
- Python API, CLI, YAML workflow, and orchestration layer
- report generation, export, visualization integration, and output directory conventions
- tests, CI, benchmark framework, validation metrics, and release readiness
- cross-module interface contracts and integration across Members B and C
- GitHub workflow hygiene, PR templates, issue templates, branch strategy, and project documentation
- OpenClaw/GitHub/Feishu coordination where it affects delivery and task tracking

Member A should not deeply own the primary scientific algorithms for Member B or Member C, but must define the interfaces they implement against, integrate their results, and verify their outputs are reproducible and reportable.

Current team mapping:

| Member | GitHub | Repository Role | Responsibility |
|---|---|---|---|
| A | `leima-max` | owner / admin | Platform architecture, data flow, workflow/API/CLI, reporting, validation, delivery |
| B | `wangzhe221` | collaborator / write | Input, calibration, FFT, diffraction, hkl, phase matching, simulation, 4D-STEM |
| C | `yswgityes` | collaborator / write | Atomic structure, GPA/strain, defects, Z-contrast, spectroscopy, AI |

Detailed planning references:

- `configs/three-person-module-distribution.md` for A/B/C module grouping and sequential task chains.
- `configs/team-responsibility-task-board.md` for the full task board and acceptance-oriented deliverables.
- `configs/openclaw-feishu-project-agent-identity.md` for OpenClaw/Feishu project-assistant identity and communication rules.
- `MEMORY.md` for long-term durable memory.
- `memory/YYYY-MM-DD.md` for daily running memory.

## Project Memory Protocol

Use project memory proactively. The project has two memory layers:

- `memory/YYYY-MM-DD.md`: daily running notes for important events, time points, decisions, corrections, lessons, and short-lived context.
- `MEMORY.md`: long-term valid project memory that should remain true across future sessions.

Write memory to files when:

- The user says "记住这个" or otherwise asks for something to be remembered.
- The user corrects the assistant or establishes a new preference.
- The team makes an important decision.
- A significant project time point, milestone, ownership change, deadline, or coordination rule appears.
- A new durable lesson emerges from an error or blocked workflow.

When writing memory:

- Keep entries concise and factual.
- Prefer exact dates such as `2026-05-11` over relative dates.
- Put detailed daily trail in `memory/YYYY-MM-DD.md`.
- Promote only durable summaries to `MEMORY.md`.
- Do not store secrets, tokens, private credentials, or sensitive experimental data in memory files.

## Core Scope

The skill should support these analysis layers:

| Layer | Inputs | Main Outputs |
|---|---|---|
| Image QA | TEM/HRTEM/STEM images and stacks | denoised/enhanced images, quality score, pixel calibration |
| Reciprocal space | FFT, SAED, NBED, 4D-STEM diffraction | peaks, g-vectors, d-spacing, hkl, zone axis, phase candidates |
| Microstructure | grains, interfaces, twins, precipitates, defects | masks, size statistics, orientations, defect densities |
| Strain | HRTEM/GPA, atom-column displacement, 4D-STEM strain | displacement fields, strain tensors, rotation maps, profiles |
| Atomic structure | HAADF/ABF/STEM atomic images | atom coordinates, sublattice labels, local lattice, displacement |
| Spectroscopy/composition | EDS/EELS/HAADF Z contrast | element maps, valence proxies, atomic-column intensity |
| Simulation/inversion | CIF/model structures | simulated diffraction, simulated STEM, matching scores |
| Reporting | all results | JSON, HDF5/Zarr, CSV, figures, HTML/PDF reports |

## Recommended Technical Ecosystem

Prefer adapters around these libraries where appropriate:

- `HyperSpy` for multidimensional microscopy data structures and interactive analysis.
- `pyxem` for electron diffraction and pixelated detector data.
- `py4DSTEM` for 4D-STEM processing, Bragg disk detection, strain/orientation workflows.
- `LiberTEM` for large-scale binary microscopy data and distributed/high-throughput processing.
- `Atomap` for high-quality atom-resolved STEM column detection and 2D Gaussian refinement.
- `AtomAI` for deep-learning microscopy/spectroscopy workflows and atom/defect detection.
- `abTEM`, `diffsims`, `pymatgen`, and `orix` for diffraction/STEM simulation, crystallography, orientation, and symmetry.
- Local CIF databases, ICSD/PDF-card integrations, and Materials Project API for phase candidates when available.

Before adding any dependency, check license compatibility. Keep GPL-risk components isolated behind optional adapters when commercial deployment is possible.

## Development Priorities

Implement in this order unless the user explicitly asks otherwise:

1. Foundation data models and project skeleton.
2. Image reading, metadata parsing, quality scoring, and pixel/diffraction calibration.
3. Preprocessing, denoising, FFT calculation, peak detection, d-spacing, and g-vector measurement.
4. Diffraction indexing, phase matching, zone-axis inference, and simulated diffraction overlays.
5. Atom-column detection, subpixel refinement, sublattice classification, atom spacing, and local displacement.
6. GPA strain and atom-based strain cross-validation.
7. Grain, twin, boundary, phase, and defect segmentation/statistics.
8. Z-contrast, multislice/simulation correction, 4D-STEM, EDS, and EELS expansion.
9. CLI, Python API, YAML workflows, Web/Jupyter UI, automatic reports, benchmarks, and packaging.

For Member A specifically, prioritize this delivery order:

1. A0 Engineering skeleton: package structure, `pyproject.toml`, branch/PR/Issue conventions, minimal CI.
2. A1 Core data models: `TEMDataset`, `DiffractionResult`, `AtomLattice`, `StrainResult`, `AnalysisReport`, and `Measurement` schema.
3. A2 Workflow/API/CLI baseline: `TEMAnalysisSkill`, pipeline orchestration, YAML workflow schema, command-line entrypoint.
4. A3 Reporting/export/visualization integration: JSON, CSV, figures, overlays, HTML reports, provenance, confidence, warnings.
5. A4 Validation and benchmark framework: metric interfaces, standard sample benchmark hooks, integration tests, reportable acceptance criteria.

Member A must define stable contracts for Member B and C outputs before deep integration work:

- Member B returns `TEMDataset` and `DiffractionResult`, plus FFT images, peak tables, g-vector/d-spacing tables, phase candidates, and diffraction overlays.
- Member C returns `AtomLattice` and `StrainResult`, plus atom coordinate tables, atom overlays, strain maps, sublattice/defect/Z-contrast tables when available.

For the MVP, prioritize a reliable closed loop for a single HRTEM or HAADF-STEM image:

- image quality assessment
- pixel calibration
- FFT image
- diffraction peak table
- d-spacing table
- hkl/phase candidates
- atom coordinates
- atom-distance statistics
- preliminary strain map
- automatic report

## Architecture Expectations

Use a modular Python package layout. Keep each analysis family independently testable and composable into pipelines.

Suggested package shape:

```text
autotem_skill/
  pyproject.toml
  configs/
  autotem/
    io/
    calibration/
    preprocessing/
    diffraction/
    microstructure/
    strain/
    atomic/
    fourdstem/
    spectroscopy/
    simulation/
    ai/
    workflows/
    visualization/
    reporting/
    validation/
  tests/
  notebooks/
  examples/
```

Expose the main workflow through a facade similar to:

```python
class TEMAnalysisSkill:
    def load_data(self, path, metadata=None): ...
    def assess_quality(self, dataset): ...
    def calibrate(self, dataset, method="auto", reference=None): ...
    def preprocess(self, dataset, recipe="auto"): ...
    def analyze_fft(self, image, candidate_phases=None): ...
    def index_diffraction(self, pattern, candidates=None): ...
    def segment_grains(self, image): ...
    def detect_twins(self, image, orientation_map=None): ...
    def run_gpa(self, image, reference_region=None, g_vectors=None): ...
    def detect_atoms(self, image, mode="auto"): ...
    def classify_sublattices(self, atom_lattice, candidate_structure=None): ...
    def infer_zone_axis(self, atom_lattice, candidate_phases=None): ...
    def quantify_z_contrast(self, image, atom_lattice, microscope_params=None): ...
    def simulate_diffraction(self, structure, zone_axis, params): ...
    def simulate_stem_image(self, structure, microscope_params): ...
    def fuse_results(self, results): ...
    def generate_report(self, results, format="html"): ...
```

## Data Model Requirements

Unify module input and output through structured objects. Do not pass loose arrays and dictionaries across module boundaries when a clear data class is better.

Core objects should include:

- `TEMDataset`: data array, modality, pixel size/unit, metadata, provenance.
- `DiffractionResult`: peaks, center, g-vectors, d-spacings, indexed spots, phase candidates, confidence.
- `AtomLattice`: coordinates, intensities, sublattice labels, lattice vectors, zone axis, confidence.
- `StrainResult`: displacement maps, strain tensor maps, rotation map, uncertainty.
- `AnalysisReport`: summary, figures, tables, warnings, provenance.

Every reported structure parameter must include:

```json
{
  "value": "...",
  "unit": "...",
  "uncertainty": "...",
  "method": "...",
  "parameters": {},
  "confidence": 0.0,
  "provenance": {},
  "warnings": []
}
```

Use HDF5/Zarr for large arrays and intermediate results, JSON for structured metrics, CSV for tables, PNG/SVG for figures, HTML/PDF for reports, CIF for crystal structures, and YAML for workflow recipes.

## Workflow Design

Design workflows around this data flow:

```text
raw data
  -> format detection and metadata parsing
  -> quality assessment and calibration
  -> modality detection
  -> preprocessing and enhancement
  -> FFT/diffraction, microstructure, GPA/strain, atomic, Z-contrast, 4D-STEM, or spectroscopy branches
  -> simulation and database matching
  -> result fusion and confidence scoring
  -> structure parameters, visualization, and report export
```

YAML workflows should support task enablement, candidate phases, calibration references, preprocessing recipes, output formats, and saving intermediate artifacts.

## Module Guidance

For FFT/SAED/NBED analysis:

- Detect center/direct beam robustly.
- Detect Bragg spots using classical methods first, then optional ML detectors.
- Fit peak positions at subpixel precision where possible.
- Compute g-vectors, d-spacing, angular relationships, and uncertainties.
- Match against CIF/database candidates through simulated diffraction.
- Return top-N phase candidates with matched spot counts, mean d-error, angle error, zone axis, and confidence.

For GPA:

- Select two strong non-collinear g-vectors automatically, with user override support.
- Support automatic and manual reference-region selection.
- Save Bragg masks, phase maps, displacement maps, strain tensors, rotation maps, line profiles, uncertainty, and warnings.
- Cross-check GPA strain with atom-column displacement where both are available.

For atomic structure:

- Use LoG/DoG plus `peak_local_max` for baseline atom-column detection.
- Use Atomap Gaussian refinement for high-quality atom-resolved STEM.
- Use AtomAI or other ML segmentation when images are complex or labels/models are available.
- Classify sublattices with geometry, intensity, local environment, and candidate CIF constraints.
- Compute atom spacing, displacement vectors, local lattice parameters, atomic strain, and defect candidates.

For microstructure and defects:

- Combine local FFT orientation, image texture, segmentation models, and crystallographic constraints.
- Avoid trusting pure image segmentation without physical checks.
- Output masks, boundary curves, grain statistics, twin planes, misorientation, defect density, and confidence.

For Z-contrast:

- Treat quantitative composition carefully.
- Separate relative Z-contrast outputs from semi-quantitative or quantitative composition estimates.
- Warn when thickness, tilt, detector angle, beam current, or simulation calibration is missing.

## Validation Targets

Use realistic benchmark data and expert annotations. Suggested targets:

- FFT peak recall/precision: > 90% on standard images.
- d-spacing relative error: < 1-2% on standard samples.
- hkl indexing top-3 accuracy: > 85% when candidate phases are constrained.
- phase recognition top-3 accuracy: > 80% on multi-candidate systems.
- atom-coordinate RMSE: < 5-10 pm on high-quality STEM.
- atom-distance relative error: < 1-2%.
- grain segmentation Dice/IoU: > 0.80.
- twin detection precision: > 80% against expert labels.
- common 4k image analysis: < 30-60 seconds for routine pipelines, measured separately on CPU/GPU.

Report these metrics with dataset assumptions. Do not present them as universal guarantees.

## Risk Handling

Always surface warnings for:

- missing or inconsistent metadata
- unknown pixel calibration or camera length
- low SNR, drift, contamination, astigmatism, undersampling, or defocus issues
- ambiguous phase candidates
- unreliable Z-contrast due to unknown thickness, tilt, detector geometry, or light-element invisibility
- questionable GPA reference regions
- AI model domain shift or low-confidence predictions

When automation is uncertain, preserve an expert-in-the-loop path for calibration, ROI selection, reference-region selection, phase candidate filtering, and result approval.

## Coding Standards

- Prefer small, tested modules over monolithic notebooks.
- Keep algorithms deterministic where possible and record random seeds when randomness is used.
- Keep all outputs reproducible from saved workflow config, input paths, versions, and parameters.
- Add tests for numerical correctness, schema stability, edge cases, and benchmark workflows.
- Preserve raw data and intermediate artifacts unless the user explicitly requests a lightweight run.
- Keep documentation close to executable examples, but avoid bloating the core skill instructions.
