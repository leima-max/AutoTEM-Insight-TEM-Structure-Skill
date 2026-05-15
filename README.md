# AutoTEM-Insight / Auto TEM Skill

AutoTEM-Insight is a planned full-pipeline TEM data-analysis skill for converting raw TEM/HRTEM/STEM/4D-STEM data into reproducible structural parameters, diffraction indexing, atom coordinates, strain maps, defect statistics, phase candidates, simulations, and reports.

See [AGENTS.md](AGENTS.md) for the project-level agent guide, architecture expectations, MVP priorities, validation targets, and development standards.

## Initial MVP

The first reliable workflow should support a single HRTEM or HAADF-STEM image and produce:

- image quality assessment
- pixel calibration
- FFT image and diffraction peak table
- d-spacing and g-vector measurements
- hkl / phase candidates
- atom coordinates and atom-distance statistics
- preliminary strain map
- automatic JSON/CSV/HTML report outputs

## A2 Baseline Usage

Run the workflow from CLI:

```bash
autotem analyze examples/sample.npy --workflow configs/default.yaml --output outputs/sample
```

Override output formats/title from CLI when needed:

```bash
autotem analyze examples/sample.npy --output outputs/sample --formats json,csv,html,svg,pdf --report-title "A3 custom report"
```

Control A3 HTML sections from CLI:

```bash
autotem analyze examples/sample.npy --output outputs/sample --formats json,html --report-sections overview,workflow_details,interface_checks --open-step-sections
```

Control PDF structure from CLI:

```bash
autotem analyze examples/sample.npy --output outputs/sample --formats json,pdf --pdf-sections cover,overview,figures --pdf-max-figures 6 --pdf-cover-field sample_id=S01 --pdf-cover-field operator=member-a
```

Key A2/A3 outputs:

- `analysis_report.json`
- `workflow_used.yaml`
- `run_summary.json`
- `run.log`
- `tables/*.csv` (when `outputs.formats` contains `csv`)
- `report.html` (when `outputs.formats` contains `html`)
- `figures/*.png` (A3 baseline plots and overlays)
- `figures/*.svg` (when `svg` format is requested)
- `json/*.json` (A3 module snapshots and bundle summary)
  - includes A4 `validation_summary.json` / `benchmark_report.json` when `validate` / `benchmark` steps are enabled
- `report.pdf` (when `pdf` format is requested)
- `intermediates/*` (when `save_intermediates: true`)
  - large arrays are tiered to HDF5 when available, with preview + metadata sidecars

Runtime contract outputs:

- `run_summary.json`: stable schema summary for CI/benchmark ingestion (`workflow_name`, step statuses, warning counts, provenance fingerprints, paths).
- `run.log`: JSONL structured logs (one `RunLogEntry` per line with `timestamp_utc`, `level`, `message`, `context`).

See docs for contract details:

- `docs/runtime-contracts.md`
- `docs/a2-reserved-step-contracts.md`
- `docs/a3-output-conventions.md`
- `docs/a4-validation-benchmark.md`

## A2 Workflow Parameters

Example parameterized workflow block:

```yaml
workflow:
  name: tuned_a2
  steps: [load, assess_quality, calibrate, preprocess, analyze, report]
  parameters:
    calibration:
      method: reference
      diffraction_calibration_engine: internal  # internal | py4dstem | pyxem | auto
      pixel_size_nm: 0.05
      pixel_size_uncertainty_nm: 0.001
      voltage_kv: 300
      known_d_spacing: 0.213
      known_d_spacing_unit: nm
      known_d_uncertainty_nm: 0.002
      measured_radius_px: 46.5
      measured_radius_uncertainty_px: 0.25
      detector_pixel_size_um: 15.0
    preprocessing:
      recipe: b2_fft
      clip_percentile_low: 1.0
      clip_percentile_high: 99.0
      denoise_sigma: 1.0
      background_sigma: 12.0
    diffraction:
      top_k_peaks: 24
      min_peak_distance: 3.0
      center_exclusion_radius: 3.0
      indexing_mode: auto  # auto | baseline | lattice_fit | b4_mvp | b4_strict
      peak_threshold_strategy: auto
      peak_threshold_value: 0.995
      subpixel_refine: true
      subpixel_window_radius: 2
      subpixel_max_shift: 1.5
      max_hkl: 4
      d_spacing_tolerance_rel: 0.08
      angle_tolerance_deg: 12.0
      phase_top_n: 3
      candidate_phases: [Gold, Silicon]
      candidate_phase_params:
        - name: Silicon_like
          a: 0.543
          b: 0.543
          c: 0.543
          alpha: 90
          beta: 90
          gamma: 90
          space_group: Fd-3m
      # cif_candidates_dir: ./cif_candidates  # Used when indexing_mode=b4_strict
outputs:
  formats: [json, csv, html]
  save_intermediates: true
```

Validation is enforced during workflow loading; invalid ranges or unsupported formats fail fast with clear `ValueError` messages.

### B1 Minimal Reproducible Calibration

```bash
autotem analyze examples/sample.npy --workflow configs/hrtem_fft_phase.yaml --output outputs/hrtem_b1
```

When `known_d_spacing + measured_radius_px` are provided, the calibration step now propagates
uncertainty to:

- `calibration.diffraction_scale.uncertainty`
- downstream `diffraction.g_vectors[].magnitude.uncertainty`
- downstream `diffraction.d_spacings[].uncertainty`

Optional adapter switch:

- `diffraction_calibration_engine: internal` (default, no extra dependency)
- `diffraction_calibration_engine: py4dstem`
- `diffraction_calibration_engine: pyxem`
- `diffraction_calibration_engine: auto` (try py4dstem, then fallback)

If an optional adapter is unavailable, the workflow degrades gracefully to internal calibration and records a warning.

### A4 Uncertainty Metrics

A4 validation/benchmark reports now include d-spacing uncertainty quality metrics:

- `d_spacing_uncertainty_coverage`
- `d_spacing_uncertainty_mean_relative`
- `d_spacing_uncertainty_plausible_fraction`
- `d_spacing_uncertainty_outlier_count`

These are emitted automatically when diffraction d-spacing values are present.

## A4 Validation & Benchmark

Enable A4 steps directly in workflow:

```yaml
workflow:
  steps: [load, assess_quality, calibrate, preprocess, analyze, validate, benchmark, report]
  parameters:
    validation:
      acceptance_criteria:
        - metric: workflow_no_failed_steps
          op: ">="
          threshold: 1
    benchmark:
      case_id: hrtem_mvp
      reference:
        d_spacings_nm: [0.235, 0.204]
      acceptance_criteria:
        - metric: d_spacing_mean_relative_error
          op: "<="
          threshold: 0.02
```

A4 outputs (when enabled):

- `json/validation_summary.json`
- `json/benchmark_report.json`
- `analysis_report.json -> benchmark` snapshot

Multi-case A4 suite example:

- `configs/benchmarks/suite_smoke.yaml`
- `benchmarks/suite_public_minimal.yaml` (public minimal runnable package)
- programmatic runner:
  - `from autotem.validation import run_benchmark_suite_from_file`
- CLI:
  - `autotem benchmark --suite configs/benchmarks/suite_smoke.yaml --fail-on-failed`
  - `autotem benchmark --suite benchmarks/suite_public_minimal.yaml --output outputs/benchmark-public-minimal`
- Trend artifacts (auto-generated per suite output dir):
  - `benchmark_suite_history.jsonl`
  - `benchmark_suite_trend.json`
  - `benchmark_suite_trend.html`
  - includes case-level `regression_alerts` leaderboard for degraded cases
  - severity is configurable by suite `regression_weights`
  - modality-level defaults are configurable via `suite.regression_weight_templates[]`
  - each case can override weights via `cases[].regression_weights`
  - effective precedence is `suite default -> modality template -> case override`
- package doc:
  - `benchmarks/README.md`

Nightly CI benchmark:

- workflow: `.github/workflows/nightly-benchmark.yml`
- schedule: daily at `02:00 UTC` (manual trigger also supported)
- uploads artifacts:
  - `benchmark_suite_report.json/html`
  - `benchmark_suite_history.jsonl`
  - `benchmark_suite_trend.json/html`
  - `cases/**/benchmark_report.json`
- notification behavior:
  - always publishes `Top Regressions` to GitHub Job Summary
  - when regressions are detected, auto creates/updates issue:
    - title: `[Nightly Benchmark] Regressions detected (<suite_id>)`
  - when regressions drop to zero, the workflow auto closes that issue

## Local UI (Atom Detection C0)

Run a simple UI for atom-column detection (supports `dm/dm3/dm4/tif/tiff`):

```powershell
.\.venv\Scripts\python.exe -m streamlit run apps\atom_detection_ui.py
```

Then open the local URL shown by Streamlit (usually `http://localhost:8501`).

UI supports:

- realtime parameter update (toggle `Realtime update`)
- direct click add/remove on image canvas
- irregular polygon ROI selection
- CSV export of edited atom coordinates
- subpixel refinement (`gaussian` / `com`) with displacement summary
- optional lattice-constraint correction (can be disabled for defect/GPA workflows)

## Local UI (B2 Diffraction / FFT)

Run a parameter-tuning UI for B2 diffraction testing:

```powershell
.\.venv\Scripts\python.exe -m streamlit run apps\b2_diffraction_ui.py
```

UI supports:

- upload `npy/tif/tiff/png/jpg/dm3/dm4/mrc/h5` and other baseline reader formats
- built-in demo sample generation for quick smoke testing
- calibration controls (`pixel_size_nm`, `voltage_kv`)
- preprocess controls (recipe + optional percentile clipping)
- diffraction controls (`top_k_peaks`, `min_peak_distance`, `center_exclusion_radius`)
- phase candidate filtering and `phase_top_n`
- FFT peak overlay visualization and CSV export for peaks/phase candidates

## C0 Contract Alignment

`detect_atoms(...)` returns `AtomDetectionResult`, and you can convert it to the shared
`AtomLattice` contract used by core workflows:

```python
from autotem.atomic.atom_detection import detect_atoms

result = detect_atoms("example.dm4", mode="auto")
atom_lattice = result.to_atom_lattice(coordinate_unit="pixel")
```

## Dependency Notes

- Core install (includes C0 atom detection + DM/TIFF support):
  - `python -m pip install -e ".[dev]"`
- UI install (Streamlit tooling):
  - `python -m pip install -e ".[dev,ui]"`
- Optional GPL adapter install (py4DSTEM/pyxem calibration hooks):
  - `python -m pip install -e ".[gpl_adapters]"`

## Test Quickstart

- Run tests via project virtual environment (recommended on Windows):
  - `.\run-tests.cmd`
- PowerShell variant:
  - `.\run-tests.ps1`
- Run a single test file:
  - `.\run-tests.cmd tests\test_io.py`
- If `.venv` is missing:
  - `python -m venv .venv`
  - `.\.venv\Scripts\python.exe -m pip install -e ".[dev,io]"`
