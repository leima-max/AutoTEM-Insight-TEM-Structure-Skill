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

Key A2 outputs:

- `analysis_report.json`
- `workflow_used.yaml`
- `run_summary.json`
- `run.log`
- `tables/*.csv` (when `outputs.formats` contains `csv`)
- `report.html` (when `outputs.formats` contains `html`)

Runtime contract outputs:

- `run_summary.json`: stable schema summary for CI/benchmark ingestion (`workflow_name`, step statuses, warning counts, provenance fingerprints, paths).
- `run.log`: JSONL structured logs (one `RunLogEntry` per line with `timestamp_utc`, `level`, `message`, `context`).

See docs for contract details:

- `docs/runtime-contracts.md`
- `docs/a2-reserved-step-contracts.md`

## A2 Workflow Parameters

Example parameterized workflow block:

```yaml
workflow:
  name: tuned_a2
  steps: [load, assess_quality, calibrate, preprocess, analyze, report]
  parameters:
    calibration:
      method: auto
      pixel_size_nm: 0.05
      voltage_kv: 300
    preprocessing:
      recipe: min_max
      clip_percentile_low: 1.0
      clip_percentile_high: 99.0
    diffraction:
      top_k_peaks: 24
      min_peak_distance: 3.0
      center_exclusion_radius: 3.0
      phase_top_n: 3
      candidate_phases: [Gold, Silicon]
outputs:
  formats: [json, csv, html]
  save_intermediates: true
```

Validation is enforced during workflow loading; invalid ranges or unsupported formats fail fast with clear `ValueError` messages.

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
