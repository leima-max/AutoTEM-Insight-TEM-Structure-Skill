# A3 Output Conventions

This document defines the baseline A3 output directory structure and machine-readable manifest.

## Output Root

Each run writes into one output root (CLI `--output` or API `output_dir`):

- `analysis_report.json`
- `run_summary.json`
- `run.log`
- `workflow_used.yaml`
- `report.html` (when `html` is requested)

## Standard Subdirectories

Auto-created for A3:

- `figures/`: PNG plots and overlays (`input_image.png`, `fft_magnitude.png`, etc.)
  including B5 `simulation_diffraction_overlay.png` when simulated and measured
  reciprocal-space spots are available.
- `tables/`: CSV tables (`diffraction_peaks.csv`, `d_spacings.csv`,
  `indexed_spots.csv`, etc.)
- `json/`: JSON module snapshots and manifests
- `intermediates/`: reserved for future intermediate arrays and debug artifacts

`save_intermediates: true` writes baseline arrays such as:

- `intermediates/input_image.npy`
- `intermediates/preprocessed_image.npy`
- `intermediates/fft_magnitude.npy`
- `intermediates/diffraction_peaks.json`
- `intermediates/workflow_steps.json`

For larger arrays, A3 uses layered storage:

- preferred tier: HDF5 dataset (`intermediates/large_arrays.h5::/dataset_name`)
- fallback tier: `.npy` (when HDF5 adapter is unavailable)
- both tiers include preview (`*_preview.png`) and metadata (`*_metadata.json`)

## JSON Bundle and Manifest

Inside `json/`:

- `report_bundle_summary.json`: high-level summary of report modules and counts
- `output_manifest.json`: canonical artifact inventory for downstream automation
- `interface_completeness_bc.json`: B/C interface completeness contract summary and checks
- module snapshots: `workflow.json`, `dataset.json`, `diffraction.json`, `atom_lattice.json`, etc. (when present)

`output_manifest.json` includes runtime contract artifacts as well, including
`analysis_report.json`, `run_summary.json`, `run.log`, and `workflow_used.yaml`.

Inside `tables/`, A3 also writes:

- `interface_completeness_bc.csv`: flat contract table for project board and acceptance automation
- `phase_candidates.csv`: B4 phase-candidate ranking, lattice parameters,
  crystal source, atom-site count, Materials Project ranking metadata when
  available, matched spot counts, and d/angle errors.
- `simulation_phase_matches.csv`: B5 simulation-vs-measurement closed-loop summary,
  including engine, adapter status, phase-matching status, selected crystal
  source, matched spot count, mean d-error, mean angle-error, and rotation
  evidence plus the selected zone-axis metadata when a B5 zone-axis sweep is run.
- `simulated_spots.csv`: reciprocal-space spots extracted from simulated
  diffraction patterns, including zone-axis/rotation metadata and hkl labels
  when the simulator provides them.

Both `output_manifest.json` and `report_bundle_summary.json` are validated against Pydantic
contract models and have exportable JSON Schemas.

## `output_manifest.json` Keys

Minimum fields:

- `schema_version`
- `report_summary`
- `requested_formats`
- `output_root`
- `directories`
- `counts`
- `artifact_kind_counts`
- `artifacts`
- `provenance`

## Integration Notes (A/B/C)

- Member B outputs should be reflected in diffraction tables and overlays in `tables/` and `figures/`.
- Member C outputs should be reflected in atom/strain overlays and tables.
- A3 keeps a stable artifact index so A4 benchmark and CI checks can locate outputs without hard-coded paths.

## Stage 3 HTML Behavior

- HTML report supports configurable sections via API `report_sections_override` or CLI `--report-sections`.
- Available section names:
  - `overview`
  - `workflow_steps`
  - `workflow_details`
  - `interface_checks`
  - `simulation`
  - `figures`
  - `tables`
  - `warnings`
  - `provenance`
- Workflow detail sections render as collapsible `details` blocks.
- Interface completeness includes Member B/C contract checks and marks each item as `pass` or `missing`.
- The `simulation` HTML section includes a closed-loop summary, a candidate
  comparison table that combines B4/Materials Project ranking metadata with
  diffsims match scores, inferred zone-axis metadata, and the final matched
  phase candidates.

## Quality Gates

A3 appends `BLOCKING:` warnings and `provenance.quality_gates` when required contract gates fail
(for example failed workflow steps or missing mandatory outputs for requested stages).

## Extended Formats

- `svg`: export SVG companions for figure outputs.
- `pdf`: export `report.pdf` summary with embedded figure pages.

PDF report supports stage-4 style template options:

- `sections`: choose from `cover`, `overview`, `workflow_steps`, `interface_checks`, `warnings`, `provenance`, `figures`
- `max_figure_pages`: cap figure pages in PDF
- `cover_fields`: key-value fields shown on PDF cover

These can be supplied via workflow `workflow.parameters.reporting.pdf` or CLI flags.
