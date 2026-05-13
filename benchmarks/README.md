# Public Minimal Benchmark Package (A4)

This directory provides a runnable, repository-local A4 benchmark package that goes beyond
schema-only examples.

## Contents

- `suite_public_minimal.yaml`
  - two public mini cases
  - B-side diffraction metrics: `d-spacing / hkl / phase`
  - C-side atom/strain metrics: `atom_coordinate / atom_distance / strain_xx`
- `cases/b_diffraction_public/`
  - `input.npy` synthetic HRTEM-like toy data
  - `analysis_report.json` report fixture used by benchmark runner
  - `case.yaml` single-case spec
  - `reference.json` transparent reference payload
- `cases/c_atomic_strain_public/`
  - `input.npy` synthetic HAADF-like toy data
  - `analysis_report.json` report fixture used by benchmark runner
  - `case.yaml` single-case spec
  - `reference.json` transparent reference payload

## Run

```bash
autotem benchmark --suite benchmarks/suite_public_minimal.yaml --output outputs/benchmark-public-minimal
```

Expected outputs:

- `benchmark_suite_report.json`
- `benchmark_suite_report.html`
- `benchmark_suite_history.jsonl`
- `benchmark_suite_trend.json`
- `benchmark_suite_trend.html`
- `cases/<case_id>/benchmark_report.json`

## Notes

- The sample arrays are deterministic synthetic data, intended for public CI/demo and contract checks.
- They are intentionally small to keep benchmark runtime low.
- For production-quality scientific validation, replace references with instrument-ground-truth data.
