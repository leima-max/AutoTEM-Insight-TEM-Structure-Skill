# Public Minimal Benchmark Package (A4)

This directory provides a runnable, repository-local A4 benchmark package.

## Contents

- `suite_public_minimal.yaml`
  - three public minimal cases after task cleanup
  - B-side diffraction metrics: `d_spacing / hkl / phase`
  - B5 simulation metrics: `diffsims / simulated_spots / matched_phase`
  - C-side atom/strain metrics: `atom_coordinate / atom_distance / strain_xx`
- `cases/b_diffraction_public/`
  - `analysis_report.json` report fixture used by benchmark runner
  - `case.yaml` single-case spec
  - `reference.json` transparent reference payload
- `cases/b5_diffsims_phase_matching_public/`
  - `analysis_report.json` report fixture used by benchmark runner
  - `case.yaml` single-case spec
  - `reference.json` transparent reference payload
- `cases/c_atomic_strain_public/`
  - `analysis_report.json` report fixture used by benchmark runner
  - `case.yaml` single-case spec
  - `reference.json` transparent reference payload

## Run

```bash
autotem benchmark --suite benchmarks/suite_public_minimal.yaml --output outputs/benchmark-public-minimal
```
