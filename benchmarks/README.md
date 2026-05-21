# Public Minimal Benchmark Package (A4)

This directory provides a runnable, repository-local A4 benchmark package.

## Contents

- `suite_public_minimal.yaml`
  - one public minimal case after task cleanup
  - C-side atom/strain metrics: `atom_coordinate / atom_distance / strain_xx`
- `cases/c_atomic_strain_public/`
  - `analysis_report.json` report fixture used by benchmark runner
  - `case.yaml` single-case spec
  - `reference.json` transparent reference payload

## Run

```bash
autotem benchmark --suite benchmarks/suite_public_minimal.yaml --output outputs/benchmark-public-minimal
```
