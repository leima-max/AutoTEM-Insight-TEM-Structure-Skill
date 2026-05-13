# A4 Validation & Benchmark Contracts

This document defines the A4 machine-readable outputs generated when workflow steps include
`validate` and/or `benchmark`.

## `json/validation_summary.json`

Produced by `validate` step.

Top-level fields:

- `schema_version`: report schema version.
- `metric_count`: number of emitted validation metrics.
- `acceptance`: acceptance evaluation summary (`evaluated`, `passed`, `failed`, `all_passed`).
- `metrics`: serialized `ValidationMetric[]`.

## `json/benchmark_report.json`

Produced by `benchmark` step.

Top-level fields:

- `schema_version`: report schema version.
- `case_id`: benchmark case identifier.
- `summary`: human-readable acceptance summary.
- `passed`: benchmark pass/fail boolean.
- `acceptance`: acceptance evaluation summary (`evaluated`, `passed`, `failed`, `all_passed`).
- `metric_count`: number of benchmark metrics.
- `metrics`: serialized `ValidationMetric[]` with acceptance targets.

## Workflow Parameter Contract

Use these parameter blocks under `workflow.parameters`:

```yaml
validation:
  acceptance_criteria:
    - metric: workflow_no_failed_steps
      op: ">="
      threshold: 1

benchmark:
  case_id: hrtem_mvp
  reference:
    d_spacings_nm: [0.235, 0.204]
    atom_coordinates: [[10.2, 14.6], [18.0, 14.4]]
  acceptance_criteria:
    - metric: d_spacing_mean_relative_error
      op: "<="
      threshold: 0.02
```

Supported `op` values: `>=`, `<=`, `>`, `<`, `==`.
