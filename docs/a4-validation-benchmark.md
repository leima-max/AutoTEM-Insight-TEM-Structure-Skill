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

## Benchmark Case/Suite Contract

Case and suite YAML schema models:

- `BenchmarkCaseSpec`
- `BenchmarkSuiteSpec`
- schema version: `2026-05-13.a4.case-schema.v1`

Recommended benchmark directory shape:

```text
benchmarks/
  suite.yaml
  cases/
    case_a/
      case.yaml
      input.npy
      reference.json
```

Repository public minimal package:

- `benchmarks/suite_public_minimal.yaml`
- `benchmarks/cases/b_diffraction_public/`
- `benchmarks/cases/c_atomic_strain_public/`
- `benchmarks/README.md`

Suite YAML shape:

```yaml
suite:
  suite_id: mvp_suite
  default_output_dir: outputs/benchmark-suite
  regression_weights:
    status_flip_pass_to_fail: 5
    acceptance_failed_increase: 2
    warning_count_increase: 1
    acceptance_pass_rate_drop: 2
  regression_weight_templates:
    - modality: HRTEM
      weights:
        status_flip_pass_to_fail: 8
        acceptance_failed_increase: 3
cases:
  - case_id: hrtem_case
    report_path: outputs/hrtem_case/analysis_report.json
    regression_weights:
      status_flip_pass_to_fail: 8
      warning_count_increase: 0
    acceptance_criteria:
      - metric: workflow_no_failed_steps
        op: ">="
        threshold: 1
```

Runner APIs:

- `run_benchmark_case(...)`
- `run_benchmark_suite(...)`
- `run_benchmark_suite_from_file(...)`
- CLI: `autotem benchmark --suite <path> [--output <dir>] [--fail-on-failed]`

Suite output:

- `benchmark_suite_report.json`
- `benchmark_suite_report.html`
- `benchmark_suite_history.jsonl`
- `benchmark_suite_trend.json`
- `benchmark_suite_trend.html`
- `cases/<case_id>/benchmark_report.json`

`benchmark_suite_report.html` homepage summary includes:

- `Top Regressions (Current Run)` (severity-ranked, top-3)
- per-item `weight_source` for fast triage of suite/template/case override provenance

Trend payload includes case-level regression leaderboard:

- `regression_alert_count`
- `regression_alerts[]` with severity-ranked entries and reasons (PASS->FAIL, failed checks increase, warning increase, pass-rate drop)
- `regression_weights` reflecting the active per-suite severity weights
- each alert includes case `weights` and `weight_source` for traceability

Case-level override precedence:

- suite default: `suite.regression_weights.*`
- modality template: first enabled match from `suite.regression_weight_templates[]` by `case.modality`
- case override: `cases[i].regression_weights.*`
- effective rule: `suite default -> modality template -> case override`
- override rule detail: later layers only replace non-null keys, unspecified keys keep earlier values

## Extended B/C Reference Metrics

Supported `reference` keys for benchmark comparisons:

- `d_spacings_nm`: list of reference d-spacing values.
- `hkl_truth`: list of reference hkls (`[h,k,l]` or `"h k l"`).
- `hkl_top_k`: top-k indexed spots to evaluate (`default=3`).
- `phase_truth`: list of expected phase names.
- `phase_top_k`: top-k phase candidates to evaluate (`default=3`).
- `atom_coordinates`: list of reference atom coordinates `[x, y]`.
- `atom_nearest_distances`: list of reference nearest-neighbor distances.
- `strain_maps`: per-component map references:
  - component names: `strain_xx`, `strain_yy`, `strain_xy`, `rotation_xy`
  - each component payload supports `values` and optional `mask`

Additional metric names emitted when references are provided:

- `hkl_topk_accuracy`
- `phase_topk_accuracy`
- `atom_nearest_distance_mean_relative_error`
- `<strain_component>_mae`
- `<strain_component>_rmse`
