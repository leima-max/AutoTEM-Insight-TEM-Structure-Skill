from pathlib import Path

from autotem.core import (
    AnalysisReport,
    ArrayReference,
    AtomLattice,
    AtomSite,
    DiffractionResult,
    IndexedSpot,
    Measurement,
    PhaseCandidate,
    StrainResult,
    TEMDataset,
    WorkflowConfig,
    WorkflowStep,
    write_json,
)
from autotem.validation import (
    BenchmarkCaseSpec,
    BenchmarkSuiteSpec,
    RegressionAlertWeightOverrides,
    RegressionAlertWeights,
    RegressionWeightTemplate,
    load_benchmark_suite,
    run_benchmark_suite,
    run_benchmark_suite_from_file,
    run_benchmark_summary,
    write_benchmark_case_schemas,
)


def _build_report(case_id: str, *, modality: str = "unknown") -> AnalysisReport:
    return AnalysisReport(
        summary=f"report-{case_id}",
        workflow=WorkflowConfig(
            name="suite_case",
            steps=[
                WorkflowStep(name="load", status="done"),
                WorkflowStep(name="fft", status="done"),
                WorkflowStep(name="benchmark", status="done"),
            ],
        ),
        dataset=TEMDataset(
            data=ArrayReference(shape=(8, 8), dtype="float32", uri=f"{case_id}.npy"),
            modality=modality,
        ),
        diffraction=DiffractionResult(
            d_spacings=[
                Measurement(value=0.2, unit="nm"),
                Measurement(value=0.4, unit="nm"),
            ],
            indexed_spots=[
                IndexedSpot(peak_index=0, h=1, k=1, l=1),
                IndexedSpot(peak_index=1, h=2, k=0, l=0),
            ],
            phase_candidates=[
                PhaseCandidate(name="Gold", score=0.9),
                PhaseCandidate(name="Silicon", score=0.7),
            ],
        ),
        atom_lattice=AtomLattice(
            atoms=[
                AtomSite(x=0.0, y=0.0),
                AtomSite(x=2.0, y=2.0),
                AtomSite(x=4.0, y=0.0),
            ]
        ),
        strain=StrainResult(
            strain_xx=ArrayReference(
                shape=(2, 2),
                dtype="float64",
                inline=[[0.10, 0.20], [0.30, 0.40]],
            )
        ),
        provenance={"quality_gates": {"passed": True, "blocking_warning_count": 0}},
    )


def _build_regressed_report(case_id: str) -> AnalysisReport:
    report = _build_report(case_id)
    if report.workflow is not None and report.workflow.steps:
        report.workflow.steps[0].status = "failed"
        report.workflow.steps[0].warnings = ["forced regression"]
    report.warnings = ["forced regression warning"]
    return report


def test_a4_benchmark_suite_loader_and_schema_export(tmp_path: Path) -> None:
    suite_yaml = tmp_path / "suite.yaml"
    suite_yaml.write_text(
        "\n".join(
            [
                "suite:",
                "  suite_id: quick_suite",
                "  description: smoke",
                "  regression_weights:",
                "    status_flip_pass_to_fail: 9",
                "    acceptance_failed_increase: 4",
                "  regression_weight_templates:",
                "    - modality: HRTEM",
                "      weights:",
                "        status_flip_pass_to_fail: 11",
                "cases:",
                "  - case_id: c1",
                "    report_path: reports/c1.json",
                "    regression_weights:",
                "      status_flip_pass_to_fail: 12",
                "    acceptance_criteria:",
                "      - metric: workflow_no_failed_steps",
                "        op: \">=\"",
                "        threshold: 1",
            ]
        ),
        encoding="utf-8",
    )

    suite = load_benchmark_suite(suite_yaml)
    assert suite.suite_id == "quick_suite"
    assert len(suite.cases) == 1
    assert suite.cases[0].acceptance_criteria[0].metric == "workflow_no_failed_steps"
    assert suite.regression_weights.status_flip_pass_to_fail == 9
    assert suite.regression_weights.acceptance_failed_increase == 4
    assert len(suite.regression_weight_templates) == 1
    assert suite.regression_weight_templates[0].modality == "HRTEM"
    assert suite.cases[0].regression_weights.status_flip_pass_to_fail == 12

    schema_dir = tmp_path / "schemas"
    write_benchmark_case_schemas(schema_dir)
    assert (schema_dir / "BenchmarkCaseSpec.schema.json").exists()
    assert (schema_dir / "BenchmarkSuiteSpec.schema.json").exists()
    assert (schema_dir / "benchmark_case_schema_manifest.json").exists()


def test_a4_multi_case_runner_with_existing_reports(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report1_path = reports_dir / "case1.json"
    report2_path = reports_dir / "case2.json"
    write_json(report1_path, _build_report("case1"))
    write_json(report2_path, _build_report("case2"))

    suite = BenchmarkSuiteSpec(
        suite_id="multi_case",
        cases=[
            BenchmarkCaseSpec(
                case_id="case1",
                report_path=str(report1_path),
                acceptance_criteria=[
                    {"metric": "workflow_no_failed_steps", "op": ">=", "threshold": 1}
                ],
            ),
            BenchmarkCaseSpec(
                case_id="case2",
                report_path=str(report2_path),
                acceptance_criteria=[
                    {"metric": "workflow_no_failed_steps", "op": ">=", "threshold": 1}
                ],
            ),
        ],
    )
    summary = run_benchmark_suite(
        suite,
        suite_root=tmp_path,
        output_dir=tmp_path / "benchmark_out",
    )

    assert summary["cases_executed"] == 2
    assert summary["cases_failed"] == 0
    assert (tmp_path / "benchmark_out" / "benchmark_suite_report.json").exists()
    assert (tmp_path / "benchmark_out" / "benchmark_suite_report.html").exists()
    assert (tmp_path / "benchmark_out" / "benchmark_suite_trend.json").exists()
    assert (tmp_path / "benchmark_out" / "benchmark_suite_trend.html").exists()
    assert (tmp_path / "benchmark_out" / "benchmark_suite_history.jsonl").exists()
    assert "suite_report_html" in summary
    assert "trend_report_json" in summary
    assert "trend_report_html" in summary
    assert "history_path" in summary
    assert (
        tmp_path / "benchmark_out" / "cases" / "case1" / "benchmark_report.json"
    ).exists()
    assert (
        tmp_path / "benchmark_out" / "cases" / "case2" / "benchmark_report.json"
    ).exists()


def test_a4_public_minimal_package_suite_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite_path = repo_root / "benchmarks" / "suite_public_minimal.yaml"
    assert suite_path.exists()

    summary = run_benchmark_suite_from_file(
        suite_path,
        output_dir=tmp_path / "public_suite_out",
    )
    assert summary["suite_id"] == "public_minimal_suite"
    assert summary["cases_executed"] == 2
    assert summary["cases_failed"] == 0
    assert (tmp_path / "public_suite_out" / "benchmark_suite_report.json").exists()
    assert (tmp_path / "public_suite_out" / "benchmark_suite_trend.json").exists()

    case_ids = {str(item["case_id"]) for item in summary["cases"]}
    assert case_ids == {"b_diffraction_public", "c_atomic_strain_public"}


def test_a4_extended_bc_reference_metrics_are_reported() -> None:
    report = _build_report("bc_metrics")
    benchmark, _, warnings = run_benchmark_summary(
        report,
        benchmark_parameters={
            "case_id": "bc_metrics",
            "reference": {
                "d_spacings_nm": [0.2, 0.5],
                "hkl_truth": [[1, 1, 1], [3, 1, 1]],
                "hkl_top_k": 2,
                "phase_truth": ["Gold", "TiO2"],
                "phase_top_k": 2,
                "atom_coordinates": [[0.0, 0.0], [2.0, 2.0], [4.0, 0.0]],
                "atom_nearest_distances": [2.82842712, 2.82842712, 2.82842712],
                "strain_maps": {
                    "strain_xx": {
                        "values": [[0.10, 0.10], [0.25, 0.50]],
                        "mask": [[True, True], [False, True]],
                    }
                },
            },
            "acceptance_criteria": [
                {"metric": "hkl_topk_accuracy", "op": ">=", "threshold": 0.4},
                {"metric": "phase_topk_accuracy", "op": ">=", "threshold": 0.4},
                {
                    "metric": "atom_nearest_distance_mean_relative_error",
                    "op": "<=",
                    "threshold": 0.1,
                },
                {"metric": "strain_xx_mae", "op": "<=", "threshold": 0.1},
            ],
        },
    )

    assert not warnings
    metric_map = {item.name: item for item in benchmark.metrics}
    assert "hkl_topk_accuracy" in metric_map
    assert "phase_topk_accuracy" in metric_map
    assert "atom_nearest_distance_mean_relative_error" in metric_map
    assert "strain_xx_mae" in metric_map
    assert "strain_xx_rmse" in metric_map
    assert "d_spacing_uncertainty_coverage" in metric_map
    assert "d_spacing_uncertainty_mean_relative" in metric_map
    assert "d_spacing_uncertainty_plausible_fraction" in metric_map
    assert benchmark.passed is True


def test_a4_benchmark_trend_accumulates_history(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report1_path = reports_dir / "case1.json"
    write_json(report1_path, _build_report("case1"))

    suite = BenchmarkSuiteSpec(
        suite_id="trend_suite",
        cases=[
            BenchmarkCaseSpec(
                case_id="case1",
                report_path=str(report1_path),
                acceptance_criteria=[
                    {"metric": "workflow_no_failed_steps", "op": ">=", "threshold": 1}
                ],
            ),
        ],
    )
    out_dir = tmp_path / "trend_out"
    first = run_benchmark_suite(suite, suite_root=tmp_path, output_dir=out_dir)
    write_json(report1_path, _build_regressed_report("case1"))
    second = run_benchmark_suite(suite, suite_root=tmp_path, output_dir=out_dir)

    assert first["trend"]["history_count"] >= 1
    assert second["trend"]["history_count"] >= 2
    delta = second["trend"]["delta"]
    assert "case_pass_rate_delta" in delta
    assert second["trend"]["regression_alert_count"] >= 1
    alerts = second["trend"]["regression_alerts"]
    assert isinstance(alerts, list)
    assert any(item["case_id"] == "case1" for item in alerts)


def test_a4_benchmark_trend_uses_custom_regression_weights(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report1_path = reports_dir / "case1.json"
    write_json(report1_path, _build_report("case1"))

    custom_weights = RegressionAlertWeights(
        status_flip_pass_to_fail=10,
        acceptance_failed_increase=7,
        warning_count_increase=0,
        acceptance_pass_rate_drop=4,
    )
    suite = BenchmarkSuiteSpec(
        suite_id="trend_weighted_suite",
        regression_weights=custom_weights,
        cases=[
            BenchmarkCaseSpec(
                case_id="case1",
                report_path=str(report1_path),
                acceptance_criteria=[
                    {"metric": "workflow_no_failed_steps", "op": ">=", "threshold": 1}
                ],
            ),
        ],
    )

    out_dir = tmp_path / "trend_weighted_out"
    run_benchmark_suite(suite, suite_root=tmp_path, output_dir=out_dir)
    write_json(report1_path, _build_regressed_report("case1"))
    second = run_benchmark_suite(suite, suite_root=tmp_path, output_dir=out_dir)

    assert second["regression_weights"]["status_flip_pass_to_fail"] == 10
    assert second["trend"]["regression_weights"]["acceptance_failed_increase"] == 7
    alerts = second["trend"]["regression_alerts"]
    assert isinstance(alerts, list)
    case1 = next(item for item in alerts if item["case_id"] == "case1")
    # PASS->FAIL (10) + failed increase (7) + pass-rate drop (4) = 21
    assert case1["severity"] == 21


def test_a4_benchmark_case_weight_override_precedence(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report1_path = reports_dir / "case1.json"
    write_json(report1_path, _build_report("case1"))

    suite = BenchmarkSuiteSpec(
        suite_id="trend_case_override_suite",
        regression_weights=RegressionAlertWeights(
            status_flip_pass_to_fail=1,
            acceptance_failed_increase=1,
            warning_count_increase=1,
            acceptance_pass_rate_drop=1,
        ),
        cases=[
            BenchmarkCaseSpec(
                case_id="case1",
                report_path=str(report1_path),
                regression_weights=RegressionAlertWeightOverrides(
                    status_flip_pass_to_fail=10,
                    acceptance_failed_increase=7,
                    warning_count_increase=0,
                    acceptance_pass_rate_drop=4,
                ),
                acceptance_criteria=[
                    {"metric": "workflow_no_failed_steps", "op": ">=", "threshold": 1}
                ],
            ),
        ],
    )
    out_dir = tmp_path / "trend_case_override_out"
    run_benchmark_suite(suite, suite_root=tmp_path, output_dir=out_dir)
    write_json(report1_path, _build_regressed_report("case1"))
    second = run_benchmark_suite(suite, suite_root=tmp_path, output_dir=out_dir)

    alerts = second["trend"]["regression_alerts"]
    assert isinstance(alerts, list)
    case1 = next(item for item in alerts if item["case_id"] == "case1")
    assert case1["severity"] == 21
    assert case1["weights"]["status_flip_pass_to_fail"] == 10
    assert case1["weights"]["warning_count_increase"] == 0


def test_a4_benchmark_modality_template_weights_apply(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report1_path = reports_dir / "case1.json"
    write_json(report1_path, _build_report("case1", modality="HRTEM"))

    suite = BenchmarkSuiteSpec(
        suite_id="trend_template_suite",
        regression_weights=RegressionAlertWeights(
            status_flip_pass_to_fail=1,
            acceptance_failed_increase=1,
            warning_count_increase=1,
            acceptance_pass_rate_drop=1,
        ),
        regression_weight_templates=[
            RegressionWeightTemplate(
                modality="HRTEM",
                weights=RegressionAlertWeightOverrides(
                    status_flip_pass_to_fail=8,
                    acceptance_failed_increase=5,
                    warning_count_increase=0,
                    acceptance_pass_rate_drop=3,
                ),
            )
        ],
        cases=[
            BenchmarkCaseSpec(
                case_id="case1",
                modality="HRTEM",
                report_path=str(report1_path),
                acceptance_criteria=[
                    {"metric": "workflow_no_failed_steps", "op": ">=", "threshold": 1}
                ],
            ),
        ],
    )
    out_dir = tmp_path / "trend_template_out"
    run_benchmark_suite(suite, suite_root=tmp_path, output_dir=out_dir)
    write_json(report1_path, _build_regressed_report("case1"))
    second = run_benchmark_suite(suite, suite_root=tmp_path, output_dir=out_dir)

    alerts = second["trend"]["regression_alerts"]
    case1 = next(item for item in alerts if item["case_id"] == "case1")
    # PASS->FAIL (8) + failed increase (5) + pass-rate drop (3) = 16
    assert case1["severity"] == 16
    assert case1["weights"]["status_flip_pass_to_fail"] == 8
    assert case1["weights"]["warning_count_increase"] == 0
    assert "template:HRTEM" in case1["weight_source"]
    suite_report_html = Path(str(second["suite_report_html"])).read_text(encoding="utf-8")
    assert "Top Regressions (Current Run)" in suite_report_html
    assert "Weight Source" in suite_report_html
    assert "template:HRTEM" in suite_report_html
