import json
from pathlib import Path

import pytest

from autotem.core import (
    AnalysisReport,
    ArrayReference,
    AtomLattice,
    AtomSite,
    DiffractionResult,
    IndexedSpot,
    Measurement,
    OutputArtifact,
    PhaseCandidate,
    SimulatedPattern,
    SimulatedSpot,
    SimulationResult,
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

pytestmark = pytest.mark.smoke


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
            coordinate_unit="pixel",
            atoms=[
                AtomSite(x=0.0, y=0.0),
                AtomSite(x=2.0, y=2.0),
                AtomSite(x=4.0, y=0.0),
            ]
        ),
        simulation=SimulationResult(
            patterns=[
                SimulatedPattern(
                    kind="diffraction",
                    structure_id="Gold",
                    zone_axis=(0, 0, 1),
                    simulation_parameters={
                        "engine": "diffsims",
                        "simulated_spot_count": 4,
                        "phase_match": {
                            "matched_spot_count": 3,
                            "rotation_deg": 0.0,
                        },
                    },
                    spots=[
                        SimulatedSpot(
                            gx=Measurement(value=2.5, unit="1/nm"),
                            gy=Measurement(value=0.0, unit="1/nm"),
                        ),
                        SimulatedSpot(
                            gx=Measurement(value=0.0, unit="1/nm"),
                            gy=Measurement(value=2.5, unit="1/nm"),
                        ),
                    ],
                )
            ],
            matched_phase_candidates=[
                PhaseCandidate(
                    name="Gold",
                    score=0.96,
                    matched_spot_count=3,
                    mean_d_error=Measurement(
                        value=0.005,
                        unit="relative",
                        method="diffsims_phase_match",
                    ),
                    mean_angle_error=Measurement(
                        value=0.3,
                        unit="deg",
                        method="diffsims_phase_match",
                    ),
                )
            ],
            overlays=[
                OutputArtifact(
                    kind="image",
                    uri="figures/simulation_diffraction_overlay.png",
                    label="Simulation diffraction overlay",
                )
            ],
            provenance={
                "method": "diffsims",
                "phase_matching": {
                    "active": True,
                    "method": "diffsims_template_spot_match",
                    "selected_phase": "Gold",
                    "measured_spot_count": 3,
                    "simulated_pattern_count": 1,
                },
            },
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


def test_a4_suite_propagates_missing_metric_evidence(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    write_json(report_path, _build_report("missing"))
    suite = BenchmarkSuiteSpec(
        suite_id="missing_metric",
        cases=[BenchmarkCaseSpec(
            case_id="missing",
            report_path=str(report_path),
            acceptance_criteria=[
                {"metric": "workflow_no_failed_steps", "op": ">=", "threshold": 1},
                {"metric": "required_but_absent", "op": ">=", "threshold": 1},
            ],
        )],
    )
    summary = run_benchmark_suite(suite, suite_root=tmp_path, output_dir=tmp_path / "out")
    acceptance = summary["acceptance"]
    assert summary["cases_failed"] == 1
    assert acceptance["requested"] == 2
    assert acceptance["evaluated"] == 1
    assert acceptance["unevaluated"] == 1
    assert acceptance["failed"] == 0
    assert acceptance["all_passed"] is False
    assert acceptance["errors"][0]["case_id"] == "missing"
    saved = json.loads(Path(str(summary["suite_report_json"])).read_text(encoding="utf-8"))
    assert saved["acceptance"]["all_passed"] is False
    assert saved["cases"][0]["acceptance"]["errors"][0]["code"] == "missing_metric"
    html = Path(str(summary["suite_report_html"])).read_text(encoding="utf-8")
    assert "requested=2" in html
    assert "unevaluated=1" in html
    assert "Acceptance Evidence Errors" in html
    assert "required_but_absent" in html


@pytest.mark.parametrize("criteria", [
    [],
    [{"metric": "workflow_no_failed_steps", "op": ">=", "threshold": 1, "unit": "nm"}],
    [{"metric": "workflow_no_failed_steps", "op": ">=", "threshold": float("inf")}],
])
def test_a4_suite_never_passes_unevaluated_criteria(tmp_path: Path, criteria) -> None:
    report_path = tmp_path / "report.json"
    write_json(report_path, _build_report("invalid"))
    suite = BenchmarkSuiteSpec(cases=[BenchmarkCaseSpec(
        case_id="invalid", report_path=str(report_path), acceptance_criteria=criteria,
    )])
    summary = run_benchmark_suite(suite, suite_root=tmp_path, output_dir=tmp_path / "out")
    assert summary["cases_passed"] == 0
    assert summary["cases_failed"] == 1
    assert summary["acceptance"]["evaluated"] == 0
    assert summary["acceptance"]["all_passed"] is False
    assert summary["acceptance"]["error_count"] == 1


@pytest.mark.parametrize("disabled_case", [False, True])
def test_a4_suite_without_enabled_cases_is_not_evaluated(
    tmp_path: Path, disabled_case: bool,
) -> None:
    cases = [BenchmarkCaseSpec(case_id="disabled", enabled=False)] if disabled_case else []
    summary = run_benchmark_suite(
        BenchmarkSuiteSpec(cases=cases), suite_root=tmp_path, output_dir=tmp_path / "out",
    )
    assert summary["cases_executed"] == 0
    assert summary["acceptance"]["status"] == "not_evaluated"
    assert summary["acceptance"]["all_passed"] is False
    assert summary["acceptance"]["errors"][0]["code"] == "no_enabled_cases"


def test_a4_suite_omitted_criteria_uses_default_gates(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report = AnalysisReport(
        workflow=WorkflowConfig(steps=[WorkflowStep(name="load", status="done")]),
        provenance={"quality_gates": {"passed": True}},
    )
    write_json(report_path, report)
    summary = run_benchmark_suite(
        BenchmarkSuiteSpec(cases=[
            BenchmarkCaseSpec(case_id="default", report_path=str(report_path)),
        ]),
        suite_root=tmp_path,
        output_dir=tmp_path / "out",
    )
    assert summary["cases_passed"] == 1
    assert summary["acceptance"]["requested"] == 2
    assert summary["acceptance"]["all_passed"] is True


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


def test_a4_public_minimal_package_suite_blocks_unknown_coordinate_units(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    suite_path = repo_root / "benchmarks" / "suite_public_minimal.yaml"
    assert suite_path.exists()

    summary = run_benchmark_suite_from_file(
        suite_path,
        output_dir=tmp_path / "public_suite_out",
    )
    assert summary["suite_id"] == "public_minimal_suite"
    assert summary["cases_executed"] == 3
    # The frozen atomic report declares unknown units; matching numbers alone
    # cannot establish a valid pixel-coordinate benchmark.
    assert summary["cases_passed"] == 2
    assert summary["cases_failed"] == 1
    assert summary["acceptance"]["requested"] == 24
    assert summary["acceptance"]["evaluated"] == 22
    assert summary["acceptance"]["unevaluated"] == 2
    assert summary["acceptance"]["all_passed"] is False
    assert {error["metric"] for error in summary["acceptance"]["errors"]} == {
        "atom_coordinate_rmse", "atom_nearest_distance_mean_relative_error",
    }
    assert {error["case_id"] for error in summary["acceptance"]["errors"]} == {
        "c_atomic_strain_public",
    }
    assert (tmp_path / "public_suite_out" / "benchmark_suite_report.json").exists()
    assert (tmp_path / "public_suite_out" / "benchmark_suite_trend.json").exists()

    case_ids = {str(item["case_id"]) for item in summary["cases"]}
    assert case_ids == {
        "b_diffraction_public",
        "b5_diffsims_phase_matching_public",
        "c_atomic_strain_public",
    }
    case_reports = {
        str(item["case_id"]): json.loads(
            Path(str(item["benchmark_report_path"])).read_text(encoding="utf-8")
        )
        for item in summary["cases"]
    }
    b_metric_map = {
        str(item["name"]): item for item in case_reports["b_diffraction_public"]["metrics"]
    }
    c_metric_map = {
        str(item["name"]): item for item in case_reports["c_atomic_strain_public"]["metrics"]
    }
    b5_metric_map = {
        str(item["name"]): item
        for item in case_reports["b5_diffsims_phase_matching_public"]["metrics"]
    }
    assert b_metric_map["b_interface_pass_rate"]["value"]["value"] == 1.0
    assert b_metric_map["b_interface_missing_count"]["value"]["value"] == 0
    assert b5_metric_map["b5_phase_matching_active_flag"]["value"]["value"] == 1
    assert b5_metric_map["b5_simulated_spot_count"]["value"]["value"] >= 4
    assert b5_metric_map["b5_best_matched_spot_count"]["value"]["value"] >= 3
    assert b5_metric_map["b5_phase_topk_accuracy"]["value"]["value"] == 1.0
    assert c_metric_map["c_interface_pass_rate"]["value"]["value"] == 1.0
    assert c_metric_map["c_interface_missing_count"]["value"]["value"] == 0


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
    assert "b3_b4_indexed_nonzero_fraction" in metric_map
    assert "b3_b4_top_phase_score" in metric_map
    assert "b5_simulation_pattern_count" in metric_map
    assert "b5_simulated_spot_count" in metric_map
    assert "b5_matched_phase_candidate_count" in metric_map
    assert "b5_phase_matching_active_flag" in metric_map
    assert "b5_simulation_overlay_count" in metric_map
    assert "b5_best_phase_match_score" in metric_map
    assert "b5_best_matched_spot_count" in metric_map
    assert "b5_best_mean_d_error" in metric_map
    assert "b5_best_mean_angle_error_deg" in metric_map
    assert "b_interface_pass_rate" in metric_map
    assert "b_interface_missing_count" in metric_map
    assert "c_interface_pass_rate" in metric_map
    assert "c_interface_missing_count" in metric_map
    assert benchmark.passed is True


def test_a4_benchmark_reference_hook_reports_b5_phase_accuracy() -> None:
    report = _build_report("b5_metrics", modality="HRTEM")

    benchmark, _, warnings = run_benchmark_summary(
        report,
        benchmark_parameters={
            "case_id": "b5_metrics",
            "reference": {
                "b5_phase_truth": ["Gold"],
                "b5_phase_top_k": 1,
            },
            "acceptance_criteria": [
                {"metric": "b5_phase_topk_accuracy", "op": ">=", "threshold": 1.0},
                {"metric": "b5_phase_matching_active_flag", "op": ">=", "threshold": 1},
                {"metric": "b5_best_matched_spot_count", "op": ">=", "threshold": 3},
                {"metric": "b5_best_mean_d_error", "op": "<=", "threshold": 0.01},
                {"metric": "b5_best_mean_angle_error_deg", "op": "<=", "threshold": 1.0},
            ],
        },
    )

    assert not warnings
    metric_map = {item.name: item for item in benchmark.metrics}
    assert metric_map["b5_phase_topk_accuracy"].value.value == 1.0
    assert metric_map["b5_best_phase_match_score"].value.value == 0.96
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
