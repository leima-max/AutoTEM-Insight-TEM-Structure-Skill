"""A4 benchmark suite runner for multi-case validation and regression checks."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from autotem.core import AnalysisReport, read_json
from autotem.validation.benchmarks import run_benchmark_summary
from autotem.validation.cases import (
    BenchmarkCaseSpec,
    BenchmarkSuiteSpec,
    RegressionAlertWeightOverrides,
    RegressionAlertWeights,
    RegressionWeightTemplate,
    load_benchmark_suite,
)

if TYPE_CHECKING:
    from autotem.workflows.pipeline import TEMAnalysisSkill


@dataclass
class BenchmarkCaseRunResult:
    """Execution result for one benchmark case."""

    case_id: str
    report_path: Path
    benchmark_report_path: Path
    passed: bool
    metric_count: int
    warning_count: int
    acceptance: dict[str, object]
    regression_weights: dict[str, int]
    regression_weight_source: str


def _resolve_path(raw_path: str | None, *, base_dir: Path) -> Path | None:
    if raw_path is None or not raw_path.strip():
        return None
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else (base_dir / candidate)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_case_regression_weights(
    suite_weights: RegressionAlertWeights,
    templates: list[RegressionWeightTemplate],
    case_modality: str,
    case_overrides: RegressionAlertWeightOverrides | None,
) -> tuple[RegressionAlertWeights, str]:
    source_parts = ["suite_default"]
    if case_overrides is None:
        case_overrides = RegressionAlertWeightOverrides()

    resolved = suite_weights.model_dump(mode="json")
    normalized_modality = case_modality.strip().lower()
    if normalized_modality:
        for template in templates:
            if not template.enabled:
                continue
            if template.modality.strip().lower() != normalized_modality:
                continue
            template_payload = template.weights.model_dump(mode="json")
            for key, value in template_payload.items():
                if value is not None:
                    resolved[key] = int(value)
            source_parts.append(f"template:{template.modality}")
            break

    override_payload = case_overrides.model_dump(mode="json")
    for key, value in override_payload.items():
        if value is not None:
            resolved[key] = int(value)
    if any(value is not None for value in override_payload.values()):
        source_parts.append("case_override")

    return RegressionAlertWeights.model_validate(resolved), " > ".join(source_parts)


def _format_delta(value: int | float) -> str:
    if isinstance(value, int):
        return f"{value:+d}"
    return f"{value:+.4f}"


def _html_escape(value: object) -> str:
    return html.escape(str(value))


def _load_history(history_path: Path) -> list[dict[str, object]]:
    if not history_path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _write_history(history_path: Path, rows: list[dict[str, object]]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(item, sort_keys=True) for item in rows)
    if payload:
        payload += "\n"
    history_path.write_text(payload, encoding="utf-8")


def _snapshot_from_summary(summary: dict[str, object]) -> dict[str, object]:
    acceptance = summary.get("acceptance", {})
    if isinstance(acceptance, dict):
        accepted = int(acceptance.get("passed", 0))
        evaluated = int(acceptance.get("evaluated", 0))
        failed = int(acceptance.get("failed", 0))
        all_passed = bool(acceptance.get("all_passed", False))
    else:
        accepted = 0
        evaluated = 0
        failed = 0
        all_passed = False

    executed_cases = int(summary.get("cases_executed", 0))
    passed_cases = int(summary.get("cases_passed", 0))
    pass_rate = (passed_cases / executed_cases) if executed_cases > 0 else 0.0
    case_snapshots: list[dict[str, object]] = []
    cases = summary.get("cases", [])
    if isinstance(cases, list):
        for item in cases:
            if not isinstance(item, dict):
                continue
            acceptance_row = item.get("acceptance", {})
            if isinstance(acceptance_row, dict):
                acceptance_evaluated = int(acceptance_row.get("evaluated", 0))
                acceptance_passed = int(acceptance_row.get("passed", 0))
                acceptance_failed = int(acceptance_row.get("failed", 0))
            else:
                acceptance_evaluated = 0
                acceptance_passed = 0
                acceptance_failed = 0
            case_snapshots.append(
                {
                    "case_id": str(item.get("case_id", "")),
                    "passed": bool(item.get("passed", False)),
                    "warning_count": int(item.get("warning_count", 0)),
                    "metric_count": int(item.get("metric_count", 0)),
                    "acceptance_evaluated": acceptance_evaluated,
                    "acceptance_passed": acceptance_passed,
                    "acceptance_failed": acceptance_failed,
                    "regression_weights": item.get("regression_weights", {}),
                    "regression_weight_source": str(
                        item.get("regression_weight_source", "")
                    ),
                }
            )
    return {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_id": str(summary.get("suite_id", "")),
        "cases_executed": executed_cases,
        "cases_passed": passed_cases,
        "cases_failed": int(summary.get("cases_failed", 0)),
        "acceptance_evaluated": evaluated,
        "acceptance_passed": accepted,
        "acceptance_failed": failed,
        "acceptance_all_passed": all_passed,
        "case_pass_rate": pass_rate,
        "cases": case_snapshots,
    }


def _build_trend_payload(
    *,
    suite_id: str,
    history_rows: list[dict[str, object]],
    regression_weights: RegressionAlertWeights,
    history_limit: int = 30,
) -> dict[str, object]:
    recent = history_rows[-history_limit:]
    latest = recent[-1] if recent else {}
    previous = recent[-2] if len(recent) >= 2 else {}

    def _num(row: dict[str, object], key: str) -> float:
        value = row.get(key, 0)
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    if latest and previous:
        delta = {
            "cases_passed_delta": int(
                _num(latest, "cases_passed") - _num(previous, "cases_passed")
            ),
            "cases_failed_delta": int(
                _num(latest, "cases_failed") - _num(previous, "cases_failed")
            ),
            "acceptance_passed_delta": int(
                _num(latest, "acceptance_passed") - _num(previous, "acceptance_passed")
            ),
            "acceptance_failed_delta": int(
                _num(latest, "acceptance_failed") - _num(previous, "acceptance_failed")
            ),
            "case_pass_rate_delta": float(
                _num(latest, "case_pass_rate") - _num(previous, "case_pass_rate")
            ),
        }
    else:
        delta = {
            "cases_passed_delta": 0,
            "cases_failed_delta": 0,
            "acceptance_passed_delta": 0,
            "acceptance_failed_delta": 0,
            "case_pass_rate_delta": 0.0,
        }

    def _case_map(row: dict[str, object]) -> dict[str, dict[str, object]]:
        payload = row.get("cases", [])
        if not isinstance(payload, list):
            return {}
        case_lookup: dict[str, dict[str, object]] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            case_id = str(item.get("case_id", "")).strip()
            if case_id:
                case_lookup[case_id] = item
        return case_lookup

    latest_cases = _case_map(latest) if isinstance(latest, dict) else {}
    previous_cases = _case_map(previous) if isinstance(previous, dict) else {}
    default_weight_pass_flip = int(regression_weights.status_flip_pass_to_fail)
    default_weight_failed_increase = int(regression_weights.acceptance_failed_increase)
    default_weight_warning_increase = int(regression_weights.warning_count_increase)
    default_weight_pass_rate_drop = int(regression_weights.acceptance_pass_rate_drop)
    alerts: list[dict[str, object]] = []
    for case_id in sorted(set(latest_cases).intersection(previous_cases)):
        cur = latest_cases[case_id]
        prev = previous_cases[case_id]

        cur_weights = cur.get("regression_weights", {})
        if isinstance(cur_weights, dict):
            weight_pass_flip = int(
                cur_weights.get("status_flip_pass_to_fail", default_weight_pass_flip)
            )
            weight_failed_increase = int(
                cur_weights.get("acceptance_failed_increase", default_weight_failed_increase)
            )
            weight_warning_increase = int(
                cur_weights.get("warning_count_increase", default_weight_warning_increase)
            )
            weight_pass_rate_drop = int(
                cur_weights.get("acceptance_pass_rate_drop", default_weight_pass_rate_drop)
            )
        else:
            weight_pass_flip = default_weight_pass_flip
            weight_failed_increase = default_weight_failed_increase
            weight_warning_increase = default_weight_warning_increase
            weight_pass_rate_drop = default_weight_pass_rate_drop

        reasons: list[str] = []
        severity = 0

        cur_passed = bool(cur.get("passed", False))
        prev_passed = bool(prev.get("passed", False))
        if prev_passed and not cur_passed:
            reasons.append("status flip: PASS -> FAIL")
            severity += weight_pass_flip

        cur_failed = int(cur.get("acceptance_failed", 0))
        prev_failed = int(prev.get("acceptance_failed", 0))
        if cur_failed > prev_failed:
            reasons.append(
                f"acceptance failed increased: {prev_failed} -> {cur_failed}"
            )
            severity += weight_failed_increase

        cur_warnings = int(cur.get("warning_count", 0))
        prev_warnings = int(prev.get("warning_count", 0))
        if cur_warnings > prev_warnings:
            reasons.append(
                f"warning count increased: {prev_warnings} -> {cur_warnings}"
            )
            severity += weight_warning_increase

        cur_eval = max(int(cur.get("acceptance_evaluated", 0)), 1)
        prev_eval = max(int(prev.get("acceptance_evaluated", 0)), 1)
        cur_pass_rate = int(cur.get("acceptance_passed", 0)) / cur_eval
        prev_pass_rate = int(prev.get("acceptance_passed", 0)) / prev_eval
        if cur_pass_rate < prev_pass_rate:
            reasons.append(
                f"acceptance pass rate dropped: {prev_pass_rate:.3f} -> {cur_pass_rate:.3f}"
            )
            severity += weight_pass_rate_drop

        if reasons:
            alerts.append(
                {
                    "case_id": case_id,
                    "severity": severity,
                    "reasons": reasons,
                    "weights": {
                        "status_flip_pass_to_fail": weight_pass_flip,
                        "acceptance_failed_increase": weight_failed_increase,
                        "warning_count_increase": weight_warning_increase,
                        "acceptance_pass_rate_drop": weight_pass_rate_drop,
                    },
                    "weight_source": str(cur.get("regression_weight_source", "")),
                    "previous": prev,
                    "current": cur,
                }
            )
    alerts.sort(key=lambda item: (-int(item.get("severity", 0)), str(item.get("case_id", ""))))

    return {
        "suite_id": suite_id,
        "history_count": len(history_rows),
        "recent_count": len(recent),
        "latest": latest,
        "previous": previous,
        "delta": delta,
        "regression_weights": regression_weights.model_dump(mode="json"),
        "regression_alert_count": len(alerts),
        "regression_alerts": alerts,
        "recent_runs": recent,
    }


def _write_trend_html(path: Path, trend: dict[str, object]) -> None:
    recent_runs = trend.get("recent_runs", [])
    rows_html: list[str] = []
    if isinstance(recent_runs, list):
        for item in recent_runs:
            if not isinstance(item, dict):
                continue
            rows_html.append(
                "<tr>"
                f"<td>{item.get('run_timestamp_utc', '')}</td>"
                f"<td>{item.get('cases_executed', 0)}</td>"
                f"<td>{item.get('cases_passed', 0)}</td>"
                f"<td>{item.get('cases_failed', 0)}</td>"
                f"<td>{item.get('acceptance_passed', 0)}/{item.get('acceptance_evaluated', 0)}</td>"
                f"<td>{item.get('case_pass_rate', 0.0):.4f}</td>"
                "</tr>"
            )

    delta = trend.get("delta", {})
    if isinstance(delta, dict):
        delta_cases_passed = _format_delta(int(delta.get("cases_passed_delta", 0)))
        delta_cases_failed = _format_delta(int(delta.get("cases_failed_delta", 0)))
        delta_acceptance_passed = _format_delta(int(delta.get("acceptance_passed_delta", 0)))
        delta_acceptance_failed = _format_delta(int(delta.get("acceptance_failed_delta", 0)))
        delta_pass_rate = _format_delta(float(delta.get("case_pass_rate_delta", 0.0)))
    else:
        delta_cases_passed = "+0"
        delta_cases_failed = "+0"
        delta_acceptance_passed = "+0"
        delta_acceptance_failed = "+0"
        delta_pass_rate = "+0.0000"

    alerts = trend.get("regression_alerts", [])
    alerts_html_rows: list[str] = []
    if isinstance(alerts, list):
        for item in alerts:
            if not isinstance(item, dict):
                continue
            case_id = str(item.get("case_id", ""))
            severity = int(item.get("severity", 0))
            reasons = item.get("reasons", [])
            if isinstance(reasons, list):
                reason_text = ", ".join(str(row) for row in reasons)
            else:
                reason_text = ""
            alerts_html_rows.append(
                "<tr class='bad-row'>"
                f"<td>{case_id}</td>"
                f"<td>{severity}</td>"
                f"<td>{reason_text}</td>"
                "</tr>"
            )

    if alerts_html_rows:
        alerts_html = (
            "<h2>Regression Alerts (Case Leaderboard)</h2>"
            "<table><thead><tr>"
            "<th>Case ID</th><th>Severity</th><th>Reasons</th>"
            "</tr></thead><tbody>"
            f"{''.join(alerts_html_rows)}"
            "</tbody></table>"
        )
    else:
        alerts_html = "<h2>Regression Alerts (Case Leaderboard)</h2><p>No regressions detected.</p>"

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>AutoTEM A4 Benchmark Trend</title>"
        "<style>"
        "body{font-family:Segoe UI,Arial,sans-serif;padding:24px;line-height:1.45;}"
        "table{border-collapse:collapse;width:100%;margin-top:12px;}"
        "th,td{border:1px solid #d0d7de;padding:8px;text-align:left;}"
        "th{background:#f6f8fa;}"
        ".bad-row td{color:#b42318;font-weight:600;}"
        "code{font-family:Consolas,monospace;font-size:12px;}"
        "</style></head><body>"
        "<h1>AutoTEM A4 Benchmark Trend</h1>"
        f"<p><strong>Suite:</strong> {trend.get('suite_id', '')}</p>"
        f"<p><strong>History Count:</strong> {trend.get('history_count', 0)}</p>"
        "<h2>Delta vs Previous Run</h2>"
        "<ul>"
        f"<li>cases_passed_delta: <code>{delta_cases_passed}</code></li>"
        f"<li>cases_failed_delta: <code>{delta_cases_failed}</code></li>"
        f"<li>acceptance_passed_delta: <code>{delta_acceptance_passed}</code></li>"
        f"<li>acceptance_failed_delta: <code>{delta_acceptance_failed}</code></li>"
        f"<li>case_pass_rate_delta: <code>{delta_pass_rate}</code></li>"
        "</ul>"
        f"{alerts_html}"
        "<h2>Recent Runs</h2>"
        "<table><thead><tr>"
        "<th>run_timestamp_utc</th>"
        "<th>cases_executed</th>"
        "<th>cases_passed</th>"
        "<th>cases_failed</th>"
        "<th>acceptance_passed/evaluated</th>"
        "<th>case_pass_rate</th>"
        "</tr></thead><tbody>"
        f"{''.join(rows_html)}"
        "</tbody></table>"
        "</body></html>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _write_suite_html(path: Path, summary: dict[str, object]) -> None:
    cases = summary.get("cases", [])
    rows_html: list[str] = []
    if isinstance(cases, list):
        for item in cases:
            if not isinstance(item, dict):
                continue
            case_id = str(item.get("case_id", ""))
            passed = bool(item.get("passed", False))
            metric_count = int(item.get("metric_count", 0))
            warning_count = int(item.get("warning_count", 0))
            report_path = str(item.get("report_path", ""))
            benchmark_report_path = str(item.get("benchmark_report_path", ""))
            status = "PASS" if passed else "FAIL"
            rows_html.append(
                "<tr>"
                f"<td>{case_id}</td>"
                f"<td>{status}</td>"
                f"<td>{metric_count}</td>"
                f"<td>{warning_count}</td>"
                f"<td><code>{report_path}</code></td>"
                f"<td><code>{benchmark_report_path}</code></td>"
                "</tr>"
            )

    acceptance = summary.get("acceptance", {})
    if isinstance(acceptance, dict):
        evaluated = int(acceptance.get("evaluated", 0))
        passed_checks = int(acceptance.get("passed", 0))
        failed_checks = int(acceptance.get("failed", 0))
        all_passed = bool(acceptance.get("all_passed", False))
    else:
        evaluated = 0
        passed_checks = 0
        failed_checks = 0
        all_passed = False

    generated_at = datetime.now(timezone.utc).isoformat()
    trend = summary.get("trend", {})
    if isinstance(trend, dict):
        trend_delta = trend.get("delta", {})
        trend_alerts = trend.get("regression_alerts", [])
        top_regression_html = ""
        if isinstance(trend_delta, dict):
            cases_passed_delta = _format_delta(int(trend_delta.get("cases_passed_delta", 0)))
            cases_failed_delta = _format_delta(int(trend_delta.get("cases_failed_delta", 0)))
            acceptance_passed_delta = _format_delta(
                int(trend_delta.get("acceptance_passed_delta", 0))
            )
            acceptance_failed_delta = _format_delta(
                int(trend_delta.get("acceptance_failed_delta", 0))
            )
            case_pass_rate_delta = _format_delta(
                float(trend_delta.get("case_pass_rate_delta", 0.0))
            )
            trend_delta_html = (
                "<h2>Trend Delta vs Previous Run</h2>"
                "<ul>"
                f"<li>cases_passed_delta: <code>{cases_passed_delta}</code></li>"
                f"<li>cases_failed_delta: <code>{cases_failed_delta}</code></li>"
                f"<li>acceptance_passed_delta: <code>{acceptance_passed_delta}</code></li>"
                f"<li>acceptance_failed_delta: <code>{acceptance_failed_delta}</code></li>"
                f"<li>case_pass_rate_delta: <code>{case_pass_rate_delta}</code></li>"
                "</ul>"
            )
        else:
            trend_delta_html = ""
        alert_rows: list[str] = []
        if isinstance(trend_alerts, list):
            top_rows: list[str] = []
            for item in trend_alerts:
                if not isinstance(item, dict):
                    continue
                case_id = str(item.get("case_id", ""))
                severity = int(item.get("severity", 0))
                reasons = item.get("reasons", [])
                weight_source = str(item.get("weight_source", ""))
                reason_text = (
                    ", ".join(str(entry) for entry in reasons)
                    if isinstance(reasons, list)
                    else ""
                )
                if len(top_rows) < 3:
                    top_rows.append(
                        "<li>"
                        f"<strong>{_html_escape(case_id)}</strong> "
                        f"(severity={severity}, source="
                        f"<code>{_html_escape(weight_source or 'suite_default')}</code>)"
                        f": {_html_escape(reason_text or 'regression detected')}"
                        "</li>"
                    )
                alert_rows.append(
                    "<tr class='bad-row'>"
                    f"<td>{_html_escape(case_id)}</td>"
                    f"<td>{severity}</td>"
                    f"<td><code>{_html_escape(weight_source or 'suite_default')}</code></td>"
                    f"<td>{_html_escape(reason_text)}</td>"
                    "</tr>"
                )
            if top_rows:
                top_regression_html = (
                    "<h2>Top Regressions (Current Run)</h2>"
                    "<ul>"
                    f"{''.join(top_rows)}"
                    "</ul>"
                )
            else:
                top_regression_html = (
                    "<h2>Top Regressions (Current Run)</h2>"
                    "<p>No regressions detected.</p>"
                )
        else:
            top_regression_html = (
                "<h2>Top Regressions (Current Run)</h2>"
                "<p>No regressions detected.</p>"
            )
        if alert_rows:
            trend_alert_html = (
                "<h2>Regression Alerts (Case Leaderboard)</h2>"
                "<table><thead><tr>"
                "<th>Case ID</th><th>Severity</th><th>Weight Source</th><th>Reasons</th>"
                "</tr></thead><tbody>"
                f"{''.join(alert_rows)}"
                "</tbody></table>"
            )
        else:
            trend_alert_html = (
                "<h2>Regression Alerts (Case Leaderboard)</h2>"
                "<p>No regressions detected.</p>"
            )
    else:
        trend_delta_html = ""
        top_regression_html = ""
        trend_alert_html = ""

    html = (
        "<!doctype html>"
        "<html><head><meta charset='utf-8'>"
        "<title>AutoTEM A4 Benchmark Suite Report</title>"
        "<style>"
        "body{font-family:Segoe UI,Arial,sans-serif;padding:24px;line-height:1.45;}"
        "table{border-collapse:collapse;width:100%;margin-top:12px;}"
        "th,td{border:1px solid #d0d7de;padding:8px;text-align:left;vertical-align:top;}"
        "th{background:#f6f8fa;}"
        ".ok{color:#116329;font-weight:600;}.bad{color:#b42318;font-weight:600;}"
        ".bad-row td{color:#b42318;font-weight:600;}"
        "code{font-family:Consolas,monospace;font-size:12px;}"
        "</style></head><body>"
        "<h1>AutoTEM A4 Benchmark Suite Report</h1>"
        f"<p><strong>Suite:</strong> {summary.get('suite_id', '')}</p>"
        f"<p><strong>Description:</strong> {summary.get('description', '')}</p>"
        f"<p><strong>Generated (UTC):</strong> {generated_at}</p>"
        f"<p><strong>Cases:</strong> executed={summary.get('cases_executed', 0)}, "
        f"passed={summary.get('cases_passed', 0)}, failed={summary.get('cases_failed', 0)}</p>"
        f"<p><strong>Acceptance checks:</strong> evaluated={evaluated}, "
        f"passed={passed_checks}, failed={failed_checks}, "
        f"all_passed=<span class='{'ok' if all_passed else 'bad'}'>{all_passed}</span></p>"
        f"{trend_delta_html}"
        f"{top_regression_html}"
        f"{trend_alert_html}"
        "<h2>Case Details</h2>"
        "<table><thead><tr>"
        "<th>Case ID</th><th>Status</th><th>Metric Count</th><th>Warnings</th>"
        "<th>Report Path</th><th>Benchmark Report Path</th>"
        "</tr></thead><tbody>"
        f"{''.join(rows_html)}"
        "</tbody></table>"
        "</body></html>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _load_report_for_case(
    case: BenchmarkCaseSpec,
    *,
    suite_root: Path,
    case_output_dir: Path,
    skill: TEMAnalysisSkill | None,
) -> tuple[AnalysisReport, Path]:
    report_path = _resolve_path(case.report_path, base_dir=suite_root)
    if report_path is not None:
        if not report_path.exists():
            raise FileNotFoundError(
                f"Benchmark case '{case.case_id}' missing report: {report_path}"
            )
        return read_json(report_path, AnalysisReport), report_path

    input_path = _resolve_path(case.input_path, base_dir=suite_root)
    if input_path is None:
        raise ValueError(
            f"Benchmark case '{case.case_id}' requires either `report_path` "
            "or `input_path`."
        )
    workflow_path = _resolve_path(case.workflow_path, base_dir=suite_root)
    if skill is None:
        from autotem.workflows.pipeline import TEMAnalysisSkill

        skill = TEMAnalysisSkill()
    run_result = skill.run(
        input_path=input_path,
        workflow_path=workflow_path,
        output_dir=case_output_dir,
    )
    resolved_report_path = Path(str(run_result["report_path"]))
    return read_json(resolved_report_path, AnalysisReport), resolved_report_path


def run_benchmark_case(
    case: BenchmarkCaseSpec,
    *,
    suite_root: str | Path,
    suite_output_dir: str | Path,
    suite_regression_weights: RegressionAlertWeights | None = None,
    suite_regression_templates: list[RegressionWeightTemplate] | None = None,
    skill: TEMAnalysisSkill | None = None,
) -> BenchmarkCaseRunResult:
    """Run one benchmark case and persist its benchmark report JSON."""
    suite_root_path = Path(suite_root)
    suite_output_path = Path(suite_output_dir)
    if suite_regression_weights is None:
        suite_regression_weights = RegressionAlertWeights()
    if suite_regression_templates is None:
        suite_regression_templates = []
    effective_weights, effective_source = _resolve_case_regression_weights(
        suite_regression_weights,
        suite_regression_templates,
        case.modality,
        case.regression_weights,
    )
    case_dir_name = case.output_subdir.strip() or case.case_id
    case_output_dir = suite_output_path / "cases" / case_dir_name
    case_output_dir.mkdir(parents=True, exist_ok=True)

    report, report_path = _load_report_for_case(
        case,
        suite_root=suite_root_path,
        case_output_dir=case_output_dir,
        skill=skill,
    )
    benchmark, payload, warnings = run_benchmark_summary(
        report,
        benchmark_parameters={
            "case_id": case.case_id,
            "reference": case.reference,
            "acceptance_criteria": [
                criterion.model_dump(mode="json") for criterion in case.acceptance_criteria
            ],
        },
        report_base_dir=report_path.parent,
    )
    benchmark_payload = {
        **payload,
        "warnings": warnings,
        "report_path": str(report_path),
        "regression_weights": effective_weights.model_dump(mode="json"),
        "regression_weight_source": effective_source,
    }
    benchmark_report_path = case_output_dir / "benchmark_report.json"
    _write_json(benchmark_report_path, benchmark_payload)

    return BenchmarkCaseRunResult(
        case_id=case.case_id,
        report_path=report_path,
        benchmark_report_path=benchmark_report_path,
        passed=bool(benchmark.passed),
        metric_count=len(benchmark.metrics),
        warning_count=len(warnings),
        acceptance=dict(payload.get("acceptance", {})),
        regression_weights=effective_weights.model_dump(mode="json"),
        regression_weight_source=effective_source,
    )


def run_benchmark_suite(
    suite: BenchmarkSuiteSpec,
    *,
    suite_root: str | Path,
    output_dir: str | Path | None = None,
    skill: TEMAnalysisSkill | None = None,
    ) -> dict[str, object]:
    """Run all enabled cases in a benchmark suite and return aggregate summary."""
    suite_root_path = Path(suite_root)
    suite_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else (suite_root_path / Path(suite.default_output_dir))
    )
    suite_output_dir.mkdir(parents=True, exist_ok=True)

    case_results: list[BenchmarkCaseRunResult] = []
    skipped_cases: list[str] = []
    for case in suite.cases:
        if not case.enabled:
            skipped_cases.append(case.case_id)
            continue
        case_results.append(
            run_benchmark_case(
                case,
                suite_root=suite_root_path,
                suite_output_dir=suite_output_dir,
                suite_regression_weights=suite.regression_weights,
                suite_regression_templates=suite.regression_weight_templates,
                skill=skill,
            )
        )

    total_evaluated = 0
    total_passed = 0
    total_failed = 0
    for case_result in case_results:
        acceptance = case_result.acceptance
        evaluated = int(acceptance.get("evaluated", 0))
        passed = int(acceptance.get("passed", 0))
        failed = int(acceptance.get("failed", max(evaluated - passed, 0)))
        total_evaluated += evaluated
        total_passed += passed
        total_failed += failed

    summary = {
        "schema_version": suite.schema_version,
        "suite_id": suite.suite_id,
        "description": suite.description,
        "output_dir": str(suite_output_dir),
        "regression_weights": suite.regression_weights.model_dump(mode="json"),
        "cases_total": len(suite.cases),
        "cases_executed": len(case_results),
        "cases_skipped": len(skipped_cases),
        "case_ids_skipped": skipped_cases,
        "cases_passed": sum(1 for item in case_results if item.passed),
        "cases_failed": sum(1 for item in case_results if not item.passed),
        "acceptance": {
            "evaluated": total_evaluated,
            "passed": total_passed,
            "failed": total_failed,
            "all_passed": total_failed == 0,
        },
        "cases": [
            {
                "case_id": item.case_id,
                "passed": item.passed,
                "metric_count": item.metric_count,
                "warning_count": item.warning_count,
                "report_path": str(item.report_path),
                "benchmark_report_path": str(item.benchmark_report_path),
                "acceptance": item.acceptance,
                "regression_weights": item.regression_weights,
                "regression_weight_source": item.regression_weight_source,
            }
            for item in case_results
        ],
    }
    history_path = suite_output_dir / "benchmark_suite_history.jsonl"
    trend_json_path = suite_output_dir / "benchmark_suite_trend.json"
    trend_html_path = suite_output_dir / "benchmark_suite_trend.html"
    history_rows = _load_history(history_path)
    history_rows.append(_snapshot_from_summary(summary))
    if len(history_rows) > 200:
        history_rows = history_rows[-200:]
    _write_history(history_path, history_rows)
    trend = _build_trend_payload(
        suite_id=suite.suite_id,
        history_rows=history_rows,
        regression_weights=suite.regression_weights,
        history_limit=30,
    )
    _write_json(trend_json_path, trend)
    _write_trend_html(trend_html_path, trend)
    summary["trend"] = trend
    summary["trend_report_json"] = str(trend_json_path)
    summary["trend_report_html"] = str(trend_html_path)
    summary["history_path"] = str(history_path)

    suite_report_json = suite_output_dir / "benchmark_suite_report.json"
    suite_report_html = suite_output_dir / "benchmark_suite_report.html"
    _write_json(suite_report_json, summary)
    _write_suite_html(suite_report_html, summary)
    summary["suite_report_json"] = str(suite_report_json)
    summary["suite_report_html"] = str(suite_report_html)
    return summary


def run_benchmark_suite_from_file(
    suite_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    skill: TEMAnalysisSkill | None = None,
) -> dict[str, object]:
    """Load benchmark suite YAML and execute it."""
    suite_path_obj = Path(suite_path)
    suite = load_benchmark_suite(suite_path_obj)
    suite_root = suite_path_obj.parent
    return run_benchmark_suite(
        suite,
        suite_root=suite_root,
        output_dir=output_dir,
        skill=skill,
    )
