"""A4 validation metrics derived from workflow report outputs."""

from __future__ import annotations

from autotem.core import AnalysisReport, Measurement, ValidationMetric
from autotem.reporting.interface_contracts import evaluate_bc_interface_completeness


def _metric(
    *,
    name: str,
    value: float | int | bool,
    unit: str,
    method: str,
    direction: str = "informational",
    passed: bool | None = None,
    details: dict[str, object] | None = None,
) -> ValidationMetric:
    return ValidationMetric(
        name=name,
        value=Measurement(value=value, unit=unit, method=method),
        direction=direction,  # type: ignore[arg-type]
        passed=passed,
        details=dict(details or {}),
    )


def collect_validation_metrics(report: AnalysisReport) -> tuple[list[ValidationMetric], list[str]]:
    """Build baseline A4 validation metrics from a finished analysis report."""
    warnings: list[str] = []
    metrics: list[ValidationMetric] = []

    failed_steps = 0
    total_steps = 0
    if report.workflow is not None:
        total_steps = len(report.workflow.steps)
        failed_steps = sum(1 for step in report.workflow.steps if step.status == "failed")
    no_failed_steps = failed_steps == 0

    metrics.append(
        _metric(
            name="workflow_no_failed_steps",
            value=1 if no_failed_steps else 0,
            unit="flag",
            method="workflow_step_status_scan",
            direction="higher_is_better",
            passed=no_failed_steps,
            details={"failed_steps": failed_steps, "total_steps": total_steps},
        )
    )

    gate_payload = report.provenance.get("quality_gates", {})
    gate_passed = bool(gate_payload.get("passed")) if isinstance(gate_payload, dict) else False
    blocking_count = (
        int(gate_payload.get("blocking_warning_count", 0))
        if isinstance(gate_payload, dict)
        else 0
    )
    metrics.append(
        _metric(
            name="quality_gates_passed_flag",
            value=1 if gate_passed else 0,
            unit="flag",
            method="quality_gate_summary",
            direction="higher_is_better",
            passed=gate_passed,
            details={"blocking_warning_count": blocking_count},
        )
    )

    metrics.append(
        _metric(
            name="report_warning_count",
            value=len(report.warnings),
            unit="count",
            method="analysis_report",
            direction="lower_is_better",
        )
    )

    diffraction = report.diffraction
    peak_count = len(diffraction.peaks) if diffraction is not None else 0
    d_spacing_count = len(diffraction.d_spacings) if diffraction is not None else 0
    phase_count = len(diffraction.phase_candidates) if diffraction is not None else 0
    metrics.append(
        _metric(
            name="diffraction_peak_count",
            value=peak_count,
            unit="count",
            method="diffraction_result",
            direction="higher_is_better",
        )
    )
    metrics.append(
        _metric(
            name="d_spacing_count",
            value=d_spacing_count,
            unit="count",
            method="diffraction_result",
            direction="higher_is_better",
        )
    )
    if d_spacing_count > 0 and diffraction is not None:
        rel_uncertainties: list[float] = []
        uncertainty_present_count = 0
        outlier_count = 0
        for item in diffraction.d_spacings:
            if item.uncertainty is None:
                continue
            try:
                value = float(item.value)
                uncertainty = float(item.uncertainty)
            except (TypeError, ValueError):
                continue
            if value == 0:
                continue
            if uncertainty < 0:
                continue
            uncertainty_present_count += 1
            rel_unc = abs(uncertainty / value)
            rel_uncertainties.append(rel_unc)
            if rel_unc > 0.5:
                outlier_count += 1

        coverage = uncertainty_present_count / d_spacing_count
        if rel_uncertainties:
            mean_rel_unc = float(sum(rel_uncertainties) / len(rel_uncertainties))
            plausible_fraction = float(
                sum(1 for rel in rel_uncertainties if rel <= 0.5) / len(rel_uncertainties)
            )
        else:
            mean_rel_unc = 0.0
            plausible_fraction = 0.0

        metrics.append(
            _metric(
                name="d_spacing_uncertainty_coverage",
                value=coverage,
                unit="fraction",
                method="diffraction_result_uncertainty_scan",
                direction="higher_is_better",
                details={
                    "with_uncertainty": uncertainty_present_count,
                    "total_d_spacings": d_spacing_count,
                },
            )
        )
        metrics.append(
            _metric(
                name="d_spacing_uncertainty_mean_relative",
                value=mean_rel_unc,
                unit="fraction",
                method="diffraction_result_uncertainty_scan",
                direction="lower_is_better",
                details={"sample_count": len(rel_uncertainties)},
            )
        )
        metrics.append(
            _metric(
                name="d_spacing_uncertainty_plausible_fraction",
                value=plausible_fraction,
                unit="fraction",
                method="diffraction_result_uncertainty_scan",
                direction="higher_is_better",
                details={
                    "criterion": "relative_uncertainty <= 0.5",
                    "sample_count": len(rel_uncertainties),
                },
            )
        )
        metrics.append(
            _metric(
                name="d_spacing_uncertainty_outlier_count",
                value=outlier_count,
                unit="count",
                method="diffraction_result_uncertainty_scan",
                direction="lower_is_better",
                details={"criterion": "relative_uncertainty > 0.5"},
            )
        )
    metrics.append(
        _metric(
            name="phase_candidate_count",
            value=phase_count,
            unit="count",
            method="diffraction_result",
            direction="higher_is_better",
        )
    )
    if diffraction is not None:
        indexed_nonzero = sum(
            1
            for spot in diffraction.indexed_spots
            if (int(spot.h), int(spot.k), int(spot.l)) != (0, 0, 0)
        )
        indexed_fraction = indexed_nonzero / peak_count if peak_count > 0 else 0.0
        metrics.append(
            _metric(
                name="b3_b4_indexed_nonzero_fraction",
                value=indexed_fraction,
                unit="fraction",
                method="diffraction.indexed_spots_scan",
                direction="higher_is_better",
                details={
                    "indexed_nonzero": indexed_nonzero,
                    "peak_count": peak_count,
                },
            )
        )
        fft_payload = diffraction.provenance.get("fft", {})
        if isinstance(fft_payload, dict):
            center_payload = fft_payload.get("center", {})
            if isinstance(center_payload, dict) and isinstance(
                center_payload.get("offset_px"),
                (int, float),
            ):
                metrics.append(
                    _metric(
                        name="diffraction_center_offset_px",
                        value=float(center_payload["offset_px"]),
                        unit="pixel",
                        method="fft.center.offset_px",
                        direction="lower_is_better",
                    )
                )
        consistency_payload = diffraction.provenance.get("b3_b4_consistency", {})
        top_phase_score = None
        if isinstance(consistency_payload, dict) and isinstance(
            consistency_payload.get("top_phase_score"),
            (int, float),
        ):
            top_phase_score = float(consistency_payload["top_phase_score"])
            score_method = "diffraction.provenance.b3_b4_consistency"
        else:
            candidate_scores = [
                float(item.score)
                for item in diffraction.phase_candidates
                if isinstance(item.score, (int, float))
            ]
            if candidate_scores:
                top_phase_score = max(candidate_scores)
                score_method = "diffraction.phase_candidates"
            else:
                score_method = "diffraction.phase_candidates"
        if top_phase_score is not None:
            metrics.append(
                _metric(
                    name="b3_b4_top_phase_score",
                    value=float(top_phase_score),
                    unit="score",
                    method=score_method,
                    direction="higher_is_better",
                )
            )

    atom_count = len(report.atom_lattice.atoms) if report.atom_lattice is not None else 0
    metrics.append(
        _metric(
            name="atom_coordinate_count",
            value=atom_count,
            unit="count",
            method="atom_lattice",
            direction="higher_is_better",
        )
    )

    checks = evaluate_bc_interface_completeness(report)
    total_checks = len(checks)
    pass_checks = sum(1 for item in checks if item["status"] == "pass")
    pass_rate = (pass_checks / total_checks) if total_checks > 0 else 0.0
    metrics.append(
        _metric(
            name="bc_interface_pass_rate",
            value=pass_rate,
            unit="fraction",
            method="reporting.interface_contracts.evaluate_bc_interface_completeness",
            direction="higher_is_better",
            details={"pass": pass_checks, "total": total_checks},
        )
    )

    if report.dataset is None:
        warnings.append("Validation notice: TEMDataset is missing from AnalysisReport.")
    if diffraction is None:
        warnings.append("Validation notice: DiffractionResult is missing from AnalysisReport.")

    return metrics, warnings

