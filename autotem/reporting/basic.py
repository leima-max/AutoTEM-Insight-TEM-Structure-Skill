"""Baseline report exporters for CSV tables and HTML summary pages."""

from __future__ import annotations

import csv
import html
from pathlib import Path

from autotem.core import AnalysisReport, OutputArtifact


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def export_csv_tables(report: AnalysisReport, output_dir: str | Path) -> list[OutputArtifact]:
    """Export key baseline analysis tables as CSV files."""
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    artifacts: list[OutputArtifact] = []

    if report.diffraction is not None:
        peaks_rows = [
            {
                "index": idx,
                "x": peak.x,
                "y": peak.y,
                "intensity": peak.intensity,
                "confidence": peak.confidence,
            }
            for idx, peak in enumerate(report.diffraction.peaks)
        ]
        peaks_path = tables_dir / "diffraction_peaks.csv"
        _write_csv(peaks_path, ["index", "x", "y", "intensity", "confidence"], peaks_rows)
        artifacts.append(
            OutputArtifact(
                kind="table",
                uri=str(peaks_path),
                label="Diffraction Peaks",
                mime_type="text/csv",
            )
        )

        d_spacing_rows = [
            {
                "index": idx,
                "value": measurement.value,
                "unit": measurement.unit,
                "uncertainty": measurement.uncertainty,
                "method": measurement.method,
                "confidence": measurement.confidence,
            }
            for idx, measurement in enumerate(report.diffraction.d_spacings)
        ]
        d_spacing_path = tables_dir / "d_spacings.csv"
        _write_csv(
            d_spacing_path,
            ["index", "value", "unit", "uncertainty", "method", "confidence"],
            d_spacing_rows,
        )
        artifacts.append(
            OutputArtifact(
                kind="table",
                uri=str(d_spacing_path),
                label="d-Spacings",
                mime_type="text/csv",
            )
        )

        phase_rows = [
            {
                "name": candidate.name,
                "formula": candidate.formula,
                "identifier": candidate.identifier,
                "score": candidate.score,
                "matched_spot_count": candidate.matched_spot_count,
                "mean_d_error": candidate.mean_d_error.value if candidate.mean_d_error else "",
                "mean_d_error_unit": candidate.mean_d_error.unit if candidate.mean_d_error else "",
            }
            for candidate in report.diffraction.phase_candidates
        ]
        phases_path = tables_dir / "phase_candidates.csv"
        _write_csv(
            phases_path,
            [
                "name",
                "formula",
                "identifier",
                "score",
                "matched_spot_count",
                "mean_d_error",
                "mean_d_error_unit",
            ],
            phase_rows,
        )
        artifacts.append(
            OutputArtifact(
                kind="table",
                uri=str(phases_path),
                label="Phase Candidates",
                mime_type="text/csv",
            )
        )

    warnings_rows = [
        {"index": idx, "warning": warning} for idx, warning in enumerate(report.warnings)
    ]
    warnings_path = tables_dir / "warnings.csv"
    _write_csv(warnings_path, ["index", "warning"], warnings_rows)
    artifacts.append(
        OutputArtifact(
            kind="table",
            uri=str(warnings_path),
            label="Warnings",
            mime_type="text/csv",
        )
    )
    return artifacts


def export_html_report(report: AnalysisReport, output_dir: str | Path) -> OutputArtifact:
    """Export a simple baseline HTML report."""
    output_dir = Path(output_dir)
    report_path = output_dir / "report.html"

    workflow_name = report.workflow.name if report.workflow is not None else "unknown"
    dataset_path = report.dataset.source_path if report.dataset is not None else "unknown"
    peak_count = len(report.diffraction.peaks) if report.diffraction is not None else 0
    phase_count = len(report.diffraction.phase_candidates) if report.diffraction is not None else 0

    warnings_html = "".join(f"<li>{html.escape(item)}</li>" for item in report.warnings)
    if not warnings_html:
        warnings_html = "<li>None</li>"

    body = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>AutoTEM A2 Report</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
      h1 {{ margin-bottom: 0.2rem; }}
      .meta {{ color: #4b5563; margin-bottom: 1rem; }}
      table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
      th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; }}
      th {{ background: #f3f4f6; }}
    </style>
  </head>
  <body>
    <h1>AutoTEM A2 Analysis Report</h1>
    <div class="meta">{html.escape(report.summary)}</div>
    <table>
      <tr><th>Workflow</th><td>{html.escape(workflow_name)}</td></tr>
      <tr><th>Input</th><td>{html.escape(dataset_path)}</td></tr>
      <tr><th>Diffraction Peaks</th><td>{peak_count}</td></tr>
      <tr><th>Phase Candidates</th><td>{phase_count}</td></tr>
      <tr><th>Warnings</th><td>{len(report.warnings)}</td></tr>
    </table>
    <h2>Warnings</h2>
    <ul>{warnings_html}</ul>
  </body>
</html>
"""
    report_path.write_text(body, encoding="utf-8")
    return OutputArtifact(
        kind="html",
        uri=str(report_path),
        label="HTML Report",
        mime_type="text/html",
    )
