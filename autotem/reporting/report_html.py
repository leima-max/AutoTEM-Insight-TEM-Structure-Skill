"""A3 HTML report rendering with configurable sections and interface checks."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from autotem.core import AnalysisReport, OutputArtifact
from autotem.reporting.interface_contracts import evaluate_bc_interface_completeness

_DEFAULT_SECTIONS: tuple[str, ...] = (
    "overview",
    "workflow_steps",
    "workflow_details",
    "interface_checks",
    "figures",
    "tables",
    "warnings",
    "provenance",
)
_DEFAULT_PDF_SECTIONS: tuple[str, ...] = (
    "cover",
    "overview",
    "workflow_steps",
    "interface_checks",
    "warnings",
    "provenance",
    "figures",
)


def _as_rel(path: str, base: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def _normalize_sections(requested: list[str] | None) -> list[str]:
    if not requested:
        return list(_DEFAULT_SECTIONS)
    normalized: list[str] = []
    allowed = set(_DEFAULT_SECTIONS)
    for item in requested:
        name = str(item).strip().lower()
        if name in allowed and name not in normalized:
            normalized.append(name)
    return normalized or list(_DEFAULT_SECTIONS)


def _normalize_pdf_sections(requested: list[str] | None) -> list[str]:
    if not requested:
        return list(_DEFAULT_PDF_SECTIONS)
    normalized: list[str] = []
    allowed = set(_DEFAULT_PDF_SECTIONS)
    for item in requested:
        name = str(item).strip().lower()
        if name in allowed and name not in normalized:
            normalized.append(name)
    return normalized or list(_DEFAULT_PDF_SECTIONS)


def _step_module_snapshot(report: AnalysisReport, step_name: str) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "load": report.dataset.model_dump(mode="json") if report.dataset is not None else None,
        "assess_quality": report.quality.model_dump(mode="json")
        if report.quality is not None
        else None,
        "calibrate": report.calibration.model_dump(mode="json")
        if report.calibration is not None
        else None,
        "preprocess": report.preprocessing.model_dump(mode="json")
        if report.preprocessing is not None
        else None,
        "fft": report.diffraction.model_dump(mode="json")
        if report.diffraction is not None
        else None,
        "diffraction_peaks": report.diffraction.model_dump(mode="json")
        if report.diffraction is not None
        else None,
        "index_diffraction": report.diffraction.model_dump(mode="json")
        if report.diffraction is not None
        else None,
        "phase_match": report.diffraction.model_dump(mode="json")
        if report.diffraction is not None
        else None,
        "simulate": report.simulation.model_dump(mode="json")
        if report.simulation is not None
        else None,
        "detect_atoms": report.atom_lattice.model_dump(mode="json")
        if report.atom_lattice is not None
        else None,
        "run_gpa": report.strain.model_dump(mode="json") if report.strain is not None else None,
        "report": {
            "figure_count": len(report.figures),
            "table_count": len(report.tables),
            "artifact_count": len(report.artifacts),
            "warning_count": len(report.warnings),
        },
    }
    payload = mapping.get(step_name)
    if isinstance(payload, dict):
        return payload
    return {}


def _render_workflow_details(report: AnalysisReport, open_by_default: bool) -> str:
    if report.workflow is None:
        return "<p>No workflow steps recorded.</p>"

    blocks: list[str] = []
    for step in report.workflow.steps:
        snapshot = _step_module_snapshot(report, step.name)
        snapshot_json = html.escape(json.dumps(snapshot, indent=2, sort_keys=True))
        step_warnings = "".join(f"<li>{html.escape(item)}</li>" for item in step.warnings)
        if not step_warnings:
            step_warnings = "<li>None</li>"
        open_attr = " open" if open_by_default or step.status in {"warning", "failed"} else ""
        blocks.append(
            "<details class=\"step-detail\""
            f"{open_attr}>"
            "<summary>"
            f"{html.escape(step.name)} | status={html.escape(step.status)} "
            f"| warnings={len(step.warnings)}"
            "</summary>"
            "<div class=\"detail-grid\">"
            "<div>"
            "<h3>Warnings</h3>"
            f"<ul>{step_warnings}</ul>"
            "</div>"
            "<div>"
            "<h3>Snapshot</h3>"
            f"<pre>{snapshot_json}</pre>"
            "</div>"
            "</div>"
            "</details>"
        )
    return "".join(blocks)


def export_html_report(
    report: AnalysisReport,
    output_dir: str | Path,
    *,
    options: dict[str, Any] | None = None,
) -> OutputArtifact:
    """Export a configurable HTML report for A3+ stages."""
    output_dir = Path(output_dir)
    report_path = output_dir / "report.html"
    options = options or {}
    sections = _normalize_sections(options.get("sections"))
    open_step_sections = bool(options.get("open_step_sections", False))

    workflow_name = report.workflow.name if report.workflow is not None else "unknown"
    dataset_path = report.dataset.source_path if report.dataset is not None else "unknown"
    step_rows = ""
    if report.workflow is not None:
        step_rows = "".join(
            "<tr>"
            f"<td>{html.escape(step.name)}</td>"
            f"<td>{html.escape(step.status)}</td>"
            f"<td>{len(step.warnings)}</td>"
            "</tr>"
            for step in report.workflow.steps
        )
    if not step_rows:
        step_rows = "<tr><td colspan='3'>No workflow steps recorded.</td></tr>"

    warnings_html = "".join(f"<li>{html.escape(item)}</li>" for item in report.warnings)
    if not warnings_html:
        warnings_html = "<li>None</li>"

    figures_html = "".join(
        "<figure>"
        f"<img src=\"{html.escape(_as_rel(figure.uri, output_dir))}\" "
        f"alt=\"{html.escape(figure.label)}\" />"
        f"<figcaption><strong>{html.escape(figure.label)}</strong>: "
        f"{html.escape(figure.caption)}</figcaption>"
        "</figure>"
        for figure in report.figures
    )
    if not figures_html:
        figures_html = "<p>No figures exported.</p>"

    tables_html = "".join(
        "<li>"
        f"{html.escape(table.label)}"
        + (
            f" - <a href=\"{html.escape(_as_rel(table.uri, output_dir))}\">CSV</a>"
            if table.uri
            else ""
        )
        + f" (rows: {len(table.rows)})"
        "</li>"
        for table in report.tables
    )
    if not tables_html:
        tables_html = "<li>No tables exported.</li>"

    provenance_rows = "".join(
        "<tr>"
        f"<th>{html.escape(str(key))}</th>"
        f"<td>{html.escape(str(value))}</td>"
        "</tr>"
        for key, value in report.provenance.items()
    )
    if not provenance_rows:
        provenance_rows = "<tr><td colspan='2'>No provenance details.</td></tr>"

    interface_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['owner'])}</td>"
        f"<td>{html.escape(item['item'])}</td>"
        f"<td>{html.escape(item['status'])}</td>"
        f"<td>{html.escape(item['detail'])}</td>"
        "</tr>"
        for item in evaluate_bc_interface_completeness(report)
    )
    if not interface_rows:
        interface_rows = "<tr><td colspan='4'>No interface checks.</td></tr>"

    workflow_details_html = _render_workflow_details(report, open_step_sections)

    section_blocks: list[str] = []
    if "overview" in sections:
        section_blocks.append(
            "<div class=\"card\">"
            "<h2>Overview</h2>"
            "<table>"
            f"<tr><th>Workflow</th><td>{html.escape(workflow_name)}</td></tr>"
            f"<tr><th>Input</th><td>{html.escape(dataset_path)}</td></tr>"
            f"<tr><th>Warnings</th><td>{len(report.warnings)}</td></tr>"
            f"<tr><th>Figures</th><td>{len(report.figures)}</td></tr>"
            f"<tr><th>Tables</th><td>{len(report.tables)}</td></tr>"
            f"<tr><th>Artifacts</th><td>{len(report.artifacts)}</td></tr>"
            "</table>"
            "</div>"
        )
    if "workflow_steps" in sections:
        section_blocks.append(
            "<div class=\"card\">"
            "<h2>Workflow Steps</h2>"
            "<table>"
            "<tr><th>Step</th><th>Status</th><th>Warnings</th></tr>"
            f"{step_rows}"
            "</table>"
            "</div>"
        )
    if "workflow_details" in sections:
        section_blocks.append(
            "<div class=\"card\">"
            "<h2>Workflow Step Details</h2>"
            f"{workflow_details_html}"
            "</div>"
        )
    if "interface_checks" in sections:
        section_blocks.append(
            "<div class=\"card\">"
            "<h2>Interface Completeness (B/C)</h2>"
            "<table>"
            "<tr><th>Owner</th><th>Contract Item</th><th>Status</th><th>Detail</th></tr>"
            f"{interface_rows}"
            "</table>"
            "</div>"
        )
    if "figures" in sections:
        section_blocks.append(
            "<div class=\"card\">"
            "<h2>Figures</h2>"
            f"<div class=\"gallery\">{figures_html}</div>"
            "</div>"
        )
    if "tables" in sections:
        section_blocks.append(
            "<div class=\"card\">"
            "<h2>Tables</h2>"
            f"<ul>{tables_html}</ul>"
            "</div>"
        )
    if "warnings" in sections:
        section_blocks.append(
            "<div class=\"card\">"
            "<h2>Warnings</h2>"
            f"<ul>{warnings_html}</ul>"
            "</div>"
        )
    if "provenance" in sections:
        section_blocks.append(
            "<div class=\"card\">"
            "<h2>Provenance</h2>"
            f"<table>{provenance_rows}</table>"
            "</div>"
        )

    body = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>AutoTEM A3 Report</title>
    <style>
      :root {{
        --bg: #f8f9fb;
        --card: #ffffff;
        --ink: #1f2937;
        --sub: #5b6472;
        --line: #d8dee9;
        --head: #edf2f7;
      }}
      body {{
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        margin: 0;
        background: var(--bg);
        color: var(--ink);
      }}
      .page {{
        max-width: 1080px;
        margin: 24px auto;
        padding: 0 16px 24px 16px;
      }}
      .card {{
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 16px;
        margin-top: 16px;
      }}
      h1, h2, h3 {{
        margin: 0 0 12px 0;
      }}
      .meta {{
        color: var(--sub);
      }}
      table {{
        border-collapse: collapse;
        width: 100%;
      }}
      th, td {{
        border: 1px solid var(--line);
        padding: 8px;
        text-align: left;
        vertical-align: top;
      }}
      th {{
        background: var(--head);
      }}
      .gallery {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 12px;
      }}
      figure {{
        margin: 0;
      }}
      img {{
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 8px;
      }}
      figcaption {{
        margin-top: 6px;
        color: var(--sub);
        font-size: 0.92rem;
      }}
      details.step-detail {{
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
      }}
      details.step-detail > summary {{
        cursor: pointer;
        font-weight: 600;
      }}
      .detail-grid {{
        margin-top: 8px;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 10px;
      }}
      pre {{
        white-space: pre-wrap;
        word-break: break-word;
        margin: 0;
        font-size: 12px;
        background: #f7fafc;
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 8px;
      }}
    </style>
  </head>
  <body>
    <div class="page">
      <div class="card">
        <h1>AutoTEM A3 Analysis Report</h1>
        <div class="meta">{html.escape(report.summary)}</div>
      </div>
      {"".join(section_blocks)}
    </div>
  </body>
</html>
"""
    report_path.write_text(body, encoding="utf-8")
    return OutputArtifact(
        kind="html",
        uri=str(report_path),
        label="HTML report",
        mime_type="text/html",
    )


def _export_pdf_report_internal(
    report: AnalysisReport,
    output_dir: str | Path,
    *,
    options: dict[str, Any] | None,
) -> OutputArtifact:
    output_dir = Path(output_dir)
    pdf_path = output_dir / "report.pdf"
    options = options or {}
    sections = _normalize_pdf_sections(options.get("sections"))
    max_figure_pages_raw = options.get("max_figure_pages", 8)
    try:
        max_figure_pages = max(0, int(max_figure_pages_raw))
    except (TypeError, ValueError):
        max_figure_pages = 8
    cover_fields = options.get("cover_fields")
    if not isinstance(cover_fields, dict):
        cover_fields = {}

    with PdfPages(pdf_path) as pdf:
        if "cover" in sections:
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=140)
            ax.axis("off")
            lines = [
                "AutoTEM A3 Analysis Report",
                "",
                f"Summary: {report.summary}",
                f"Warnings: {len(report.warnings)}",
                f"Figures: {len(report.figures)}",
                f"Tables: {len(report.tables)}",
                f"Artifacts: {len(report.artifacts)}",
            ]
            if report.workflow is not None:
                lines.append(f"Workflow: {report.workflow.name}")
            if cover_fields:
                lines.append("")
                lines.append("Cover Fields:")
                for key, value in cover_fields.items():
                    lines.append(f"  - {key}: {value}")
            ax.text(
                0.02,
                0.98,
                "\n".join(lines),
                va="top",
                ha="left",
                fontsize=10,
                family="monospace",
            )
            fig.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.02)
            pdf.savefig(fig)
            plt.close(fig)

        if "overview" in sections:
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=140)
            ax.axis("off")
            workflow_name = report.workflow.name if report.workflow is not None else "unknown"
            dataset_path = report.dataset.source_path if report.dataset is not None else "unknown"
            lines = [
                "Overview",
                "",
                f"Workflow: {workflow_name}",
                f"Input: {dataset_path}",
                f"Warnings: {len(report.warnings)}",
                f"Figures: {len(report.figures)}",
                f"Tables: {len(report.tables)}",
            ]
            ax.text(
                0.02,
                0.98,
                "\n".join(lines),
                va="top",
                ha="left",
                fontsize=10,
                family="monospace",
            )
            fig.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.02)
            pdf.savefig(fig)
            plt.close(fig)

        if "workflow_steps" in sections and report.workflow is not None:
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=140)
            ax.axis("off")
            lines = ["Workflow Steps", ""]
            for step in report.workflow.steps:
                lines.append(
                    f"- {step.name}: {step.status} (warnings={len(step.warnings)})"
                )
            ax.text(
                0.02,
                0.98,
                "\n".join(lines),
                va="top",
                ha="left",
                fontsize=10,
                family="monospace",
            )
            fig.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.02)
            pdf.savefig(fig)
            plt.close(fig)

        if "interface_checks" in sections:
            checks = evaluate_bc_interface_completeness(report)
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=140)
            ax.axis("off")
            lines = ["Interface Completeness (B/C)", ""]
            for item in checks:
                lines.append(
                    f"[{item['status']}] {item['owner']}.{item['item']} - {item['detail']}"
                )
            ax.text(
                0.02,
                0.98,
                "\n".join(lines),
                va="top",
                ha="left",
                fontsize=9,
                family="monospace",
            )
            fig.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.02)
            pdf.savefig(fig)
            plt.close(fig)

        if "warnings" in sections:
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=140)
            ax.axis("off")
            lines = ["Warnings", ""]
            if report.warnings:
                for warning in report.warnings:
                    lines.append(f"- {warning}")
            else:
                lines.append("- None")
            ax.text(
                0.02,
                0.98,
                "\n".join(lines),
                va="top",
                ha="left",
                fontsize=10,
                family="monospace",
            )
            fig.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.02)
            pdf.savefig(fig)
            plt.close(fig)

        if "provenance" in sections:
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=140)
            ax.axis("off")
            lines = ["Provenance", ""]
            for key, value in report.provenance.items():
                lines.append(f"- {key}: {value}")
            ax.text(
                0.02,
                0.98,
                "\n".join(lines),
                va="top",
                ha="left",
                fontsize=9,
                family="monospace",
            )
            fig.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.02)
            pdf.savefig(fig)
            plt.close(fig)

        if "figures" in sections:
            for figure in report.figures[:max_figure_pages]:
                figure_path = Path(figure.uri)
                if not figure_path.exists():
                    continue
                try:
                    image = plt.imread(figure_path)
                except Exception:  # noqa: BLE001
                    continue
                fig_img, ax_img = plt.subplots(figsize=(8.27, 11.69), dpi=140)
                ax_img.imshow(image)
                ax_img.set_title(f"{figure.label}\n{figure.caption}", fontsize=10)
                ax_img.axis("off")
                fig_img.tight_layout()
                pdf.savefig(fig_img)
                plt.close(fig_img)

    return OutputArtifact(
        kind="other",
        uri=str(pdf_path),
        label="PDF report",
        mime_type="application/pdf",
    )


def export_pdf_report(
    report: AnalysisReport,
    output_dir: str | Path,
    *,
    options: dict[str, Any] | None = None,
) -> OutputArtifact:
    """Export a configurable PDF report using available figure outputs."""
    return _export_pdf_report_internal(report, output_dir, options=options)
