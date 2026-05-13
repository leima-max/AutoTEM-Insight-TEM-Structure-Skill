"""Interface completeness checks for Member B/C integration contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, TypedDict

from autotem.core import AnalysisReport, OutputArtifact, ReportTable
from autotem.reporting.schema_models import InterfaceCompletenessContractModel


class InterfaceCheck(TypedDict):
    owner: str
    item: str
    status: str
    detail: str


def evaluate_bc_interface_completeness(report: AnalysisReport) -> list[InterfaceCheck]:
    """Evaluate baseline completeness for B/C output contracts."""
    checks: list[InterfaceCheck] = []

    has_diffraction_overlay = any(
        figure.figure_type == "diffraction_overlay" for figure in report.figures
    )
    has_atom_overlay = any("atom_overlay" in figure.figure_type for figure in report.figures) or (
        report.atom_lattice is not None
        and report.atom_lattice.overlay is not None
        and bool(report.atom_lattice.overlay.uri)
    )

    def add(owner: str, item: str, ok: bool, detail: str) -> None:
        checks.append(
            InterfaceCheck(
                owner=owner,
                item=item,
                status="pass" if ok else "missing",
                detail=detail,
            )
        )

    add(
        "B",
        "TEMDataset",
        report.dataset is not None,
        "Input dataset object for downstream calibration/diffraction.",
    )
    add(
        "B",
        "DiffractionResult",
        report.diffraction is not None,
        "Reciprocal-space container for peaks, d-spacing, and phase candidates.",
    )
    add(
        "B",
        "Diffraction peaks",
        report.diffraction is not None and len(report.diffraction.peaks) > 0,
        "At least one detected peak is expected.",
    )
    add(
        "B",
        "d-spacing table",
        report.diffraction is not None and len(report.diffraction.d_spacings) > 0,
        "d-spacing entries should be exported for phase matching.",
    )
    add(
        "B",
        "Phase candidates",
        report.diffraction is not None and len(report.diffraction.phase_candidates) > 0,
        "Top-N phase candidates for downstream review.",
    )
    add(
        "B",
        "Diffraction overlay figure",
        has_diffraction_overlay,
        "FFT peak overlay in figures.",
    )

    add(
        "C",
        "AtomLattice",
        report.atom_lattice is not None,
        "Atom lattice object for atom/strain pipelines.",
    )
    add(
        "C",
        "Atom coordinates",
        report.atom_lattice is not None and len(report.atom_lattice.atoms) > 0,
        "At least one atom coordinate expected when atom step is active.",
    )
    add(
        "C",
        "Atom overlay figure",
        bool(has_atom_overlay),
        "Atom overlay visual for reportability.",
    )
    add(
        "C",
        "StrainResult",
        report.strain is not None,
        "Structured strain result object.",
    )
    add(
        "C",
        "Strain map availability",
        report.strain is not None
        and any(
            field is not None
            for field in [
                report.strain.strain_xx,
                report.strain.strain_yy,
                report.strain.strain_xy,
                report.strain.rotation_xy,
            ]
        ),
        "At least one strain/rotation map should be provided.",
    )

    return checks


def interface_check_rows(report: AnalysisReport) -> list[dict[str, str]]:
    """Return normalized interface-check rows for CSV/JSON contract exports."""
    rows: list[dict[str, str]] = []
    for check in evaluate_bc_interface_completeness(report):
        rows.append(
            {
                "owner": check["owner"],
                "item": check["item"],
                "status": check["status"],
                "detail": check["detail"],
            }
        )
    return rows


def summarize_bc_interface_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Summarize pass/missing counts for B/C interface checks."""
    total = len(rows)
    passed = sum(1 for row in rows if row["status"] == "pass")
    missing = sum(1 for row in rows if row["status"] == "missing")
    by_owner: dict[str, dict[str, int]] = {}
    for row in rows:
        owner = row["owner"]
        bucket = by_owner.setdefault(owner, {"total": 0, "passed": 0, "missing": 0})
        bucket["total"] += 1
        if row["status"] == "pass":
            bucket["passed"] += 1
        else:
            bucket["missing"] += 1
    return {
        "total": total,
        "passed": passed,
        "missing": missing,
        "pass_rate": (passed / total) if total > 0 else 0.0,
        "by_owner": by_owner,
    }


def export_bc_interface_contract_csv(
    report: AnalysisReport,
    output_dir: str | Path,
) -> tuple[OutputArtifact, ReportTable]:
    """Export B/C interface completeness contract table as CSV."""
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    path = tables_dir / "interface_completeness_bc.csv"
    rows = interface_check_rows(report)
    fieldnames = ["owner", "item", "status", "detail"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    artifact = OutputArtifact(
        kind="table",
        uri=str(path),
        label="B/C Interface Completeness",
        mime_type="text/csv",
    )
    table = ReportTable(
        label="B/C Interface Completeness",
        uri=str(path),
        rows=rows,
        table_type="interface_completeness_bc",
    )
    return artifact, table


def export_bc_interface_contract_json(
    report: AnalysisReport,
    output_dir: str | Path,
) -> OutputArtifact:
    """Export B/C interface completeness contract payload as JSON."""
    output_dir = Path(output_dir)
    json_dir = output_dir / "json"
    path = json_dir / "interface_completeness_bc.json"
    rows = interface_check_rows(report)
    summary = summarize_bc_interface_rows(rows)
    payload = InterfaceCompletenessContractModel(
        schema_version=report.schema_version,
        summary={
            "total": int(summary["total"]),
            "passed": int(summary["passed"]),
            "missing": int(summary["missing"]),
            "pass_rate": float(summary["pass_rate"]),
        },
        checks=rows,
    ).model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return OutputArtifact(
        kind="json",
        uri=str(path),
        label="B/C interface completeness contract",
        mime_type="application/json",
    )
