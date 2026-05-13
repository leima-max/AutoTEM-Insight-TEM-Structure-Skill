"""A3 table export helpers."""

from __future__ import annotations

import csv
from pathlib import Path

from autotem.core import AnalysisReport, OutputArtifact, ReportTable


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def export_csv_tables(report: AnalysisReport, output_dir: str | Path) -> list[OutputArtifact]:
    """Backward-compatible wrapper that returns CSV artifacts only."""
    artifacts, _ = export_csv_tables_with_models(report, output_dir)
    return artifacts


def export_csv_tables_with_models(
    report: AnalysisReport,
    output_dir: str | Path,
) -> tuple[list[OutputArtifact], list[ReportTable]]:
    """Export key analysis tables as CSV files and report-table metadata."""
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    artifacts: list[OutputArtifact] = []
    report_tables: list[ReportTable] = []

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
        report_tables.append(
            ReportTable(
                label="Diffraction Peaks",
                uri=str(peaks_path),
                rows=peaks_rows,
                table_type="diffraction_peaks",
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
        report_tables.append(
            ReportTable(
                label="d-Spacings",
                uri=str(d_spacing_path),
                rows=d_spacing_rows,
                table_type="d_spacings",
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
        report_tables.append(
            ReportTable(
                label="Phase Candidates",
                uri=str(phases_path),
                rows=phase_rows,
                table_type="phase_candidates",
            )
        )

    atom_rows = []
    if report.atom_lattice is not None:
        atom_rows = [
            {
                "index": idx,
                "x": atom.x,
                "y": atom.y,
                "intensity": atom.intensity.value if atom.intensity is not None else "",
                "intensity_unit": atom.intensity.unit if atom.intensity is not None else "",
                "confidence": atom.confidence,
            }
            for idx, atom in enumerate(report.atom_lattice.atoms)
        ]
    atom_path = tables_dir / "atom_coordinates.csv"
    _write_csv(
        atom_path,
        ["index", "x", "y", "intensity", "intensity_unit", "confidence"],
        atom_rows,
    )
    artifacts.append(
        OutputArtifact(
            kind="table",
            uri=str(atom_path),
            label="Atom Coordinates",
            mime_type="text/csv",
        )
    )
    report_tables.append(
        ReportTable(
            label="Atom Coordinates",
            uri=str(atom_path),
            rows=atom_rows,
            table_type="atom_coordinates",
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
    report_tables.append(
        ReportTable(
            label="Warnings",
            uri=str(warnings_path),
            rows=warnings_rows,
            table_type="warnings",
        )
    )
    return artifacts, report_tables
