"""A3 table export helpers."""

from __future__ import annotations

import csv
from pathlib import Path

from autotem.core import AnalysisReport, OutputArtifact, ReportTable
from autotem.reporting.interface_contracts import export_bc_interface_contract_csv


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _as_mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _list_count(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


def _phase_match_method(candidate: object) -> str:
    mean_d_error = getattr(candidate, "mean_d_error", None)
    mean_angle_error = getattr(candidate, "mean_angle_error", None)
    method = getattr(mean_d_error, "method", "") or getattr(mean_angle_error, "method", "")
    return str(method or "")


def _phase_match_rotation(candidate: object) -> object:
    mean_d_error = getattr(candidate, "mean_d_error", None)
    provenance = _as_mapping(getattr(mean_d_error, "provenance", {}))
    return provenance.get("rotation_deg", "")


def _phase_match_raw_score(candidate: object) -> object:
    mean_d_error = getattr(candidate, "mean_d_error", None)
    provenance = _as_mapping(getattr(mean_d_error, "provenance", {}))
    return provenance.get("raw_diffsims_score", "")


def _candidate_keys(candidate: object) -> set[str]:
    values = (getattr(candidate, "identifier", None), getattr(candidate, "name", None))
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _pattern_keys(pattern: object) -> set[str]:
    structure_id = getattr(pattern, "structure_id", None)
    return {str(structure_id).strip().lower()} if str(structure_id).strip() else set()


def _best_pattern_for_candidate(report: AnalysisReport, candidate: object) -> object | None:
    if report.simulation is None:
        return None
    keys = _candidate_keys(candidate)
    fallback = None
    for pattern in report.simulation.patterns:
        if fallback is None:
            fallback = pattern
        if keys.intersection(_pattern_keys(pattern)):
            phase_match = _as_mapping(pattern.simulation_parameters.get("phase_match", {}))
            if phase_match:
                return pattern
            fallback = pattern
    return fallback


def _phase_matching_rows_by_name(report: AnalysisReport) -> dict[str, dict[str, object]]:
    if report.simulation is None:
        return {}
    provenance = _as_mapping(report.simulation.provenance.get("phase_matching", {}))
    per_phase = provenance.get("per_phase", [])
    if not isinstance(per_phase, list):
        return {}
    rows: dict[str, dict[str, object]] = {}
    for row in per_phase:
        if not isinstance(row, dict):
            continue
        name = row.get("phase_name")
        if str(name).strip():
            rows[str(name)] = dict(row)
    return rows


def _simulation_phase_row(
    *,
    report: AnalysisReport,
    candidate: object,
    adapter_provenance: dict[str, object],
    phase_matching_provenance: dict[str, object],
    per_phase_rows: dict[str, dict[str, object]],
) -> dict[str, object]:
    pattern = _best_pattern_for_candidate(report, candidate)
    pattern_parameters = (
        _as_mapping(getattr(pattern, "simulation_parameters", {}))
        if pattern is not None
        else {}
    )
    pattern_match = _as_mapping(pattern_parameters.get("phase_match", {}))
    per_phase = per_phase_rows.get(str(getattr(candidate, "name", "")), {})
    mean_d_error = getattr(candidate, "mean_d_error", None)
    mean_angle_error = getattr(candidate, "mean_angle_error", None)
    engine = pattern_parameters.get("engine") or report.simulation.provenance.get("method", "")
    return {
        "name": getattr(candidate, "name", ""),
        "formula": getattr(candidate, "formula", ""),
        "source": getattr(candidate, "source", ""),
        "identifier": getattr(candidate, "identifier", ""),
        "engine": engine,
        "adapter_active": adapter_provenance.get("active", ""),
        "phase_matching_active": phase_matching_provenance.get("active", ""),
        "phase_matching_method": phase_matching_provenance.get(
            "method",
            _phase_match_method(candidate),
        ),
        "score": getattr(candidate, "score", ""),
        "matched_spot_count": getattr(candidate, "matched_spot_count", ""),
        "mean_d_error": getattr(mean_d_error, "value", ""),
        "mean_d_error_unit": getattr(mean_d_error, "unit", ""),
        "mean_angle_error": getattr(mean_angle_error, "value", ""),
        "mean_angle_error_unit": getattr(mean_angle_error, "unit", ""),
        "measured_spot_count": per_phase.get(
            "measured_spot_count",
            phase_matching_provenance.get("measured_spot_count", ""),
        ),
        "simulated_pattern_count": per_phase.get(
            "simulated_pattern_count",
            phase_matching_provenance.get("simulated_pattern_count", ""),
        ),
        "simulated_spot_count": per_phase.get(
            "simulated_spot_count",
            pattern_parameters.get("simulated_spot_count", ""),
        ),
        "best_pattern_index": per_phase.get("best_pattern_index", ""),
        "phase_match_rotation_deg": pattern_match.get(
            "rotation_deg",
            per_phase.get("rotation_deg", _phase_match_rotation(candidate)),
        ),
        "simulation_rotation_angle_deg": pattern_parameters.get(
            "rotation_angle_deg",
            per_phase.get("simulation_rotation_angle_deg", ""),
        ),
        "zone_axis_index": per_phase.get(
            "zone_axis_index",
            pattern_parameters.get("zone_axis_index", ""),
        ),
        "d_tolerance_rel": phase_matching_provenance.get("d_tolerance_rel", ""),
        "angle_tolerance_deg": phase_matching_provenance.get("angle_tolerance_deg", ""),
        "zone_axis": per_phase.get("zone_axis", getattr(candidate, "zone_axis", "")),
    }


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

        indexed_rows = [
            {
                "peak_index": spot.peak_index,
                "h": spot.h,
                "k": spot.k,
                "l": spot.l,
                "d_spacing": spot.d_spacing.value if spot.d_spacing else "",
                "d_spacing_unit": spot.d_spacing.unit if spot.d_spacing else "",
                "error": spot.error.value if spot.error else "",
                "error_unit": spot.error.unit if spot.error else "",
                "error_method": spot.error.method if spot.error else "",
                "confidence": spot.confidence,
            }
            for spot in report.diffraction.indexed_spots
        ]
        indexed_path = tables_dir / "indexed_spots.csv"
        _write_csv(
            indexed_path,
            [
                "peak_index",
                "h",
                "k",
                "l",
                "d_spacing",
                "d_spacing_unit",
                "error",
                "error_unit",
                "error_method",
                "confidence",
            ],
            indexed_rows,
        )
        artifacts.append(
            OutputArtifact(
                kind="table",
                uri=str(indexed_path),
                label="Indexed Spots",
                mime_type="text/csv",
            )
        )
        report_tables.append(
            ReportTable(
                label="Indexed Spots",
                uri=str(indexed_path),
                rows=indexed_rows,
                table_type="indexed_spots",
            )
        )

        phase_rows = [
            {
                "name": candidate.name,
                "formula": candidate.formula,
                "source": candidate.source,
                "identifier": candidate.identifier,
                "score": candidate.score,
                "matched_spot_count": candidate.matched_spot_count,
                "mean_d_error": candidate.mean_d_error.value if candidate.mean_d_error else "",
                "mean_d_error_unit": candidate.mean_d_error.unit if candidate.mean_d_error else "",
                "mean_d_error_method": (
                    candidate.mean_d_error.method if candidate.mean_d_error else ""
                ),
                "mean_angle_error": (
                    candidate.mean_angle_error.value if candidate.mean_angle_error else ""
                ),
                "mean_angle_error_unit": (
                    candidate.mean_angle_error.unit if candidate.mean_angle_error else ""
                ),
                "mean_angle_error_method": (
                    candidate.mean_angle_error.method if candidate.mean_angle_error else ""
                ),
                "lattice_a": candidate.lattice_parameters.get("a", ""),
                "lattice_b": candidate.lattice_parameters.get("b", ""),
                "lattice_c": candidate.lattice_parameters.get("c", ""),
                "lattice_unit": candidate.lattice_parameters.get("unit", ""),
                "space_group": candidate.lattice_parameters.get("space_group", ""),
                "space_group_number": candidate.lattice_parameters.get(
                    "space_group_number",
                    "",
                ),
                "atom_site_count": _list_count(
                    candidate.lattice_parameters.get("atom_sites", [])
                ),
                "mp_rank": _as_mapping(
                    candidate.lattice_parameters.get("materials_project", {})
                ).get("rank", ""),
                "mp_energy_above_hull": _as_mapping(
                    candidate.lattice_parameters.get("materials_project", {})
                ).get("energy_above_hull", ""),
                "mp_is_stable": _as_mapping(
                    candidate.lattice_parameters.get("materials_project", {})
                ).get("is_stable", ""),
                "mp_theoretical": _as_mapping(
                    candidate.lattice_parameters.get("materials_project", {})
                ).get("theoretical", ""),
                "mp_deprecated": _as_mapping(
                    candidate.lattice_parameters.get("materials_project", {})
                ).get("deprecated", ""),
                "mp_site_count": _as_mapping(
                    candidate.lattice_parameters.get("materials_project", {})
                ).get("site_count", ""),
                "phase_match_method": _phase_match_method(candidate),
                "phase_match_rotation_deg": _phase_match_rotation(candidate),
                "phase_match_raw_diffsims_score": _phase_match_raw_score(candidate),
            }
            for candidate in report.diffraction.phase_candidates
        ]
        phases_path = tables_dir / "phase_candidates.csv"
        _write_csv(
            phases_path,
            [
                "name",
                "formula",
                "source",
                "identifier",
                "score",
                "matched_spot_count",
                "mean_d_error",
                "mean_d_error_unit",
                "mean_d_error_method",
                "mean_angle_error",
                "mean_angle_error_unit",
                "mean_angle_error_method",
                "lattice_a",
                "lattice_b",
                "lattice_c",
                "lattice_unit",
                "space_group",
                "space_group_number",
                "atom_site_count",
                "mp_rank",
                "mp_energy_above_hull",
                "mp_is_stable",
                "mp_theoretical",
                "mp_deprecated",
                "mp_site_count",
                "phase_match_method",
                "phase_match_rotation_deg",
                "phase_match_raw_diffsims_score",
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

    if report.simulation is not None:
        adapter_provenance = _as_mapping(report.simulation.provenance.get("adapter", {}))
        phase_matching_provenance = _as_mapping(
            report.simulation.provenance.get("phase_matching", {})
        )
        per_phase_rows = _phase_matching_rows_by_name(report)
        simulation_phase_rows = [
            _simulation_phase_row(
                report=report,
                candidate=candidate,
                adapter_provenance=adapter_provenance,
                phase_matching_provenance=phase_matching_provenance,
                per_phase_rows=per_phase_rows,
            )
            for candidate in report.simulation.matched_phase_candidates
        ]
        simulation_phase_path = tables_dir / "simulation_phase_matches.csv"
        _write_csv(
            simulation_phase_path,
            [
                "name",
                "formula",
                "source",
                "identifier",
                "engine",
                "adapter_active",
                "phase_matching_active",
                "phase_matching_method",
                "score",
                "matched_spot_count",
                "mean_d_error",
                "mean_d_error_unit",
                "mean_angle_error",
                "mean_angle_error_unit",
                "measured_spot_count",
                "simulated_pattern_count",
                "simulated_spot_count",
                "best_pattern_index",
                "phase_match_rotation_deg",
                "simulation_rotation_angle_deg",
                "zone_axis_index",
                "d_tolerance_rel",
                "angle_tolerance_deg",
                "zone_axis",
            ],
            simulation_phase_rows,
        )
        artifacts.append(
            OutputArtifact(
                kind="table",
                uri=str(simulation_phase_path),
                label="Simulation Phase Matches",
                mime_type="text/csv",
            )
        )
        report_tables.append(
            ReportTable(
                label="Simulation Phase Matches",
                uri=str(simulation_phase_path),
                rows=simulation_phase_rows,
                table_type="simulation_phase_matches",
            )
        )
        simulated_spot_rows = []
        for pattern_idx, pattern in enumerate(report.simulation.patterns):
            for spot_idx, spot in enumerate(pattern.spots):
                simulated_spot_rows.append(
                    {
                        "pattern_index": pattern_idx,
                        "structure_id": pattern.structure_id,
                        "rotation_index": pattern.simulation_parameters.get(
                            "rotation_index",
                            "",
                        ),
                        "rotation_angle_deg": pattern.simulation_parameters.get(
                            "rotation_angle_deg",
                            "",
                        ),
                        "zone_axis_index": pattern.simulation_parameters.get(
                            "zone_axis_index",
                            "",
                        ),
                        "zone_axis": pattern.zone_axis or "",
                        "spot_index": spot_idx,
                        "gx": spot.gx.value,
                        "gy": spot.gy.value,
                        "unit": spot.gx.unit,
                        "magnitude": spot.magnitude.value if spot.magnitude else "",
                        "d_spacing": spot.d_spacing.value if spot.d_spacing else "",
                        "d_spacing_unit": spot.d_spacing.unit if spot.d_spacing else "",
                        "intensity": spot.intensity,
                        "h": spot.h,
                        "k": spot.k,
                        "l": spot.l,
                    }
                )
        simulated_spots_path = tables_dir / "simulated_spots.csv"
        _write_csv(
            simulated_spots_path,
            [
                "pattern_index",
                "structure_id",
                "rotation_index",
                "rotation_angle_deg",
                "zone_axis_index",
                "zone_axis",
                "spot_index",
                "gx",
                "gy",
                "unit",
                "magnitude",
                "d_spacing",
                "d_spacing_unit",
                "intensity",
                "h",
                "k",
                "l",
            ],
            simulated_spot_rows,
        )
        artifacts.append(
            OutputArtifact(
                kind="table",
                uri=str(simulated_spots_path),
                label="Simulated Spots",
                mime_type="text/csv",
            )
        )
        report_tables.append(
            ReportTable(
                label="Simulated Spots",
                uri=str(simulated_spots_path),
                rows=simulated_spot_rows,
                table_type="simulated_spots",
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

    interface_artifact, interface_table = export_bc_interface_contract_csv(report, output_dir)
    artifacts.append(interface_artifact)
    report_tables.append(interface_table)
    return artifacts, report_tables
