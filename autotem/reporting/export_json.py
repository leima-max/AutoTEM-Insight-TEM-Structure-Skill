"""A3 JSON exports for report bundles and module snapshots."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from autotem.core import AnalysisReport, OutputArtifact
from autotem.reporting.schema_models import ReportBundleSummaryModel


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_model(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def export_json_bundle(report: AnalysisReport, output_dir: str | Path) -> list[OutputArtifact]:
    """Write per-module JSON snapshots and a bundle summary."""
    output_dir = Path(output_dir)
    json_dir = output_dir / "json"
    artifacts: list[OutputArtifact] = []

    model_exports: dict[str, BaseModel | None] = {
        "workflow": report.workflow,
        "dataset": report.dataset,
        "quality": report.quality,
        "calibration": report.calibration,
        "preprocessing": report.preprocessing,
        "diffraction": report.diffraction,
        "atom_lattice": report.atom_lattice,
        "strain": report.strain,
        "microstructure": report.microstructure,
        "z_contrast": report.z_contrast,
        "spectroscopy": report.spectroscopy,
        "fourdstem": report.fourdstem,
        "simulation": report.simulation,
        "ai_inference": report.ai_inference,
        "benchmark": report.benchmark,
    }

    exported_modules: list[str] = []
    for name, model in model_exports.items():
        if model is None:
            continue
        path = json_dir / f"{name}.json"
        _write_model(path, model)
        exported_modules.append(name)
        artifacts.append(
            OutputArtifact(
                kind="json",
                uri=str(path),
                label=f"{name} snapshot",
                mime_type="application/json",
            )
        )

    summary = ReportBundleSummaryModel(
        schema_version=report.schema_version,
        summary=report.summary,
        module_snapshots=exported_modules,
        figure_count=len(report.figures),
        table_count=len(report.tables),
        warning_count=len(report.warnings),
        warnings=list(report.warnings),
        provenance=dict(report.provenance),
    )
    summary_path = json_dir / "report_bundle_summary.json"
    _write_payload(summary_path, summary.model_dump(mode="json"))
    artifacts.append(
        OutputArtifact(
            kind="json",
            uri=str(summary_path),
            label="A3 report bundle summary",
            mime_type="application/json",
        )
    )

    return artifacts
