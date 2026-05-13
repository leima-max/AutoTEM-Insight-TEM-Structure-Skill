"""A3 output directory conventions and manifest export."""

from __future__ import annotations

import json
from pathlib import Path

from autotem.core import AnalysisReport, OutputArtifact
from autotem.reporting.schema_models import OutputManifestModel

_A3_DIRECTORIES: tuple[str, ...] = (
    "figures",
    "tables",
    "json",
    "intermediates",
)


def ensure_a3_output_layout(output_dir: str | Path) -> dict[str, str]:
    """Create and return canonical A3 output subdirectories."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, str] = {}
    for directory in _A3_DIRECTORIES:
        path = output_dir / directory
        path.mkdir(parents=True, exist_ok=True)
        resolved[directory] = str(path)
    return resolved


def write_output_manifest(
    report: AnalysisReport,
    output_dir: str | Path,
    *,
    requested_formats: set[str],
) -> OutputArtifact:
    """Write output manifest describing report artifacts and directory conventions."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "json" / "output_manifest.json"

    by_kind: dict[str, int] = {}
    for artifact in report.artifacts:
        by_kind[artifact.kind] = by_kind.get(artifact.kind, 0) + 1

    payload = OutputManifestModel(
        schema_version=report.schema_version,
        report_summary=report.summary,
        requested_formats=sorted(requested_formats),
        output_root=str(output_dir),
        directories={
            "figures": str(output_dir / "figures"),
            "tables": str(output_dir / "tables"),
            "json": str(output_dir / "json"),
            "intermediates": str(output_dir / "intermediates"),
        },
        counts={
            "warnings": len(report.warnings),
            "figures": len(report.figures),
            "tables": len(report.tables),
            "artifacts": len(report.artifacts),
        },
        artifact_kind_counts=by_kind,
        artifacts=[artifact.model_dump(mode="json") for artifact in report.artifacts],
        provenance={
            "source": "autotem.reporting.output_contract.write_output_manifest",
            "report_provenance": report.provenance,
        },
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return OutputArtifact(
        kind="json",
        uri=str(manifest_path),
        label="A3 output manifest",
        mime_type="application/json",
    )
