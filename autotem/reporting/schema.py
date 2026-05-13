"""Schema export helpers for reporting contract models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autotem.core.models import CURRENT_SCHEMA_VERSION
from autotem.reporting.schema_models import reporting_contract_schemas


def reporting_schema_manifest() -> dict[str, Any]:
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "models": ["ReportBundleSummary", "OutputManifest"],
    }


def write_reporting_json_schemas(directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    schemas = reporting_contract_schemas()
    for name, schema in schemas.items():
        path = directory / f"{name}.schema.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(reporting_schema_manifest(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
