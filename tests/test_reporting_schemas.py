from pathlib import Path

import numpy as np

from autotem import TEMAnalysisSkill
from autotem.reporting import (
    OutputManifestModel,
    ReportBundleSummaryModel,
    reporting_schema_manifest,
    write_reporting_json_schemas,
)


def test_reporting_schema_manifest_and_export(tmp_path: Path) -> None:
    schema_dir = tmp_path / "reporting_schemas"
    write_reporting_json_schemas(schema_dir)

    assert (schema_dir / "manifest.json").exists()
    assert (schema_dir / "OutputManifest.schema.json").exists()
    assert (schema_dir / "ReportBundleSummary.schema.json").exists()

    manifest = reporting_schema_manifest()
    assert manifest["schema_version"] == "0.1.0"
    assert manifest["models"] == ["ReportBundleSummary", "OutputManifest"]


def test_generated_reporting_json_validate_against_models(tmp_path: Path) -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[16, 24] = 7.0
    image[16, 8] = 7.0
    input_path = tmp_path / "input.npy"
    np.save(input_path, image)

    output_dir = tmp_path / "outputs"
    TEMAnalysisSkill().run(
        input_path=input_path,
        output_dir=output_dir,
        output_formats_override=["json", "html"],
    )

    manifest_payload = (output_dir / "json" / "output_manifest.json").read_text(encoding="utf-8")
    bundle_payload = (output_dir / "json" / "report_bundle_summary.json").read_text(
        encoding="utf-8"
    )

    manifest = OutputManifestModel.model_validate_json(manifest_payload)
    bundle = ReportBundleSummaryModel.model_validate_json(bundle_payload)

    assert manifest.schema_version == "0.1.0"
    assert "figures" in manifest.directories
    assert bundle.schema_version == "0.1.0"
    assert bundle.warning_count >= 0
