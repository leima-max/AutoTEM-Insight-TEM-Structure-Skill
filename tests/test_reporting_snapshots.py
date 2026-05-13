import json
import re
from pathlib import Path

import numpy as np

from autotem import TEMAnalysisSkill

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _normalize_html_signature(html_text: str) -> dict[str, object]:
    section_titles = re.findall(r"<h2>([^<]+)</h2>", html_text)
    step_rows = re.findall(
        r"<tr><td>([^<]+)</td><td>([^<]+)</td><td>(\d+)</td></tr>",
        html_text,
    )
    interface_rows = re.findall(
        r"<tr><td>([BC])</td><td>([^<]+)</td><td>(pass|missing)</td><td>([^<]+)</td></tr>",
        html_text,
    )
    return {
        "title_present": "AutoTEM A3 Analysis Report" in html_text,
        "section_titles": section_titles,
        "step_rows": [list(item) for item in step_rows],
        "interface_rows": [list(item) for item in interface_rows],
        "contains_step_detail_class": "class=\"step-detail\"" in html_text,
    }


def _normalize_manifest_signature(payload: dict[str, object]) -> dict[str, object]:
    directories = payload.get("directories", {})
    if not isinstance(directories, dict):
        directories = {}
    counts = payload.get("counts", {})
    if not isinstance(counts, dict):
        counts = {}
    kind_counts = payload.get("artifact_kind_counts", {})
    if not isinstance(kind_counts, dict):
        kind_counts = {}
    return {
        "schema_version": payload.get("schema_version"),
        "requested_formats": payload.get("requested_formats"),
        "directories_keys": sorted(directories.keys()),
        "counts_keys": sorted(counts.keys()),
        "artifact_kind_counts": kind_counts,
    }


def test_a3_html_snapshot_signature(tmp_path: Path) -> None:
    image = np.zeros((36, 36), dtype=np.float32)
    image[18, 8] = 5.0
    image[18, 28] = 5.0
    input_path = tmp_path / "snapshot_html.npy"
    np.save(input_path, image)
    output_dir = tmp_path / "outputs"

    TEMAnalysisSkill().run(
        input_path=input_path,
        output_dir=output_dir,
        output_formats_override=["json", "html"],
        report_title="Snapshot Report",
        report_sections_override=["workflow_steps", "interface_checks"],
    )

    html_text = (output_dir / "report.html").read_text(encoding="utf-8")
    signature = _normalize_html_signature(html_text)
    expected = json.loads((SNAPSHOT_DIR / "a3_html_signature.json").read_text(encoding="utf-8"))
    assert signature == expected


def test_a3_manifest_snapshot_signature(tmp_path: Path) -> None:
    image = np.zeros((36, 36), dtype=np.float32)
    image[18, 8] = 5.0
    image[18, 28] = 5.0
    input_path = tmp_path / "snapshot_manifest.npy"
    np.save(input_path, image)
    output_dir = tmp_path / "outputs"

    TEMAnalysisSkill().run(
        input_path=input_path,
        output_dir=output_dir,
        output_formats_override=["json", "html", "svg", "pdf"],
        report_title="Snapshot Report",
        report_sections_override=["workflow_steps", "interface_checks"],
        pdf_sections_override=["cover", "figures"],
        pdf_max_figure_pages=2,
        pdf_cover_fields={"sample_id": "SNAP-001", "operator": "member-a"},
    )

    manifest_text = (output_dir / "json" / "output_manifest.json").read_text(
        encoding="utf-8"
    )
    manifest_payload = json.loads(manifest_text)
    signature = _normalize_manifest_signature(manifest_payload)
    expected = json.loads(
        (SNAPSHOT_DIR / "a3_manifest_signature.json").read_text(encoding="utf-8")
    )
    assert signature == expected
