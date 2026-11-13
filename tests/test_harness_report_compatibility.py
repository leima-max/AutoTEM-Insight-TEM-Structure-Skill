"""Registered report delivery and the legacy/natively-shaped YAML entry points."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from autotem import TEMAnalysisSkill
from autotem.cli import EXIT_CONFIG_ERROR, EXIT_OK, EXIT_RUNTIME_ERROR, main
from autotem.core import AnalysisReport, RunSummary, StrainResult, TEMDataset
from autotem.harness import Harness
from autotem.harness.compiler import compile_legacy_recipe
from autotem.harness.contracts import HarnessError
from autotem.preprocessing.quality import load_dataset_array
from autotem.reporting.harness_evidence import delivery_index

pytestmark = pytest.mark.smoke


def write_recipe(path, value):
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def published(report):
    files = {a["role"]: Path(a["path"]) for a in report["artifacts"]}
    return json.loads(files["report_manifest"].read_bytes()), files


def test_repeated_role_and_exported_reports_do_not_replace_evidence(tmp_path):
    path = tmp_path / "field.npy"
    np.save(path, np.arange(256).reshape(16, 16))
    harness = Harness(tmp_path / "runs")
    plan = {
        "input_path": str(path),
        "steps": [
            {"id": "load", "operation_id": "input.load"},
            {"id": "first", "operation_id": "quality.assess"},
        ],
    }
    state = harness.submit_analysis(plan, user_request="Inspect <field>", asynchronous=False)
    run_id = state["run_id"]
    first, _ = published(harness.export_report(run_id, ["json"]))
    assert first["completeness"]["required_complete"]
    followup = {**plan, "steps": [{"id": "second", "operation_id": "quality.assess"}]}
    harness.submit_analysis(
        followup,
        run_id=run_id,
        user_request="Check again",
        asynchronous=False,
    )
    report, files = published(harness.export_report(run_id, ["json", "html"]))
    assert report["report_type"] == "analysis"
    assert report["completeness"]["delivered_required_steps"] == 3
    assert [q["step_id"] for q in report["results_by_role"]["quality"]] == ["first", "second"]
    assert report["claim_evidence"][1]["user_request"] == "Inspect <field>"
    assert report["claim_evidence"][2]["user_request"] == "Check again"
    assert report["claim_evidence"][2]["consumed_artifacts"]["raw"] == report["result_index"]["raw"]
    assert all(a["step_id"] != "_report" for a in report["artifacts"])
    html = files["report_html"].read_text(encoding="utf-8")
    assert "Inspect &lt;field&gt;" in html
    assert all(q["artifact_id"] in html for q in report["results_by_role"]["quality"])
    for row in report["claim_evidence"]:
        for measurement in row["measurement_examples"]:
            stored = harness.store.load_json(measurement["artifact_id"])
            assert stored[measurement["pointer"][1:]]["value"] == measurement["value"]
            assert measurement["unit"] and measurement["method"]


def test_incomplete_request_exports_diagnostic_and_retains_pending_result(tmp_path):
    path = tmp_path / "uncalibrated.npy"
    np.save(path, np.zeros((16, 16), dtype="float32"))
    harness = Harness(tmp_path / "runs")
    state = harness.submit_analysis(
        {
            "input_path": str(path),
            "steps": [
                {"id": "read", "operation_id": "input.load"},
                {
                    "id": "scale",
                    "operation_id": "calibration.resolve",
                    "parameters": {"require_physical": True},
                },
                {"id": "denoise", "operation_id": "preprocessing.apply"},
            ],
        },
        asynchronous=False,
    )
    report, files = published(harness.export_report(state["run_id"], ["json", "html"]))
    assert state["status"] == "needs_input" and report["report_type"] == "diagnostic"
    assert report["completeness"]["unmet_required_steps"] == ["scale", "denoise"]
    by_id = {row["step_id"]: row for row in report["claim_evidence"]}
    assert by_id["scale"]["error"]["code"] == "NEEDS_INPUT"
    assert by_id["denoise"]["artifacts_by_role"] == {}
    assert "denoise" in files["report_html"].read_text(encoding="utf-8")


def test_two_strain_methods_keep_complete_arrays_and_choose_plan_order(tmp_path):
    y, x = np.indices((80, 80), dtype=float)
    image = np.full((80, 80), 10.0)
    for cy in range(12, 61, 8):
        for cx in range(12, 61, 8):
            image += 100 * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 2)
    path = tmp_path / "lattice.npy"
    np.save(path, image)
    harness = Harness(tmp_path / "runs")
    state = harness.submit_analysis(
        {
            "input_path": str(path),
            "modality": "HAADF-STEM",
            "steps": [
                {"id": "load", "operation_id": "input.load"},
                {"id": "detect", "operation_id": "atomic.detect"},
                {"id": "refine", "operation_id": "atomic.refine"},
                {
                    "id": "atom",
                    "operation_id": "strain.atom",
                    "parameters": {"reference_region": [8, 36, 8, 68]},
                },
                {
                    "id": "gpa",
                    "operation_id": "strain.gpa",
                    "parameters": {
                        "g_vectors": [[0.125, 0.0], [0.0, 0.125]],
                        "reference_region": [16, 48, 16, 48],
                    },
                },
            ],
        },
        asynchronous=False,
    )
    assert state["status"] == "completed", state["error"]
    report, _ = published(harness.export_report(state["run_id"], ["json"]))
    strain = report["strain_results"]
    assert strain["primary"]["role"] == "atom_strain"
    assert [s["role"] for s in strain["all"]] == ["atom_strain", "gpa"]
    for result in strain["all"]:
        strict = StrainResult.model_validate(harness.store.load_json(result["artifact_id"]))
        assert strict.schema_version == "0.1.0" and strict.warnings
        assert np.load(strict.strain_xx.uri, allow_pickle=False).shape == image.shape


def test_missing_committed_id_is_reported_and_optional_failure_cannot_satisfy_a_claim():
    run = {
        "request": {
            "steps": [
                {"id": "missing", "operation_id": "quality.assess", "parameters": {}},
                {"id": "opt", "operation_id": "strain.gpa", "parameters": {}, "optional": True},
            ]
        }
    }
    state = {
        "steps": [
            {"step_id": "missing", "status": "completed", "result": {"artifact_ids": ["absent"]}},
            {"step_id": "opt", "status": "skipped"},
        ],
        "events": [],
    }
    evidence = [{"step_id": "opt", "role": "gpa", "artifact_id": "diag", "execution_usable": False}]
    report = delivery_index(run, state, evidence, lambda aid: None)
    assert report["claim_evidence"][1]["diagnostic_artifact_ids"] == ["diag"]
    assert report["completeness"]["unmet_required_steps"] == ["missing"]
    assert report["completeness"]["optional_skipped"] == ["opt"]
    assert report["strain_results"]["primary"] is None


def test_minimal_legacy_and_registered_facades_agree_on_disk_input(tmp_path, capsys):
    path = tmp_path / "frame.npy"
    data = np.arange(256, dtype="float32").reshape(16, 16)
    np.save(path, data)
    legacy = write_recipe(
        tmp_path / "legacy.yaml",
        {
            "workflow": {"name": "read", "steps": ["load", "report"]},
            "outputs": {"formats": ["json"]},
        },
    )
    skill = TEMAnalysisSkill()
    old_root = tmp_path / "legacy-output"
    old = skill.run(path, legacy, old_root)
    old_model = AnalysisReport.model_validate_json(Path(old["report_path"]).read_bytes())
    assert old_model.schema_version == "0.1.0" and old_model.dataset.data.shape == data.shape
    np.testing.assert_array_equal(load_dataset_array(old_model.dataset), data)
    migrated = skill.run_harness(path, legacy, tmp_path / "migrated")
    report, _ = published(migrated["report"])
    assert report["completeness"]["required_steps"] == 1
    assert [r["operation_id"] for r in report["claim_evidence"]] == ["input.load"]
    current_root = tmp_path / "formal"
    native = write_recipe(
        tmp_path / "formal.yaml",
        {
            "input_path": "INPUT_FILE",
            "steps": [
                {"id": "read", "operation_id": "input.load"},
            ],
        },
    )
    code = main(
        [
            "analyze",
            str(path),
            "--workflow",
            str(native),
            "--output",
            str(current_root),
            "--formats",
            "json",
        ]
    )
    captured = capsys.readouterr()
    assert code == EXIT_OK, captured.err
    _, formal_json = captured.out.split("\n", 1)
    formal = json.loads(formal_json)
    for root, result in [(tmp_path / "migrated", migrated), (current_root, formal)]:
        store = Harness(root).store
        dataset = TEMDataset.model_validate(
            store.load_json(store.role(result["run_id"], "dataset"))
        )
        assert dataset.data.shape == old_model.dataset.data.shape
        np.testing.assert_array_equal(store.load_array(store.role(result["run_id"], "raw")), data)


@pytest.mark.parametrize("kind", ["channel", "frame", "dataset"])
def test_legacy_selection_reloads_the_requested_pixels(tmp_path, capsys, kind):
    grid = np.arange(78 * 90, dtype=np.int16).reshape(78, 90)
    if kind == "channel":
        source = tmp_path / "image.tif"
        pages = [
            np.stack([(grid + offset + c) % 251 for c in (0, 7, 13)], axis=-1).astype(np.uint8)
            for offset in (0, 50)
        ]
        Image.fromarray(pages[0]).save(
            source, save_all=True, append_images=[Image.fromarray(pages[1])]
        )
        params, plane = {"frame": 1, "channel": 2}, pages[1][..., 2]
    elif kind == "frame":
        source = tmp_path / "stack.npy"
        np.save(source, np.stack((grid + 23, grid)))
        params, plane = {"frame": 1}, grid
    else:
        source = tmp_path / "sets.npz"
        np.savez(source, first=grid + 18, second=np.stack((grid + 23, grid)))
        params, plane = {"dataset_path": "second", "selection_indices": [1]}, grid
    params["roi"] = [3, 71, 5, 80]
    expected = plane[3:71, 5:80]
    original = source.read_bytes()
    out = tmp_path / "out"
    recipe = write_recipe(
        tmp_path / "recipe.yaml",
        {
            "workflow": {"steps": ["load", "report"], "parameters": {"load": params}},
            "outputs": {"formats": ["json"], "save_intermediates": kind != "frame"},
        },
    )
    if kind == "channel":
        rc = main(["analyze", str(source), "--workflow", str(recipe), "--output", str(out)])
        captured = capsys.readouterr()
        assert rc == EXIT_OK, captured.err
        result = json.loads(captured.out[captured.out.index("{") :])
    else:
        result = TEMAnalysisSkill().run(source, recipe, out)
    report = AnalysisReport.model_validate_json(Path(result["report_path"]).read_bytes())
    summary = RunSummary.model_validate_json((out / "run_summary.json").read_bytes())
    assert (
        result["execution_status"] == "completed"
        and summary.schema_version == report.schema_version == "0.1.0"
    )
    raw_path = Path(report.dataset.data.uri)
    assert raw_path.is_relative_to(out) and report.dataset.source_path == str(source)
    assert report.dataset.data.inline is None and expected.size > 4096
    assert any(a.uri == str(raw_path) and a.kind == "array" for a in report.artifacts)
    stored = np.load(raw_path, allow_pickle=False)
    assert stored.dtype == expected.dtype and stored.tobytes() == expected.tobytes()
    np.testing.assert_array_equal(load_dataset_array(report.dataset), expected)
    assert report.dataset.metadata["selection"]["roi_origin_yx"] == [3, 5]
    assert report.workflow.steps[0].parameters == params and source.read_bytes() == original
    if kind == "channel":
        unselected = write_recipe(
            tmp_path / "read.yaml", {"workflow": {"steps": ["load", "report"]}}
        )
        old = TEMAnalysisSkill().run(source, unselected, tmp_path / "unselected")
        original_report = AnalysisReport.model_validate_json(Path(old["report_path"]).read_bytes())
        np.testing.assert_array_equal(
            load_dataset_array(original_report.dataset), pages[0].mean(axis=2)
        )
        assert any(
            "IMAGE_MULTIFRAME_TIFF" in warning for warning in original_report.dataset.warnings
        )


@pytest.mark.parametrize(
    ("options", "code"),
    [
        ({"frame": 1}, "NEEDS_INPUT"),
        ({"channel": 2}, "NEEDS_INPUT"),
        ({"frame": 2, "channel": 0}, "INVALID_SELECTION"),
        ({"frame": 1, "channel": 3}, "INVALID_SELECTION"),
        ({"frame": 1, "channel": 2, "roi": [0, 17, 0, 4]}, "INVALID_ROI"),
        ({"dataset_index": 0}, "INVALID_SELECTION"),
        ({"frame": True}, "INVALID_PARAMETERS"),
        ({"bad_option": 1}, "INVALID_PARAMETERS"),
    ],
)
def test_legacy_selection_refuses_ambiguous_and_invalid_requests(tmp_path, capsys, options, code):
    source = tmp_path / "image.tif"
    page = Image.fromarray(np.zeros((16, 24, 3), dtype=np.uint8))
    page.save(source, save_all=True, append_images=[page.copy()])
    recipe = write_recipe(
        tmp_path / "recipe.yaml",
        {
            "workflow": {"steps": ["load", "report"], "parameters": {"load": options}},
        },
    )
    with pytest.raises(HarnessError) as error:
        TEMAnalysisSkill().run(source, recipe, tmp_path / "api")
    assert error.value.code == code
    rc = main(
        ["analyze", str(source), "--workflow", str(recipe), "--output", str(tmp_path / "cli")]
    )
    assert rc == (EXIT_RUNTIME_ERROR if code == "NEEDS_INPUT" else EXIT_CONFIG_ERROR)
    assert code in capsys.readouterr().err
    assert not list(tmp_path.rglob("analysis_report.json"))
    assert not list(tmp_path.rglob("selected-*.npy"))


@pytest.mark.parametrize(
    "parameters",
    [
        {"unused": {"mystery": 1}},
        {"calibration": {"pixel_size_nm": 0.2}},
        {"load": {"bad_option": True}},
    ],
)
def test_unbound_or_unknown_legacy_options_are_rejected(tmp_path, parameters):
    recipe = write_recipe(
        tmp_path / "old.yaml",
        {
            "workflow": {"steps": ["load", "report"], "parameters": parameters},
        },
    )
    with pytest.raises(HarnessError) as exc:
        compile_legacy_recipe("input.npy", recipe)
    assert exc.value.code == "LEGACY_PARAMETER_REQUIRES_MIGRATION"


def test_legacy_diffraction_options_reach_their_declared_operations(tmp_path):
    recipe = write_recipe(
        tmp_path / "old.yaml",
        {
            "workflow": {
                "steps": ["load", "calibrate", "preprocess", "analyze", "report"],
                "parameters": {
                    "calibration": {"method": "auto", "pixel_size_nm": 0.02},
                    "diffraction": {
                        "top_k_peaks": 12,
                        "fft_center_mode": "geometric",
                        "phase_top_n": 2,
                    },
                },
            }
        },
    )
    steps = {s["operation_id"]: s for s in compile_legacy_recipe("input.npy", recipe)["steps"]}
    assert steps["diffraction.peaks"]["parameters"]["top_k"] == 12
    assert steps["diffraction.center"]["parameters"]["center_mode"] == "geometric"
    assert steps["diffraction.index"]["parameters"]["top_n"] == 2
    assert steps["calibration.resolve"]["parameters"]["pixel_size_nm"] == 0.02


def test_cli_refuses_unapplied_legacy_options_in_harness_mode(tmp_path, capsys):
    assert (
        main(["analyze", "input.npy", "--harness", "--detect-atoms", "--output", str(tmp_path)])
        == EXIT_CONFIG_ERROR
    )
    assert "plan" in capsys.readouterr().err and not list(tmp_path.iterdir())
