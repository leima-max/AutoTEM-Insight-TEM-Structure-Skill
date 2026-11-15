"""Registered report delivery and the legacy/natively-shaped YAML entry points."""

from __future__ import annotations

import hashlib
import json
import os
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
from autotem.io import read_image
from autotem.preprocessing.quality import load_dataset_array
from autotem.reporting.harness_evidence import (
    _diffraction_html,
    claim_html,
    delivery_index,
    render_harness_html,
)

pytestmark = pytest.mark.smoke


@pytest.mark.skipif(os.name != "nt", reason="Win32 extended input paths")
def test_long_tiff_keeps_ordinary_path_pixels_and_input_fingerprint(tmp_path):
    folder = tmp_path / "原始报告数据"
    while len(str(folder / "detector.tif").encode("utf-16-le")) // 2 < 270:
        folder /= "制样 0123456789"
    path = folder / "detector.tif"
    native = Path("\\\\?\\" + str(path))
    native.parent.mkdir(parents=True)
    pixels = np.arange(70, dtype=np.uint16).reshape(7, 10)
    with native.open("wb") as stream:
        Image.fromarray(pixels).save(stream, format="TIFF")
    direct = TEMAnalysisSkill().load_data(path)
    assert direct.source_path == str(path)
    np.testing.assert_array_equal(load_dataset_array(direct), pixels)
    recipe = write_recipe(tmp_path / "read.yaml", {"workflow": {"steps": ["load", "report"]}})
    out = tmp_path / "out"
    assert main(["analyze", str(path), "--workflow", str(recipe), "--output", str(out)]) == EXIT_OK
    report = AnalysisReport.model_validate_json((out / "analysis_report.json").read_bytes())
    assert report.schema_version == "0.1.0" and report.dataset.source_path == str(path)
    assert report.provenance["input_fingerprint"] == {
        "path": str(path), "exists": True, "size_bytes": native.stat().st_size,
        "sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
    }
    np.testing.assert_array_equal(load_dataset_array(report.dataset), pixels)


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
    page = files["report_html"].read_text(encoding="utf-8")
    assert "Undelivered required steps · scale, denoise" in page
    assert "<h2>Diffraction measurements</h2>" not in page
    assert page.index("scale, denoise") < page.index("Step parameters and provenance")


@pytest.mark.parametrize("has_peak_map", [True, False])
def test_html_spacings_bind_direction_to_source_peak_only_when_recorded(has_peak_map):
    first = {"value": .16, "unit": "nm", "method": "FFT", "uncertainty": None}
    second = {**first, "value": .25, "uncertainty": .002}
    result = {
        "d_spacings": [first, second],
        "g_vectors": [
            {"source_peak_index": 0, "angle": {"value": 42, "unit": "deg"}},
            {"source_peak_index": 1, "angle": {"value": 90, "unit": "deg"}},
            {"source_peak_index": 3, "angle": {"value": 0, "unit": "deg"}},
        ],
        "provenance": {"d_by_peak": [None, first, None, second]} if has_peak_map else {},
        "phase_candidates": [{"name": "X<1", "matched_spot_count": 4}],
        "warnings": ["<reference needed>"],
    }
    artifact = {"artifact_id": "abc", "step_id": "indexed", "path": "results/abc.json"}
    page = _diffraction_html(result, artifact)
    assert "0.16 nm" in page and "0.25 nm" in page
    assert "Not evaluated" in page and "0.002 nm" in page
    assert "X&lt;1 · 4 matched spots" in page and "&lt;reference needed&gt;" in page
    assert '<a href="abc.json">abc</a>' in page
    assert "42 deg" not in page
    assert ("<td>1</td><td>90 deg</td><td>0.16 nm" in page) == has_peak_map
    assert ("<td>3</td><td>0 deg</td><td>0.25 nm" in page) == has_peak_map
    if not has_peak_map:
        assert " deg" not in page


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
    report, files = published(harness.export_report(state["run_id"], ["json", "html"]))
    strain = report["strain_results"]
    assert strain["primary"]["role"] == "atom_strain"
    assert [s["role"] for s in strain["all"]] == ["atom_strain", "gpa"]
    assert "validity_mask" not in strain["primary"]
    mask_id = report["result_index"]["gpa.quality_mask"]
    mask = strain["all"][1]["validity_mask"]
    assert mask["artifact_id"] == mask_id
    full = harness.read_artifact(strain["all"][1]["artifact_id"])
    assert full["validity_mask_artifact_id"] == mask_id
    assert full["validity_interpretation"] == mask["interpretation"]
    preview = harness.read_artifact(report["result_index"]["gpa.strain_xx"])
    assert preview["validity_interpretation"] == mask["interpretation"]
    assert "Downloaded arrays can contain zeros" in mask["interpretation"]
    page = files["report_html"].read_text(encoding="utf-8")
    assert mask["interpretation"] in page
    assert f'href="{mask_id}.npy">GPA quality mask (gpa)</a>' in page
    for result in strain["all"]:
        strict = StrainResult.model_validate(harness.store.load_json(result["artifact_id"]))
        assert strict.schema_version == "0.1.0" and strict.warnings
        assert np.load(strict.strain_xx.uri, allow_pickle=False).shape == image.shape


def test_gpa_report_links_only_the_exact_mask_from_a_delivered_step():
    run = {"request": {"steps": [
        {"id": sid, "operation_id": "strain.gpa", "parameters": {}} for sid in ("one", "two")
    ]}}
    state = {"events": [], "steps": [
        {"step_id": sid, "status": "completed"} for sid in ("one", "two")
    ]}

    def item(step, role, aid, kind):
        return {"step_id": step, "role": role, "kind": kind, "artifact_id": aid, "size": 150,
                "execution_usable": True, "path": aid + (".npy" if kind == "array" else ".json")}

    evidence = [
        item("one", "gpa", "result1", "json"),
        item("one", "gpa.quality_mask", "mask1", "array"),
        item("two", "gpa", "result2", "json"),
        item("two", "gpa.quality_mask", "mask2", "array"),
    ]
    # The first JSON was copied from a Windows store. The second refers to a
    # different step's artifact, even though its own step also has a mask.
    payload = {"provenance": {"quality_mask": {"uri": r"D:\old\runs\mask1.npy"}}}
    report = delivery_index(run, state, evidence, lambda aid: payload)
    first, second = report["strain_results"]["all"]
    assert first["validity_mask"]["artifact_id"] == "mask1"
    assert second["validity_mask"]["artifact_id"] is None
    assert "unavailable" in second["validity_mask"]["interpretation"]
    html = claim_html(report, evidence)
    assert 'href="mask1.npy">GPA quality mask (one)</a>' in html
    assert 'href="mask2.npy">GPA quality mask (two)</a>' not in html
    assert second["validity_mask"]["interpretation"] in html
    evidence[0]["size"] = 256_001

    def read_bounded(aid):
        assert aid != "result1"
        return payload

    bounded = delivery_index(run, state, evidence, read_bounded)
    assert bounded["strain_results"]["primary"]["validity_mask"]["artifact_id"] is None


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

    orphan = {"step_id": "missing", "role": "peak_table", "artifact_id": "orphan",
              "execution_usable": True, "kind": "table", "path": "orphan.csv",
              "sha256": "0" * 64, "operation_status": "completed"}
    delivery = delivery_index(run, state, [orphan], lambda aid: None)
    manifest = {**delivery, "artifacts": [orphan], "run": {"status": "failed", "run_id": "run"},
                "report_type": "diagnostic", "scientific_accuracy": "not_evaluated",
                "result_index": {}}
    page = render_harness_html(manifest, lambda aid: None)
    assert 'href="orphan.csv"' in page  # Available for inspection in the full evidence register.
    assert "<h2>Data tables</h2>" not in page


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
        preview = read_image(source)
        np.testing.assert_array_equal(preview.array, pages[0].mean(axis=2).astype(np.float32))
        codes = {warning["code"] for warning in preview.warnings}
        assert {"IMAGE_MULTIFRAME_TIFF", "IMAGE_RGB_TO_GRAYSCALE"} <= codes


@pytest.mark.parametrize("kind", ["rgb", "archive"])
def test_legacy_atom_report_uses_its_selected_plane(tmp_path, capsys, kind):
    from autotem.atomic.atom_detection import detect_atoms

    y, x = np.indices((60, 82))
    other = np.exp(-((y - 52) ** 2 + (x - 72) ** 2) / 4)
    target = np.exp(-((y - 23) ** 2 + (x - 36) ** 2) / 4)
    target += np.exp(-((y - 33) ** 2 + (x - 45) ** 2) / 4)
    other, target = (np.rint(image * 240).astype(np.uint8) for image in (other, target))
    if kind == "rgb":
        source = tmp_path / "channels.tif"
        Image.fromarray(np.stack((other, other, target), axis=-1)).save(source)
        selection = {"channel": 2}
    else:
        source = tmp_path / "survey.npz"
        np.savez(source, overview=other, scan=np.stack((other, target)))
        selection = {"dataset_path": "scan", "frame": 1}
    selection["roi"] = [12, 47, 20, 60]
    recipe = write_recipe(
        tmp_path / "atoms.yaml",
        {
            "workflow": {
                "steps": ["load", "detect_atoms", "report"],
                "parameters": {"load": selection},
            },
            "outputs": {"formats": ["json", "csv"], "save_intermediates": kind == "rgb"},
        },
    )
    out = tmp_path / "out"
    if kind == "rgb":
        assert (
            main(["analyze", str(source), "--workflow", str(recipe), "--output", str(out)])
            == EXIT_OK
        )
        capsys.readouterr()
    else:
        assert TEMAnalysisSkill().run(source, recipe, out)["execution_status"] == "completed"
    report = AnalysisReport.model_validate_json((out / "analysis_report.json").read_bytes())
    plane = target[12:47, 20:60]
    np.testing.assert_array_equal(np.load(report.dataset.data.uri, allow_pickle=False), plane)
    expected = detect_atoms(plane)
    assert len(expected.coordinates) >= 2
    lattice = report.atom_lattice
    assert lattice.coordinate_unit == "pixel"
    assert lattice.provenance["selection"]["roi_origin_yx"] == [12, 20]
    assert (
        lattice.provenance["selected_npy_sha256"]
        == report.dataset.provenance["selected_npy_sha256"]
    )
    assert lattice.provenance["detector_provenance"]["input_shape"] == list(plane.shape)
    np.testing.assert_array_equal([[a.y, a.x] for a in lattice.atoms], expected.coordinates)
    np.testing.assert_array_equal([a.intensity.value for a in lattice.atoms], expected.intensities)
    table = next(t for t in report.tables if t.table_type == "atom_coordinates")
    np.testing.assert_array_equal([[r["y"], r["x"]] for r in table.rows], expected.coordinates)


@pytest.mark.parametrize("bare", [False, True], ids=["load-report", "no-yaml"])
def test_legacy_workflow_asks_for_the_real_frame_before_science(tmp_path, capsys, bare):
    source = tmp_path / "two-pages.tif"
    grid = np.arange(35, dtype=np.uint8).reshape(5, 7)
    pages = [np.stack([grid + offset + c for c in (0, 5, 9)], axis=-1) for offset in (0, 80)]
    Image.fromarray(pages[0]).save(source, save_all=True, append_images=[Image.fromarray(pages[1])])
    recipe = (
        None
        if bare
        else write_recipe(tmp_path / "read.yaml", {"workflow": {"steps": ["load", "report"]}})
    )
    with pytest.raises(HarnessError) as exc:
        TEMAnalysisSkill().run(source, recipe, tmp_path / "api")
    details = {"fields": [{"name": "frame", "count": 2}]}
    assert (exc.value.code, exc.value.details) == ("NEEDS_INPUT", details)
    command = ["analyze", str(source), "--output", str(tmp_path / "cli")]
    if recipe is not None:
        command += ["--workflow", str(recipe)]
    assert main(command) == EXIT_RUNTIME_ERROR
    stderr = capsys.readouterr().err
    assert "NEEDS_INPUT" in stderr and "workflow.parameters.load" in stderr
    assert json.loads(stderr.splitlines()[-1]) == details
    assert not list(tmp_path.rglob("analysis_report.json"))
    assert not list(tmp_path.rglob("selected-*.npy"))


def test_implicit_stack_and_archive_choices_precede_pixel_loading(tmp_path, monkeypatch):
    recipe = write_recipe(tmp_path / "read.yaml", {"workflow": {"steps": ["load", "report"]}})
    stack = np.arange(120, dtype=np.int16).reshape(2, 3, 4, 5)
    source = tmp_path / "stack.npy"
    np.save(source, stack)
    multi = tmp_path / "data.npz"
    np.savez_compressed(multi, survey=stack[0, 0], scan=stack)
    single = tmp_path / "single.npz"
    np.savez_compressed(single, scan=stack)
    with np.load(multi, allow_pickle=False) as archive:

        def fail_if_decoded(*args):
            pytest.fail("An incomplete archive selection decompressed a pixel array")

        monkeypatch.setattr(type(archive), "__getitem__", fail_if_decoded)
    indices = {"name": "selection_indices", "count": 2, "shape": [2, 3, 4, 5]}
    datasets = {"name": "dataset_path", "choices": ["survey", "scan"]}
    for path, field in [(source, indices), (multi, datasets), (single, indices)]:
        with pytest.raises(HarnessError) as exc:
            TEMAnalysisSkill().run(path, recipe, tmp_path / path.name.replace(".", "-"))
        assert exc.value.code == "NEEDS_INPUT" and exc.value.details == {"fields": [field]}


@pytest.mark.parametrize("suffix", ["png", "jpg", "tiff"])
def test_image_headers_request_a_channel_without_decoding_pixels(tmp_path, monkeypatch, suffix):
    from autotem.services.data_quality import read_selected_input

    source = tmp_path / f"color.{suffix}"
    Image.fromarray(np.zeros((12, 20, 3), dtype=np.uint8)).save(source)
    recipe = write_recipe(tmp_path / "read.yaml", {"workflow": {"steps": ["load", "report"]}})

    def unexpected_buffer(*args, **kwargs):
        pytest.fail("An incomplete channel selection decoded the image")

    monkeypatch.setattr(Image.Image, "tobytes", unexpected_buffer)
    with pytest.raises(HarnessError) as legacy:
        TEMAnalysisSkill().run(source, recipe, tmp_path / "out")
    with pytest.raises(HarnessError) as shared:
        read_selected_input(source, {}, dataset_id="color")
    assert legacy.value.code == shared.value.code == "NEEDS_INPUT"
    assert (
        legacy.value.details
        == shared.value.details
        == {"fields": [{"name": "channel", "count": 3}]}
    )


@pytest.mark.parametrize("suffix", ["tif", "npy", "txt", "csv", "npz"])
def test_unique_2d_inputs_retain_pixels_and_strict_report_contract(tmp_path, suffix):
    grid = np.arange(80, dtype=np.uint16).reshape(8, 10)
    source = tmp_path / f"single.{suffix}"
    if suffix == "tif":
        Image.fromarray(grid).save(source, dpi=(50_800_000, 50_800_000))
    elif suffix == "npy":
        np.save(source, grid)
    elif suffix == "npz":
        np.savez_compressed(source, only=grid)
    else:
        np.savetxt(source, grid)
    recipe = write_recipe(tmp_path / "read.yaml", {"workflow": {"steps": ["load", "report"]}})
    out = tmp_path / "report"
    result = TEMAnalysisSkill().run(source, recipe, out)
    report = AnalysisReport.model_validate_json(Path(result["report_path"]).read_bytes())
    summary = RunSummary.model_validate_json((out / "run_summary.json").read_bytes())
    assert result["execution_status"] == "completed"
    assert report.schema_version == summary.schema_version == "0.1.0"
    assert report.dataset.source_path == str(source)
    np.testing.assert_array_equal(load_dataset_array(report.dataset), grid)
    if suffix == "npz":
        assert report.dataset.metadata["selection"]["dataset_path"] == "only"
    else:
        direct = read_image(source)
        assert report.dataset.provenance == direct.provenance
        assert report.dataset.metadata["raw"] == direct.metadata_raw
    if suffix == "tif":
        assert report.dataset.pixel_size.value == pytest.approx(0.5)
        assert {"XResolution", "ResolutionUnit", "282", "296"} <= set(
            report.dataset.metadata["raw"]
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
    stderr = capsys.readouterr().err
    assert code in stderr
    if code == "NEEDS_INPUT":
        field = (
            {"name": "channel", "count": 3} if "frame" in options else {"name": "frame", "count": 2}
        )
        assert json.loads(stderr.splitlines()[-1]) == error.value.details == {"fields": [field]}
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
