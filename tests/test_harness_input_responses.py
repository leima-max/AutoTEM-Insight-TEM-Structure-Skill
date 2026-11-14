"""Disk-to-child acceptance for versioned human input through the public Harness."""

from __future__ import annotations

import copy
import json
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from autotem.harness import Harness, HarnessError
from autotem.harness.cli import dispatch_request
from autotem.harness.store import file_hash


def call(harness, tool, **arguments):
    response = dispatch_request(
        harness,
        {
            "schema_version": "1.0.0",
            "request_id": "test",
            "tool": tool,
            "arguments": arguments,
        },
    )
    assert response["ok"], response["error"]
    return response["result"]


def calibration_plan(path, mode="snapshot"):
    return {
        "input_path": str(path),
        "input_mode": mode,
        "modality": "HRTEM",
        "steps": [
            {"id": "load", "operation_id": "input.load"},
            {
                "id": "cal",
                "operation_id": "calibration.resolve",
                "parameters": {"require_physical": True},
            },
        ],
    }


def answer_plan(plan, request, **answers):
    return {
        **copy.deepcopy(plan),
        "user_input": {
            "response": {
                "request_id": request["request_id"],
                "source": "user",
                "basis": "Acquisition log",
                "answers": answers,
            }
        },
    }


def missing_calibration(tmp_path, mode="snapshot"):
    harness = Harness(tmp_path / "h")
    source = tmp_path / "lattice.npy"
    yy, xx = np.mgrid[:48, :64]
    np.save(source, np.cos(np.pi * xx / 4) + np.cos(np.pi * yy / 3))
    plan = calibration_plan(source, mode)
    state = call(harness, "submit_analysis", plan=plan, asynchronous=False)
    assert state["status"] == "needs_input", state
    (request,) = state["input_requests"]
    assert request["request_id"] == state["error"]["details"]["request_id"]
    return harness, source, plan, state, request


def test_disk_calibration_concurrent_answer_and_changed_answer(tmp_path):
    h, source, plan, parent, request = missing_calibration(tmp_path)
    original = np.load(source)
    parent_id = parent["run_id"]
    evidence = {a["path"]: a["sha256"] for a in call(h, "list_artifacts", run_id=parent_id)}
    assert request["status"] == "open" and request["schema_version"] == "1.0.0"
    assert request["fields"][0]["answer_schema"]["pixel_size_nm"]["value"]["exclusiveMinimum"] == 0
    assert h.store.role(parent_id, "raw") in request["evidence_artifact_ids"]
    calibrated = answer_plan(plan, request, pixel_size_nm={"value": 25, "unit": "pm/pixel"})
    calibrated["user_input"]["response"]["raw_reply"] = "25 pm per pixel"
    validated = call(h, "validate_plan", plan=calibrated)
    assert validated["plan"]["steps"][1]["parameters"]["pixel_size_nm"] == 0.025
    np.save(source, np.zeros((48, 64)))  # the parent owns the input for this replay

    def submit(i):
        return call(
            h,
            "submit_analysis",
            plan=calibrated if i else validated["plan"],
            parent_run_id=parent_id,
            idempotency_key="first-answer",
            asynchronous=False,
        )

    with ThreadPoolExecutor(2) as pool:
        first, retry = pool.map(submit, (0, 1))
    assert first["run_id"] == retry["run_id"] and first["status"] == "completed"
    child_id = first["run_id"]
    np.testing.assert_array_equal(original, h.store.load_array(h.store.role(child_id, "raw")))
    scale = h.store.load_json(h.store.role(child_id, "calibration"))["pixel_size"]
    assert scale["value"] == pytest.approx(0.025) and scale["unit"] == "nm/pixel"
    (accepted,) = first["input_responses"]
    assert accepted["child_run_id"] == child_id and accepted["parent_run_id"] == parent_id
    assert accepted["response"]["answers"]["pixel_size_nm"]["unit"] == "pm/pixel"
    assert accepted["validation"]["status"] == "accepted" and first["parameter_diff"]
    assert len(list((h.store.run_dir(child_id) / "raw").iterdir())) == 1
    changed = answer_plan(plan, request, pixel_size_nm={"value": 0.03, "unit": "nm/pixel"})
    with pytest.raises(HarnessError) as conflict:
        h.submit_analysis(changed, parent_run_id=parent_id, idempotency_key="first-answer")
    assert conflict.value.code == "IDEMPOTENCY_CONFLICT"
    second = h.submit_analysis(
        changed, parent_run_id=parent_id, idempotency_key="amended", asynchronous=False
    )
    assert second["status"] == "completed" and second["run_id"] != child_id
    reopened = Harness(h.root).get_run_status(parent_id)
    assert (
        reopened["status"] == "needs_input"
        and reopened["input_requests"][0]["status"] == "answered"
    )
    assert len(reopened["input_responses"]) == 2
    assert all(file_hash(Path(path)) == digest for path, digest in evidence.items())
    assert reopened["next_action"] == "get_child_run_status"
    assert reopened["latest_response_child_run_id"] == second["run_id"]
    report = call(h, "export_report", run_id=child_id, formats=["json"])
    manifest = h.store.load_json(report["artifacts"][0]["artifact_id"])
    assert manifest["run"]["input_responses"][0]["request_id"] == request["request_id"]


def test_invalid_replies_do_not_submit_or_store_credentials(tmp_path, monkeypatch):
    h, _, plan, parent, request = missing_calibration(tmp_path)
    parent_id = parent["run_id"]
    monkeypatch.setattr(h, "_launch_scheduler", lambda *_: pytest.fail("Unexpected worker"))
    canonical = answer_plan(plan, request, pixel_size_nm={"value": 0.025, "unit": "nm/pixel"})

    def denied(plan, code="INVALID_INPUT_RESPONSE", **kwargs):
        with pytest.raises(HarnessError) as error:
            h.submit_analysis(plan, **kwargs)
        assert error.value.code == code

    submit = {"parent_run_id": parent_id, "idempotency_key": "invalid"}
    denied(plan, "INPUT_RESPONSE_REQUIRED", **submit)
    denied(canonical, "IDEMPOTENCY_KEY_REQUIRED", parent_run_id=parent_id)
    denied(canonical, "INPUT_REQUEST_PARENT_MISMATCH", idempotency_key="no-parent")
    for bad in [
        {"value": -0.2, "unit": "nm/pixel"},
        {"value": True, "unit": "nm/pixel"},
        {"value": 0.025},
        {"value": 0.025, "unit": "nm"},
    ]:
        denied(answer_plan(plan, request, pixel_size_nm=bad), **submit)
    for key, value in [
        ("source", "guess"),
        ("basis", " "),
        ("source", "artifact"),
        ("answers", {"unrequested": {"value": 1}}),
    ]:
        bad = copy.deepcopy(canonical)
        bad["user_input"]["response"][key] = value
        denied(bad, **submit)
    bad = copy.deepcopy(canonical)
    bad["user_input"]["response"]["raw_reply"] = "password=sentinel_secret_99"
    denied(bad, "CREDENTIAL_PARAMETER_FORBIDDEN", **submit)
    bad = copy.deepcopy(canonical)
    bad["steps"][1]["parameters"]["pixel_size_nm"] = 9
    denied(bad, **submit)
    with h.store.connect() as db:
        assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
    reopened = Harness(h.root).get_run_status(parent_id)
    assert reopened["input_requests"][0]["status"] == "open"
    assert reopened["input_responses"] and "sentinel_secret_99" not in str(reopened)
    assert all(
        r["validation"]["status"] == "rejected" and r["child_run_id"] is None
        for r in reopened["input_responses"]
    )


def test_reference_reply_rechecks_missing_or_changed_source(tmp_path):
    h, source, plan, parent, request = missing_calibration(tmp_path, mode="reference")
    answered = answer_plan(plan, request, pixel_size_nm={"value": 10, "unit": "pm/pixel"})
    np.save(source, np.zeros((48, 64)))
    kwargs = {"parent_run_id": parent["run_id"], "idempotency_key": "reference"}
    with pytest.raises(HarnessError) as changed:
        h.submit_analysis(answered, **kwargs)
    assert changed.value.code == "INPUT_CHANGED"
    source.unlink()
    with pytest.raises(HarnessError) as missing:
        h.submit_analysis(answered, **kwargs)
    assert missing.value.code == "INPUT_NOT_FOUND"
    assert not h.get_run_status(parent["run_id"])["input_responses"]


def test_disconnected_reply_reuses_interrupted_child_and_discards_partial_snapshot(tmp_path):
    harness, source, plan, parent, request = missing_calibration(tmp_path)
    parent_id = parent["run_id"]
    original_hash = file_hash(source)
    parent_artifacts = {a["path"]: a["sha256"] for a in harness.list_artifacts(parent_id)}
    source.rename(tmp_path / "moved.npy")
    answer = answer_plan(plan, request, pixel_size_nm={"value": 25, "unit": "pm/pixel"})
    args = {"plan": answer, "parent_run_id": parent_id, "idempotency_key": "lost-reply"}
    signal = tmp_path / "snapshot-started"
    disconnect = tmp_path / "disconnect"
    # Crash the actual submitting interpreter, including when Windows uses a venv launcher.
    submit = """
import json, os, pathlib, sys, time
from autotem.harness import Harness, service
root, signal, disconnect = map(pathlib.Path, sys.argv[1:])
def partial_copy(src, dst):
    with open(src, "rb") as original, open(dst, "xb") as staging:
        staging.write(original.read(16))
        staging.flush()
        signal.touch()
        deadline = time.monotonic() + 60
        while not disconnect.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        os._exit(27)
service.shutil.copyfile = partial_copy
Harness(root).submit_analysis(**json.load(sys.stdin))
"""
    with subprocess.Popen(
        [sys.executable, "-B", "-c", submit, str(harness.root), str(signal), str(disconnect)],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as client:
        try:
            client.stdin.write(json.dumps(args).encode())
            client.stdin.close()
            deadline = time.monotonic() + 45
            while not signal.exists():
                assert client.poll() is None, client.stderr.read()
                assert time.monotonic() < deadline, "The snapshot writer did not reach staging"
                time.sleep(0.05)
            reopened = Harness(harness.root)
            live = call(reopened, "submit_analysis", **args, asynchronous=True)
            child_id, job_id = live["run_id"], live["job_id"]
            assert live["status"] == "preparing" and live["jobs"][0]["status"] == "queued"
            directory = reopened.store.run_dir(child_id)
            partial = directory / f".snapshot-{job_id}.tmp"
            assert partial.stat().st_size == 16 < (tmp_path / "moved.npy").stat().st_size
            other = directory / ".snapshot-unrelated.tmp"
            other.write_bytes(b"other job")
        finally:
            disconnect.touch()
            assert client.wait(timeout=15) == 27

    def retry(_):
        return call(Harness(harness.root), "submit_analysis", **args, asynchronous=False)

    # A regression must fail promptly rather than leaving a synchronous retry hung.
    assert reopened.get_run_status(child_id)["status"] == "interrupted"
    with ThreadPoolExecutor(2) as pool:
        replies = list(pool.map(retry, range(2)))
    for result in replies:
        assert (result["run_id"], result["job_id"], result["status"]) == (
            child_id,
            job_id,
            "interrupted",
        )
        assert result["error"]["code"] == "PREPARATION_INTERRUPTED"
        assert result["jobs"][0]["status"] == "interrupted"
        assert all(step["status"] == "interrupted" for step in result["steps"])
        assert [e["kind"] for e in result["events"]] == ["interrupted"]
    assert (
        not partial.exists()
        and not list((directory / "raw").iterdir())
        and other.read_bytes() == b"other job"
    )
    assert all(a["step_id"] == "_request" for a in harness.list_artifacts(child_id))
    original = harness.get_run_status(parent_id)
    assert (
        original["latest_response_child_run_id"] == child_id
        and len(original["input_responses"]) == 1
    )
    changed = answer_plan(plan, request, pixel_size_nm={"value": 30, "unit": "pm/pixel"})
    with pytest.raises(HarnessError) as err:
        harness.submit_analysis(**{**args, "plan": changed})
    assert err.value.code == "IDEMPOTENCY_CONFLICT"
    with harness.store.connect() as db:
        assert db.execute("SELECT count(*) FROM runs").fetchone()[0] == 2
        assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 2
    replacement = harness.submit_analysis(
        **{**args, "idempotency_key": "replacement"}, asynchronous=False
    )
    assert replacement["status"] == "completed" and replacement["run_id"] != child_id
    assert replacement["parent_run_id"] == parent_id
    assert (
        file_hash(Path(harness.store.get_run(replacement["run_id"])["input_path"])) == original_hash
    )
    assert all(file_hash(Path(path)) == digest for path, digest in parent_artifacts.items())


def test_unknown_preparation_owner_remains_available_for_manual_cancel(tmp_path, monkeypatch):
    h = Harness(tmp_path / "h")
    path = tmp_path / "image.npy"
    np.save(path, np.arange(64).reshape(8, 8))
    plan = calibration_plan(path)
    monkeypatch.setattr(h, "_launch_scheduler", lambda *_: None)
    state = h.submit_analysis(plan, idempotency_key="previous-client")
    with h.store.connect() as db:
        db.execute(
            "UPDATE runs SET status='preparing',worker_pid=NULL,worker_identity=NULL "
            "WHERE run_id=?",
            (state["run_id"],),
        )
    assert (
        h.get_run_status(state["run_id"])["next_action"] == "check_submitting_client_or_cancel_run"
    )
    retry = h.submit_analysis(plan, idempotency_key="previous-client", asynchronous=False)
    assert retry["run_id"] == state["run_id"] and retry["status"] == "preparing"
    assert retry["next_action"] == "check_submitting_client_or_cancel_run"
    assert h.cancel_run(retry["run_id"])["status"] == "cancelled"


def test_inspection_dataset_and_axis_replies_validate_actual_choices(tmp_path):
    h = Harness(tmp_path / "h")
    arrays = np.arange(4 * 7 * 9, dtype=float).reshape((4, 7, 9))
    source = tmp_path / "sets.npz"
    np.savez(source, first=arrays + 300, second=arrays)
    initial = call(h, "inspect_input", path=str(source))
    claim = initial["input_requests"][0]
    selection_id = claim["fields"][0]["evidence"]["selection_artifact_id"]
    assert selection_id in claim["evidence_artifact_ids"]
    assert h.read_artifact(selection_id)["value"]["fields"][0]["choices"] == ["first", "second"]
    assert claim["fields"][0]["answer_schema"]["dataset_path"]["value"]["enum"] == [
        "first",
        "second",
    ]
    load = {"input_path": str(source), "steps": [{"id": "load", "operation_id": "input.load"}]}
    with pytest.raises(HarnessError):
        h.validate_plan(answer_plan(load, claim, dataset_path={"value": "absent"}))
    selected = answer_plan(load, claim, dataset_path={"value": "second"})
    stack = h.submit_analysis(
        selected, parent_run_id=initial["run_id"], idempotency_key="dataset", asynchronous=False
    )
    axis = stack["input_requests"][0]
    assert stack["status"] == "needs_input" and axis["fields"][0]["shape"] == list(arrays.shape)
    selected = h.store.get_run(stack["run_id"])["request"]
    for wrong in [4, -1, True, 0.5]:
        with pytest.raises(HarnessError):
            h.validate_plan(answer_plan(selected, axis, frame={"value": wrong}))
    good = answer_plan(selected, axis, selection_indices={"value": [2]})
    final = h.submit_analysis(
        good, parent_run_id=stack["run_id"], idempotency_key="axis", asynchronous=False
    )
    assert final["status"] == "completed", final
    np.testing.assert_array_equal(
        h.store.load_array(h.store.role(final["run_id"], "raw")), arrays[2]
    )


def test_gpa_questions_use_actual_fft_and_transform_source_roi(tmp_path):
    h = Harness(tmp_path / "h")
    source = tmp_path / "hrtem.npy"
    yy, xx = np.mgrid[:96, :112]
    np.save(source, np.cos(2 * np.pi * xx / 8) + np.cos(2 * np.pi * yy / 8))
    initial = {
        "input_path": str(source),
        "modality": "HRTEM",
        "selection": {"roi": [8, 72, 16, 96]},
        "steps": [
            {"id": "load", "operation_id": "input.load"},
            {"id": "prep", "operation_id": "preprocessing.apply"},
            {"id": "strain", "operation_id": "strain.gpa"},
        ],
    }
    state = h.submit_analysis(initial, asynchronous=False)
    question = state["input_requests"][0]
    field = question["fields"][0]
    table = h.read_artifact(field["evidence"]["peak_table_artifact_id"])
    rows = table["rows"]
    assert len(rows) >= 4 and not table["artifact"]["execution_usable"]
    vectors = np.array(
        [[float(r["gx_cycles_per_pixel"]), float(r["gy_cycles_per_pixel"])] for r in rows]
    )
    partner = next(
        i for i, vector in enumerate(vectors) if abs(np.linalg.det([vectors[0], vector])) > 0.005
    )
    with pytest.raises(HarnessError, match="distinct"):
        h.validate_plan(answer_plan(initial, question, g_peak_indices={"value": [0, 0]}))
    with pytest.raises(HarnessError, match="non-collinear"):
        h.validate_plan(
            answer_plan(
                initial, question, g_vectors={"value": [[0.1, 0], [0.2, 0]], "unit": "1/pixel"}
            )
        )
    by_index = answer_plan(initial, question, g_peak_indices={"value": [0, partner]})
    changed_image = copy.deepcopy(by_index)
    changed_image["steps"][-1]["parameters"] = {"image_role": "processed"}
    with pytest.raises(HarnessError, match="image role"):
        h.validate_plan(changed_image)
    choice = h.submit_analysis(
        by_index, parent_run_id=state["run_id"], idempotency_key="g-pair", asynchronous=False
    )
    assert choice["status"] == "needs_input", choice
    region = choice["input_requests"][0]
    frame = region["fields"][0]["answer_schema"]["reference_region"]["coordinates"]
    assert frame["origin_yx"] == [8, 16] and frame["shape_yx"] == [64, 80]
    child_plan = h.store.get_run(choice["run_id"])["request"]
    for bounds in ([2, 50, 20, 60], [20, 21, 20, 50], [9, 7, 15, 30], [10, 40, 20]):
        with pytest.raises(HarnessError):
            h.validate_plan(
                answer_plan(child_plan, region, reference_region={"value": bounds, "unit": "pixel"})
            )
    roi = answer_plan(
        child_plan, region, reference_region={"value": [24, 56, 32, 72], "unit": "pixel"}
    )
    valid = h.validate_plan(roi)
    assert valid["plan"]["steps"][-1]["parameters"]["reference_region"] == [16, 48, 16, 56]
    final = h.submit_analysis(
        valid["plan"], parent_run_id=choice["run_id"], idempotency_key="roi", asynchronous=False
    )
    assert final["status"] == "completed", final
    assert h.store.role(final["run_id"], "gpa")

    exported = h.export_report(final["run_id"], ["json", "html"])
    files = {a["role"]: Path(a["path"]) for a in exported["artifacts"]}
    manifest = json.loads(files["report_manifest"].read_bytes())
    peak_answer, roi_answer = manifest["input_response_history"]
    assert [
        peak_answer["parent_run_id"],
        peak_answer["child_run_id"],
        roi_answer["child_run_id"],
    ] == [state["run_id"], choice["run_id"], final["run_id"]]
    assert roi_answer["parent_run_id"] == choice["run_id"]
    assert peak_answer["request_id"] == question["request_id"]
    assert peak_answer["source"] == "user" and peak_answer["basis"] == "Acquisition log"
    assert peak_answer["requested_fields"][0]["evidence"] == field["evidence"]
    assert peak_answer["answers"]["g_peak_indices"]["value"] == [0, partner]
    np.testing.assert_allclose(
        peak_answer["adopted_values"]["g_peak_indices"]["parameters"]["g_vectors"],
        vectors[[0, partner]],
    )
    assert roi_answer["answers"]["reference_region"] == {"value": [24, 56, 32, 72], "unit": "pixel"}
    assert roi_answer["adopted_values"]["reference_region"]["parameters"]["reference_region"] == [
        16,
        48,
        16,
        56,
    ]
    page = files["report_html"].read_text(encoding="utf-8")
    assert all(row["request_id"] in page for row in (peak_answer, roi_answer))


def test_gpa_preview_uses_the_mask_from_the_same_step(tmp_path):
    h = Harness(tmp_path / "h")
    path = tmp_path / "field.npy"
    np.save(path, np.ones((8, 8)))
    run = h.submit_analysis(
        {"input_path": str(path), "steps": [{"id": "load", "operation_id": "input.load"}]},
        asynchronous=False,
    )
    rid = run["run_id"]
    field = np.array([[0.0, 0.0, 2.0], [3.0, 4.0, 5.0]])
    valid = np.array([[False, True, True], [False, True, True]])
    role = "gpa.quality_mask"
    mask = h.store.write_array(rid, "gpa1", role, valid)
    h.store.write_array(rid, "gpa2", role, ~valid)
    stored = h.store.write_array(
        rid,
        "gpa1",
        "gpa.strain_xx",
        field,
        {
            "validity_role": role,
            "validity_sha256": mask["metadata"]["array_sha256"],
        },
    )
    preview = h.read_artifact(stored["artifact_id"], offset=0, limit=4)
    assert preview["values"] == [None, 0.0, 2.0, None]
    assert preview["validity_mask_artifact_id"] == mask["artifact_id"]
    assert "Downloaded arrays can contain zeros" in preview["validity_interpretation"]
    assert h.read_artifact(stored["artifact_id"], offset=3, limit=10)["values"] == [None, 4.0, 5.0]
    assert h.read_artifact(mask["artifact_id"])["values"] == valid.ravel().tolist()
    np.testing.assert_array_equal(h.store.load_array(stored["artifact_id"]), field)
    large = h.store.write_json(
        rid, "gpa1", "gpa",
        {"provenance": {"quality_mask": {"uri": mask["path"]}}, "notes": "x" * 256_001},
    )
    bounded = h.read_artifact(large["artifact_id"])
    assert "value" not in bounded and "256 kB" in bounded["preview_limitation"]
    assert bounded["validity_mask_artifact_id"] is None
    assert "unavailable" in bounded["validity_interpretation"]
    Path(mask["path"]).write_bytes(b"damaged")
    with pytest.raises(HarnessError):
        h.read_artifact(stored["artifact_id"])


def test_gpa_comparison_replies_and_report_for_empty_overlap(tmp_path):
    from autotem.acceptance.cases import write_fixture

    source, _ = write_fixture("J03", tmp_path)
    h = Harness(tmp_path / "h")
    good = {"g_vectors": [[0.125, 0], [0, 0.125]], "reference_region": [16, 48, 16, 48]}
    parameters = {
        "reference_config": good,
        "candidate_config": {**good, "reference_region": [0, 4, 0, 4]},
        "sample_step": 100,
    }
    plan = {
        "input_path": str(source),
        "modality": "HRTEM",
        "steps": [
            {"id": "load", "operation_id": "input.load"},
            {"id": "compare", "operation_id": "strain.gpa_compare", "parameters": parameters},
        ],
    }
    parent = call(h, "submit_analysis", plan=plan, asynchronous=False)
    assert parent["status"] == "needs_input"
    (request,) = parent["input_requests"]
    (field,) = request["fields"]
    assert (field["name"], field["parameter_group"]) == ("reference_region", "candidate_config")
    mask = h.store.load_array(field["evidence"]["quality_mask_artifact_id"])
    assert mask[0:4, 0:4].sum() == 0
    assert mask[16:48, 16:48].all()
    assert h.store.role(parent["run_id"], "gpa_comparison") is None
    invalid = answer_plan(plan, request, reference_region={"value": [0, 4, 0, 4], "unit": "pixel"})
    with pytest.raises(HarnessError, match="0 valid phase pixels"):
        h.validate_plan(invalid)

    reply = answer_plan(
        plan, request, reference_region={"value": good["reference_region"], "unit": "pixel"}
    )
    final = call(
        h,
        "submit_analysis",
        plan=reply,
        parent_run_id=parent["run_id"],
        idempotency_key="clean-candidate-reference",
        asynchronous=False,
    )
    assert final["status"] == "completed"
    reports = call(h, "export_report", run_id=final["run_id"], formats=["json", "html"])
    files = {a["role"]: Path(a["path"]) for a in reports["artifacts"]}
    manifest = json.loads(files["report_manifest"].read_bytes())
    claim = next(c for c in manifest["claim_evidence"] if c["step_id"] == "compare")
    (artifact_id,) = claim["artifacts_by_role"]["gpa_comparison"]
    comparison = call(h, "read_artifact", artifact_id=artifact_id)["value"]
    assert comparison["quality"]["intersection_fraction"] == 0
    assert comparison["diagnosis"]["configs_aligned"] is None
    reason = comparison["diagnosis"]["reason"]
    assert reason in claim["warnings"]
    assert reason in files["report_html"].read_text("utf-8")
    assert h.get_run_status(parent["run_id"])["status"] == "needs_input"


def test_numbered_atom_replies_verify_indices_geometry_and_evidence(tmp_path):
    h = Harness(tmp_path / "h")
    yy, xx = np.mgrid[:72, :80]
    points = [(y, x) for y in range(12, 64, 8) for x in range(12, 72, 8)]
    array = sum(np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 2) for y, x in points)
    path = tmp_path / "atoms.npy"
    np.save(path, array)
    initial = {
        "input_path": str(path),
        "modality": "HAADF-STEM",
        "steps": [
            {"id": "load", "operation_id": "input.load"},
            {"id": "detect", "operation_id": "atomic.detect"},
            {"id": "refine", "operation_id": "atomic.refine", "parameters": {"method": "com"}},
            {"id": "spacing", "operation_id": "atomic.spacing.basis"},
        ],
    }
    parent = h.submit_analysis(initial, asynchronous=False)
    claim = parent["input_requests"][0]
    table_id = claim["fields"][0]["evidence"]["table_artifact_id"]
    table = h.read_artifact(table_id)["rows"]
    coords = np.array([[float(r["y_px"]), float(r["x_px"])] for r in table])
    b = next(
        i
        for i in range(2, len(coords))
        if abs(np.linalg.det((coords[[1, i]] - coords[0]).T)) > 0.01
    )
    invalid_plan = answer_plan(
        initial, claim, base_index={"value": 0}, a_index={"value": 0}, b_index={"value": b}
    )
    with pytest.raises(HarnessError, match="distinct"):
        h.validate_plan(invalid_plan)
    good = answer_plan(
        initial, claim, base_index={"value": 0}, a_index={"value": 1}, b_index={"value": b}
    )
    good["user_input"]["response"].update(source="artifact", evidence_artifact_ids=[table_id])
    revised = copy.deepcopy(good)
    revised["steps"][1]["parameters"] = {"min_distance": 15}
    with pytest.raises(HarnessError, match="evidence changed"):
        h.validate_plan(revised)
    final = h.submit_analysis(
        good, parent_run_id=parent["run_id"], idempotency_key="basis", asynchronous=False
    )
    assert final["status"] == "completed", final
    foreign = h.store.role(final["run_id"], "raw")
    good["user_input"]["response"]["evidence_artifact_ids"] = [foreign]
    with pytest.raises(HarnessError) as out_of_run:
        h.validate_plan(good)
    assert out_of_run.value.code == "CROSS_RUN_ARTIFACT"
    good["user_input"]["response"]["evidence_artifact_ids"] = [table_id]
    Path(h.store.artifact(table_id)["path"]).write_bytes(b"changed table")
    with pytest.raises(HarnessError) as corrupted:
        h.validate_plan(good)
    assert corrupted.value.code == "ARTIFACT_CHANGED"
    retry = h.submit_analysis(good, parent_run_id=parent["run_id"], idempotency_key="basis")
    assert retry["run_id"] == final["run_id"] and retry["status"] == "completed"


def test_historical_missing_input_record_remains_answerable_after_upgrade(tmp_path):
    h, _, plan, parent, original = missing_calibration(tmp_path)
    run_id = parent["run_id"]
    deadline = time.monotonic() + 30
    while any(j["status"] == "running" for j in h.get_run_status(run_id)["jobs"]):
        assert time.monotonic() < deadline, "Legacy worker must finalize before migration"
        time.sleep(0.05)
    # Recreate the old DB shape and needs_input payload from before H10.
    with h.store.connect() as db:
        import json

        error = copy.deepcopy(parent["error"])
        error["details"].pop("request_id")
        db.execute("UPDATE runs SET error=? WHERE run_id=?", (json.dumps(error), run_id))
        db.execute(
            "UPDATE steps SET error=? WHERE run_id=? AND status='needs_input'",
            (json.dumps(error), run_id),
        )
        db.execute("DROP TABLE input_requests")
        db.execute("DROP TABLE input_responses")
        db.execute("PRAGMA user_version=0")
    reopened = Harness(h.root)
    migrated = reopened.get_run_status(run_id)["input_requests"][0]
    assert (
        migrated["request_id"] != original["request_id"]
        and migrated["operation_version"] == "1.0.0"
    )
    same = Harness(h.root).get_run_status(run_id)["input_requests"][0]
    assert same == migrated
    valid = reopened.validate_plan(
        answer_plan(plan, migrated, pixel_size_nm={"value": 1, "unit": "nm/pixel"})
    )
    assert valid["input_response_validation"]["parent_run_id"] == run_id


def test_schema_migration_backup_concurrency_recovery_and_future_refusal(tmp_path):
    from contextlib import closing

    from autotem.harness.store import STORE_SCHEMA_VERSION, Store

    store = Store(tmp_path / "h")
    backups = store.root / "schema-backups"
    assert not backups.exists()
    with closing(sqlite3.connect(store.db_path)) as db:
        db.execute("DROP TABLE input_requests")
        db.execute("DROP TABLE input_responses")
        db.execute("ALTER TABLE runs DROP COLUMN submitted_hash")
        db.execute("ALTER TABLE runs DROP COLUMN worker_identity")
        db.execute("PRAGMA user_version=0")
        db.execute(
            "INSERT INTO runs(run_id,request_hash,request,status,created,updated) "
            "VALUES('old','hash','{}','running',1,1)"
        )
        db.commit()
        with pytest.raises(HarnessError) as busy:
            Store(store.root)
        assert busy.value.code == "STORE_MIGRATION_BUSY" and not backups.exists()
        db.execute("UPDATE runs SET status='completed'")
        db.commit()
        # Keep this connection and committed WAL state alive during simultaneous first opens.
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda _: Store(store.root), range(3)))
        assert db.execute("PRAGMA user_version").fetchone()[0] == STORE_SCHEMA_VERSION
        assert {"worker_identity", "submitted_hash"} <= {
            r[1] for r in db.execute("PRAGMA table_info(runs)")
        }
    (saved,) = backups.glob("harness-v0-*.sqlite3")
    assert saved.stem == "harness-v0-" + file_hash(saved)
    with closing(sqlite3.connect(saved.as_uri() + "?mode=ro", uri=True)) as backup:
        assert backup.execute("PRAGMA user_version").fetchone()[0] == 0
        assert (
            backup.execute("SELECT status FROM runs WHERE run_id='old'").fetchone()[0]
            == "completed"
        )
        assert not backup.execute(
            "SELECT 1 FROM sqlite_master WHERE name='input_requests'"
        ).fetchall()
    with closing(sqlite3.connect(store.db_path)) as db:
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("PRAGMA user_version=97")
    before = file_hash(store.db_path)
    with pytest.raises(HarnessError) as future:
        Store(store.root)
    assert future.value.code == "STORE_SCHEMA_UNSUPPORTED" and before == file_hash(store.db_path)
    assert list(backups.iterdir()) == [saved]
