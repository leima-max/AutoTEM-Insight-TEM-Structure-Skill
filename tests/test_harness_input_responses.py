"""Disk-to-child acceptance for versioned human input through the public Harness."""

from __future__ import annotations

import copy
import sqlite3
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
