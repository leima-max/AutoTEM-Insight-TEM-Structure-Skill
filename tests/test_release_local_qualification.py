"""The maintainer certificate must preserve an expected failure and reject modified evidence."""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "acceptance" / "qualify_local.py"


def load_driver():
    spec = importlib.util.spec_from_file_location("test_installed_rg2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.smoke
def test_full_array_controls_and_registration_boundaries():
    q = load_driver()
    module = q.component("numerics")
    numbers = np.array([1.0, 2.0, np.nan])
    module.arrays_equal(numbers, numbers.copy())
    with pytest.raises(AssertionError):
        module.arrays_equal(numbers, np.array([1.0, 3.0, np.nan]))
    with pytest.raises(ValueError, match="shape"):
        module.arrays_equal(numbers, numbers[None])
    with pytest.raises(ValueError, match="keys"):
        module.equal({"status": "ok"}, {"availability": "ok"})
    result = q.captured(lambda: module.equal({"unit": "pixel"}, {"unit": "nm"}))
    assert result["verdict"] == "failed" and "Value differs" in result["reason"]
    assert set(q.verifier_hashes()) == {
        "qualify_local.py",
        "qualify_local_numerics.py",
        "qualify_local_faults.py",
        "qualify_local_state_probe.py",
    }


@pytest.mark.smoke
def test_sqlite_evidence_freezes_uncheckpointed_wal_and_rejects_stale_certificate(tmp_path):
    q = load_driver()
    fault = "owned_jobs_and_isolation"
    directory = tmp_path / fault / "harness"
    directory.mkdir(parents=True)
    with sqlite3.connect(directory / "harness.sqlite3") as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE runs(run_id TEXT,status TEXT,error TEXT,run_seconds_used REAL)")
        db.execute("INSERT INTO runs VALUES ('owned', 'interrupted', NULL, 1.0)")
        db.commit()
        inventory = q.supporting_files(tmp_path, [fault])
    snapshot = tmp_path / "sqlite-snapshots" / fault / "harness" / "harness.sqlite3"
    assert inventory[snapshot.relative_to(tmp_path).as_posix()] == q.digest(snapshot)
    with sqlite3.connect(snapshot) as db:
        assert db.execute("SELECT run_id,status FROM runs").fetchall() == [("owned", "interrupted")]
    with pytest.raises(ValueError, match="Unverified"):
        q.component("faults").recheck(fault, tmp_path, {"verdict": "failed"}, inventory)
    cert = tmp_path / "local-execution.json"
    cert.write_text(
        json.dumps({"schema_version": "1.0.0", "gate": "RG2", "case_verdict": "failed"}), "utf-8"
    )
    with pytest.raises(ValueError, match="Incomplete certificate"):
        q.recheck(cert, tmp_path, tmp_path, tmp_path)


@pytest.fixture
def interrupted_artifact(tmp_path):
    rid, aid = "1" * 32, "a" * 32
    store = tmp_path / "fault" / "harness"
    files = store / "runs" / rid / "artifacts"
    files.mkdir(parents=True)
    temporary = files / f"{aid}.png.tmp"
    temporary.write_bytes(b"interrupted write")
    published = files / ("b" * 32 + ".png")
    published.write_bytes(b"registered evidence")
    database = store / "harness.sqlite3"
    with sqlite3.connect(database) as db:
        db.executescript(
            "CREATE TABLE runs(run_id TEXT,status TEXT,active_started REAL,current_step TEXT);"
            "CREATE TABLE jobs(run_id TEXT,status TEXT,ended REAL);"
            "CREATE TABLE artifacts(artifact_id TEXT,run_id TEXT,path TEXT);"
        )
        db.execute("INSERT INTO runs VALUES (?, 'cancelled', NULL, NULL)", (rid,))
        db.execute("INSERT INTO jobs VALUES (?, 'cancelled', 2.0)", (rid,))
        db.execute("INSERT INTO artifacts VALUES (?,?,?)", ("b" * 32, rid, str(published)))
    return database, temporary, published


@pytest.mark.parametrize("status", ["cancelled", "interrupted", "timed_out"])
def test_support_keeps_registry_and_science_but_excludes_abandoned_stream(
    tmp_path, interrupted_artifact, status
):
    database, temporary, published = interrupted_artifact
    with sqlite3.connect(database) as db:
        db.execute("UPDATE runs SET status=?", (status,))
        db.execute("UPDATE jobs SET status=?", (status,))
    q = load_driver()
    manifest = q.supporting_files(tmp_path, ["fault"])
    assert manifest[published.relative_to(tmp_path).as_posix()] == q.digest(published)
    assert ("sqlite-snapshots/" + database.relative_to(tmp_path).as_posix()) in manifest
    assert temporary.relative_to(tmp_path).as_posix() not in manifest
    assert temporary.read_bytes() == b"interrupted write"


@pytest.mark.parametrize(
    "variant",
    [
        "running",
        "unsettled",
        "different_run",
        "registered",
        "registered_path",
        "crossrun_path",
        "unknown_type",
        "wrong_parent",
        "unknown_name",
        "hard_link",
        "final_exists",
        "directory",
    ],
)
def test_support_rejects_ambiguous_temporary_paths(tmp_path, interrupted_artifact, variant):
    database, temporary, published = interrupted_artifact
    if variant in {
        "running",
        "unsettled",
        "different_run",
        "registered",
        "registered_path",
        "crossrun_path",
    }:
        with sqlite3.connect(database) as db:
            if variant == "running":
                db.execute("UPDATE runs SET status='running',active_started=1,current_step='load'")
            elif variant == "unsettled":
                db.execute("INSERT INTO jobs VALUES (?, 'running', NULL)", ("1" * 32,))
            elif variant == "different_run":
                db.execute("UPDATE runs SET run_id=?", ("2" * 32,))
            else:
                aid = "a" * 32 if variant == "registered" else "c" * 32
                path = published if variant == "registered" else temporary
                rid = "2" * 32 if variant == "crossrun_path" else "1" * 32
                db.execute("INSERT INTO artifacts VALUES (?,?,?)", (aid, rid, str(path)))
    elif variant == "hard_link":
        os.link(temporary, tmp_path / "linked-copy")
    elif variant == "final_exists":
        temporary.with_suffix("").write_bytes(b"also present")
    elif variant == "directory":
        temporary.unlink()
        temporary.mkdir()
    else:
        destination = {
            "unknown_type": temporary.with_name("a" * 32 + ".exe.tmp"),
            "wrong_parent": temporary.parent.parent / temporary.name,
            "unknown_name": temporary.with_name("unrelated.png.tmp"),
        }[variant]
        temporary.rename(destination)
    with pytest.raises(ValueError, match="staging|Staging|temporary"):
        load_driver().supporting_files(tmp_path, ["fault"])


@pytest.mark.smoke
def test_private_runtime_is_required_and_failed_gate_is_saved(tmp_path):
    q = load_driver()
    result = q.qualify(tmp_path, tmp_path, tmp_path, tmp_path / "attempt")
    assert (
        result["case_verdict"] == "failed"
        and result["checks"]["qualification_initialization"]["verdict"] == "failed"
    )
    assert result == q.read(tmp_path / "attempt" / "local-execution.json")


@pytest.mark.smoke
def test_terminal_report_requires_job_settlement_for_every_terminal_state():
    module = load_driver().component("state_probe")
    settled = {"status": "failed", "jobs": [{"status": "failed"}]}
    trial = {
        "run_id": "same",
        "public_failed": settled,
        "immediate_export": {"error": "RUN_NOT_FINAL"},
        "settled_export": {"report_type": "diagnostic", "run": settled},
        "append": {"error": "RUN_NOT_READY"},
    }
    rows = {
        state: {
            "early": {"error": "RUN_NOT_FINAL"},
            "settled": {"run": state, "job": state, "type": "diagnostic"},
        }
        for state in ("failed", "needs_input", "timed_out", "cancelled", "interrupted")
    }
    trace = {"trials": [trial], "controlled_scheduler_boundaries": rows}
    assert module.assessment(trace)["verdict"] == "passed"
    rows["cancelled"]["early"] = {"job": "running"}
    assert module.assessment(trace)["gaps"] == [
        "cancelled diagnostic export accepted before job settled"
    ]
