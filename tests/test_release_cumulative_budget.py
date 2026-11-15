"""Exercise both timing paths without slowing down the machine or running a worker."""

import importlib.util
import json
import sqlite3
from collections import deque
from copy import deepcopy
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke
SCRIPT = Path(__file__).resolve().parents[1] / "tools/acceptance/qualify_local_faults.py"


@pytest.fixture
def probe(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("budget_qualifier", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    folder = tmp_path / "cumulative_run_budget"
    folder.mkdir()
    source = folder / "fixture.npy"
    source.write_bytes(b"input fixture")
    monkeypatch.setattr(module, "sample", lambda _: source)
    results = deque()
    states = {}

    class Harness:
        def submit_analysis(self, request, run_id=None):
            event, seconds = results.popleft()
            if event.isupper():
                raise module.HarnessError(event, "observed rejection")
            rid = run_id or f"run{len(states)}"
            jobs = states.get(rid, {}).get("jobs", [])
            jid = f"{rid}-job{len(jobs)}"
            status = "timed_out" if "timeout" in event else event
            error = None
            if status == "timed_out":
                error = {
                    "code": "TIMEOUT",
                    "message": "Run budget exceeded"
                    if event == "run_timeout"
                    else "Operation budget exceeded",
                }
            if status == "failed":
                error = {"code": "DEPENDENCY_BROKEN"}
            new_job = {
                "run_id": rid,
                "job_id": jid,
                "status": status,
                "ended": seconds,
                "created": float(len(jobs)),
            }
            states[rid] = {
                "run_id": rid,
                "job_id": jid,
                "status": status,
                "error": error,
                "execution_seconds": seconds,
                "jobs": [*jobs, new_job],
                "events": ["observed"],
            }
            return {"run_id": rid, "job_id": jid}

    monkeypatch.setattr(module, "Harness", lambda _: Harness())
    monkeypatch.setattr(module, "terminal", lambda h, rid: deepcopy(states[rid]))
    return module, folder, results


def snapshot(probe, check):
    module, folder, _ = probe
    db_path = folder.parent / "sqlite-snapshots" / folder.name / "harness/harness.sqlite3"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as db:
        db.executescript(
            "CREATE TABLE runs(run_id TEXT,request TEXT,source_hash TEXT,"
            "run_seconds_used REAL,status TEXT,error TEXT);"
            "CREATE TABLE jobs(run_id TEXT,job_id TEXT,status TEXT,ended REAL,created REAL);"
        )
        for ref in check["attempts"]:
            path = folder / ref["file"]
            trace = json.loads(path.read_bytes())
            calls = [c for c in trace["calls"] if "state" in c]
            for call in calls:
                state = call["state"]
                db.execute(
                    "INSERT INTO jobs VALUES (:run_id,:job_id,:status,:ended,:created)",
                    state["jobs"][-1],
                )
            plan = {**calls[0]["request"], "steps": [c["request"]["steps"][0] for c in calls]}
            db.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?)",
                (
                    state["run_id"],
                    json.dumps(plan),
                    trace["input_sha256"],
                    state["execution_seconds"],
                    state["status"],
                    json.dumps(state["error"]) if state["error"] else None,
                ),
            )
    return db_path, {
        p.relative_to(folder.parent).as_posix(): module.digest(p)
        for p in folder.parent.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("slow_first", [None, "completed", "run_timeout"])
def test_default_and_slow_first_keep_real_enforcement_and_recheck(probe, slow_first):
    module, folder, results = probe
    if slow_first:
        results.append((slow_first, 12.14))
        if slow_first == "completed":
            results.append(("RUN_BUDGET_EXHAUSTED", 0))
    selected = 31 if slow_first else 12
    results.extend([("completed", 6), ("completed", selected + 0.1), ("RUN_BUDGET_EXHAUSTED", 0)])
    check = {**module.cumulative(folder, None, None), "verdict": "passed"}
    assert check["budget_seconds"] == selected
    assert len(check["attempts"]) == (2 if slow_first else 1) and not results
    assert [x[1] for x in check["job_measurements"]] == [6, selected + 0.1]
    db, inventory = snapshot(probe, check)
    module.recheck(folder.name, folder.parent, check, inventory)
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM jobs WHERE job_id=?", (check["job_measurements"][0][0],))
    with pytest.raises(ValueError, match="ledger"):
        module.recheck(folder.name, folder.parent, check, inventory)


@pytest.mark.parametrize(
    "events,reason",
    [
        ([("completed", 3), ("QUEUE_FULL", 0)], "Unexpected append rejection"),
        ([("completed", 3), ("RUN_BUDGET_EXHAUSTED", 0)], "Unexpected append rejection"),
        ([("operation_timeout", 12.2)], "Unexpected timeout"),
        ([("failed", 12.2)], "Unexpected budget job result"),
        (
            [("completed", 12.2), ("RUN_BUDGET_EXHAUSTED", 0), ("run_timeout", 31.2)],
            "before a second job",
        ),
        ([("completed", 4), ("run_timeout", 3)], "reset or repeated"),
    ],
)
def test_non_budget_faults_and_single_job_are_not_accepted(probe, events, reason):
    module, folder, results = probe
    results.extend(events)
    with pytest.raises(ValueError, match=reason):
        module.cumulative(folder, None, None)
    assert not results and list(folder.glob("budget-attempt-*.json"))


def test_append_run_timeout_is_a_valid_terminal_enforcement(probe):
    module, folder, results = probe
    results.extend([("completed", 8.1), ("run_timeout", 12.1)])
    check = module.cumulative(folder, None, None)
    assert check["run"]["execution_status"] == "timed_out" and check["budget_seconds"] == 12
    assert module.retry_budget(300) == 90
