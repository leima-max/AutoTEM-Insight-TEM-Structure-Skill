"""Record public run/job state at scientific failure and during report export."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from autotem.acceptance.cases import write_fixture
from autotem.acceptance.driver import runtime_record, write_json
from autotem.acceptance.evidence import digest, require
from autotem.harness import Harness, HarnessError


def transition_matrix(folder, source):
    """Hold only disposable queue records at each persisted scheduler bookkeeping boundary."""
    h = Harness(folder / "controlled")
    result = {}
    for terminal in ("failed", "needs_input", "timed_out", "cancelled", "interrupted"):
        request = {
            "input_path": str(source),
            "steps": [{"id": "read", "operation_id": "input.load"}],
        }
        with patch.object(h, "_start_scheduler", lambda: None):
            state = h.submit_analysis(request)
        rid = state["run_id"]
        with h.store.connect() as db:
            db.execute("UPDATE runs SET status=? WHERE run_id=?", (terminal, rid))
            db.execute("UPDATE jobs SET status='running' WHERE run_id=?", (rid,))
        observed = h.get_run_status(rid)
        try:
            attempt = h.export_report(rid, formats=["json"])
        except HarnessError as exc:
            early = {"error": exc.code}
        else:
            artifact = attempt["artifacts"][0]
            early = {
                "artifact_id": artifact["artifact_id"],
                "sha256": artifact["sha256"],
                "job": h.store.load_json(artifact["artifact_id"])["run"]["jobs"][-1]["status"],
            }
        with h.store.connect() as db:
            db.execute(
                "UPDATE jobs SET status=?,ended=? WHERE run_id=?", (terminal, time.time(), rid)
            )
        export = h.export_report(rid, formats=["json"])
        artifact = export["artifacts"][0]
        settled = h.store.load_json(artifact["artifact_id"])
        result[terminal] = {
            "run_id": rid,
            "public_before": observed["status"],
            "job_before": observed["jobs"][-1]["status"],
            "early": early,
            "settled": {
                "artifact_id": artifact["artifact_id"],
                "sha256": artifact["sha256"],
                "run": settled["run"]["status"],
                "job": settled["run"]["jobs"][-1]["status"],
                "type": settled["report_type"],
            },
        }
    return result


def assessment(data):
    errors = []
    require(data["trials"], "No real failure trial")
    for t in data["trials"]:
        settled = t["settled_export"]
        require(
            settled["report_type"] == "diagnostic"
            and settled["run"]["status"] == settled["run"]["jobs"][-1]["status"] == "failed"
            and t["append"]["error"] == "RUN_NOT_READY",
            "Failed parent or final report changed",
        )
        before = t["public_failed"]
        exported = t["immediate_export"]
        if before["status"] == "failed" and before["jobs"][-1]["status"] != "failed":
            errors.append("failed run returned before job settled")
        if "run" in exported and exported["run"]["jobs"][-1]["status"] != "failed":
            errors.append("diagnostic artifact persisted with running job")
    matrix = data["controlled_scheduler_boundaries"]
    require(
        set(matrix) == {"failed", "needs_input", "timed_out", "cancelled", "interrupted"},
        "Missing controlled terminal boundary",
    )
    for terminal, row in matrix.items():
        require(
            row["settled"]["run"] == row["settled"]["job"] == terminal
            and row["settled"]["type"] == "diagnostic",
            "Final diagnostic outcome mismatch",
        )
        if row["early"].get("error") != "RUN_NOT_FINAL":
            errors.append(terminal + " diagnostic export accepted before job settled")
    return {
        "verdict": "failed" if errors else "passed",
        "gaps": errors,
        "real_run_ids": [t["run_id"] for t in data["trials"]],
        "controlled_boundaries": sorted(matrix),
    }


def observe(folder, bundle):
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    require(not any(folder.iterdir()), "A fresh probe output is required")
    runtime = runtime_record(True, bundle)
    fn = Path(__file__).resolve()
    source, _ = write_fixture("J03", folder)
    h = Harness(folder / "harness")
    plan = {
        "input_path": str(source),
        "modality": "HRTEM",
        "steps": [
            {"id": "read", "operation_id": "input.load"},
            {
                "id": "reject",
                "operation_id": "strain.gpa",
                "parameters": {
                    "g_vectors": [[0.125, 0], [0.25, 0]],
                    "reference_region": [16, 48, 16, 48],
                },
            },
        ],
    }
    follow = {**plan, "steps": [{"id": "extra", "operation_id": "quality.assess"}]}
    result = {
        "schema_version": "1.0.0",
        "release_manifest_sha256": runtime["release_identity"]["manifest_sha256"],
        "verifier_sha256": digest(fn),
        "trials": [],
    }
    for _ in range(3):
        run = h.submit_analysis(plan)
        rid = run["run_id"]
        end = time.monotonic() + 90
        seen = []
        transitions = []
        while time.monotonic() < end:
            state = h.get_run_status(rid)
            pair = (state["status"], state["jobs"][-1]["status"])
            if pair not in seen:
                seen.append(pair)
                transitions.append(
                    {
                        "state": pair,
                        "execution_seconds": state["execution_seconds"],
                        "time_ns": time.time_ns(),
                    }
                )
            if (
                state["status"] in {"failed", "finalizing"}
                and (state.get("error") or {}).get("code") == "INVALID_SCIENTIFIC_INPUT"
            ):
                break
            time.sleep(0.01)
        require(
            state["status"] in {"failed", "finalizing"}
            and state["error"]["code"] == "INVALID_SCIENTIFIC_INPUT",
            "Expected controlled failure",
        )
        before = state

        def read_report(rid=rid):
            try:
                value = h.export_report(rid, formats=["json"])
            except HarnessError as exc:
                return {"error": exc.code}
            art = value["artifacts"][0]
            data = h.store.load_json(art["artifact_id"])
            return {
                "artifact_id": art["artifact_id"],
                "sha256": art["sha256"],
                "report_type": data["report_type"],
                "run": data["run"],
            }

        def append(rid=rid):
            try:
                value = h.submit_analysis(follow, run_id=rid)
            except HarnessError as exc:
                return {"error": exc.code, "message": str(exc)}
            return {"status": value["status"]}

        with ThreadPoolExecutor(max_workers=2) as pool:
            report = pool.submit(read_report)
            addition = pool.submit(append)
            artifact, append_result = report.result(), addition.result()
        while time.monotonic() < end:
            after = h.get_run_status(rid)
            if after["jobs"][-1]["status"] == "failed":
                break
            time.sleep(0.02)
        require(after["jobs"][-1]["status"] == "failed", "Final scheduler accounting missing")
        final = read_report()
        require(
            final["run"]["jobs"][-1]["status"] == "failed" and final["report_type"] == "diagnostic",
            "Expected finalized diagnostic",
        )
        old_hash = artifact.get("sha256")
        retained = (
            (old_hash == digest(Path(h.store.artifact(artifact["artifact_id"])["path"])))
            if old_hash
            else None
        )
        require(append_result.get("error") == "RUN_NOT_READY", "Failed parent accepted append")
        require(
            [s["step_id"] for s in after["steps"]] == ["read", "reject"],
            "Failed parent graph changed",
        )
        result["trials"].append(
            {
                "run_id": rid,
                "transitions": transitions,
                "public_failed": before,
                "immediate_export": artifact,
                "append": append_result,
                "after_finalization": after,
                "settled_export": final,
                "first_report_hash_retained": retained,
            }
        )
        write_json(folder / "failure-finalization.json", result)
        early = artifact.get("run", {}).get("jobs", [])
        if early and early[-1]["status"] == "running" or before["status"] == "finalizing":
            break
    result["controlled_scheduler_boundaries"] = transition_matrix(folder, source)
    write_json(folder / "failure-finalization.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = observe(args.output, args.release_bundle)
    print(
        json.dumps(
            [
                {
                    "run": v["run_id"],
                    "at_failure": [
                        v["public_failed"]["status"],
                        v["public_failed"]["jobs"][-1]["status"],
                    ],
                    "at_export": v["immediate_export"]
                    .get("run", {})
                    .get("jobs", [{}])[-1]
                    .get("status"),
                    "append": v["append"].get("error"),
                    "settled": v["settled_export"]["run"]["jobs"][-1]["status"],
                }
                for v in data["trials"]
            ]
        )
    )
    sys.exit(0)
