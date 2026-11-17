"""Re-audit historical failed releases from immutable, same-release attempts and reports."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from autotem.acceptance.driver import runtime_record, write_json
from autotem.acceptance.evidence import digest, require


def common():
    target = Path(__file__).with_name("qualify_local.py")
    spec = importlib.util.spec_from_file_location("local_qualification_common", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def failure_proof(q, proof, controlled, identity):
    actual = q.read(proof)
    require(
        actual["release_manifest_sha256"] == identity["manifest_sha256"], "Failure proof release"
    )
    observations = []
    for trial in actual["trials"]:
        rid = trial["run_id"]
        for when in ("immediate_export", "settled_export"):
            capture = trial[when]
            path = (
                proof.parent
                / "harness"
                / "runs"
                / rid
                / "artifacts"
                / (capture["artifact_id"] + ".json")
            )
            document = q.read(path)
            require(
                digest(path) == capture["sha256"]
                and document["run"] == capture["run"]
                and document["report_type"] == "diagnostic",
                "Original diagnostic bytes changed",
            )
        first = trial["immediate_export"]["run"]
        settled = trial["settled_export"]["run"]
        require(
            first["status"] == settled["status"] == "failed"
            and first["jobs"][-1]["status"] == "running"
            and settled["jobs"][-1]["status"] == "failed"
            and trial["append"]["error"] == "RUN_NOT_READY"
            and trial["first_report_hash_retained"],
            "This audit requires an actually reproduced premature diagnostic",
        )
        observations.append(
            {
                "run_id": rid,
                "early_sha256": trial["immediate_export"]["sha256"],
                "final_sha256": trial["settled_export"]["sha256"],
                "early_job": "running",
                "settled_job": "failed",
                "append": "RUN_NOT_READY",
            }
        )
    require(observations, "Missing public failure proof")
    database = controlled / "controlled" / "harness.sqlite3"
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = list(
            db.execute(
                "SELECT r.run_id,r.status,a.artifact_id,a.sha256 FROM runs r "
                "JOIN artifacts a ON r.run_id=a.run_id "
                "WHERE a.role='report_manifest' ORDER BY a.created"
            )
        )
    staged = {}
    for rid, status, aid, sha in rows:
        path = database.parent / "runs" / rid / "artifacts" / (aid + ".json")
        doc = q.read(path)
        require(
            doc["run"]["run_id"] == rid
            and doc["report_type"] == "diagnostic"
            and digest(path) == sha,
            "Controlled diagnostic evidence changed",
        )
        staged.setdefault(status, []).append(
            {"run_id": rid, "report_sha256": sha, "reported_job": doc["run"]["jobs"][-1]["status"]}
        )
    require(
        set(staged) == {"failed", "cancelled", "needs_input", "interrupted", "timed_out"}
        and all(
            [r["reported_job"] for r in reports] == ["running", status]
            for status, reports in staged.items()
        ),
        "Controlled finalization matrix changed",
    )
    return {
        "verdict": "failed",
        "reason": "Diagnostic manifests retained a running job before scheduler settlement",
        "live": observations,
        "controlled": staged,
        "trace_sha256": digest(proof),
        "reproduction_verifier_sha256": actual["verifier_sha256"],
    }


def audit(bundle, science, variants, attempts, proof, controlled):
    q = common()
    runtime = runtime_record(True, bundle)
    identity = runtime["release_identity"]
    base, matrix, extra, frames = q.bindings(science, variants, runtime)
    numerical = q.component("numerics")
    faults = q.component("faults")
    checks = {
        "registered_files_and_release_binding": {
            "verdict": "passed",
            "runs": len(frames),
            "files": sum(len(f.audit) for f in frames.values()),
        },
        "domain_roles_and_input_child": q.roles_and_child(science, base, frames, matrix),
    }
    archive = []
    candidates = {n: [] for n in faults.CHECKS}
    selected = {}
    for path in attempts:
        e = q.read(path)
        require(
            e["runtime"]["release_identity"] == identity
            and e["runtime"]["private_install_verified"]
            and e["source_evidence_sha256"]
            == {
                "science/evidence.json": digest(science / "evidence.json"),
                "science/operations.json": digest(science / "operations.json"),
                "variants/operations.json": digest(variants / "operations.json"),
            },
            "Source attempt is a different release or input set",
        )
        index = path.parent / "supporting-files.json"
        require(digest(index) == e["supporting_files_sha256"], "Source attempt inventory SHA")
        files = q.read(index)
        for rel, sha in files.items():
            item = (path.parent / rel).resolve()
            require(
                item.is_relative_to(path.parent) and digest(item) == sha,
                "Changed stored evidence: " + rel,
            )
        label = path.parent.name
        archive.append(
            {
                "directory": label,
                "certificate_sha256": digest(path),
                "supporting_files_sha256": digest(index),
                "verifier_files": e["verifier_files"],
                "observed_gaps": {
                    k: {f: v[f] for f in ("error_type", "reason", "gaps") if f in v}
                    for k, v in e["checks"].items()
                    if v["verdict"] != "passed"
                },
            }
        )
        for name in faults.CHECKS:
            result = e["checks"][name]
            if result["verdict"] != "passed":
                candidates[name].append(
                    {"attempt": label, "observed": "failed", "reason": result.get("reason")}
                )
                continue
            rerun = q.captured(
                lambda n=name, r=result, p=path.parent, inv=files: faults.recheck(n, p, r, inv)
            )
            candidates[name].append(
                {"attempt": label, "observed": rerun["verdict"], "reason": rerun.get("reason")}
            )
            if rerun["verdict"] == "passed":
                selected[name] = label
    for name in faults.CHECKS:
        checks[name] = {
            "verdict": "passed" if name in selected else "failed",
            "selected_attempt": selected.get(name),
            "attempts": candidates[name],
        }
    operations = {}
    with TemporaryDirectory() as temp:
        for op, contract in matrix.items():
            require(contract["negative_contract"]["verdict"] == "passed", "Wrong negative contract")
            if contract["implementation_status"] == "reserved":
                verdict = "reserved"
                result = {"negative_contract": contract["negative_contract"]}
            elif op in {"simulation.diffraction", "diffraction.match"}:
                verdict = extra["operations"][op]["verdict"]
                result = {"reason": extra["operations"][op]["reason"]}
            else:
                result = q.captured(
                    lambda op=op: numerical.verify(op, frames, matrix, extra, Path(temp))
                )
                verdict = result.pop("verdict")
            operations[op] = {
                "operation_version": contract["version"],
                "verdict": verdict,
                **result,
            }
    checked = [op for op, row in operations.items() if row["verdict"] == "passed"]
    checks["operation_numeric_wiring"] = {
        "verdict": "passed" if len(checked) == 35 else "failed",
        "count": len(checked),
    }
    require(
        operations["simulation.diffraction"]["verdict"] == "dependency_unavailable"
        and operations["diffraction.match"]["verdict"] == "blocked",
        "Explicit optional exclusions changed",
    )
    checks["profile_scoping_and_registration"] = {
        "verdict": "passed",
        "reserved": sum(row["verdict"] == "reserved" for row in operations.values()),
        "dependency_unavailable": ["simulation.diffraction"],
        "blocked_upstream": ["diffraction.match"],
    }
    checks["finalized_diagnostic_reports"] = failure_proof(q, proof, controlled, identity)
    return {
        "schema_version": "1.0.0",
        "gate": "RG2",
        "case_verdict": "failed",
        "evidence_kind": "historical_failure_audit",
        "scientific_accuracy": "not_evaluated",
        "release_ready": False,
        "release_manifest_sha256": identity["manifest_sha256"],
        "verifier_sha256": digest(Path(__file__)),
        "supporting_verifiers": q.verifier_hashes(),
        "source_attempts": archive,
        "checks": checks,
        "operations": operations,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("release-bundle", "science", "variants", "reproduction", "controlled"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--attempt", type=Path, required=True, action="append")
    dest = p.add_mutually_exclusive_group(required=True)
    dest.add_argument("--output", type=Path)
    dest.add_argument("--recheck", type=Path)
    a = p.parse_args()
    evidence = audit(
        a.release_bundle.resolve(),
        a.science.resolve(),
        a.variants.resolve(),
        [v.resolve() for v in a.attempt],
        a.reproduction.resolve(),
        a.controlled.resolve(),
    )
    if a.recheck:
        require(
            common().read(a.recheck) == evidence, "Historical audit no longer reproduces exactly"
        )
    else:
        require(not a.output.exists(), "Refuse to replace historical evidence")
        a.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(a.output, evidence)
    print(
        json.dumps(
            {
                "gate": "RG2",
                "case_verdict": evidence["case_verdict"],
                "numerical": evidence["checks"]["operation_numeric_wiring"],
                "gaps": [k for k, v in evidence["checks"].items() if v["verdict"] != "passed"],
            }
        )
    )
    sys.exit(2)
