"""RG2 qualification of an installed traditional-core-mcp release and its registered evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sqlite3
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from autotem.acceptance.driver import runtime_record, write_json
from autotem.acceptance.evidence import audit_artifact, digest, require
from autotem.harness import Harness
from autotem.harness.store import ARTIFACT_ROWS, Store

BASE_CHECKS = {
    "registered_files_and_release_binding",
    "operation_numeric_wiring",
    "profile_scoping_and_registration",
    "domain_roles_and_input_child",
    "finalized_diagnostic_reports",
}


def read(path):
    return json.loads(path.read_text("utf-8"))


def component(suffix):
    path = Path(__file__).with_name(f"qualify_local_{suffix}.py")
    spec = importlib.util.spec_from_file_location(f"_autotem_qualification_{suffix}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def captured(check):
    try:
        result = check()
        return {"verdict": "passed", **(result or {})}
    except Exception as exc:
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        return {
            "verdict": "failed",
            "error_type": type(exc).__name__,
            "reason": (str(exc) or frame.line or "check failed")[:1000],
            "check_location": f"{Path(frame.filename).name}:{frame.lineno}",
        }


class RecordedRun:
    """Use read-only registration metadata and independently validated bytes."""

    def __init__(self, root, case_id, run_id, report_id, *, store_root=None):
        self.root, self.case_id, self.run_id = root, case_id, run_id
        require(re.fullmatch("[0-9a-f]{32}", run_id), "Invalid evidence run ID")
        store = store_root if store_root is not None else root / "harness"
        database = store / "harness.sqlite3"
        with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(ARTIFACT_ROWS + " AND a.run_id=?", (run_id,)).fetchall()
        self.index, self.audit = {}, {}
        for item in rows:
            a = Store._artifact_row(item)
            a["path"] = str(store / "runs" / run_id / "artifacts" / Path(a["path"]).name)
            self.audit[a["artifact_id"]] = audit_artifact(root, run_id, a, store_root=store)
            self.index[a["artifact_id"]] = a
        self.report = read(Path(self.index[report_id]["path"]))
        require(self.report["run"]["run_id"] == run_id, "Report belongs to another run")
        wanted = {k: a["sha256"] for k, a in self.index.items() if a["step_id"] != "_report"}
        require(
            wanted == {a["artifact_id"]: a["sha256"] for a in self.report["artifacts"]},
            "Registered artifacts and report differ",
        )
        self.claims = {s["operation_id"]: s for s in self.report["claim_evidence"]}

    def ref(self, op, role):
        ids = self.claims[op]["artifacts_by_role"][role]
        require(len(ids) == 1, f"Ambiguous {op}/{role}")
        entry = self.index[ids[0]]
        require(entry["execution_usable"] and entry["role"] == role, "Unusable registered role")
        return Path(entry["path"])

    def value(self, op, role):
        return read(self.ref(op, role))

    def array(self, op, role):
        return np.load(self.ref(op, role), allow_pickle=False)


def bindings(science, variants, runtime):
    evidence = read(science / "evidence.json")
    require(
        digest(science / "operations.json") == evidence["operations"]["sha256"],
        "Science matrix SHA",
    )
    matrix = read(science / "operations.json")["operations"]
    extra = read(variants / "operations.json")
    for item in (evidence, extra):
        require(
            item["case_verdict"] == "passed" and item["scientific_accuracy"] == "not_evaluated",
            "Source gate or scientific boundary differs",
        )
        require(
            item["runtime"]["release_identity"] == runtime["release_identity"]
            and item["runtime"]["private_install_verified"] is True
            and item["runtime"]["python_source_tree_sha256"]
            == runtime["python_source_tree_sha256"],
            "Evidence source is a different installation",
        )
    require(
        extra["driver"]["source_archive_match"] is True,
        "Variant verifier differs from release archive",
    )
    frames = {}
    for fixture in evidence["fixtures"]:
        path = (science / fixture["path"]).resolve()
        require(
            path.is_relative_to(science) and digest(path) == fixture["sha256"],
            "Fixture SHA or root",
        )
    for case in evidence["cases"]:
        require(case["case_verdict"] == "passed", "Failed science case")
        frame = RecordedRun(science, case["case_id"], case["run_id"], case["report_artifact_id"])
        require(
            all(frame.audit[a["artifact_id"]]["sha256"] == a["sha256"] for a in case["artifacts"]),
            "Science case digest mismatch",
        )
        frames[case["case_id"]] = frame
    for name, ref in extra["cases"].items():
        path = (variants / ref["case_evidence"]).resolve()
        require(
            path.is_relative_to(variants) and digest(path) == ref["case_evidence_sha256"],
            "Variant case SHA mismatch",
        )
        case = read(path)
        require(
            case["runtime"]["release_identity"] == runtime["release_identity"]
            and case["run_id"] == ref["run_id"]
            and case["verdict"] == "passed",
            "Variant case binding",
        )
        frames[name] = RecordedRun(path.parent, name, case["run_id"], case["report_artifact_id"])
    return evidence, matrix, extra, frames


def roles_and_child(science, base, frames, matrix):
    profiles = set()
    for frame in frames.values():
        require(
            frame.report["scientific_accuracy"] == "not_evaluated", "Scientific boundary changed"
        )
        for claim in frame.claims.values():
            definition = matrix[claim["operation_id"]]
            require(claim["status"] == "completed" and claim["delivered"], "Undelivered claim")
            require(claim["operation_version"] == definition["version"], "Claim version mismatch")
            for role in definition["output_roles"]:
                require(role in claim["artifacts_by_role"], "Declared output role missing: " + role)
            for role, aid in claim["consumed_artifacts"].items():
                require(
                    aid in frame.index
                    and frame.index[aid]["role"] == role.removeprefix("reference:")
                    and frame.index[aid]["execution_usable"],
                    "Foreign or unusable input role",
                )
            profiles.add(definition["profile_id"])
    require(
        {"data-quality", "diffraction", "atomic-strain"}.issubset(profiles), "Missing domain role"
    )
    child = next(c for c in base["cases"] if c["case_id"] == "J06")
    reply = child["input_response"]
    parent = RecordedRun(
        science, "J06-parent", reply["parent_run_id"], reply["diagnostic_report_id"]
    )
    require(
        parent.report["report_type"] == "diagnostic"
        and parent.report["run"]["status"] == "needs_input",
        "Parent diagnostic status",
    )
    for artifact in reply["parent_artifacts"]:
        require(
            parent.audit[artifact["artifact_id"]]["sha256"] == artifact["sha256"],
            "J06 parent changed",
        )
    accepted = reply["accepted_reply"]
    require(
        child["parent_run_id"] == parent.run_id
        and reply["invalid_reply_code"] == "INVALID_INPUT_RESPONSE"
        and reply["adopted"]["parameters"] == {"pixel_size_nm": 0.02}
        and accepted["source"] == "user"
        and accepted["answers"] == {"pixel_size_nm": {"value": 20, "unit": "pm/pixel"}},
        "J06 calibration response or units",
    )
    require(
        accepted["request_id"] == reply["request_id"]
        and all(a in parent.index for a in accepted["evidence_artifact_ids"]),
        "J06 answer links",
    )
    return {
        "verdict": "passed",
        "profiles": sorted(profiles),
        "parent_run_id": parent.run_id,
        "child_run_id": child["run_id"],
        "request_id": reply["request_id"],
        "invalid_reply_code": reply["invalid_reply_code"],
        "adopted": reply["adopted"],
    }


def supporting_files(output, fault_names):
    """Freeze mutable SQLite/WAL state through SQLite's backup transaction."""
    snapshots = output / "sqlite-snapshots"
    snapshots.mkdir()
    files = {}
    for name in sorted(fault_names):
        directory = output / name
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(output)
            if {"__pycache__", "cache"}.intersection(relative.parts) or path.suffix in {
                ".pyc",
                ".pyo",
            }:
                continue
            if "extension-venv" in relative.parts and path.name not in {
                "qualification_pixel.py",
                "qualification_broken.py",
                "METADATA",
                "entry_points.txt",
                "release-libraries.pth",
            }:
                continue
            if path.name == "harness.sqlite3":
                copy = snapshots / relative
                copy.parent.mkdir(parents=True, exist_ok=True)
                with (
                    sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as src,
                    sqlite3.connect(copy) as dst,
                ):
                    src.backup(dst)
                file = copy
            elif "harness.sqlite3-" in path.name or path.suffix == ".lock":
                continue
            else:
                file = path
            files[file.relative_to(output).as_posix()] = digest(file)
    return files


def verifier_hashes():
    return {
        p.name: digest(p)
        for p in (
            Path(__file__),
            *(
                Path(__file__).with_name(f"qualify_local_{s}.py")
                for s in ("numerics", "faults", "state_probe")
            ),
        )
    }


def recheck(certificate, bundle, science, variants):
    """Audit an unpacked RC, both original evidence roots and captured fault evidence offline."""
    from autotem.installation.release import verify_release

    bundle, science, variants = map(lambda p: Path(p).resolve(), (bundle, science, variants))
    certificate = Path(certificate).resolve()
    base_dir = certificate.parent
    ev = read(certificate)
    hashes = verifier_hashes()
    require(
        ev["schema_version"] == "1.0.0"
        and ev["gate"] == "RG2"
        and ev["case_verdict"] in {"passed", "failed"}
        and ev.get("scientific_accuracy") == "not_evaluated",
        "Incomplete certificate",
    )
    require(
        ev["verifier_files"] == hashes
        and ev["verifier_sha256"] == hashes[Path(__file__).name]
        and ev["verifier_set_sha256"]
        == hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "Verification source SHA changed",
    )
    release = verify_release(bundle, profile="traditional-core-mcp")
    identity = ev["runtime"]["release_identity"]
    require(
        ev["release_manifest_sha256"] == release.manifest_sha256 == identity["manifest_sha256"]
        and identity["wheel_sha256"] == digest(release.wheel)
        and identity["lock_sha256"] == digest(release.lock),
        "Release SHA changed",
    )
    base, matrix, extra, frames = bindings(science, variants, ev["runtime"])
    expected_sources = {
        "science/evidence.json": digest(science / "evidence.json"),
        "science/operations.json": digest(science / "operations.json"),
        "variants/operations.json": digest(variants / "operations.json"),
    }
    require(ev["source_evidence_sha256"] == expected_sources, "Changed science source")
    faults = component("faults")
    numerical = component("numerics")
    required = BASE_CHECKS | faults.CHECKS.keys()
    require(
        required == ev["checks"].keys()
        and all(
            c["verdict"] == "passed"
            for name, c in ev["checks"].items()
            if name != "finalized_diagnostic_reports"
        ),
        "Missing or failed internal gate",
    )
    require(
        set(ev["operations"]) == matrix.keys() and len(matrix) == 49, "Operation inventory changed"
    )
    numeric = set()
    excluded = {}
    for op, row in ev["operations"].items():
        contract = matrix[op]
        require(
            row["operation_version"] == contract["version"]
            and row["negative_contract"] == contract["negative_contract"],
            "Operation version or error contract changed",
        )
        if contract["implementation_status"] == "reserved":
            require(
                row["verdict"] == "reserved"
                and row["negative_contract"]["observed_code"] == "DEFERRED_THIS_PHASE",
                "Reserved result",
            )
        elif op in {"simulation.diffraction", "diffraction.match"}:
            excluded[op] = row["verdict"]
            require(
                extra["operations"][op]["verdict"] == row["verdict"],
                "Optional operation result changed",
            )
        else:
            require(
                row["verdict"] == "passed"
                and row["disk_execution"] == "passed"
                and row["comparisons"],
                "Missing numerical comparison: " + op,
            )
            numeric.add(op)
            require(
                [c["case"] for c in row["comparisons"]] == numerical.expected_cases(op, frames),
                "Missing operation/case comparison: " + op,
            )
            for compare in row["comparisons"]:
                f = frames[compare["case"]]
                claim = f.claims[op]
                require(
                    compare["run_id"] == f.run_id
                    and compare["consumed_artifacts"] == claim["consumed_artifacts"],
                    "Scientific input provenance",
                )
                require(
                    compare["outputs"]
                    == {
                        role: {i: f.audit[i]["sha256"] for i in ids}
                        for role, ids in claim["artifacts_by_role"].items()
                    },
                    "Scientific output SHA or role",
                )
    require(
        len(numeric) == 35
        and excluded
        == {"simulation.diffraction": "dependency_unavailable", "diffraction.match": "blocked"},
        "Profile exclusions or numerical scope",
    )
    require(
        ev["checks"]["registered_files_and_release_binding"]
        == {
            "verdict": "passed",
            "runs": len(frames),
            "files": sum(len(f.audit) for f in frames.values()),
        }
        and ev["checks"]["operation_numeric_wiring"]
        == {"verdict": "passed", "in_scope": len(numeric), "gaps": []}
        and ev["checks"]["profile_scoping_and_registration"]
        == {
            "verdict": "passed",
            "reserved": 12,
            "dependency_unavailable": ["simulation.diffraction"],
            "blocked_upstream": ["diffraction.match"],
        },
        "Inconsistent registration or numeric subgate",
    )
    require(
        ev["checks"]["domain_roles_and_input_child"]
        == roles_and_child(science, base, frames, matrix),
        "Changed domain/input chain",
    )
    manifest = base_dir / "supporting-files.json"
    require(ev["supporting_files_sha256"] == digest(manifest), "Supporting inventory digest")
    files = read(manifest)
    require(files, "Missing local qualification files")
    for relative, sha in files.items():
        path = (base_dir / relative).resolve()
        require(
            path.is_relative_to(base_dir) and digest(path) == sha,
            "Changed local evidence: " + relative,
        )
    for name in faults.CHECKS:
        faults.recheck(name, base_dir, ev["checks"][name], files)
    finalization = component("state_probe")
    proof = base_dir / "finalized_diagnostic_reports" / "failure-finalization.json"
    raw = read(proof)
    require(
        raw["release_manifest_sha256"] == release.manifest_sha256
        and raw["verifier_sha256"] == hashes["qualify_local_state_probe.py"],
        "Terminal-state proof release",
    )
    derived = finalization.assessment(raw)
    require(
        ev["checks"]["finalized_diagnostic_reports"] == {**derived, "trace_sha256": digest(proof)}
        and ev["case_verdict"] == derived["verdict"],
        "Finalized-report proof/verdict",
    )
    for trial in raw["trials"]:
        for capture in (trial["immediate_export"], trial["settled_export"]):
            if "artifact_id" in capture:
                ref = (
                    "finalized_diagnostic_reports/harness/runs/"
                    + trial["run_id"]
                    + "/artifacts/"
                    + capture["artifact_id"]
                    + ".json"
                )
                require(
                    files[ref] == capture["sha256"]
                    and read(base_dir / ref)["run"] == capture["run"],
                    "Captured diagnostic manifest differs from committed bytes",
                )
    for item in raw["controlled_scheduler_boundaries"].values():
        for capture in (item["early"], item["settled"]):
            if "artifact_id" in capture:
                path = (
                    proof.parent
                    / "controlled"
                    / "runs"
                    / item["run_id"]
                    / "artifacts"
                    / (capture["artifact_id"] + ".json")
                )
                document = read(path)
                require(
                    files[path.relative_to(base_dir).as_posix()] == capture["sha256"]
                    and document["run"]["jobs"][-1]["status"] == capture["job"]
                    and document["report_type"] == "diagnostic",
                    "Controlled report changed",
                )
    return {
        "gate": "RG2",
        "case_verdict": derived["verdict"],
        "finalized_report_gaps": derived["gaps"],
        "certificate_sha256": digest(certificate),
        "release_manifest_sha256": release.manifest_sha256,
        "checks": sorted(required),
        "physical_fault_files": len(files),
        "scientific_runs": len(frames),
        "numerical_operations": len(numeric),
        "excluded": excluded,
    }


def qualify(bundle, science, variants, output):
    require(__debug__, "Run qualification without -O")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "A new or empty output directory is required")
    hashes = verifier_hashes()
    evidence = {
        "schema_version": "1.0.0",
        "gate": "RG2",
        "case_verdict": "failed",
        "scientific_accuracy": "not_evaluated",
        "release_ready": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": digest(Path(__file__)),
        "verifier_files": hashes,
        "verifier_set_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        "checks": {},
        "operations": {},
    }
    checks = evidence["checks"]
    try:
        runtime = runtime_record(True, bundle)
        evidence.update(
            runtime=runtime, release_manifest_sha256=runtime["release_identity"]["manifest_sha256"]
        )
        science, variants = science.resolve(), variants.resolve()
        base, matrix, extra, frames = bindings(science, variants, runtime)
        evidence["source_evidence_sha256"] = {
            "science/evidence.json": digest(science / "evidence.json"),
            "science/operations.json": digest(science / "operations.json"),
            "variants/operations.json": digest(variants / "operations.json"),
        }
        checks["registered_files_and_release_binding"] = {
            "verdict": "passed",
            "runs": len(frames),
            "files": sum(len(frame.audit) for frame in frames.values()),
        }
        checks["domain_roles_and_input_child"] = captured(
            lambda: roles_and_child(science, base, frames, matrix)
        )
        registry = {
            v["operation_id"]: v
            for v in Harness(output / "registry").list_capabilities()["operations"]
        }
        require(registry.keys() == matrix.keys(), "Installed registry differs from source matrix")
        numeric = component("numerics")
        for op, definition in registry.items():
            baseline = matrix[op]
            require(definition["version"] == baseline["version"], f"{op} version differs")
            version = definition["version"]
            negative = baseline["negative_contract"]
            row = {
                "operation_version": version,
                "negative_contract": negative,
                "scientific_limitations": definition["scientific_limitations"],
            }
            require(negative["verdict"] == "passed", f"{op} negative contract missing")
            if definition["implementation_status"] == "reserved":
                require(negative["observed_code"] == "DEFERRED_THIS_PHASE", f"{op} reserved code")
                row.update(
                    verdict="reserved", reason="Reserved operation rejected by installed CLI"
                )
            elif op in ("simulation.diffraction", "diffraction.match"):
                expected = "dependency_unavailable" if op == "simulation.diffraction" else "blocked"
                require(
                    extra["operations"][op]["verdict"] == expected, f"{op} profile scope changed"
                )
                if op == "simulation.diffraction":
                    require(
                        definition["availability"]["state"] == "dependency_unavailable",
                        "Optional backend state changed",
                    )
                row.update(
                    verdict=expected,
                    reason=extra["operations"][op]["reason"],
                    availability=definition["availability"]["state"],
                )
            else:
                v = extra["operations"].get(op, {})
                require(
                    baseline["disk_execution"] == "passed" or v.get("verdict") == "passed",
                    f"{op} has no installed disk evidence",
                )
                if v:
                    require(
                        v["release_binding"]["manifest_sha256"]
                        == evidence["release_manifest_sha256"]
                        and v["operation_version"] == version,
                        f"{op} variant version/binding",
                    )
                result = captured(lambda op=op: numeric.verify(op, frames, matrix, extra, output))
                row.update(result, disk_execution="passed")
            evidence["operations"][op] = row
        checked = {
            op: row
            for op, row in evidence["operations"].items()
            if row["verdict"] not in {"reserved", "dependency_unavailable", "blocked"}
        }
        missing = [op for op, row in checked.items() if row["verdict"] != "passed"]
        checks["operation_numeric_wiring"] = {
            "verdict": "failed" if missing else "passed",
            "in_scope": len(checked),
            "gaps": missing,
        }
        checks["profile_scoping_and_registration"] = {
            "verdict": "passed",
            "reserved": sum(d["implementation_status"] == "reserved" for d in registry.values()),
            "dependency_unavailable": ["simulation.diffraction"],
            "blocked_upstream": ["diffraction.match"],
        }
        faults = component("faults")
        for name, fn in faults.CHECKS.items():
            folder = output / name
            folder.mkdir()
            checks[name] = captured(lambda fn=fn, folder=folder: fn(folder, frames, base))
        finalization = component("state_probe")
        terminal_dir = output / "finalized_diagnostic_reports"

        def terminal_qualification():
            record = finalization.observe(terminal_dir, bundle)
            return {
                **finalization.assessment(record),
                "trace_sha256": digest(terminal_dir / "failure-finalization.json"),
            }

        checks["finalized_diagnostic_reports"] = captured(terminal_qualification)
        files = supporting_files(output, [*faults.CHECKS, "finalized_diagnostic_reports"])
        write_json(output / "supporting-files.json", files)
        evidence["supporting_files_sha256"] = digest(output / "supporting-files.json")
        evidence["case_verdict"] = (
            "passed" if all(c["verdict"] == "passed" for c in checks.values()) else "failed"
        )
    except Exception as exc:
        checks["qualification_initialization"] = {"verdict": "failed", "reason": str(exc)[:1000]}
    evidence["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "local-execution.json", evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("release-bundle", "science", "variants"):
        parser.add_argument("--" + field, required=True, type=Path)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path)
    target.add_argument("--recheck", type=Path)
    a = parser.parse_args()
    if a.recheck:
        try:
            observed = recheck(a.recheck, a.release_bundle, a.science, a.variants)
            print(json.dumps(observed))
        except (KeyError, TypeError, OSError, ValueError, AssertionError) as exc:
            print(json.dumps({"gate": "RG2", "case_verdict": "failed", "reason": str(exc)}))
            sys.exit(2)
        sys.exit(0 if observed["case_verdict"] == "passed" else 2)
    result = qualify(a.release_bundle, a.science, a.variants, a.output)
    print(
        json.dumps(
            {
                "gate": "RG2",
                "case_verdict": result["case_verdict"],
                "checks": {k: v["verdict"] for k, v in result["checks"].items()},
            }
        )
    )
    sys.exit(0 if result["case_verdict"] == "passed" else 2)
