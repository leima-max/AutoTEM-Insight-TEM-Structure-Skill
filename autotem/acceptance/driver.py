"""Exercise the public Harness tools using only package code and generated disk fixtures."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import sysconfig
import time
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np

import autotem

from .bridge import UnexpectedResponse, open_bridge, submit_and_wait
from .calibration_child import answer_calibration
from .cases import CASE_MODALITIES, CORE_CASES, REQUESTS, plan_for, read_fixture, write_fixture
from .evidence import RunEvidence, check_atoms, check_diffraction, digest, require
from .strain_pattern import check_atom_strain, check_direct_pattern, check_gpa


def runtime_record(require_installed, bundle=None):
    package = Path(autotem.__file__).resolve().parent
    lib = Path(sysconfig.get_path("purelib")).resolve()
    installed = package.is_relative_to(lib)
    distribution = None
    editable = None
    try:
        distribution = importlib.metadata.distribution("autotem-insight")
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        editable = direct.get("dir_info", {}).get("editable", False)
    except importlib.metadata.PackageNotFoundError:
        pass
    if require_installed:
        require(sys.flags.isolated, "Use the private Python interpreter with -I")
        require(sys.prefix != sys.base_prefix, "A private virtual environment is required")
        config = (Path(sys.prefix) / "pyvenv.cfg").read_text(encoding="utf-8")
        settings = {
            k.strip().lower(): v.strip().lower()
            for line in config.splitlines()
            if "=" in line
            for k, v in [line.split("=", 1)]
        }
        require(
            settings.get("include-system-site-packages") == "false",
            "System site-packages must be disabled",
        )
        require(
            lib.is_relative_to(Path(sys.prefix).resolve())
            and installed
            and distribution is not None
            and not editable,
            "An installed wheel is required",
        )
        require(
            Path(distribution.locate_file("autotem")).resolve() == package,
            "Distribution metadata belongs to a different import",
        )
        require(
            not (Path.cwd() / "autotem" / "__init__.py").is_file(),
            "Run installed acceptance from a directory outside the checkout",
        )
    marker = Path(sys.prefix) / ".autotem-runtime.json"
    identity = {}
    if marker.is_file():
        state = json.loads(marker.read_text("utf-8"))
        identity = {
            k: state[k]
            for k in ("manifest_sha256", "wheel_sha256", "package_version", "profile")
            if k in state
        }
    if bundle is not None:
        from autotem.installation.release import PROFILE, verify_release

        require(require_installed, "Use --require-installed when binding a release bundle")
        verified = verify_release(bundle)
        require(
            identity.get("manifest_sha256") == verified.manifest_sha256
            and identity.get("wheel_sha256") == digest(verified.wheel)
            and identity.get("profile", PROFILE) == verified.profile,
            "Installed runtime marker and supplied release bundle differ",
        )
        with ZipFile(verified.wheel) as wheel:
            members = {
                m.filename
                for m in wheel.infolist()
                if m.filename.startswith("autotem/") and not m.is_dir()
            }
            installed_code = {p.relative_to(lib).as_posix() for p in package.rglob("*.py")}
            require(
                installed_code == {m for m in members if m.endswith(".py")},
                "Installed Python file inventory differs from the release wheel",
            )
            for member in sorted(members):
                require(
                    (lib / member).read_bytes() == wheel.read(member),
                    f"Installed file differs from the release wheel: {member}",
                )
        manifest = json.loads((bundle / "release-manifest.json").read_text("utf-8"))
        identity.update(
            release_id=verified.release_id,
            profile=verified.profile,
            source_archive_sha256=manifest["source"]["archive_sha256"],
            git_commit=manifest["source"]["git_commit"],
            lock_sha256=digest(verified.lock),
        )
    fingerprint = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        fingerprint.update(f"{path.relative_to(package).as_posix()} {digest(path)}\n".encode())
    skill = package / "harness" / "resources" / "skill"
    resources = {
        f.relative_to(skill).as_posix(): digest(f) for f in sorted(skill.rglob("*")) if f.is_file()
    }
    require("SKILL.md" in resources, "Missing installed Skill")
    for name in ("data-quality", "diffraction", "atomic-strain"):
        require(f"references/profiles/{name}.yaml" in resources, f"Missing {name} profile")
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "autotem_version": autotem.__version__,
        "python_source_tree_sha256": fingerprint.hexdigest(),
        "installed_origin": installed,
        "editable": editable,
        "isolated": bool(sys.flags.isolated),
        "private_install_verified": require_installed,
        "release_identity": identity,
        "skill_resource_sha256": resources,
    }


def check_contracts(call, capabilities, source):
    rows = {}
    for definition in capabilities["operations"]:
        name = definition["operation_id"]
        implemented = definition["implementation_status"] == "implemented"
        modality = next(iter(definition["modality"]), "HRTEM")
        plan = {
            "input_path": str(source),
            "modality": modality,
            "steps": [
                {
                    "id": "contract",
                    "operation_id": name,
                    "operation_version": definition["version"],
                    "parameters": {"unexpected_release_parameter": 1} if implemented else {},
                }
            ],
        }
        expected = "INVALID_PARAMETERS" if implemented else "DEFERRED_THIS_PHASE"
        try:
            error = call("validate_plan", {"plan": plan}, expect_error=expected)
            negative = {"verdict": "passed", "observed_code": error["code"]}
        except UnexpectedResponse as exc:
            negative = {"verdict": "failed", "expected_code": expected, "error": str(exc)}
        # The full schema can be large. It is recorded once per operation in this file.
        row = {
            key: definition[key]
            for key in (
                "operation_id",
                "version",
                "implementation_status",
                "implementation_source",
                "profile_id",
                "parameter_schema",
                "input_roles",
                "output_roles",
                "modality",
                "coordinate_contract",
                "permission",
                "budget_seconds",
                "dependencies",
                "scientific_limitations",
                "result_level",
            )
        }
        row.update(
            {
                "availability": {
                    k: v
                    for k, v in definition["availability"].items()
                    if k in {"state", "dependency", "code"}
                },
                "negative_contract": negative,
                "disk_execution": "not_evaluated",
                "direct_api_equivalence": "not_evaluated",
                "execution_evidence": [],
            }
        )
        rows[name] = row
    require(rows, "Empty capability registry")
    return rows


def run_case(call, output: Path, fixture, source, matrix, *, store_root: Path | None = None):
    case_id = fixture["case_id"]
    plan = plan_for(case_id, source)
    inspection = call(
        "inspect_input", {"path": str(source), "selection": {"modality": fixture["modality"]}}
    )
    require(
        inspection["status"] == "completed",
        f"Input inspection run {inspection['run_id']} {inspection['status']} "
        f"({(inspection.get('error') or {}).get('code')})",
    )
    require(inspection["dataset"]["data"]["shape"] == fixture["shape"], "Wrong inspected shape")
    valid = call("validate_plan", {"plan": plan})
    require(valid["valid"], "Fixture plan was rejected")
    args = {
        "plan": plan,
        "asynchronous": True,
        "idempotency_key": case_id,
        "user_request": REQUESTS[case_id],
    }
    submitted = submit_and_wait(call, args)
    lineage = {}
    if case_id == "J06":
        submitted, args, lineage = answer_calibration(call, output, source, args, submitted)
    run_id = submitted["run_id"]
    state = call("get_run_status", {"run_id": run_id})
    require(
        state["status"] == "completed",
        f"{case_id} run {run_id}: {state['status']} ({(state.get('error') or {}).get('code')})",
    )
    require(run_id != inspection["run_id"], "Primary and inspection run were conflated")
    require(
        len(state["steps"]) == len(plan["steps"])
        and all(s["status"] == "completed" for s in state["steps"]),
        "Incomplete science steps",
    )
    require(state["scientific_accuracy"] == "not_evaluated", "Missing accuracy boundary")
    exported = call("export_report", {"run_id": run_id, "formats": ["json", "html"]})
    artifacts = call("list_artifacts", {"run_id": run_id})
    store_root = store_root or (output / "harness")
    data = RunEvidence(output, run_id, artifacts, exported, store_root=store_root)
    require(
        data.report["completeness"]["completed_steps"] == len(plan["steps"]),
        "Incorrect report completeness",
    )
    raw_preview = call("read_artifact", {"artifact_id": data.index["raw"], "limit": 2})
    require(raw_preview["is_preview"] and len(raw_preview["values"]) == 2, "Invalid raw preview")
    raw = data.array("raw")
    disk = read_fixture(source)
    np.testing.assert_array_equal(raw, disk)
    snapshot = store_root / "runs" / run_id / "raw" / source.name
    require(digest(snapshot) == digest(source) == fixture["sha256"], "Raw input hash changed")
    inspected_raw = next(a for a in inspection["artifacts"] if a["role"] == "raw")
    require(
        inspected_raw["metadata"]["array_sha256"]
        == data.by_id[data.index["raw"]]["metadata"]["array_sha256"],
        "Inspection and analysis selected different pixels",
    )
    if case_id == "J02":
        claims = check_atoms(data, fixture)
    elif case_id == "J06":
        calibration = data.value("calibration")["pixel_size"]
        require(
            calibration["value"] == 0.02 and calibration["unit"] == "nm/pixel",
            "Child calibration was not applied on disk",
        )
        claims = check_diffraction(data, with_candidate=False)
    else:
        check = {
            "J01": check_diffraction,
            "J03": check_gpa,
            "J04": check_atom_strain,
            "J05-NBED": check_direct_pattern,
            "J05-SAED": check_direct_pattern,
        }[case_id]
        claims = check(data)
    # A reused key must resolve to the same job. A changed request must conflict.
    reuse = call("submit_analysis", {**args, "asynchronous": True})
    require(
        reuse["run_id"] == run_id and reuse["job_id"] == state["job_id"],
        "Idempotent retry created another run/job",
    )
    call(
        "submit_analysis",
        {**args, "user_request": "Changed request"},
        expect_error="IDEMPOTENCY_CONFLICT",
    )
    require(
        {a["artifact_id"]: a["sha256"] for a in call("list_artifacts", {"run_id": run_id})}
        == {a["artifact_id"]: a["sha256"] for a in artifacts},
        "Retry modified evidence",
    )
    direct_ops = {
        "J01": {"diffraction.fft"},
        "J02": {"atomic.detect", "atomic.refine", "atomic.intensity"},
        "J03": {"strain.gpa"},
        "J04": {"strain.atom"},
        "J06": {"diffraction.fft"},
    }.get(case_id, set())
    for step in state["steps"]:
        name = step["operation_id"]
        row = matrix[name]
        require(row["implementation_status"] == "implemented", "Reserved operation executed")
        row["disk_execution"] = "passed"
        if name in direct_ops:
            row["direct_api_equivalence"] = "passed"
        row["execution_evidence"].append(
            {
                "case_id": case_id,
                "run_id": run_id,
                "step_id": step["step_id"],
                "consumed_artifacts": step["result"]["consumed_artifacts"],
                "artifact_ids": step["result"]["artifact_ids"],
                "scientific_level": step["result"]["scientific_level"],
            }
        )
    return {
        "case_id": case_id,
        "case_verdict": "passed",
        "execution_status": state["status"],
        "user_request": REQUESTS[case_id],
        "inspection_run_id": inspection["run_id"],
        "run_id": run_id,
        "job_id": state["job_id"],
        "parent_run_id": state["parent_run_id"],
        "input_response": lineage or None,
        "snapshot_sha256": fixture["sha256"],
        "artifacts": data.entries,
        "claims": claims,
        "report_artifact_id": data.manifest_id,
        "warnings": [
            w.replace(str(output), ".") for s in state["steps"] for w in s["result"]["warnings"]
        ],
        "scientific_accuracy": "not_evaluated",
        "idempotent_retry_and_conflict": "passed",
    }


def exercise_faults(call, fixtures, source):
    bad = fixtures / "truncated.npy"
    bad.write_bytes(b"truncated analytic file")
    plan = {
        "input_path": str(bad),
        "steps": [
            {"id": "load", "operation_id": "input.load"},
            {"id": "quality", "operation_id": "quality.assess", "depends_on": ["load"]},
        ],
    }
    state = submit_and_wait(call, {"plan": plan})
    require(
        state["status"] == "failed" and state["error"]["code"] == "INVALID_INPUT",
        f"Truncated NPY run {state['run_id']} returned {state['status']} "
        f"({(state.get('error') or {}).get('code')}), expected INVALID_INPUT",
    )
    require(
        [s["status"] for s in state["steps"]] == ["failed", "blocked"],
        "Failed input did not block its dependent",
    )
    run_id = state["run_id"]
    report = call("export_report", {"run_id": run_id, "formats": ["json"]})
    require(report["report_type"] == "diagnostic", "Invalid data received an analysis report")
    require(
        not any(a["execution_usable"] for a in call("list_artifacts", {"run_id": run_id})),
        "Failed input produced consumable results",
    )

    submitted = call("submit_analysis", {"plan": plan_for("J01", source), "asynchronous": True})
    cancelled = call("cancel_run", {"run_id": submitted["run_id"]})
    deadline = time.monotonic() + 30
    while cancelled["status"] in {"preparing", "queued", "running", "finalizing"}:
        require(time.monotonic() < deadline, "Cancellation exceeded 30 seconds")
        time.sleep(0.2)
        cancelled = call("get_run_status", {"run_id": submitted["run_id"]})
    require(cancelled["status"] == "cancelled", "Explicitly cancelled run did not stop")
    return {
        "invalid_file": {
            "case_verdict": "passed",
            "execution_status": "failed",
            "run_id": run_id,
            "code": "INVALID_INPUT",
            "report_id": report["artifacts"][0]["artifact_id"],
        },
        "cancellation": {
            "case_verdict": "passed",
            "execution_status": "cancelled",
            "run_id": submitted["run_id"],
            "observed_at_submit": submitted["status"],
        },
    }


def write_json(path: Path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )


def run(
    output: Path,
    *,
    require_installed: bool = False,
    extended: bool = False,
    release_bundle: Path | None = None,
) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "Evidence output must be new or empty")
    fixtures = output / "fixtures"
    fixtures.mkdir()
    evidence = {
        "schema_version": "1.0.0",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "case_verdict": "failed",
        "execution_layer": "local_cli_jsonl",
        "scientific_accuracy": "not_evaluated",
        "host_llm_e2e": "not_evaluated",
        "release_ready": False,
        "cases": [],
        "tool_trace": [],
    }
    matrix = {}
    try:
        evidence["runtime"] = runtime_record(require_installed, release_bundle)
        all_fixtures = {}
        for case_id in CASE_MODALITIES if extended else CORE_CASES:
            path, fixture = write_fixture(case_id, fixtures)
            fixture.update(path=path.relative_to(output).as_posix(), sha256=digest(path))
            all_fixtures[case_id] = (path, fixture)
        evidence["fixtures"] = [fixture for _, fixture in all_fixtures.values()]
        with open_bridge(output, evidence["tool_trace"]) as call:
            capabilities = call("list_capabilities")
            first = all_fixtures["J01"][0]
            matrix = check_contracts(call, capabilities, first)
            for case_id, (source, fixture) in all_fixtures.items():
                trace_start = len(evidence["tool_trace"])
                try:
                    evidence["cases"].append(run_case(call, output, fixture, source, matrix))
                except Exception as exc:
                    trace = evidence["tool_trace"][trace_start:]
                    submitted = next((r for r in trace if r["tool"] == "submit_analysis"), {})
                    evidence["cases"].append(
                        {
                            "case_id": case_id,
                            "case_verdict": "failed",
                            "run_id": submitted.get("run_id"),
                            "execution_status": submitted.get("status", "not_evaluated"),
                            "error": str(exc).replace(str(output), ".")[:1000],
                        }
                    )
            evidence["reliability"] = exercise_faults(call, fixtures, first)
        require(
            all(c["case_verdict"] == "passed" for c in evidence["cases"]),
            "One or more disk-to-report cases failed",
        )
        require(
            all(row["negative_contract"]["verdict"] == "passed" for row in matrix.values()),
            "One or more operation contracts accepted unexpected input",
        )
        evidence["case_verdict"] = "passed"
    except Exception as exc:
        evidence["error"] = str(exc).replace(str(output), ".")[:1000]
    finally:
        if matrix:
            write_json(
                output / "operations.json", {"schema_version": "1.0.0", "operations": matrix}
            )
            evidence["operations"] = {
                "path": "operations.json",
                "sha256": digest(output / "operations.json"),
                "implemented": sum(
                    r["implementation_status"] == "implemented" for r in matrix.values()
                ),
                "disk_execution_passed": sum(
                    r["disk_execution"] == "passed" for r in matrix.values()
                ),
                "reserved_rejected": sum(
                    r["implementation_status"] == "reserved"
                    and r["negative_contract"]["verdict"] == "passed"
                    for r in matrix.values()
                ),
            }
        evidence["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "evidence.json", evidence)
    return evidence
