"""Recompute release gates from a Windows bundle and private, release-bound evidence."""

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
STATES = ("failed", "blocked", "not_evaluated", "passed")
DECISIONS = {"integration": "leima-max", "diffraction": "wangzhe221", "atomic_strain": "yswgityes"}
DELIVERY_CASES = ("J06", "J07", "J08", "J09", "J10", "J11", "J12", "J13")
HOST_RECONNECT_CHECK = "post_switch_host_failure_recovery"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            require(k not in result, "Duplicate JSON member")
            result[k] = v
        return result

    return json.loads(path.read_bytes(), object_pairs_hook=unique)


def sha(path):
    with path.open("rb") as stream:
        checksum = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
        return checksum.hexdigest()


def fingerprint(value):
    data = json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(data.encode("ascii")).hexdigest()


def load_module(path):
    spec = importlib.util.spec_from_file_location("autotem_maintainer_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tool(name):
    return load_module(TOOLS / (name + ".py"))


def gate(checks):
    require(checks and all(v in STATES for v in checks.values()), "Invalid gate observations")
    return {"verdict": next(v for v in STATES if v in checks.values()), "checks": checks}


def operation_status(outstanding):
    if not outstanding:
        return "passed"
    if any(
        v in {"failed", "dependency_broken", "numerical_failure", "timeout"}
        for v in outstanding.values()
    ):
        return "failed"
    if any(v in {"blocked", "dependency_unavailable"} for v in outstanding.values()):
        return "blocked"
    return "not_evaluated"


def source_collection(report, data, cases):
    schema = data.get("schema_version", "1.0.0")
    require(schema in {"1.0.0", "1.1.0", "1.2.0", "1.3.0"}, "Unknown source regression schema")
    if schema == "1.0.0" and "collection_sha256" not in data:
        return {"verdict": "not_evaluated"}
    inventory = report.with_suffix(".collection.json")
    require(sha(inventory) == data["collection_sha256"], "Source selection inventory changed")
    selection = read(inventory)
    selected, deselected = selection["selected"], selection["deselected"]
    nodes = selected + deselected
    require(
        all(isinstance(node, str) and node for node in nodes)
        and len(set(nodes)) == len(nodes)
        and data["results"]["collected"] == len(nodes)
        and data["results"]["deselected"] == len(deselected),
        "Source collection counts or node IDs are inconsistent",
    )
    node_fields = [case.findall("./properties/property[@name='autotem_nodeid']") for case in cases]
    complete = (
        selection["targets"] == ["tests"]
        and not (selection["marker"] or selection["keyword"] or deselected)
        and all(len(fields) == 1 for fields in node_fields)
        and Counter(fields[0].get("value") for fields in node_fields) == Counter(selected)
    )
    result = {
        "verdict": "passed" if complete else "failed",
        "collected": len(nodes),
        "deselected": len(deselected),
        "sha256": data["collection_sha256"],
    }
    if schema == "1.3.0":
        require(
            selection.get("source_import", {}).get("path") == "autotem/__init__.py"
            and re.fullmatch(r"[0-9a-f]{64}", selection["source_import"].get("sha256", "")),
            "Missing actual pytest source import",
        )
        result["source_import"] = selection["source_import"]
    return result


def source_tests(
    report, identity, source_hash, file_count, *, bundle=None, simulation_identity=None
):
    data = read(report)
    xml = report.with_suffix(".junit.xml")
    require(
        data["release_manifest_sha256"] == identity
        and data["source_archive_sha256"] == source_hash
        and data["source_files_byte_verified"] == file_count
        and sha(xml) == data["junit_sha256"],
        "Source tests differ from this release or JUnit file",
    )
    root = ET.parse(xml).getroot()
    suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
    cases = root.findall(".//testcase")
    counts = Counter()
    passed_names, passed_nodes = set(), set()
    for case in cases:
        outcomes = [t for t in ("failure", "error", "skipped") if case.find(t) is not None]
        require(len(outcomes) <= 1, "Ambiguous JUnit result")
        status = outcomes[0] if outcomes else "passed"
        counts[status] += 1
        if status == "passed":
            passed_names.add(case.attrib["name"])
            fields = case.findall("./properties/property[@name='autotem_nodeid']")
            if len(fields) == 1:
                passed_nodes.add(fields[0].get("value"))
    totals = {
        "tests": len(cases),
        "passed": counts["passed"],
        "skipped": counts["skipped"],
        "failures": counts["failure"],
        "errors": counts["error"],
    }
    require(
        cases
        and suites
        and all(data["results"][k] == v for k, v in totals.items())
        and all(
            sum(int(s.attrib[k]) for s in suites) == totals[k]
            for k in ("tests", "skipped", "failures", "errors")
        ),
        "JUnit counts do not reproduce the source summary",
    )
    expected = {
        "test_real_diffsims_workflow_closes_simulation_phase_matching_loop",
        "test_optional_real_simulation_and_matching_from_disk",
    }
    optional = expected <= passed_names and expected <= set(data.get("optional_tests_passed", []))
    collection = source_collection(report, data, cases)
    environment = {"verdict": "not_evaluated"}
    if data.get("schema_version") in {"1.2.0", "1.3.0"}:
        nodes = {
            field.get("value")
            for case in cases
            if case.get("name") in expected
            for field in case.findall("./properties/property[@name='autotem_nodeid']")
        }
        optional = (
            optional
            and data["optional_diffsims_enabled"] is True
            and nodes
            == {
                "tests/test_b4_workflow_integration.py::"
                "test_real_diffsims_workflow_closes_simulation_phase_matching_loop",
                "tests/test_harness_release_operation_variants.py::"
                "test_optional_real_simulation_and_matching_from_disk",
            }
        )
        test_env = load_module(Path(__file__).with_name("source_test_environment.py"))
        environment = test_env.recheck(
            report,
            data,
            identity,
            source_hash,
            bundle=bundle,
            simulation_identity=simulation_identity,
        )
        if data["schema_version"] == "1.3.0":
            wheel_passed = test_env.DEPLOYMENT_NODE in passed_nodes
            environment["wheel_entrypoint"] = "passed" if wheel_passed else "failed"
            source_hash_observed = collection["source_import"]["sha256"]
            require(
                environment["wheel_init_sha256"] == source_hash_observed,
                "Collected source import differs from the issued wheel",
            )
            optional = optional and wheel_passed
    passed = (
        data["case_verdict"] == "passed"
        and data["execution_layer"] == "source_checkout"
        and data["pytest_selection"] == "all"
        and data["pytest_default_smoke_overridden"] is True
        and not (totals["errors"] or totals["failures"] or totals["skipped"])
        and optional
        and collection["verdict"] != "failed"
    )
    return {
        "verdict": "passed" if passed else "failed",
        **totals,
        "optional_source_execution": optional,
        "collection": collection,
        "environment": environment,
        "junit_sha256": sha(xml),
        "report_sha256": sha(report),
    }


def review_decisions(path, identity, basis):
    statuses = dict.fromkeys([*DECISIONS, "redistribution", "publisher_origin"], "not_evaluated")
    proofs = {}
    if path is None:
        return statuses, proofs
    index = read(path)
    require(
        index["schema_version"] == "1.0.0" and index["release_manifest_sha256"] == identity,
        "Review index belongs to another release",
    )
    for kind, entry in index["decisions"].items():
        require(
            kind in statuses
            and entry["reviewer"] == DECISIONS.get(kind, "leima-max")
            and entry["verdict"] in ("passed", "failed", "blocked")
            and datetime.fromisoformat(entry["recorded_at"]).utcoffset() is not None,
            "Unknown decision, reviewer, time or outcome",
        )
        if kind == "integration":
            require(entry.get("qualification_basis_sha256") == basis, "Integration review is stale")
        rel = Path(entry["record"])
        require(not rel.is_absolute() and ".." not in rel.parts, "Review record must be local")
        record = (path.parent / rel).resolve()
        require(
            record.is_relative_to(path.parent.resolve())
            and record != path.resolve()
            and record.stat().st_size > 0
            and sha(record) == entry["record_sha256"],
            "The original review record is missing or changed",
        )
        statuses[kind] = entry["verdict"]
        proofs[kind] = entry["record_sha256"]
    return statuses, proofs


def observed_host(verifier, raw, project, bundle, identity):
    from autotem.installation.host_entry import active_python

    raw, project = raw.resolve(), project.resolve()
    root = project / ".autotem/runs"
    if read(raw)["observation"] == "recorded":
        python = active_python(root.parent)
        require(
            read(root.parent / "active-runtime.json")["manifest_sha256"] == identity,
            "Captured project has another active release",
        )
        if python != Path(sys.executable):
            require(
                sha(python) == sha(Path(sys.executable)), "Unexpected project interpreter launcher"
            )
            result = subprocess.run(
                [
                    str(python),
                    "-I",
                    "-X",
                    "utf8",
                    str(Path(verifier.__file__).resolve()),
                    "verify",
                    str(raw),
                    "--project",
                    str(project),
                    "--release",
                    str(bundle),
                    "--harness-root",
                    str(root),
                ],
                cwd=project,
                capture_output=True,
                timeout=300,
            )
            require(result.returncode in (0, 1), "Project interpreter could not verify the capture")
            record = json.loads(result.stdout)
            require(
                (result.returncode == 0) == (record["case_verdict"] == "passed"),
                "Host verifier exit code conflicts with its capture",
            )
            return record
    return verifier.verify(raw, bundle=bundle, project=project, root=root)


def host_results(bundle, observations, native):
    index = tool("host_acceptance/index")
    verifier = tool("host_acceptance/verify")
    captures = []
    native = set(native)
    identity = sha(bundle / "release-manifest.json")
    for raw, project in observations:
        result = observed_host(verifier, raw, project, bundle, identity)
        require(result["evidence"]["case_record_sha256"] == sha(raw), "Host capture changed")
        if (
            result.get("variant") == "native"
            and result["case_id"] == "J14"
            and result["case_verdict"] == "passed"
        ):
            native.add(result["host"])
        if result["case_verdict"] in ("blocked", "failed"):
            require(
                result.get("attempted_release_manifest_sha256") == identity
                or result.get("runtime", {}).get("release_identity", {}).get("manifest_sha256")
                == identity,
                "Host observation lacks a release binding",
            )
        captures.append(result)
    table = index.build(bundle, native=native, observations=captures)
    hosts = {}
    for name, group in table["hosts"].items():
        verified = [r for r in captures if r["host"] == name and r["case_verdict"] == "passed"]
        hosts[name] = {
            "host_version": group["host_version"],
            "journeys": group["journey_verdict"],
            "simulation_cases": group["cases"]["J15"]["verdict"],
            "delivery_cases": gate({k: group["cases"][k]["verdict"] for k in DELIVERY_CASES})[
                "verdict"
            ],
            "first_authorization": "passed"
            if any(r["checks"].get("first_authorization") == "passed" for r in verified)
            else "not_evaluated",
            "nine_tools_discovered": "passed"
            if any(r["checks"].get("full_tool_inventory") == "passed" for r in verified)
            else "not_evaluated",
        }
    return {
        "hosts": hosts,
        "required_variants": table["required_variant_counts"],
        "matrix_sha256": table["matrix_sha256"],
        "verifier_sha256": sha(Path(verifier.__file__)),
        "capture_sha256": sorted({r["evidence"]["case_record_sha256"] for r in captures}),
    }


def maintenance(evidence, disk, identity):
    rows = [json.loads(line) for line in (evidence / "maintenance.jsonl").read_bytes().splitlines()]
    rid = "rt-" + identity[:20]
    expected = [
        ("planned", False),
        ("host_setup_required", True),
        ("host_setup_required", True),
        ("rolled_back", True),
        ("planned", False),
        ("uninstalled", True),
    ]
    statuses = [(r["state"], r["applied"]) for r in rows]
    require(statuses == expected or statuses == expected[:3], "Unexpected installer history")
    require(
        rows[1]["runtime_id"] == rid
        and rows[2]["runtime_id"] == rid
        and rows[2]["activation"] == "already_active",
        "Installer activation differs from release",
    )
    if len(rows) == 3:
        require(
            "uninstall_preserved_store" not in disk["checks"], "Maintenance transcript incomplete"
        )
        return "not_evaluated"
    require(
        all(r["runtime_id"] == rid for r in rows[3:])
        and rows[-1]["removed_file_count"] > 0
        and {"rollback_preserved_store", "uninstall_preserved_store"} <= set(disk["checks"]),
        "Maintenance evidence is incomplete",
    )
    return "passed"


def upgrade_reconnect_verdict(report, verified):
    expected = "not_evaluated"
    if "host_reconnect_recovery" in report:
        summary = report["host_reconnect_recovery"]
        require(
            isinstance(summary, dict)
            and summary.get("verdict") == "passed"
            and report["case_verdict"] == "passed",
            "Invalid host reconnect recovery summary in the upgrade receipt",
        )
        expected = "passed"
    require(
        verified.get(HOST_RECONNECT_CHECK, "not_evaluated") == expected,
        "Host reconnect recovery differs from its upgrade receipt",
    )
    return expected


def certificates(args, bundle, evidence, identity):
    values = {}
    if args.contract:
        values["contract"] = tool("release/contract_freeze").recheck(
            bundle, evidence, args.contract
        )
    if args.local:
        values["local"] = tool("acceptance/qualify_local").recheck(
            args.local, bundle, evidence / "science", evidence / "variants"
        )
    if args.headless:
        values["headless"] = tool("acceptance/verify_headless").recheck(bundle, args.headless)
    if args.upgrade:
        report, previous, project = args.upgrade
        values["upgrade"] = tool("setup/verify_upgrade").recheck(report, bundle, previous, project)
    if args.recovery:
        report, project = args.recovery
        values["recovery"] = tool("setup/verify_recovery").recheck(report, bundle, project)
    results = {}
    for key, value in values.items():
        source = getattr(args, key)
        report = source[0] if key in {"upgrade", "recovery"} else source
        data = read(report)
        require(
            data["schema_version"] == "1.0.0"
            and data["release_manifest_sha256"] == identity
            and value["release_manifest_sha256"] == identity
            and re.fullmatch("[0-9a-f]{64}", data["verifier_sha256"])
            and data["case_verdict"] == value["case_verdict"]
            and data["case_verdict"] in STATES,
            f"Unbound or invalid {key} certificate",
        )
        results[key] = {
            "verdict": value["case_verdict"],
            "report_sha256": sha(report),
            "verifier_sha256": data["verifier_sha256"],
        }
        if key == "upgrade":
            results[key][HOST_RECONNECT_CHECK] = upgrade_reconnect_verdict(data, value)
    return results


def compatible_operations(data, source_sha, wheel_sha, outstanding):
    expected = {"simulation.diffraction": "dependency_unavailable", "diffraction.match": "blocked"}
    require(
        data["case_verdict"] == "passed"
        and data["profile"] == "traditional-simulation"
        and data["source_archive_sha256"] == source_sha
        and data["wheel_sha256"] == wheel_sha
        and set(data["operations"]) == set(expected),
        "Simulation edition has different source, wheel or operations",
    )
    return {op: state for op, state in outstanding.items() if expected.get(op) != state}


def simulation_fault_evidence(report, bundle, project, positive):
    if report is None:
        return {"verdict": "not_evaluated"}
    report = report.resolve()
    failures = tool("acceptance/qualify_simulation_faults").recheck(report, bundle, project)
    expected = {
        "missing_dependency",
        "broken_dependency",
        "backend_timeout",
        "worker_timeout",
        "missing_structure",
    }
    require(
        failures["release_manifest_sha256"] == positive["release_manifest_sha256"]
        and failures["positive_run_id"] == positive["run_id"]
        and set(failures["cases"]) == expected
        and failures["case_verdict"] in STATES,
        "Simulation failure cases differ from the positive release or run",
    )
    return {
        "verdict": failures["case_verdict"],
        "report_sha256": sha(report),
        "positive_run_id": failures["positive_run_id"],
        "cases": sorted(expected),
    }


def failed_install(bundle, evidence):
    """A native installer failure before activation has no usable private test runtime."""
    verifier = load_module(TOOLS.parent / "autotem/installation/release.py")
    manifest = read(bundle / "release-manifest.json")
    binding = verifier.verify_release(bundle)
    identity = binding.manifest_sha256
    install = read(evidence / "installation.json")
    history = evidence / "maintenance.jsonl"
    events = [json.loads(line) for line in history.read_bytes().splitlines()]
    require(
        install["release_manifest_sha256"] == identity
        and install["case_verdict"] == "failed"
        and install["checks"]
        in (
            [],
            ["read_only_preview"],
            ["read_only_preview", "project_space_and_host_write_preview"],
        )
        and len(events) >= 1,
        "Expected a pre-activation cold-install failure",
    )
    if "project_space_and_host_write_preview" in install["checks"]:
        tool("setup/verify_install").cold_preview(bundle, manifest, install["project_preview"])
    last = events[-1]
    control = evidence / "project/.autotem"
    hashes = {
        "installation.json": sha(evidence / "installation.json"),
        "maintenance.jsonl": sha(history),
    }
    transport_code = None
    code = last["error"]["code"]
    if code == "DISK_SPACE_LOW" and "space_preview" in last:
        tool("setup/verify_install").cold_preview(
            bundle, manifest, last["space_preview"], fits=False
        )
        require(
            last["state"] == "blocked"
            and code in install["error"]
            and not last.get("installation_id")
            and not control.exists(),
            "Cold space rejection unexpectedly changed the project",
        )
    else:
        instance = last["installation_id"]
        require(
            re.fullmatch("[a-f0-9]{32}", instance)
            and last["state"] == "blocked"
            and last["applied"] is True
            and last["runtime_id"] == "rt-" + identity[:20],
            "Failed installer transcript does not identify this transaction",
        )
        receipt = control / "installations" / instance / "transaction.json"
        txn = read(receipt)
        require(
            txn["state"] == "failed"
            and txn["manifest_sha256"] == identity
            and txn["installation_id"] == instance
            and txn["error"]["code"] == code
            and re.fullmatch("[A-Z][A-Z0-9_]+", code)
            and code in install["error"],
            "Installer failure conflicts with its original transaction",
        )
        hashes["failed_transaction"] = sha(receipt)
        transport_code = txn.get("runtime_diagnostic", {}).get("local_mcp", {}).get("code")
    require(
        transport_code is None or re.fullmatch("[A-Z][A-Z0-9_]+", transport_code),
        "Invalid local transport diagnostic",
    )
    checks = {
        "RG0": gate({"contracts": "not_evaluated"}),
        "RG1": gate({"cold_install": "failed"}),
        "RG2": gate({"installed_execution": "blocked"}),
        "RG3": gate({"host_journeys": "not_evaluated"}),
        "RG4": gate({"user_installation": "failed", HOST_RECONNECT_CHECK: "not_evaluated"}),
        "RG5": gate({"RG1": "failed"}),
    }
    return {
        "schema_version": "1.0.0",
        "execution_layer": "release_qualification",
        "release_id": manifest["release_id"],
        "release_manifest_sha256": identity,
        "target": manifest["target"],
        "profile": verifier.PROFILE,
        "gates": checks,
        "source_archive_sha256": manifest["source"]["archive_sha256"],
        "failure_code": code,
        "local_transport_code": transport_code,
        "disk_evidence_sha256": hashes,
        "qualification_verifier_sha256": sha(Path(__file__)),
        "scientific_accuracy": "not_evaluated",
        "release_ready": False,
    }


def evaluate(args):
    bundle, evidence = args.bundle.resolve(), args.evidence.resolve()
    attempt = read(evidence / "installation.json")
    if (
        attempt["case_verdict"] == "failed"
        and "cold_installed_private_cpython" not in attempt["checks"]
    ):
        require(
            not any(
                [
                    args.contract,
                    args.local,
                    args.headless,
                    args.upgrade,
                    args.recovery,
                    args.simulation,
                    args.simulation_headless,
                    args.simulation_faults,
                    args.simulation_host_capture,
                    args.host_capture,
                    args.source_tests,
                    args.decisions,
                ]
            ),
            "Record a failed cold attempt separately from later qualifications",
        )
        return failed_install(bundle, evidence)

    from autotem.acceptance.driver import runtime_record
    from autotem.installation.release import verify_release

    verified = verify_release(bundle)
    runtime = runtime_record(True, bundle)
    manifest = read(bundle / "release-manifest.json")
    identity = sha(bundle / "release-manifest.json")
    require(
        identity == verified.manifest_sha256 == runtime["release_identity"]["manifest_sha256"],
        "Active interpreter is not this release",
    )
    slice_result = tool("release/check_first_slice").summarize(bundle, evidence)
    health = certificates(args, bundle, evidence, identity)

    def result(name, field="verdict"):
        return health.get(name, {}).get(field, "not_evaluated")

    coverage = slice_result["operations"]
    outstanding = {
        k: v["verdict"] for k, v in coverage["by_operation"].items() if v["verdict"] != "passed"
    }
    host = host_results(bundle, args.host_capture, args.native)
    remaining = outstanding
    simulation_checks, simulation_hosts = {}, {}
    require(
        args.simulation
        or not (args.simulation_headless or args.simulation_faults or args.simulation_host_capture),
        "A companion simulation edition is required for its supplementary evidence",
    )
    if args.simulation:
        report, companion, project = (p.resolve() for p in args.simulation)
        sample = tool("acceptance/qualify_simulation").recheck(report, companion, project)
        remaining = compatible_operations(
            sample, manifest["source"]["archive_sha256"], slice_result["wheel_sha256"], outstanding
        )
        health["simulation_edition"] = {
            "verdict": sample["case_verdict"],
            "report_sha256": sha(report),
            "release_manifest_sha256": sample["release_manifest_sha256"],
            "lock_sha256": sample["lock_sha256"],
            "run_id": sample["run_id"],
            "job_id": sample["job_id"],
            "operations": sample["operations"],
        }
        health["simulation_faults"] = simulation_fault_evidence(
            args.simulation_faults, companion, project, sample
        )
        simulation_checks["simulation_failure_paths"] = health["simulation_faults"]["verdict"]
        simulation_checks["simulation_headless_processes"] = "not_evaluated"
        if args.simulation_headless:
            monitored = tool("acceptance/verify_headless").recheck(
                companion, args.simulation_headless
            )
            require(
                monitored["release_manifest_sha256"] == sample["release_manifest_sha256"],
                "Import audit has another simulation edition",
            )
            simulation_checks["simulation_headless_processes"] = monitored["case_verdict"]
            health["simulation_headless"] = {
                "verdict": monitored["case_verdict"],
                "report_sha256": sha(args.simulation_headless),
            }
        conditional = host_results(companion, args.simulation_host_capture, [])
        host["simulation_edition"] = {
            "manifest_sha256": sample["release_manifest_sha256"],
            "capture_sha256": conditional["capture_sha256"],
            "hosts": {},
        }
        for name, value in conditional["hosts"].items():
            host["simulation_edition"]["hosts"][name] = {
                "host_version": value["host_version"],
                "J15": value["simulation_cases"],
            }
            simulation_hosts[name + ".J15.simulation"] = value["simulation_cases"]
    delivery = {}
    for name, h in host["hosts"].items():
        delivery.update(
            {
                name + "." + k: h[k]
                for k in ("delivery_cases", "first_authorization", "nine_tools_discovered")
            }
        )
    checks = {
        "RG0": gate({"contracts": result("contract")}),
        "RG1": gate({"installed_disk_journeys": slice_result["private_install_and_disk_verdict"]}),
        "RG2": gate(
            {
                "local_execution": result("local"),
                "headless_processes": result("headless"),
                "all_implemented_executed_across_profiles"
                if args.simulation
                else "all_implemented_executed_in_release": operation_status(remaining),
                **simulation_checks,
            }
        ),
        "RG3": gate({**{k: h["journeys"] for k, h in host["hosts"].items()}, **simulation_hosts}),
        "RG4": gate(
            {
                "installation_and_preserved_uninstall": maintenance(
                    evidence, read(evidence / "installation.json"), identity
                ),
                "cross_release_upgrade_recovery": result("upgrade"),
                HOST_RECONNECT_CHECK: result("upgrade", HOST_RECONNECT_CHECK),
                "interrupted_install_recovery": result("recovery"),
                **delivery,
            }
        ),
    }
    source = manifest["source"]
    regression = (
        source_tests(
            args.source_tests, identity, source["archive_sha256"], source["file_count"],
            bundle=bundle,
            simulation_identity=health.get("simulation_edition", {}).get("release_manifest_sha256"),
        )
        if args.source_tests
        else {"verdict": "not_evaluated"}
    )
    evidence_hashes = {
        **slice_result["evidence_sha256"],
        "maintenance.jsonl": sha(evidence / "maintenance.jsonl"),
    }
    basis = fingerprint(
        {
            "release": identity,
            "gates": checks,
            "certificates": health,
            "host": host,
            "source_tests": regression,
            "disk_evidence": evidence_hashes,
        }
    )
    reviews, proofs = review_decisions(args.decisions, identity, basis)
    checks["RG5"] = gate(
        {
            **{k: v["verdict"] for k, v in checks.items()},
            "package_source_lock_inventory": "passed",
            "source_regression": regression["verdict"],
            **reviews,
        }
    )
    return {
        "schema_version": "1.0.0",
        "execution_layer": "release_qualification",
        "release_id": manifest["release_id"],
        "release_manifest_sha256": identity,
        "target": manifest["target"],
        "profile": slice_result["profile"],
        "wheel_sha256": slice_result["wheel_sha256"],
        "source_archive_sha256": source["archive_sha256"],
        "lock_sha256": slice_result["lock_sha256"],
        "license_inventory_sha256": manifest["licenses"]["sha256"],
        "manifest_redistribution_review": manifest["licenses"]["redistribution_review"],
        "manifest_publisher_signature": manifest["publisher"]["signature_status"],
        "gates": checks,
        "release_ready": all(v["verdict"] == "passed" for v in checks.values()),
        "operations": {
            "implemented": coverage["implemented"],
            "disk_passed": coverage["disk_execution_passed"],
            "reserved_rejected": coverage["reserved_rejected"],
            "outstanding": outstanding,
            "remaining_across_evaluated_profiles": remaining,
        },
        "host_evidence": host,
        "disk_evidence_sha256": evidence_hashes,
        "certificates": health,
        "source_tests": regression,
        "review_record_sha256": proofs,
        "qualification_basis_sha256": basis,
        "qualification_verifier_sha256": sha(Path(__file__)),
        "scientific_accuracy": "not_evaluated",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "evidence", "out"):
        parser.add_argument("--" + key, required=True, type=Path)
    for key in ("contract", "local", "headless", "source-tests", "decisions"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument(
        "--upgrade", nargs=3, type=Path, metavar=("REPORT", "PREVIOUS_BUNDLE", "PROJECT")
    )
    parser.add_argument("--recovery", nargs=2, type=Path, metavar=("REPORT", "PROJECT"))
    parser.add_argument(
        "--simulation", nargs=3, type=Path, metavar=("REPORT", "SIMULATION_BUNDLE", "PROJECT")
    )
    parser.add_argument("--simulation-headless", type=Path)
    parser.add_argument("--simulation-faults", type=Path)
    parser.add_argument(
        "--simulation-host-capture",
        nargs=2,
        type=Path,
        action="append",
        default=[],
        metavar=("CASE_JSON", "PROJECT"),
    )
    parser.add_argument(
        "--host-capture",
        nargs=2,
        type=Path,
        action="append",
        default=[],
        metavar=("CASE_JSON", "PROJECT"),
    )
    parser.add_argument(
        "--native", choices=["codex", "openclaw", "workbuddy"], action="append", default=[]
    )
    parser.add_argument(
        "--require", choices=[f"RG{i}" for i in range(6)], action="append", default=[]
    )
    parser.add_argument(
        "--require-check", action="append", default=[], metavar="RGn.CHECK",
        help="Require a named check in the index, such as RG4.interrupted_install_recovery.",
    )
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    for name in args.require_check:
        if not re.fullmatch(r"RG[0-5]\.[A-Za-z][A-Za-z0-9_.]*", name):
            parser.error(f"Invalid gate check {name!r}. Use RGn.CHECK.")
    try:
        report = evaluate(args)
        with args.out.open("x", encoding="utf-8") as stream:
            print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), file=stream)
    except (
        ValueError,
        RuntimeError,
        OSError,
        KeyError,
        TypeError,
        ET.ParseError,
        subprocess.SubprocessError,
    ) as exc:
        parser.exit(2, f"Release qualification error: {exc}\n")
    print(json.dumps({k: v["verdict"] for k, v in report["gates"].items()}))
    missing = any(report["gates"][k]["verdict"] != "passed" for k in args.require)
    for name in args.require_check:
        group, key = name.split(".", 1)
        state = report["gates"][group]["checks"].get(key, "absent from this index")
        if state != "passed":
            print(f"Required check {name}: {state}", file=sys.stderr)
            missing = True
    return int(missing or args.require_ready and not report["release_ready"])


if __name__ == "__main__":
    sys.exit(main())
