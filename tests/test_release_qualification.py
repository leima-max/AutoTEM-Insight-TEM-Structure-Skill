import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from tools.acceptance import verify_headless as imports
from tools.release import qualify_ci, run_source_tests
from tools.release import qualify_release as release


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_partial_and_conditional_gates():
    assert release.operation_status({}) == "passed"
    assert release.operation_status({"simulation": "dependency_unavailable"}) == "blocked"
    assert release.operation_status({"simulation": "dependency_broken"}) == "failed"
    assert (
        release.gate({"cold": "passed", "approval": "not_evaluated"})["verdict"] == "not_evaluated"
    )
    assert release.gate({"host": "failed", "license": "blocked"})["verdict"] == "failed"
    with pytest.raises(ValueError):
        release.gate({})


def test_compatible_edition_only_covers_its_verified_conditional_operations():
    gaps = {"simulation.diffraction": "dependency_unavailable", "diffraction.match": "blocked"}
    result = {
        "case_verdict": "passed",
        "profile": "traditional-simulation",
        "source_archive_sha256": "source",
        "wheel_sha256": "wheel",
        "operations": list(gaps),
    }
    assert release.compatible_operations(result, "source", "wheel", gaps) == {}
    failed = {**gaps, "diffraction.match": "numerical_failure", "input.load": "failed"}
    assert release.compatible_operations(result, "source", "wheel", failed) == {
        "diffraction.match": "numerical_failure",
        "input.load": "failed",
    }
    for changed in (
        {"source_archive_sha256": "other"},
        {"wheel_sha256": "other"},
        {"operations": ["diffraction.match"]},
        {"profile": "traditional-cli"},
    ):
        with pytest.raises(ValueError, match="Simulation edition"):
            release.compatible_operations({**result, **changed}, "source", "wheel", gaps)


def test_staged_ci_files_use_the_pinned_archive(tmp_path):
    snapshot = tmp_path / "source.zip"
    members = {
        "tools/acceptance/driver.py": b"frozen",
        "configs/harness/contract.json": b"{}",
        "autotem/science.py": b"wheel-only",
    }
    files = {key: hashlib.sha256(value).hexdigest() for key, value in members.items()}
    index = {"schema_version": "1.0.0", "git_commit": "commit", "files": files}
    with ZipFile(snapshot, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
        z.writestr("index.json", json.dumps(index))
    source = {
        "archive": snapshot.name,
        "archive_sha256": release.sha(snapshot),
        "inventory_member": "index.json",
        "file_count": 3,
        "git_commit": "commit",
    }
    tools = qualify_ci.stage_verifiers(tmp_path, source, tmp_path / "detached")
    assert (tools / "acceptance/driver.py").read_bytes() == b"frozen"
    assert (tools.parent / "configs/harness/contract.json").is_file()
    assert not (tools.parent / "autotem/science.py").exists()
    snapshot.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        qualify_ci.stage_verifiers(tmp_path, source, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_companion_failure_cases_require_same_positive_run_and_complete_matrix(
    tmp_path, monkeypatch
):
    positive = {"run_id": "run1", "release_manifest_sha256": "edition"}
    expected = {
        "missing_dependency",
        "broken_dependency",
        "backend_timeout",
        "worker_timeout",
        "missing_structure",
    }
    result = {
        "positive_run_id": "run1",
        "release_manifest_sha256": "edition",
        "case_verdict": "passed",
        "cases": dict.fromkeys(expected, {}),
    }
    report = write(tmp_path / "faults.json", result)
    module = SimpleNamespace(recheck=lambda path, bundle, project: release.read(path))
    monkeypatch.setattr(release, "tool", lambda name: module)
    assert release.simulation_fault_evidence(None, tmp_path, tmp_path, positive) == {
        "verdict": "not_evaluated"
    }
    assert release.simulation_fault_evidence(report, tmp_path, tmp_path, positive)[
        "cases"
    ] == sorted(expected)
    for change in (
        {"positive_run_id": "other"},
        {"release_manifest_sha256": "other"},
        {"cases": {}},
    ):
        write(report, {**result, **change})
        with pytest.raises(ValueError, match="positive release or run"):
            release.simulation_fault_evidence(report, tmp_path, tmp_path, positive)


def test_original_review_required_and_integration_bound_to_basis(tmp_path):
    manifest, basis = "a" * 64, "b" * 64
    report = tmp_path / "decisions.json"
    record = tmp_path / "actual-review.txt"
    record.write_text("Unit-test fixture only.", encoding="utf-8")
    entry = {
        "reviewer": "leima-max",
        "recorded_at": "2026-11-12T10:00:00+08:00",
        "record": record.name,
        "record_sha256": release.sha(record),
        "verdict": "passed",
        "qualification_basis_sha256": basis,
    }
    data = {
        "schema_version": "1.0.0",
        "release_manifest_sha256": manifest,
        "decisions": {"integration": entry},
    }
    statuses, proofs = release.review_decisions(write(report, data), manifest, basis)
    assert statuses["integration"] == "passed" and statuses["diffraction"] == "not_evaluated"
    assert proofs == {"integration": release.sha(record)}
    with pytest.raises(ValueError, match="stale"):
        release.review_decisions(report, manifest, "c" * 64)
    data["decisions"]["diffraction"] = {**entry, "reviewer": "leima-max"}
    with pytest.raises(ValueError, match="reviewer"):
        release.review_decisions(write(report, data), manifest, basis)
    del data["decisions"]["diffraction"]
    record.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        release.review_decisions(write(report, data), manifest, basis)


def test_empty_host_matrix_retains_each_host_and_conditional_case(tmp_path):
    write(
        tmp_path / "release-manifest.json",
        {"release_id": "test", "profiles": {"traditional-core-mcp": {}}},
    )
    matrix = release.host_results(tmp_path, [], [])
    assert matrix["required_variants"]["not_evaluated"] == 81
    assert set(matrix["hosts"]) == {"codex", "openclaw", "workbuddy"}
    assert all(
        h["journeys"] == h["first_authorization"] == "not_evaluated"
        for h in matrix["hosts"].values()
    )
    native = release.host_results(tmp_path, [], ["codex"])
    assert native["required_variants"]["not_evaluated"] == 82


def test_host_capture_reopens_the_selected_projects_own_interpreter(tmp_path, monkeypatch):
    from autotem.installation import host_entry

    bundle, project = tmp_path / "bundle", tmp_path / "project"
    control = project / ".autotem"
    control.mkdir(parents=True)
    pointer = write(control / "active-runtime.json", {"manifest_sha256": "release"})
    record = write(tmp_path / "case.json", {"observation": "recorded"})
    python = control / "python.exe"
    shutil.copyfile(sys.executable, python)
    monkeypatch.setattr(host_entry, "active_python", lambda _: python)
    verifier = SimpleNamespace(__file__=__file__)
    calls = []

    def execute(cmd, **options):
        calls.append((cmd, options["cwd"]))
        return subprocess.CompletedProcess(cmd, 1, b'{"case_verdict": "blocked"}')

    monkeypatch.setattr(subprocess, "run", execute)
    assert (
        release.observed_host(verifier, record, project, bundle, "release")["case_verdict"]
        == "blocked"
    )
    cmd, cwd = calls.pop()
    assert cmd[:4] == [str(python), "-I", "-X", "utf8"] and cwd == project
    write(pointer, {"manifest_sha256": "old"})
    with pytest.raises(ValueError, match="another active release"):
        release.observed_host(verifier, record, project, bundle, "release")
    assert not calls


def test_junit_body_and_counts_are_checked(tmp_path):
    index = tmp_path / "source-regression.json"
    junit = index.with_suffix(".junit.xml")
    optional = [
        "test_real_diffsims_workflow_closes_simulation_phase_matching_loop",
        "test_optional_real_simulation_and_matching_from_disk",
    ]
    junit.write_text(
        '<testsuite tests="2" failures="0" errors="0" skipped="0">'
        + "".join(f'<testcase name="{name}"/>' for name in optional)
        + "</testsuite>",
        encoding="utf-8",
    )
    data = {
        "release_manifest_sha256": "a",
        "source_archive_sha256": "b",
        "source_files_byte_verified": 2,
        "junit_sha256": release.sha(junit),
        "results": dict(tests=2, passed=2, errors=0, skipped=0, failures=0),
        "pytest_selection": "all",
        "pytest_default_smoke_overridden": True,
        "case_verdict": "passed",
        "execution_layer": "source_checkout",
    }
    assert release.source_tests(write(index, data), "a", "b", 2)["verdict"] == "failed"
    data["optional_tests_passed"] = optional
    assert release.source_tests(write(index, data), "a", "b", 2)["verdict"] == "passed"
    junit.write_text(junit.read_text().replace('"/>', '"><failure/></testcase>', 1))
    data["junit_sha256"] = release.sha(junit)
    with pytest.raises(ValueError, match="counts"):
        release.source_tests(write(index, data), "a", "b", 2)


@pytest.mark.parametrize("filtered", [False, True])
def test_collection_inventory_links_each_real_junit_node(tmp_path, filtered):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_sample.py").write_text("def test_keep(): pass\ndef test_second(): pass\n")
    report = tmp_path / "source-regression.json"
    inventory = report.with_suffix(".collection.json")
    junit = report.with_suffix(".junit.xml")
    subprocess.run(
        [
            sys.executable,
            run_source_tests.__file__,
            "--pytest-inventory",
            str(inventory),
            "-o",
            "addopts=-q",
            "--rootdir",
            str(tmp_path),
            "-p",
            "no:cacheprovider",
            "--junitxml",
            str(junit),
            *(["-k", "keep"] if filtered else []),
            "tests",
        ],
        cwd=tmp_path,
        check=True,
    )
    data = {
        "schema_version": "1.1.0",
        "collection_sha256": release.sha(inventory),
        "results": {"collected": 2, "deselected": int(filtered)},
    }
    cases = ET.parse(junit).getroot().findall(".//testcase")
    result = release.source_collection(report, data, cases)
    assert result["verdict"] == ("failed" if filtered else "passed")
    assert result["collected"] == 2 and result["deselected"] == int(filtered)
    if not filtered:
        fields = cases[0].find("./properties")
        fields.remove(fields.find("property[@name='autotem_nodeid']"))
        assert release.source_collection(report, data, cases)["verdict"] == "failed"


def test_headless_watch_covers_work_packages_and_attempted_imports(tmp_path):
    def log(role, pid, run=None, events=()):
        items = [dict(role=role, pid=pid, run_id=run, isolated=1), *events]
        (tmp_path / f"{pid}.jsonl").write_text("\n".join(json.dumps(i) for i in items))

    controls = [dict(module=imports.CONTROL, event=e) for e in ["find_spec", "import"]]
    log("positive_control", 1, events=controls)
    for pid, role in enumerate(("bridge", "scheduler", "disk_driver", "variant_driver"), start=2):
        log(role, pid)
    log("worker", 6, "run", [dict(module="PyQt5", event="find_spec")])
    log("worker", 7, "run")
    roles, violations = imports.inspect_logs(tmp_path, {("run", 6), ("run", 7)})
    assert roles["worker"] == 2 and violations == [
        {"role": "worker", "module": "PyQt5", "event": "find_spec"}
    ]
    with pytest.raises(ValueError, match="diffsims subprocess"):
        imports.inspect_logs(tmp_path, {("run", 6), ("run", 7)}, simulation_runs={"run"})
    log("diffsims_worker", 8, "run")
    roles, _ = imports.inspect_logs(tmp_path, {("run", 6), ("run", 7)}, simulation_runs={"run"})
    assert roles["diffsims_worker"] == 1
    with pytest.raises(ValueError, match="persisted"):
        imports.inspect_logs(tmp_path, {("run", 7)})
    log("positive_control", 1)
    with pytest.raises(ValueError, match="positive"):
        imports.inspect_logs(tmp_path, {("run", 6), ("run", 7)})


def test_cli_preserves_existing_evidence_and_require_ready(tmp_path, monkeypatch):
    report = {"gates": {"RG1": {"verdict": "passed"}}, "release_ready": False}
    monkeypatch.setattr(release, "evaluate", lambda args: report)
    out = tmp_path / "published.json"
    base = ["--bundle", str(tmp_path), "--evidence", str(tmp_path), "--out", str(out)]
    assert release.main([*base, "--require-ready"]) == 1
    assert json.loads(out.read_bytes()) == report
    with pytest.raises(SystemExit) as exc:
        release.main(base)
    assert exc.value.code == 2
    assert json.loads(out.read_bytes()) == report
