import copy
import importlib.util
import json
import sqlite3
from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.fixture
def faults():
    path = Path(__file__).parents[1] / "tools/acceptance/qualify_simulation_faults.py"
    spec = importlib.util.spec_from_file_location("test_simulation_fault_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def positive_plan():
    return {
        "input_path": "positive.npy",
        "budgets": {"simulation_seconds": 120.0},
        "steps": [
            {
                "operation_id": "diffraction.index",
                "parameters": {
                    "candidate_phase_params": [{"name": "test", "atom_sites": [{"element": "Si"}]}],
                },
            },
            {"operation_id": "simulation.diffraction", "parameters": {"diffsims_timeout_s": 90.0}},
        ],
    }


def test_faults_change_only_the_requested_bound_or_sites(faults, positive_plan):
    immutable = copy.deepcopy(positive_plan)
    plans = {name: faults.fault_plan(positive_plan, name, "fixture.npy") for name in faults.CASES}
    assert positive_plan == immutable
    clean = {**positive_plan, "input_path": "fixture.npy"}
    assert plans["missing_dependency"] == plans["broken_dependency"] == clean
    expected = copy.deepcopy(clean)
    expected["steps"][1]["parameters"]["diffsims_timeout_s"] = 1.0
    assert plans["backend_timeout"] == expected
    assert plans["worker_timeout"] == {**clean, "budgets": {"simulation_seconds": 1.0}}
    expected = copy.deepcopy(clean)
    expected["steps"][0]["parameters"]["candidate_phase_params"][0]["atom_sites"] = []
    assert plans["missing_structure"] == expected
    with pytest.raises(ValueError, match="Unknown"):
        faults.fault_plan(positive_plan, "other", "")


def test_disposable_broken_image_omits_exactly_the_real_member(faults, tmp_path):
    wheel = tmp_path / "diffsims.whl"
    with ZipFile(wheel, "w") as z:
        z.writestr("diffsims/__init__.py", "version=1")
        z.writestr("diffsims/generators/__init__.py", "from . import simulation_generator")
        z.writestr(faults.BACKEND_FILE, "class SimulationGenerator: pass")
    before = wheel.read_bytes()
    mirror = tmp_path / "private"
    mirror.mkdir()
    faults.install_broken_image(mirror, wheel)
    ledger = {p.relative_to(mirror).as_posix(): faults.digest(p) for p in mirror.rglob("*.py")}
    assert ledger == {
        k: v for k, v in faults.expected_image(wheel).items() if k != faults.BACKEND_FILE
    }
    with pytest.raises(FileExistsError):
        faults.install_broken_image(mirror, wheel)
    assert wheel.read_bytes() == before
    with ZipFile(wheel, "a") as z:
        z.writestr("diffsims/../escaped.py", "unsafe relative name")
    with pytest.raises(ValueError, match="wheel member"):
        faults.expected_image(wheel)


def test_rejected_submission_requires_public_replies_empty_registry_and_missing_files(
    faults, positive_plan, tmp_path
):
    name = "missing_dependency"
    folder = tmp_path / name
    (folder / "harness").mkdir(parents=True)
    fixture = folder / "fixture.npy"
    fixture.write_bytes(b"test source")
    faults.write(folder / "plan.json", faults.fault_plan(positive_plan, name, fixture))
    exe = tmp_path / "private/python.exe"
    w = {
        "case": name,
        "isolated": True,
        "executable": str(exe),
        "backend": {"discoverable": False, "module_origin": None, "files": {}},
        "replies": [
            {
                "tool": t,
                "ok": False,
                "error": {"code": "DEPENDENCY_UNAVAILABLE", "details": {"dependency": "diffsims"}},
            }
            for t in ("validate_plan", "submit_analysis")
        ],
    }
    faults.write(folder / "witness.json", w)
    dbpath = folder / "harness/harness.sqlite3"
    with sqlite3.connect(dbpath) as db:
        for table in (
            "runs",
            "jobs",
            "steps",
            "events",
            "artifacts",
            "input_requests",
            "input_responses",
        ):
            db.execute("CREATE TABLE " + table + "(id INTEGER)")

    def replay():
        return faults.verify_case(folder, name, {}, positive_plan, faults.digest(fixture), {}, exe)

    assert replay()["status"] == "rejected"
    w["replies"][0]["error"]["code"] = "DEPENDENCY_BROKEN"
    (folder / "witness.json").write_text(json.dumps(w))
    with pytest.raises(ValueError, match="created work or succeeded"):
        replay()
    w["replies"][0]["error"]["code"] = "DEPENDENCY_UNAVAILABLE"
    (folder / "witness.json").write_text(json.dumps(w))
    with sqlite3.connect(dbpath) as db:
        db.execute("INSERT INTO jobs VALUES(1)")
    with pytest.raises(ValueError, match="created work or succeeded"):
        replay()
    with sqlite3.connect(dbpath) as db:
        db.execute("DELETE FROM jobs")
    (folder / "harness/runs/ghost").mkdir(parents=True)
    with pytest.raises(ValueError, match="created work or succeeded"):
        replay()


def test_recheck_compares_full_certificate(faults, monkeypatch, tmp_path):
    observed = {"case_verdict": "passed", "positive_run_id": "observed", "cases": {}}
    monkeypatch.setattr(faults, "summary", lambda *args: observed)
    report = tmp_path / "faults.json"
    faults.write(report, {**observed, "positive_run_id": "foreign"})
    with pytest.raises(ValueError, match="physical evidence"):
        faults.recheck(report, tmp_path, tmp_path)


def test_missing_execution_and_foreign_product_do_not_pass(faults, monkeypatch, tmp_path):
    q = tmp_path
    out = q / "simulation-faults"
    folder = out / "broken_dependency"
    folder.mkdir(parents=True)
    (q / "simulation").mkdir()
    positive = {"run_id": "same-positive", "release_manifest_sha256": "same-release"}
    faults.write(q / "simulation/simulation.json", positive)
    exe = q / "project/.autotem/runtimes/rt-test/Scripts/python.exe"
    faults.write(q / "private-install.json", {"python_executable": str(exe)})
    wheel = q / "backend.whl"
    with ZipFile(wheel, "w") as z:
        z.writestr(faults.BACKEND_FILE, "test generator")
    product = q / "product.whl"
    with ZipFile(product, "w") as z:
        for file in faults.PRODUCT_FILES:
            z.writestr(file, "test release")
    member = faults.expected_image(wheel)[faults.BACKEND_FILE]
    faults.write(
        out / "backend-origin.json",
        {
            "wheel": wheel.name,
            "wheel_sha256": faults.digest(wheel),
            "module": faults.BACKEND_FILE,
            "before_sha256": member,
            "after_sha256": member,
        },
    )
    monkeypatch.setattr(faults, "check_edition", lambda *args: (positive, {}, "", wheel, product))
    (folder / "driver.log").write_bytes(b"test trace")
    faults.write(
        folder / "execution.json",
        {"exit_code": 0, "log_sha256": faults.digest(folder / "driver.log")},
    )
    faults.write(folder / "witness.json", {"product_files": {"worker": "different distribution"}})
    result = faults.summary(out, q, q / "project")
    assert result["case_verdict"] == result["cases"]["broken_dependency"]["verdict"] == "failed"
    assert result["cases"]["missing_dependency"]["verdict"] == "not_evaluated"
    (folder / "driver.log").write_bytes(b"changed")
    with pytest.raises(ValueError, match="driver log changed"):
        faults.summary(out, q, q / "project")


def test_refuses_existing_fault_runtime(faults, tmp_path):
    stage = tmp_path / "simulation-fault-runtime"
    stage.mkdir()
    marker = stage / "user-file"
    marker.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        faults.make_fault_runtime(tmp_path, tmp_path / "python", tmp_path / "wheel")
    assert marker.read_bytes() == b"existing"
