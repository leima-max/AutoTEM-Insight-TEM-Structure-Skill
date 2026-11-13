"""Collect and replay the five J15 fault cases against one installed simulation edition."""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import venv
from contextlib import closing, contextmanager
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from autotem.acceptance.evidence import digest, require

SIMULATION = "simulation.diffraction"
MATCH = "diffraction.match"
BACKEND_FILE = "diffsims/generators/simulation_generator.py"
PRODUCT_FILES = (
    "autotem/harness/service.py",
    "autotem/harness/worker.py",
    "autotem/services/diffraction.py",
    "autotem/simulation/diffsims_adapter.py",
)
CASES = {
    "missing_dependency": ("rejected", "DEPENDENCY_UNAVAILABLE"),
    "broken_dependency": ("failed", "DEPENDENCY_BROKEN"),
    "backend_timeout": ("failed", "SIMULATION_TIMEOUT"),
    "worker_timeout": ("timed_out", "TIMEOUT"),
    "missing_structure": ("needs_input", "NEEDS_INPUT"),
}


def sibling(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, sort_keys=True, allow_nan=False)


@contextmanager
def database(store):
    path = (store / "harness.sqlite3").resolve(strict=True)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        yield db


def parameters(plan, op):
    (step,) = [step for step in plan["steps"] if step["operation_id"] == op]
    return step["parameters"]


def fault_plan(positive_plan, name, path):
    require(name in CASES, "Unknown J15 fault")
    plan = copy.deepcopy(positive_plan)
    plan["input_path"] = str(path)
    if name == "backend_timeout":
        parameters(plan, SIMULATION)["diffsims_timeout_s"] = 1.0
    elif name == "worker_timeout":
        # The unchanged inner parameter is clipped to this limit by the real service.
        plan["budgets"]["simulation_seconds"] = 1.0
    elif name == "missing_structure":
        (candidate,) = parameters(plan, "diffraction.index")["candidate_phase_params"]
        candidate["atom_sites"] = []
    return plan


def check_edition(bundle, q, project):
    from autotem.installation.release import verify_release, wheel_identity

    require(project == q / "project", "Expected the qualification's managed project")
    positive_path = q / "simulation/simulation.json"
    positive = sibling("qualify_simulation").recheck(positive_path, bundle, project)
    verified = verify_release(bundle, profile="traditional-simulation")
    manifest = read(bundle / "release-manifest.json")
    with ZipFile(bundle / manifest["source"]["archive"]) as archive:
        require(
            archive.read("tools/acceptance/" + Path(__file__).name) == Path(__file__).read_bytes(),
            "Fault verifier differs from the release source",
        )
    receipt = read(q / "private-install.json")
    pointer = read(project / ".autotem/active-runtime.json")
    require(
        receipt["ok"]
        and receipt["state"] == "host_setup_required"
        and receipt["runtime"] == "passed"
        and receipt["local_transport"] == "passed"
        and receipt["profile"] == "traditional-simulation"
        and receipt["runtime_id"] == pointer["runtime_id"] == "rt-" + verified.manifest_sha256[:20]
        and Path(receipt["python_executable"]).parts[-4:]
        == ("runtimes", pointer["runtime_id"], "Scripts", "python.exe")
        and receipt["manifest_sha256"]
        == positive["release_manifest_sha256"]
        == verified.manifest_sha256,
        "Positive run and native installation belong to different editions",
    )
    store = project / ".autotem/runs"
    with database(store) as db:
        row = db.execute(
            "SELECT request,source_hash FROM runs WHERE run_id=?", (positive["run_id"],)
        ).fetchone()
    (wheel,) = [w for w in verified.wheelhouse.glob("*.whl") if wheel_identity(w)[0] == "diffsims"]
    return positive, json.loads(row["request"]), row["source_hash"], wheel, verified.wheel


def expected_image(wheel):
    with ZipFile(wheel) as archive:
        files = {}
        for name in archive.namelist():
            path = PurePosixPath(name)
            if path.parts[0] != "diffsims" or name.endswith("/"):
                continue
            require(".." not in path.parts and "\\" not in name, "Invalid backend wheel member")
            files[name] = hashlib.sha256(archive.read(name)).hexdigest()
    require(BACKEND_FILE in files, "Locked wheel does not contain the real generator")
    return files


def link_directory(source, target):
    if os.name == "nt":
        # CPython's directory junction API also works without Windows symlink privileges.
        import _winapi

        _winapi.CreateJunction(str(source), str(target))
    else:
        target.symlink_to(source, target_is_directory=True)


def make_fault_runtime(qualification, python, wheel):
    """Share other installed libraries, but materialize diffsims only in a fresh scratch venv."""
    stage = qualification / "simulation-fault-runtime"
    stage.mkdir(exist_ok=False)
    venv.EnvBuilder(with_pip=False).create(stage)
    site = python.parents[1] / "Lib/site-packages"
    mirror = stage / "Lib/site-packages"
    require(mirror.is_dir() and not any(mirror.iterdir()), "Expected an empty private site")
    require(not (site / "_autotem_qualification_observer.pth").exists(), "Import audit is active")
    require(digest(site / BACKEND_FILE) == expected_image(wheel)[BACKEND_FILE], "Changed backend")
    for src in site.iterdir():
        if src.name in {"__pycache__", "diffsims"} or src.name.lower().startswith("diffsims-"):
            continue
        require(src.resolve() == src, "Unexpected redirected installed package")
        if src.is_dir():
            link_directory(src, mirror / src.name)
        elif src.suffix != ".pth" or src.name == "pywin32.pth":
            shutil.copyfile(src, mirror / src.name)
    # Protect shared package directories from bytecode writes by any fault subprocess.
    (mirror / "_autotem_fault_bytecode.pth").write_text(
        "import sys; sys.dont_write_bytecode = True\n", encoding="ascii"
    )
    return stage / "Scripts/python.exe", mirror


def install_broken_image(mirror, wheel):
    image = expected_image(wheel)
    (mirror / "diffsims").mkdir(exist_ok=False)
    with ZipFile(wheel) as archive:
        for name in image:
            if name != BACKEND_FILE:
                target = mirror / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(archive.read(name))


def backend_observation():
    import sysconfig

    site = Path(sysconfig.get_path("purelib")).resolve()
    spec = importlib.util.find_spec("diffsims")
    files = {
        file.relative_to(site).as_posix(): digest(file)
        for file in (site / "diffsims").rglob("*")
        if file.is_file() and "__pycache__" not in file.parts
    }
    return {
        "discoverable": spec is not None,
        "module_origin": str(Path(spec.origin).relative_to(site)) if spec else None,
        "files": files,
    }


def execute(name, folder):
    """Run in its own isolated interpreter, through the ordinary Harness dispatch and worker."""
    import autotem
    from autotem.harness import Harness
    from autotem.harness.cli import dispatch_request

    require(sys.flags.isolated and not sys.flags.optimize, "Use CPython -I with assertions enabled")
    plan = read(folder / "plan.json")
    require(
        folder.name == name
        and folder.parent.name == "simulation-faults"
        and Path.cwd() == folder
        and Path(plan["input_path"]) == folder / "fixture.npy"
        and not (folder / "harness").exists()
        and not (folder / "witness.json").exists(),
        "Expected a fresh, dedicated case directory",
    )
    site = Path(autotem.__file__).resolve().parents[1]
    entry = {
        "case": name,
        "isolated": True,
        "executable": sys.executable,
        "product_files": {f: digest(site / f) for f in PRODUCT_FILES},
    }
    if name in {"missing_dependency", "broken_dependency"}:
        entry["backend"] = backend_observation()
    harness = Harness(folder / "harness")
    if name == "missing_dependency":
        calls = []
        for tool in ("validate_plan", "submit_analysis"):
            arguments = {"plan": plan}
            if tool == "submit_analysis":
                arguments["asynchronous"] = False
            calls.append(
                dispatch_request(
                    harness, {"request_id": tool, "tool": tool, "arguments": arguments}
                )
            )
        entry["replies"] = calls
    else:
        state = harness.submit_analysis(plan, asynchronous=False)
        rid = state["run_id"]
        entry.update(run_id=rid, job_id=state["job_id"], status=state["status"])
        export = harness.export_report(rid, formats=["json"])
        (report,) = export["artifacts"]
        entry["report_artifact_id"] = report["artifact_id"]
        if name == "missing_structure" and state["status"] == "needs_input":
            (request,) = state["input_requests"]
            basis = read(folder / "answer-source.json")
            reply = {
                "request_id": request["request_id"],
                "source": "metadata",
                "basis": "J15 explicit-site fixture, positive run " + basis["positive_run_id"],
                "answers": {"candidate_phase_params": {"value": basis["candidate_phase_params"]}},
            }
            child = harness.submit_analysis(
                {**plan, "user_input": {"response": reply}},
                parent_run_id=rid,
                idempotency_key="j15-restore-explicit-sites",
                asynchronous=False,
            )
            child_export = harness.export_report(child["run_id"], formats=["json"])
            (child_report,) = child_export["artifacts"]
            entry["child"] = {
                "run_id": child["run_id"],
                "job_id": child["job_id"],
                "report_artifact_id": child_report["artifact_id"],
                "parent_report_sha256_before_answer": report["sha256"],
            }
    write(folder / "witness.json", entry)


def verify_child(folder, db, frame, witness, positive, positive_plan, snapshot):
    row = db.execute("SELECT * FROM input_requests WHERE run_id=?", (frame.run_id,)).fetchone()
    require(row and row["operation_id"] == SIMULATION, "Missing persisted structure request")
    (field,) = json.loads(row["fields"])
    (original_request,) = frame.report["run"]["input_requests"]
    require(
        field["name"] == "atom_sites"
        and original_request["request_id"] == row["request_id"]
        and original_request["status"] == "open"
        and row["status"] == "answered"
        and ["candidate_phase_params"] in field["answer_options"]
        and ["cif_candidate_paths"] in field["answer_options"]
        and json.loads(row["evidence_artifact_ids"])
        and set(json.loads(row["evidence_artifact_ids"])).issubset(frame.index),
        "Missing structure evidence or usable response options",
    )
    child = witness["child"]
    rows = db.execute(
        "SELECT r.*,j.status AS job_status,j.ended FROM runs r JOIN jobs j ON r.job_id=j.job_id "
        "WHERE r.parent_run_id=?",
        (frame.run_id,),
    ).fetchall()
    (r,) = rows
    (answer,) = db.execute(
        "SELECT * FROM input_responses WHERE request_id=?", (row["request_id"],)
    ).fetchall()
    candidates = parameters(positive_plan, "diffraction.index")["candidate_phase_params"]
    supplied = json.loads(answer["response"])
    adopted = json.loads(answer["adopted_values"])["candidate_phase_params"]
    require(
        read(folder / "answer-source.json")
        == {"positive_run_id": positive["run_id"], "candidate_phase_params": candidates}
        and r["run_id"] == answer["child_run_id"] == child["run_id"]
        and r["job_id"] == child["job_id"]
        and r["status"] == r["job_status"] == "completed"
        and r["ended"] is not None
        and json.loads(answer["validation"])["status"] == "accepted"
        and supplied["source"] == "metadata"
        and positive["run_id"] in supplied["basis"]
        and supplied["answers"]["candidate_phase_params"]["value"] == candidates
        and adopted["operation_id"] == "diffraction.index"
        and adopted["parameters"]["candidate_phase_params"] == candidates
        and parameters(json.loads(r["request"]), "diffraction.index")["candidate_phase_params"]
        == candidates
        and digest(folder / "harness/runs" / r["run_id"] / "raw" / Path(r["input_path"]).name)
        == r["source_hash"]
        == snapshot
        and frame.audit[witness["report_artifact_id"]]["sha256"]
        == child["parent_report_sha256_before_answer"],
        "The sourced response, child snapshot or immutable parent report differs",
    )
    child_frame = sibling("qualify_local").RecordedRun(
        folder, "structure-child", r["run_id"], child["report_artifact_id"]
    )
    require(
        child_frame.report["report_type"] == "analysis"
        and child_frame.report["completeness"]["required_complete"]
        and all(
            child_frame.claims[op]["status"] == "completed" and child_frame.claims[op]["delivered"]
            for op in (SIMULATION, MATCH)
        ),
        "The child report does not deliver both restored operations",
    )
    numeric = sibling("qualify_simulation").numerical_evidence(child_frame)
    require(
        numeric == positive["numerical_evidence"],
        "Supplemented structure did not recover the positive result",
    )
    return {**child, "request_id": row["request_id"], "numerical_evidence": numeric}


def verify_case(folder, name, positive, positive_plan, snapshot, image, python):
    w = read(folder / "witness.json")
    plan = read(folder / "plan.json")
    status, code = CASES[name]
    require(
        w["case"] == name
        and w["isolated"] is True
        and Path(w["executable"]) == python
        and Path(plan["input_path"]).name == "fixture.npy"
        and plan == fault_plan(positive_plan, name, plan["input_path"])
        and digest(folder / "fixture.npy") == snapshot,
        "Changed case plan, runtime or raw bytes",
    )
    result = {
        "verdict": "passed",
        "status": status,
        "code": code,
        "witness": name + "/witness.json",
        "witness_sha256": digest(folder / "witness.json"),
        "plan_sha256": digest(folder / "plan.json"),
        "database": name + "/harness/harness.sqlite3",
    }
    if name in {"missing_dependency", "broken_dependency"}:
        missing = name == "missing_dependency"
        evidence = w["backend"]
        require(
            evidence["discoverable"] is (not missing)
            and evidence["module_origin"]
            == (None if missing else str(Path("diffsims/__init__.py")))
            and evidence["files"]
            == ({} if missing else {k: v for k, v in image.items() if k != BACKEND_FILE}),
            "Observed on-disk backend differs from the intended fault",
        )
    with database(folder / "harness") as db:
        require(db.execute("PRAGMA quick_check").fetchone()[0] == "ok", "Fault store integrity")
        ledger = hashlib.sha256()
        # Preserve domain records independently of WAL checkpoints or the scheduler heartbeat.
        for table in (
            "runs",
            "jobs",
            "steps",
            "events",
            "artifacts",
            "input_requests",
            "input_responses",
        ):
            ledger.update(table.encode() + b"\n")
            for record in db.execute("SELECT * FROM " + table + " ORDER BY rowid"):
                ledger.update(json.dumps(dict(record), sort_keys=True).encode() + b"\n")
        result["database_rows_sha256"] = ledger.hexdigest()
        if name == "missing_dependency":
            require(
                [r["tool"] for r in w["replies"]] == ["validate_plan", "submit_analysis"]
                and all(
                    r["ok"] is False
                    and r["error"]["code"] == code
                    and r["error"]["details"]["dependency"] == "diffsims"
                    for r in w["replies"]
                )
                and all(
                    db.execute("SELECT COUNT(*) FROM " + t).fetchone()[0] == 0
                    for t in ("runs", "jobs", "artifacts")
                )
                and not any((folder / "harness/runs").glob("*")),
                "Missing backend created work or succeeded",
            )
            return {**result, "run_id": None, "artifacts": 0}
        rows = db.execute("SELECT * FROM runs WHERE parent_run_id IS NULL").fetchall()
        (run,) = rows
        expected_runs = 2 if name == "missing_structure" else 1
        require(
            all(
                db.execute("SELECT COUNT(*) FROM " + table).fetchone()[0] == expected_runs
                for table in ("runs", "jobs")
            )
            and not db.execute("SELECT 1 FROM artifacts WHERE committed=0").fetchone(),
            "Unrelated runs, jobs or unfinished artifacts in the fault store",
        )
        rid = run["run_id"]
        (job,) = db.execute("SELECT * FROM jobs WHERE run_id=?", (rid,)).fetchall()
        events = {}
        for kind in ("submitted", "worker_started"):
            (row,) = db.execute(
                "SELECT payload FROM events WHERE run_id=? AND kind=?", (rid, kind)
            ).fetchall()
            events[kind] = json.loads(row[0])
        from autotem.harness.store import encode

        request_hash = hashlib.sha256(
            encode(
                {"plan": plan, "parent": None, "mode": "sequential", "user_request": None}
            ).encode()
        ).hexdigest()
        require(
            rid == w["run_id"]
            and job["job_id"] == run["job_id"] == w["job_id"]
            and events["submitted"]["request_hash"] == run["request_hash"] == request_hash
            and events["submitted"]["source_hash"] == snapshot
            and events["worker_started"]
            == {"pid": run["worker_pid"], "process_identity": run["worker_identity"]}
            and run["worker_pid"]
            and run["worker_identity"]
            and json.loads(run["request"]) == plan
            and run["status"] == job["status"] == w["status"] == status
            and job["ended"] is not None
            and run["source_hash"] == snapshot
            and digest(folder / "harness/runs" / rid / "raw" / Path(run["input_path"]).name)
            == snapshot,
            "Fault run, finalized job, registry binding or snapshot differs",
        )
        steps = {
            s["operation_id"]: s for s in db.execute("SELECT * FROM steps WHERE run_id=?", (rid,))
        }
        error = json.loads(run["error"])
        sim = steps[SIMULATION]
        require(
            steps["diffraction.index"]["status"] == "completed"
            and sim["status"] == status
            and json.loads(sim["error"])["code"] == error["code"] == code
            and steps[MATCH]["status"] == "blocked",
            "Failure occurred in a different step",
        )
        if name in {"backend_timeout", "broken_dependency"}:
            adapter = error["details"]["provenance"]["adapter"]
            expected_reason = "timeout" if name == "backend_timeout" else "import_failed"
            require(
                adapter["reason"] == expected_reason and not adapter["active"],
                "Wrong backend failure",
            )
        elif name == "worker_timeout":
            require(
                error["step_id"] == sim["step_id"] and sim["ended"] - sim["started"] >= 1,
                "Wrong outer deadline",
            )
        frame = sibling("qualify_local").RecordedRun(folder, name, rid, w["report_artifact_id"])
        require(
            frame.report["report_type"] == "diagnostic"
            and frame.report["scientific_accuracy"] == "not_evaluated"
            and frame.report["run"]["status"] == frame.report["run"]["jobs"][-1]["status"] == status
            and frame.index[w["report_artifact_id"]]["role"] == "report_manifest"
            and not any(
                a["execution_usable"] and a["step_id"] in {sim["step_id"], steps[MATCH]["step_id"]}
                for a in frame.index.values()
            ),
            "False scientific evidence or inconsistent diagnostic report",
        )
        result.update(
            run_id=rid,
            job_id=job["job_id"],
            report_artifact_id=w["report_artifact_id"],
            artifacts=len(frame.audit),
            report_sha256=frame.audit[w["report_artifact_id"]]["sha256"],
        )
        if name == "missing_structure":
            result["child"] = verify_child(folder, db, frame, w, positive, positive_plan, snapshot)
    return result


def summary(out, bundle, project):
    q = out.parent
    positive, plan, snapshot, wheel, product_wheel = check_edition(bundle, q, project)
    image = expected_image(wheel)
    receipt = read(q / "private-install.json")
    original = Path(receipt["python_executable"])
    # Persist original executable names so an archive may relocate the raw evidence.
    scratch = original.parents[5] / "simulation-fault-runtime/Scripts/python.exe"
    origin = read(out / "backend-origin.json")
    require(
        origin
        == {
            "wheel": wheel.name,
            "wheel_sha256": digest(wheel),
            "module": BACKEND_FILE,
            "before_sha256": image[BACKEND_FILE],
            "after_sha256": image[BACKEND_FILE],
        },
        "Changed managed backend evidence",
    )
    with ZipFile(product_wheel) as product:
        expected = {f: hashlib.sha256(product.read(f)).hexdigest() for f in PRODUCT_FILES}
    cases = {}
    for name in CASES:
        folder = out / name
        execution = folder / "execution.json"
        if not execution.exists():
            cases[name] = {"verdict": "not_evaluated", "reason": "No case execution receipt"}
            continue
        recorded = read(execution)
        log_hash = digest(folder / "driver.log")
        require(recorded["log_sha256"] == log_hash, "Fault driver log changed")
        try:
            require(recorded["exit_code"] == 0, "Fault driver failed or exceeded its deadline")
            require(
                read(folder / "witness.json")["product_files"] == expected,
                "Fault subprocess used different product modules",
            )
            runtime = scratch if name in {"missing_dependency", "broken_dependency"} else original
            cases[name] = verify_case(folder, name, positive, plan, snapshot, image, runtime)
        except (
            ValueError,
            OSError,
            KeyError,
            IndexError,
            TypeError,
            AssertionError,
            sqlite3.Error,
        ) as exc:
            cases[name] = {
                "verdict": "failed",
                "reason": str(exc),
                "error_type": type(exc).__name__,
            }
        cases[name].update(
            execution=name + "/execution.json",
            execution_sha256=digest(execution),
            log_sha256=log_hash,
        )
    return {
        "schema_version": "1.0.0",
        "case_verdict": "passed"
        if all(c["verdict"] == "passed" for c in cases.values())
        else "failed",
        "profile": "traditional-simulation",
        "release_manifest_sha256": positive["release_manifest_sha256"],
        "positive_run_id": positive["run_id"],
        "positive_sha256": digest(q / "simulation/simulation.json"),
        "verifier_sha256": digest(Path(__file__)),
        "installation_sha256": digest(q / "private-install.json"),
        "backend_origin_sha256": digest(out / "backend-origin.json"),
        "fixture_sha256": snapshot,
        "cases": cases,
        "scientific_accuracy": "not_evaluated",
    }


def recheck(report, bundle, project):
    result = summary(report.parent, bundle, project)
    require(result == read(report), "Fault certificate differs from the physical evidence")
    return result


def collect_cases(q, python, positive, plan, fixture, snapshot, wheel):
    out = q / "simulation-faults"
    out.mkdir(exist_ok=False)
    scratch_python, mirror = make_fault_runtime(q, python, wheel)
    generator = python.parents[1] / "Lib/site-packages" / BACKEND_FILE
    before = digest(generator)
    require(digest(fixture) == snapshot, "Positive fixture changed")
    for name in CASES:
        folder = out / name
        folder.mkdir()
        src = folder / "fixture.npy"
        shutil.copyfile(fixture, src)
        write(folder / "plan.json", fault_plan(plan, name, src))
        if name == "broken_dependency":
            install_broken_image(mirror, wheel)
        if name == "missing_structure":
            write(
                folder / "answer-source.json",
                {
                    "positive_run_id": positive["run_id"],
                    "candidate_phase_params": parameters(plan, "diffraction.index")[
                        "candidate_phase_params"
                    ],
                },
            )
        executable = (
            scratch_python if name in {"missing_dependency", "broken_dependency"} else python
        )
        with (folder / "driver.log").open("xb") as log:
            try:
                result = subprocess.run(
                    [
                        str(executable),
                        "-I",
                        str(Path(__file__).resolve()),
                        "--case",
                        name,
                        str(folder),
                    ],
                    cwd=folder,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=420,
                )
                exit_code = result.returncode
            except subprocess.TimeoutExpired:
                exit_code = None
        write(
            folder / "execution.json",
            {"exit_code": exit_code, "log_sha256": digest(folder / "driver.log")},
        )
        if exit_code is None:
            break
    write(
        out / "backend-origin.json",
        {
            "wheel": wheel.name,
            "wheel_sha256": digest(wheel),
            "module": BACKEND_FILE,
            "before_sha256": before,
            "after_sha256": digest(generator),
        },
    )
    return out


def collect(bundle, q):
    from autotem.installation.host_entry import active_python

    project = q / "project"
    python = active_python(project / ".autotem")
    require(
        sys.flags.isolated and Path(sys.executable) == python,
        "Use the managed qualification interpreter",
    )
    positive, plan, snapshot, wheel, _ = check_edition(bundle, q, project)
    with database(project / ".autotem/runs") as db:
        require(
            [tuple(r) for r in db.execute("SELECT run_id,status FROM runs")]
            == [(positive["run_id"], "completed")]
            and [tuple(r) for r in db.execute("SELECT job_id,status FROM jobs")]
            == [(positive["job_id"], "completed")],
            "Fault qualification requires a dedicated, idle single-run CI project",
        )
    out = collect_cases(
        q, python, positive, plan, q / "simulation/J01 合成 HRTEM.npy", snapshot, wheel
    )
    result = summary(out, bundle, project)
    write(out / "faults.json", result)
    return recheck(out / "faults.json", bundle, project)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--case":
        require(sys.argv[2] in CASES, "Unknown fault")
        execute(sys.argv[2], Path(sys.argv[3]).resolve())
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--bundle", type=Path, required=True)
        parser.add_argument("--qualification", type=Path, required=True)
        args = parser.parse_args()
        record = collect(args.bundle.resolve(), args.qualification.resolve())
        print(json.dumps({"case_verdict": record["case_verdict"], "cases": list(record["cases"])}))
        raise SystemExit(0 if record["case_verdict"] == "passed" else 1)
