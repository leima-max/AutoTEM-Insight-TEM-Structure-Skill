"""Observe UI/DL import and discovery attempts across installed scientific processes."""

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import zipfile
from collections import Counter
from contextlib import closing
from pathlib import Path

ROOTS = (
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "gradio",
    "keras",
    "napari",
    "pytorch_lightning",
    "streamlit",
    "streamlit_drawable_canvas",
    "tensorflow",
    "tkinter",
    "torch",
    "torchvision",
)
CONTROL = "_autotem_missing_import_control"
OBSERVER = Path(__file__).with_name("import_observer.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_logs(logs, expected, *, simulation_runs=()):
    roles = Counter()
    observed = set()
    simulated = set()
    control = set()
    attempts = []
    for path in sorted(logs.glob("*.jsonl")):
        meta, *events = [json.loads(line) for line in path.read_bytes().splitlines()]
        role = meta["role"]
        roles[role] += 1
        check(meta["isolated"] == 1, "Unisolated scientific or probe process")
        if role == "worker":
            observed.add((meta["run_id"], meta["pid"]))
        if role == "diffsims_worker":
            simulated.add(meta["run_id"])
        for event in events:
            if event["module"] == CONTROL and role == "positive_control":
                control.add(event["event"])
            else:
                check(event["module"] in ROOTS, "Unexpected module in observer evidence")
                attempts.append({"role": role, **event})
    check(control == {"find_spec", "import"}, "Missing positive import/discovery control")
    check(roles["positive_control"] == 1, "Ambiguous positive control")
    check(
        all(roles[r] for r in ("bridge", "scheduler", "worker", "disk_driver", "variant_driver")),
        "Missing an expected type of scientific process",
    )
    check(expected and observed == expected, "Observed workers differ from persisted run/PID pairs")
    if simulation_runs:
        check(simulated == set(simulation_runs), "Missing or unrelated diffsims subprocess")
    return roles, attempts


def summary(bundle, out):
    manifest = bundle / "release-manifest.json"
    release_hash = sha(manifest)
    inputs = {}
    for name in ("science/evidence.json", "variants/operations.json"):
        path = out / name
        value = read(path)
        check(
            value["case_verdict"] == "passed"
            and value["runtime"]["private_install_verified"]
            and value["runtime"]["release_identity"]["manifest_sha256"] == release_hash,
            "Scientific run belongs to a different release or failed",
        )
        inputs[name] = sha(path)
    check(
        read(out / "variants/operations.json")["driver"]["source_archive_match"] is True,
        "Variant driver differs from the release",
    )
    simulation_runs = set()
    if set(read(manifest)["profiles"]) == {"traditional-simulation"}:
        case = read(out / "variants/cubic-simulation/variant-evidence.json")
        check(
            case["verdict"] == "passed"
            and case["runtime"]["release_identity"]["manifest_sha256"] == release_hash
            and all(
                case["operations"][op]["verdict"] == "passed"
                for op in ("simulation.diffraction", "diffraction.match")
            ),
            "Simulation was not exercised under the import observer",
        )
        inputs["variants/cubic-simulation/variant-evidence.json"] = sha(
            out / "variants/cubic-simulation/variant-evidence.json"
        )
        simulation_runs.add(case["run_id"])
    instances = set()
    for path in sorted(out.glob("**/harness/harness.sqlite3")):
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            # Additional work packages on the same run receive new workers. The run row
            # contains only the last PID; worker_started events preserve the entire history.
            starts = db.execute("SELECT run_id,payload FROM events WHERE kind='worker_started'")
            for run, payload in starts:
                instances.add((run, json.loads(payload)["pid"]))
            rows = db.execute("SELECT run_id,worker_pid FROM runs WHERE worker_pid IS NOT NULL")
            for run, pid in rows:
                instances.add((run, pid))
    logs = out / "processes"
    roles, attempts = inspect_logs(logs, instances, simulation_runs=simulation_runs)
    ledger = {p.name: sha(p) for p in sorted(logs.glob("*.jsonl"))}
    return {
        "schema_version": "1.0.0",
        "case_verdict": "failed" if attempts else "passed",
        "release_manifest_sha256": release_hash,
        "verifier_sha256": sha(Path(__file__)),
        "observer_sha256": sha(OBSERVER),
        "module_roots": list(ROOTS),
        "process_counts": dict(sorted(roles.items())),
        "worker_runs_verified": len({run for run, _ in instances}),
        "worker_instances_verified": len(instances),
        "worker_run_pairs_sha256": hashlib.sha256(
            json.dumps(sorted(instances)).encode()
        ).hexdigest(),
        "observed_attempts": attempts,
        "evidence_sha256": inputs,
        "process_files_sha256": ledger,
    }


def recheck(bundle, report):
    result = summary(bundle, report.parent)
    check(result == read(report), "Import audit differs from the raw process or run records")
    return result


def collect(bundle, project, out):
    pointer = read(project / ".autotem/active-runtime.json")
    identity = sha(bundle / "release-manifest.json")
    rid = "rt-" + identity[:20]
    check(
        pointer.get("runtime_id") == rid and pointer.get("manifest_sha256") == identity,
        "Disposable project does not contain this release",
    )
    runtime = project / ".autotem/runtimes" / rid
    python = runtime / "Scripts/python.exe"
    site = runtime / "Lib/site-packages"
    check(python.is_file() and site.is_dir(), "Missing private Windows runtime")
    out.mkdir(parents=True, exist_ok=False)
    logs = out / "processes"
    logs.mkdir()
    driver = out / "verify_variants.py"
    manifest = read(bundle / "release-manifest.json")
    source = bundle / manifest["source"]["archive"]
    check(sha(source) == manifest["source"]["archive_sha256"], "Source archive mismatch")
    with zipfile.ZipFile(source) as archive:
        for name in (driver.name, "operation_variants.json"):
            (out / name).write_bytes(archive.read("tools/acceptance/" + name))
    options = dict(audit_root=str(logs), monitored_roots=list(ROOTS), sentinel=CONTROL)
    observer_path = ascii(str(OBSERVER.resolve()))
    line = f"import runpy; runpy.run_path({observer_path}, init_globals={ascii(options)})\n"
    marker = site / "_autotem_qualification_observer.pth"
    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper() not in {"PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "VIRTUAL_ENV"}
    }
    env.update(PATH=os.path.join(os.environ["SystemRoot"], "System32"), PIP_NO_INDEX="1")
    calls = (
        [
            "-c",
            f"import importlib.util; importlib.util.find_spec('{CONTROL}');\n"
            f"try: __import__('{CONTROL}')\nexcept ModuleNotFoundError: pass",
            CONTROL,
        ],
        [
            "-m",
            "autotem.acceptance",
            "--require-installed",
            "--release-bundle",
            str(bundle),
            "--output",
            str(out / "science"),
            "--extended",
        ],
        [
            str(driver),
            "--require-installed",
            "--release-bundle",
            str(bundle),
            "--output",
            str(out / "variants"),
            *(["--simulation"] if set(manifest["profiles"]) == {"traditional-simulation"} else []),
        ],
    )
    with marker.open("xb") as stream:
        stream.write(line.encode("ascii"))
    try:
        for cmd in calls:
            subprocess.run(
                [str(python), "-I", *cmd],
                cwd=out,
                env=env,
                timeout=1500,
                check=True,
                capture_output=True,
            )
    finally:
        check(marker.read_bytes() == line.encode("ascii"), "Audit marker changed during run")
        marker.unlink()
    result = summary(bundle, out)
    report = out / "headless.json"
    with report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return recheck(bundle, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--project", type=Path, help="Disposable project reserved for this audit")
    parser.add_argument("--out", type=Path, help="New private evidence directory")
    parser.add_argument("--recheck", type=Path, help="Recompute an existing headless.json")
    args = parser.parse_args()
    if not args.recheck and not (args.project and args.out):
        parser.error("Collection needs both a disposable project and a new output directory")
    try:
        bundle = args.bundle.resolve()
        result = (
            recheck(bundle, args.recheck.resolve())
            if args.recheck
            else collect(bundle, args.project.resolve(), args.out.resolve())
        )
    except (ValueError, OSError, sqlite3.Error, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Import audit error: {exc}\n")
    print(
        json.dumps(
            {k: result[k] for k in ("case_verdict", "worker_runs_verified", "process_counts")}
        )
    )
    return 0 if result["case_verdict"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
