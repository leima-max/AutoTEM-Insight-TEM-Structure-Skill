"""Run the archived source suite in a new local Git checkout, retaining JUnit and source hashes."""

import argparse
import hashlib
import json
import os
import platform
import re
import runpy
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class CollectionInventory:
    """Record pytest's actual selection and link each JUnit case to its collected node."""

    def __init__(self, path):
        self.path = path
        self.deselected = []

    def pytest_deselected(self, items):
        self.deselected.extend(item.nodeid for item in items)

    def pytest_collection_finish(self, session):
        for item in session.items:
            item.user_properties.append(("autotem_nodeid", item.nodeid))
        selection = {
            "targets": session.config.args,
            "keyword": session.config.getoption("keyword"),
            "marker": session.config.getoption("markexpr"),
            "selected": [item.nodeid for item in session.items],
            "deselected": self.deselected,
        }
        with self.path.open("x", encoding="utf-8") as stream:
            print(json.dumps(selection, indent=2), file=stream)


def checked_sources(checkout, files):
    for name, expected in files.items():
        if hashlib.sha256((checkout / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source differs from archive: {name}")
    for directory in ("autotem", "tests"):
        extra = [
            p
            for p in (checkout / directory).rglob("*.py")
            if p.relative_to(checkout).as_posix() not in files
        ]
        if extra:
            raise ValueError("Unarchived product or test module in the controlled source checkout")
    return len(files)


def stage_checkout(bundle, source, checkout, support_repo, verifier):
    archive = (bundle / verifier.portable_path(source["archive"])).resolve()
    if not archive.is_relative_to(bundle) or verifier.sha256(archive) != source["archive_sha256"]:
        raise ValueError("Source archive differs from manifest")
    with zipfile.ZipFile(archive) as z:
        index = verifier.verify_snapshot(z, source["inventory_member"])
    if index["git_commit"] != source["git_commit"] or len(index["files"]) != source["file_count"]:
        raise ValueError("Source inventory differs from the manifest")
    commit = source["git_commit"]
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
        raise ValueError("Source archive has no full Git commit ID")
    git_root = subprocess.check_output(
        ["git", "-C", str(support_repo), "rev-parse", "--show-toplevel"], encoding="utf-8"
    ).strip()
    checkout.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "init", "--quiet"], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "fetch",
            "--depth=1",
            "--no-tags",
            "--no-recurse-submodules",
            Path(git_root).as_uri(),
            commit,
        ],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=checkout, check=True
    )
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, encoding="ascii")
    if actual.strip() != commit:
        raise ValueError("Detached source checkout does not match the frozen Git commit")
    with zipfile.ZipFile(archive) as z:
        for name in index["files"]:
            path = checkout / verifier.portable_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read(name))
    checked_sources(checkout, index["files"])
    return index


def run(
    bundle, checkout, out, scratch, simulation_bundle, support_repo, test_wheelhouse, test_lock
):
    qualifier = runpy.run_path(str(Path(__file__).with_name("qualify_release.py")))
    environment = qualifier["load_module"](Path(__file__).with_name("source_test_environment.py"))
    verifier = qualifier["load_module"](ROOT / "autotem/installation/release.py")
    manifest_file = bundle / "release-manifest.json"
    source = verifier.unique_json(manifest_file.read_bytes())["source"]
    targets = (checkout, out, scratch)
    if any(p.exists() for p in targets) or any(
        p.is_relative_to(other)
        for i, p in enumerate(targets)
        for j, other in enumerate((*targets, bundle, simulation_bundle))
        if i != j
    ):
        raise ValueError(
            "Checkout, output and test scratch must be distinct new directories outside the bundles"
        )
    out.mkdir(parents=True)
    scratch.mkdir(parents=True)
    index = stage_checkout(bundle, source, checkout, support_repo, verifier)
    env = environment.process_env(scratch)
    test_python = environment.prepare(
        verifier,
        {"core": bundle, "simulation": simulation_bundle},
        checkout,
        out,
        scratch,
        env,
        test_wheelhouse,
        test_lock,
    )
    junit = out / "source-regression.junit.xml"
    inventory = out / "source-regression.collection.json"
    started = time.monotonic()
    with (out / "pytest.log").open("wb") as log:
        process = subprocess.run(
            [
                str(test_python),
                "-I",
                "-B",
                "-X",
                "utf8",
                str(checkout / "tools/release/run_source_tests.py"),
                "--pytest-inventory",
                str(inventory),
                "-o",
                "addopts=-q",
                "--basetemp",
                str(scratch / "pytest"),
                "--junitxml",
                str(junit),
                "tests",
            ],
            cwd=checkout,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    count = checked_sources(checkout, index["files"])
    selection = json.loads(inventory.read_bytes())
    cases = ET.parse(junit).getroot().findall(".//testcase")
    totals = Counter()
    optional = []
    for case in cases:
        state = next(
            (k for k in ("error", "failure", "skipped") if case.find(k) is not None), "passed"
        )
        totals[state] += 1
        if state == "passed" and case.get("name") in {
            "test_real_diffsims_workflow_closes_simulation_phase_matching_loop",
            "test_optional_real_simulation_and_matching_from_disk",
        }:
            optional.append(case.get("name"))
    passed = (
        bool(cases)
        and process.returncode == 0
        and len(optional) == 2
        and totals["passed"] == len(cases)
        and not selection["deselected"]
        and len(selection["selected"]) == len(cases)
    )
    result = {
        "schema_version": "1.2.0",
        "release_manifest_sha256": verifier.sha256(manifest_file),
        "source_archive_sha256": source["archive_sha256"],
        "support_git_commit": source["git_commit"],
        "source_files_byte_verified": count,
        "execution_layer": "source_checkout",
        "python_version": platform.python_version(),
        "pytest_selection": "all",
        "pytest_default_smoke_overridden": True,
        "optional_diffsims_enabled": True,
        "optional_tests_passed": sorted(optional),
        "junit_sha256": verifier.sha256(junit),
        "collection_sha256": verifier.sha256(inventory),
        "simulation_manifest_sha256": verifier.sha256(simulation_bundle / "release-manifest.json"),
        "environment_sha256": verifier.sha256(out / "source-regression.environment.json"),
        "results": {
            "tests": len(cases),
            "collected": len(selection["selected"]) + len(selection["deselected"]),
            "deselected": len(selection["deselected"]),
            "passed": totals["passed"],
            "skipped": totals["skipped"],
            "failures": totals["failure"],
            "errors": totals["error"],
            "elapsed_seconds": round(time.monotonic() - started, 2),
        },
        "case_verdict": "passed" if passed else "failed",
    }
    output = out / "source-regression.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    checked = qualifier["source_tests"](
        output,
        result["release_manifest_sha256"],
        source["archive_sha256"],
        count,
        bundle=bundle,
        simulation_identity=result["simulation_manifest_sha256"],
    )
    print(json.dumps(result["results"]))
    return 0 if passed and checked["verdict"] == "passed" else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["--pytest-inventory"]:
        import pytest

        # Match `python -m pytest` in the detached archive checkout.
        sys.path.insert(0, str(Path.cwd()))
        inventory_plugin = CollectionInventory(Path(sys.argv[2]))
        raise SystemExit(pytest.main(sys.argv[3:], plugins=[inventory_plugin]))
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "simulation-bundle", "checkout", "out", "scratch"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--support-repo",
        type=Path,
        default=Path(os.environ.get("GITHUB_WORKSPACE", ROOT)),
        help="Git checkout for the archived support commit. Defaults to the CI or source worktree.",
    )
    parser.add_argument(
        "--test-wheelhouse",
        type=Path,
        help="Use these local test-only wheels offline. Otherwise fetch the hashed pins from PyPI.",
    )
    parser.add_argument(
        "--test-lock",
        type=Path,
        help="Explicit supplements for historical source archives without a test lock.",
    )
    args = parser.parse_args()
    raise SystemExit(
        run(
            args.bundle.resolve(),
            args.checkout.resolve(),
            args.out.resolve(),
            args.scratch.resolve(),
            args.simulation_bundle.resolve(),
            args.support_repo.resolve(),
            args.test_wheelhouse.resolve() if args.test_wheelhouse else None,
            args.test_lock.resolve() if args.test_lock else None,
        )
    )
