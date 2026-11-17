"""Run the detached Windows qualification steps on a fresh machine-local project."""

import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


def stage_verifiers(bundle, source, destination):
    """Use one immutable set of drivers and fixtures for the entire native run."""
    release_module = Path(__file__).resolve().parents[2] / "autotem/installation/release.py"
    verifier = runpy.run_path(str(release_module))
    archive = bundle / verifier["portable_path"](source["archive"])
    if verifier["sha256"](archive) != source["archive_sha256"]:
        raise ValueError("Source archive differs from the qualification manifest")
    with ZipFile(archive) as data:
        index = verifier["verify_snapshot"](data, source["inventory_member"])
        if (
            index["git_commit"] != source["git_commit"]
            or len(index["files"]) != source["file_count"]
        ):
            raise ValueError("Qualification source inventory differs from the release")
        destination.mkdir()
        for name in index["files"]:
            if (
                name.startswith(("tools/", "configs/harness/"))
                or name == "autotem/installation/release.py"
            ):
                path = destination / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data.read(name))
    return destination / "tools"


def run(bundle, evidence, out):
    manifest = json.loads((bundle / "release-manifest.json").read_bytes())
    profiles = manifest["profiles"]
    simulation = set(profiles) == {"traditional-simulation"}
    if not simulation and (set(profiles) != {"traditional-core-mcp"} or evidence is None):
        raise ValueError("Core qualification needs cold evidence; simulation uses its own project")
    out.mkdir(parents=True, exist_ok=False)
    tools = stage_verifiers(bundle, manifest["source"], out / "verifier-source")
    project = out / "project"
    project.mkdir()
    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper() not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"}
    }
    env.update(PATH=os.path.join(os.environ["SystemRoot"], "System32"), PIP_NO_INDEX="1")
    setup = [str(bundle / "AutoTEM-Setup.exe"), "--project", str(project), "--host", "codex"]
    applied = subprocess.run(
        [*setup, "--apply"], cwd=project, env=env, capture_output=True, timeout=1950
    )
    (out / "private-install.json").write_bytes(applied.stdout)
    receipt = json.loads(applied.stdout)
    if (
        applied.returncode
        or receipt["state"] != "host_setup_required"
        or receipt["runtime"] != "passed"
    ):
        raise ValueError(
            f"Private CI installation incomplete: {(receipt.get('error') or {}).get('code')}"
        )
    python = Path(receipt["python_executable"]).resolve()
    if python != project / ".autotem/runtimes" / receipt["runtime_id"] / "Scripts/python.exe":
        raise ValueError("The launcher returned a different project interpreter")
    headless = [
        tools / "acceptance/verify_headless.py",
        "--bundle",
        bundle,
        "--project",
        project,
        "--out",
        out / "headless",
    ]
    if simulation:
        commands = [
            [
                tools / "acceptance/qualify_simulation.py",
                "--bundle",
                bundle,
                "--project",
                project,
                "--out",
                out / "simulation",
            ],
            headless,
        ]
        fault_driver = tools / "acceptance/qualify_simulation_faults.py"
        if fault_driver.is_file():  # Earlier releases only contain the positive simulation driver.
            commands.append([fault_driver, "--bundle", bundle, "--qualification", out])
    else:
        commands = core_commands(tools, bundle, evidence, out, headless)
    # Collect independent diagnostics even when an earlier scientific case failed.
    failures = []
    for command in commands:
        process = subprocess.run([python, "-I", *command], cwd=out, env=env, timeout=3600)
        if process.returncode:
            failures.append((Path(command[0]).name, process.returncode))
    if failures:
        raise ValueError(f"Local qualification steps failed: {failures}")


def core_commands(tools, bundle, evidence, out, headless):
    return [
        [
            tools / "release/contract_freeze.py",
            "collect",
            "--bundle",
            bundle,
            "--evidence",
            evidence,
            "--work",
            out / "contract-work",
            "--out",
            out / "contract-freeze.json",
        ],
        [
            tools / "acceptance/qualify_local.py",
            "--release-bundle",
            bundle,
            "--science",
            evidence / "science",
            "--variants",
            evidence / "variants",
            "--output",
            out / "local",
        ],
        headless,
        [
            tools / "release/qualify_release.py",
            "--bundle",
            bundle,
            "--evidence",
            evidence,
            "--contract",
            out / "contract-freeze.json",
            "--local",
            out / "local/local-execution.json",
            "--headless",
            out / "headless/headless.json",
            "--out",
            out / "release-index.json",
            "--require",
            "RG0",
            "--require",
            "RG1",
        ],
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--evidence", type=Path, help="Cold-install evidence for traditional-core-mcp"
    )
    args = parser.parse_args()
    try:
        run(
            args.bundle.resolve(),
            args.evidence.resolve() if args.evidence else None,
            args.out.resolve(),
        )
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        sys.exit(f"Windows qualification failed: {exc}")
