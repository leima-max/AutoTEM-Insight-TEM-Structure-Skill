"""Cold-install a single traditional-cli bundle and run J01 through its project entry."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import zipfile
from contextlib import closing
from pathlib import Path

PROFILE = "traditional-cli"


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n", "utf-8")


def execute(command, project, environment, timeout, *, success=True):
    response = subprocess.run(
        command,
        cwd=project,
        env=environment,
        capture_output=True,
        timeout=timeout,
    )
    if success is not None and (response.returncode == 0) != success:
        detail = (response.stdout or response.stderr)[-1500:].decode(
            "utf-8", errors="backslashreplace"
        )
        raise ValueError(f"{Path(command[0]).name} returned {response.returncode}: {detail}")
    return json.loads(response.stdout.decode("utf-8"))


def installed(bundle, output):
    # This leg is run by the activated wheel's absolute interpreter with -I outside the checkout.
    from autotem.acceptance.bridge import open_bridge
    from autotem.acceptance.cases import write_fixture
    from autotem.acceptance.driver import run_case, runtime_record
    from autotem.acceptance.evidence import digest

    project = output / "project"
    control = project / ".autotem"
    root = control / "runs"
    runtime = runtime_record(True, bundle)
    identity = runtime["release_identity"]
    require(identity["profile"] == PROFILE, "Wrong installed profile")
    require(importlib.util.find_spec("mcp") is None, "The CLI edition unexpectedly imports MCP")
    require(not list(importlib.metadata.distributions(name="mcp")), "The MCP package is installed")
    adapter = json.loads((project / "autotem-host/codex/adapter.json").read_bytes())
    cli = adapter["cli_command"]
    base = Path(sys._base_executable).resolve()
    require(base.is_relative_to(control / "python"), "Project does not own the base interpreter")
    require(
        len(cli) == 3
        and Path(cli[0]) == base
        and cli[1] == "-I"
        and Path(cli[2]) == control / "host-entry.py",
        "Unexpected stable command",
    )
    require(adapter["jsonl_command"] == [*cli, "stdio"], "Unexpected JSONL entry")
    require(
        adapter["preferred_transport"] == "jsonl" and "mcp" not in adapter,
        "CLI assets advertise an MCP connector",
    )
    require(not (project / ".codex/config.toml").exists(), "CLI setup wrote Codex MCP config")
    capabilities = execute([*cli, "capabilities"], project, os.environ, 120)
    require(capabilities.get("ok") is True, "Stable CLI capabilities failed")
    fixtures = output / "fixtures"
    fixtures.mkdir()
    source, fixture = write_fixture("J01", fixtures)
    fixture.update(sha256=digest(source), path=source.relative_to(output).as_posix())
    trace = []
    try:
        with open_bridge(output, trace, command=adapter["jsonl_command"]) as call:
            rows = call("list_capabilities")["operations"]
            require(
                rows and rows == capabilities["result"]["operations"],
                "CLI and JSONL capabilities differ",
            )
            matrix = {row["operation_id"]: {**row, "execution_evidence": []} for row in rows}
            case = run_case(call, output, fixture, source, matrix, store_root=root)
    finally:
        write(output / "tool-trace.json", trace)
    with closing(sqlite3.connect((root / "harness.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
        events = db.execute(
            "SELECT payload FROM events WHERE run_id=? AND kind='runtime_bound'", (case["run_id"],)
        ).fetchall()
        require(len(events) == 1, "Missing managed runtime binding")
        binding = json.loads(events[0][0])
    require(
        binding["runtime_id"] == "rt-" + identity["manifest_sha256"][:20]
        and binding["job_id"] == case["job_id"],
        "Science job used the wrong runtime",
    )
    write(
        output / "science.json",
        {
            "schema_version": "1.0.0",
            "case_verdict": "passed",
            "runtime": runtime,
            "fixture": fixture,
            "case": case,
            "runtime_binding": binding,
            "capability_count": len(rows),
            "local_mcp": "not_in_profile",
        },
    )
    return {"case_verdict": "passed", "run_id": case["run_id"], "artifacts": len(case["artifacts"])}


def run(bundle, output):
    output.mkdir(parents=True, exist_ok=False)
    project = output / "project"
    project.mkdir()
    program = Path(__file__).resolve()
    manifest = bundle / "release-manifest.json"
    report = {
        "schema_version": "1.0.0",
        "case_verdict": "failed",
        "profile": PROFILE,
        "verifier_sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
        "scientific_accuracy": "not_evaluated",
        "host_connection": "not_evaluated",
        "release_ready": False,
        "checks": [],
    }
    try:
        manifest_bytes = manifest.read_bytes()
        report["release_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        release = json.loads(manifest_bytes)
        require(set(release["profiles"]) == {PROFILE}, "Expected a traditional-cli bundle")
        with zipfile.ZipFile(bundle / "source/autotem-source.zip") as archive:
            require(
                archive.read("tools/setup/verify_cli_profile.py") == program.read_bytes(),
                "CLI verifier differs from the issued source archive",
            )
        env = {
            key: value
            for key, value in os.environ.items()
            if key.upper() not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"}
        }
        env.update(PATH=str(Path(os.environ["SystemRoot"]) / "System32"), PIP_NO_INDEX="1")
        setup = [str(bundle / "AutoTEM-Setup.exe"), "--project", str(project), "--host", "codex"]
        rejected = execute(
            [*setup, "--profile", "traditional-core-mcp"], project, env, 90, success=False
        )
        require(
            rejected.get("ok") is False and not list(project.iterdir()),
            "Explicit mismatched profile was accepted or wrote project files",
        )
        preview = execute(setup, project, env, 90)
        require(
            preview["state"] == "planned"
            and preview.get("project") == str(project)
            and preview["profile"] == PROFILE
            and not list(project.iterdir()),
            "Preview did not select the only profile",
        )
        report["checks"].append("profile_selection_and_mismatch")
        receipt = execute(
            [*setup, "--profile", PROFILE, "--apply"], project, env, 1950, success=None
        )
        write(output / "installation.json", receipt)
        require(receipt.get("ok") is True, f"Native setup failed: {receipt.get('error')}")
        expected = {
            "state": "local_cli_ready",
            "profile": PROFILE,
            "runtime": "passed",
            "transport": "jsonl",
            "local_transport": "passed",
            "local_mcp": "not_in_profile",
            "host_connection": "not_evaluated",
            "manifest_sha256": report["release_manifest_sha256"],
        }
        require(
            receipt["ok"] and all(receipt.get(k) == v for k, v in expected.items()),
            "Native setup did not establish the CLI profile and its local transport",
        )
        require(
            "local_jsonl_handshake_passed" in receipt["checks"]
            and "local_mcp_handshake_passed" not in receipt["checks"],
            "Incorrect setup handshake",
        )
        runtime_id = "rt-" + report["release_manifest_sha256"][:20]
        exe = project / ".autotem/runtimes" / runtime_id / "Scripts/python.exe"
        require(Path(receipt["python_executable"]) == exe and exe.is_file(), "Wrong private Python")
        report["checks"].append("native_cold_cli_install")
        result = execute(
            [
                str(exe),
                "-I",
                str(program),
                "--installed",
                "--bundle",
                str(bundle),
                "--output",
                str(output),
            ],
            project,
            env,
            1150,
        )
        require(result["case_verdict"] == "passed", "Installed project science failed")
        science_hash = hashlib.sha256((output / "science.json").read_bytes()).hexdigest()
        report.update(
            case_verdict="passed",
            runtime_id=runtime_id,
            science=result,
            science_sha256=science_hash,
        )
        report["checks"].extend(["stable_cli_and_jsonl", "mcp_absent", "project_j01_disk_numerics"])
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        report["error"] = str(exc)
    write(output / "cli-profile.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New short output directory")
    parser.add_argument("--installed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    runner = installed if args.installed else run
    outcome = runner(args.bundle.resolve(), args.output.resolve())
    print(json.dumps(outcome, ensure_ascii=True))
    raise SystemExit(0 if outcome["case_verdict"] == "passed" else 2)
