"""Exercise the shipped Windows setup and installed scientific driver in an empty project."""

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cold_preview(bundle, manifest, preview, *, fits=True):
    def expanded(path, prefix=""):
        with zipfile.ZipFile(path) as archive:
            return sum(
                member.file_size
                for member in archive.infolist()
                if member.filename.startswith(prefix) and not member.is_dir()
            )

    wheels = [bundle / row["path"] for row in manifest["artifacts"] if row["path"].endswith(".whl")]
    base = expanded(bundle / manifest["runtime"]["archive"], "tools/")
    expected = base + sum(expanded(wheel) for wheel in wheels)
    minimum = base + 3 * sum(wheel.stat().st_size for wheel in wheels)
    target = "rt-" + sha(bundle / "release-manifest.json")[:20]
    paths = set(preview["affected_project_paths"])
    if not (
        preview["estimated_new_payload_bytes"] == expected
        and type(preview["available_bytes"]) is int
        and preview["available_bytes"] >= 0
        and preview["minimum_free_bytes"] == minimum
        and (preview["available_bytes"] >= minimum) is fits
        and preview["active_runtime_id"] is None
        and preview["existing_project_control"] is False
        and preview["target_runtime_id"] == target
        and "授权" in preview["host_authorization"]
        and {
            ".codex/config.toml",
            ".agents/skills/autotem/",
            ".autotem/active-runtime.json",
            ".autotem/runtimes/" + target + "/",
            "autotem-host/codex/",
        }
        <= paths
    ):
        raise ValueError(
            "Native preview differs from the cold project, release payload or host writes"
        )
    return {
        name: preview[name]
        for name in (
            "available_bytes",
            "minimum_free_bytes",
            "estimated_new_payload_bytes",
            "target_runtime_id",
            "active_runtime_id",
            "existing_project_control",
            "affected_project_paths",
            "host_authorization",
        )
    }


def saved_inspection(root: Path, run_id: str, runtime_id: str):
    """Reopen the managed project read-only and verify its evidence on disk."""
    uri = (root / "harness.sqlite3").resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        run = db.execute(
            "SELECT status,input_path,source_hash,request,job_id FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if not run or run[0] != "completed" or sha(Path(run[1])) != run[2]:
            raise ValueError("The managed inspection or its raw snapshot changed.")
        events = db.execute(
            "SELECT payload FROM events WHERE run_id=? AND kind='runtime_bound'", (run_id,)
        ).fetchall()
        binding = json.loads(events[0][0]) if len(events) == 1 else {}
        if (
            binding.get("runtime_id") != runtime_id
            or binding.get("job_id") != run[4]
            or binding.get("plan_sha256") != hashlib.sha256(run[3].encode()).hexdigest()
        ):
            raise ValueError("The inspection job does not match its runtime and plan.")
        artifacts = db.execute(
            "SELECT artifact_id,path,sha256,role FROM artifacts WHERE run_id=? AND committed=1",
            (run_id,),
        ).fetchall()
    if not {"raw", "dataset"} <= {row[3] for row in artifacts} or any(
        sha(Path(row[1])) != row[2] for row in artifacts
    ):
        raise ValueError("The managed inspection artifacts are missing or changed.")
    return {
        "run_id": run_id,
        "job_id": run[4],
        "plan_sha256": binding["plan_sha256"],
        "snapshot_sha256": run[2],
        "artifacts": {row[0]: row[2] for row in artifacts},
    }


def run(bundle: Path, out: Path, extended: bool, keep_active: bool):
    out.mkdir(parents=True, exist_ok=False)
    project = out / "project"
    project.mkdir()
    # The launcher, its child, and all scientific workers must find their own interpreter.
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(name, None)
    env.update(PATH=os.path.join(os.environ["SystemRoot"], "System32"), PIP_NO_INDEX="1")
    report = {
        "schema_version": "1.0.0",
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "release_manifest_sha256": sha(bundle / "release-manifest.json"),
        "case_verdict": "failed",
        "host_journey": "not_evaluated",
        "scientific_accuracy": "not_evaluated",
        "release_ready": False,
        "checks": [],
    }

    def execute(cmd, limit=900, request=None):
        proc = subprocess.run(
            cmd,
            cwd=project,
            env=env,
            capture_output=True,
            timeout=limit,
            input=(json.dumps(request, ensure_ascii=True) + "\n").encode("ascii")
            if request is not None
            else None,
        )
        try:
            row = json.loads(proc.stdout.decode("utf-8"))
        except ValueError as exc:
            (out / "invalid-response.stdout").write_bytes(proc.stdout)
            raise ValueError(
                f"Invalid UTF-8 JSON from {Path(cmd[0]).name}, exit {proc.returncode}"
            ) from exc
        if Path(cmd[0]).name == "AutoTEM-Setup.exe":
            fields = (
                "action",
                "state",
                "activation",
                "runtime_id",
                "installation_id",
                "error",
                "next_action",
                "retained",
                "active_pids",
                "removed_file_count",
            )
            event = {k: row[k] for k in fields if k in row}
            event["applied"] = "--apply" in cmd
            if (row.get("error") or {}).get(
                "code"
            ) == "DISK_SPACE_LOW" and "minimum_free_bytes" in row:
                event["space_preview"] = cold_preview(bundle, manifest, row, fits=False)
            with (out / "maintenance.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(event, ensure_ascii=True) + "\n")
        if proc.returncode:
            raise ValueError(f"{Path(cmd[0]).name}: {row.get('error') or row.get('case_verdict')}")
        return row

    try:
        manifest = json.loads((bundle / "release-manifest.json").read_bytes())
        if set(manifest["profiles"]) != {"traditional-core-mcp"}:
            raise ValueError("This installation qualification requires traditional-core-mcp.")
        command = [str(bundle / "AutoTEM-Setup.exe"), "--project", str(project), "--host", "codex"]
        preview = execute(command)
        if (
            preview["state"] != "planned"
            or preview.get("project") != str(project)
            or list(project.iterdir())
        ):
            raise ValueError("Preview must preserve the exact project path and leave it empty.")
        report["checks"].append("read_only_preview")
        report["project_preview"] = cold_preview(bundle, manifest, preview)
        report["checks"].append("project_space_and_host_write_preview")
        setup = execute([*command, "--apply"], 1950)
        if (
            setup["state"] != "host_setup_required"
            or setup["runtime"] != "passed"
            or setup["local_transport"] != "passed"
            or setup["host_assets"] != "passed"
        ):
            raise ValueError("The private runtime or local transport failed verification.")
        exe = Path(setup["python_executable"])
        rt = project / ".autotem/runtimes" / setup["runtime_id"]
        if exe.parent.parent != rt or not exe.is_file():
            raise ValueError("The launcher returned an unexpected runtime.")
        report.update(
            runtime_id=setup["runtime_id"],
            runtime_checks=setup["checks"],
            host_connection=setup["host_connection"],
        )
        report["checks"].append("cold_installed_private_cpython")
        adapter = json.loads((project / "autotem-host/codex/adapter.json").read_bytes())
        stable = [*adapter["jsonl_command"][:-1], "capabilities"]
        advertised = execute(stable, 120)
        if not advertised["ok"] or not advertised["result"]["operations"]:
            raise ValueError("The stable host entry did not execute installed tools.")
        report["checks"].append("activated_host_entry")
        args = ["--extended"] if extended else []
        result = execute(
            [
                str(exe),
                "-I",
                "-m",
                "autotem.acceptance",
                "--require-installed",
                "--release-bundle",
                str(bundle),
                "--output",
                str(out / "science"),
                *args,
            ],
            1200,
        )
        if result["case_verdict"] != "passed":
            raise ValueError("See science/evidence.json for the failed scientific contract.")
        report["science_evidence_sha256"] = sha(out / "science/evidence.json")
        report["checks"].append("installed_disk_journeys")
        if extended:
            variants = Path(__file__).resolve().parents[1] / "acceptance/verify_variants.py"
            outcome = execute(
                [
                    str(exe),
                    "-I",
                    str(variants),
                    "--require-installed",
                    "--release-bundle",
                    str(bundle),
                    "--output",
                    str(out / "variants"),
                ],
                1200,
            )
            if outcome["case_verdict"] != "passed":
                raise ValueError("See variants/operations.json for operation coverage.")
            report["variant_evidence_sha256"] = sha(out / "variants/operations.json")
            report["checks"].append("installed_operation_variants")
        evidence = json.loads((out / "science/evidence.json").read_bytes())
        fixture = next(item for item in evidence["fixtures"] if item["case_id"] == "J01")
        source = out / "science" / fixture["path"]
        inspected = execute(
            adapter["jsonl_command"],
            180,
            {
                "tool": "inspect_input",
                "request_id": "project-inspection",
                "arguments": {"path": str(source), "selection": {"modality": "HRTEM"}},
            },
        )
        if not inspected["ok"] or inspected["result"]["status"] != "completed":
            raise ValueError("The active project could not run its installed input reader.")
        root = project / ".autotem/runs"
        run_id = inspected["result"]["run_id"]
        inspection = saved_inspection(root, run_id, setup["runtime_id"])
        if inspection["snapshot_sha256"] != fixture["sha256"]:
            raise ValueError("The managed input snapshot differs from the disk fixture.")
        report["managed_project_inspection"] = inspection
        report["checks"].append("project_bound_job_and_artifacts")
        pointer = project / ".autotem/active-runtime.json"
        previous = sha(pointer)
        reused = execute([*command, "--apply"])
        if reused["activation"] != "already_active" or sha(pointer) != previous:
            raise ValueError("Repeated setup changed the active runtime.")
        report["checks"].append("idempotent_setup")
        if not keep_active:
            restored = execute([*command, "--apply", "--rollback"])
            if restored["state"] != "rolled_back" or pointer.exists():
                raise ValueError("Rollback did not restore the original project configuration.")
            if saved_inspection(root, run_id, setup["runtime_id"]) != inspection:
                raise ValueError("Rollback changed the project evidence.")
            report["checks"].append("rollback_preserved_store")
            preview_remove = execute([*command, "--uninstall"])
            if preview_remove["state"] != "planned" or not exe.exists():
                raise ValueError("Uninstall preview unexpectedly changed the runtime.")
            removed = execute([*command, "--uninstall", "--apply"])
            if (
                removed["state"] != "uninstalled"
                or rt.exists()
                or saved_inspection(root, run_id, setup["runtime_id"]) != inspection
            ):
                raise ValueError("Uninstall did not preserve the verified project evidence.")
            report["checks"].append("uninstall_preserved_store")
        report["case_verdict"] = "passed"
    except (OSError, ValueError, KeyError, sqlite3.Error, subprocess.TimeoutExpired) as exc:
        report["error"] = str(exc)
    finally:
        (out / "installation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument(
        "--keep-active", action="store_true", help="Keep setup for live host checks"
    )
    options = parser.parse_args()
    result = run(
        options.bundle.resolve(), options.output.resolve(), options.extended, options.keep_active
    )
    print(json.dumps(result, ensure_ascii=True))
    sys.exit(0 if result["case_verdict"] == "passed" else 2)
