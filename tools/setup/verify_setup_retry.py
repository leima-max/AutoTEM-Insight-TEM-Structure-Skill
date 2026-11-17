"""Record the native retry of one owned cold installation that stopped at its MCP probe."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def require(condition, label):
    if not condition:
        raise ValueError(label)


def retry(bundle, project):
    private = project / ".upgrade-verifier"
    original = private / "maintenance/install_previous-mcp-timeout.json"
    initial = private / "maintenance/install_previous.json"
    if not original.exists():
        original.write_bytes(initial.read_bytes())
    failed = read(original)
    manifest = digest(bundle / "release-manifest.json")
    rt_id = "rt-" + manifest[:20]
    root = project / ".autotem/runtimes" / rt_id
    marker = root / ".autotem-runtime.json"
    pointer = project / ".autotem/active-runtime.json"
    cfg = project / ".codex/config.toml"
    require(
        failed.get("ok") is False
        and failed["runtime_id"] == rt_id
        and failed["manifest_sha256"] == manifest
        and failed["error"]["code"] == "MCP_HANDSHAKE_FAILED"
        and failed["runtime_diagnostic"]["local_mcp"]["code"] == "MCP_PROBE_TIMEOUT"
        and {"pip_check_passed", "installed_profile_versions_and_origins_passed"}
        <= set(failed["checks"]),
        "unrecognized_failed_install",
    )
    state = read(marker)
    require(
        not pointer.exists()
        and not (project / "autotem-host/codex/adapter.json").exists()
        and state["managed_by"] == "autotem.bootstrap"
        and state["state"] == "failed"
        and state["manifest_sha256"] == manifest
        and Path(state["runtime"]) == root
        and Path(state["project"]) == project
        and state["error"]["code"] == "MCP_HANDSHAKE_FAILED",
        "unrelated_or_active_project",
    )
    out = private / "recovery"
    out.mkdir(exist_ok=True)
    known = sorted(out.glob("attempt-*.json"))
    if known and read(known[-1]).get("ok"):
        raise ValueError("successful_retry_already_recorded")
    index = len(known) + 1
    prefix = out / f"attempt-{index:03d}"
    before_marker = prefix.with_suffix(".marker")
    before_marker.write_bytes(marker.read_bytes())
    exe = root / "Scripts/python.exe"
    worker = root / "Lib/site-packages/autotem/harness/worker.py"
    original_config = digest(cfg)
    binary_identity = (
        digest(exe),
        exe.stat().st_mtime_ns,
        digest(worker),
        worker.stat().st_mtime_ns,
    )
    start = time.monotonic()
    proc = subprocess.run(
        [
            str(bundle / "AutoTEM-Setup.exe"),
            "--project",
            str(project),
            "--host",
            "codex",
            "--apply",
        ],
        cwd=project,
        capture_output=True,
        timeout=2100,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    elapsed = round(time.monotonic() - start, 3)
    native = prefix.with_suffix(".native")
    native.write_bytes(proc.stdout)
    row = read(native)
    intact = binary_identity == (
        digest(exe),
        exe.stat().st_mtime_ns,
        digest(worker),
        worker.stat().st_mtime_ns,
    )
    receipt = {
        "schema_version": "1.0.0",
        "manifest_sha256": manifest,
        "runtime_id": rt_id,
        "verifier_sha256": digest(Path(__file__)),
        "initial_failure_sha256": digest(original),
        "before_marker_sha256": digest(before_marker),
        "before_pointer": "absent",
        "user_config_before_sha256": original_config,
        "launcher_sha256": digest(bundle / "AutoTEM-Setup.exe"),
        "native_sha256": digest(native),
        "elapsed_seconds": elapsed,
        "probe_timeout_seconds": 60,
        "exit_code": proc.returncode,
        "ok": row.get("ok") is True,
        "state": row["state"],
        "error_code": (row.get("error") or {}).get("code"),
        "installed_files_reused": intact,
        "python_sha256": binary_identity[0],
        "worker_sha256": binary_identity[2],
    }
    path = prefix.with_suffix(".json")
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt), flush=True)
    if receipt["ok"]:
        require(
            proc.returncode == 0
            and row["state"] == "host_setup_required"
            and intact
            and row["manifest_sha256"] == manifest
            and row["local_transport"] == "passed"
            and read(pointer)["manifest_sha256"] == manifest
            and read(marker)["state"] == "ready",
            "native_retry_postcondition",
        )
    return receipt["ok"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if retry(args.bundle.resolve(), args.project.resolve()) else 2)
