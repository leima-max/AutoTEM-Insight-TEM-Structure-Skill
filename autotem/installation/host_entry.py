"""Stable project-local entry copied into .autotem by the managed installer."""

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path


def active_python(control: Path, command: str | None = None) -> Path:
    pointer = control / "active-runtime.json"
    if is_link(pointer):
        raise ValueError("Invalid activation pointer.")
    entry = json.loads(pointer.read_text(encoding="utf-8"))
    if command == "mcp" and entry.get("profile") == "traditional-cli":
        raise ValueError("The traditional-cli profile provides the stdio JSONL transport.")
    return runtime_python(control, entry)


def runtime_directory(control: Path, entry) -> Path:
    digest = entry.get("manifest_sha256", "")
    if (
        entry.get("schema_version") != "1.0.0"
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or entry.get("runtime_id") != f"rt-{digest[:20]}"
    ):
        raise ValueError("Invalid activation record.")
    return control / "runtimes" / entry["runtime_id"]


def runtime_python(control: Path, entry) -> Path:
    runtime = runtime_directory(control, entry)
    executable = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = runtime / ".autotem-runtime.json"
    for path in (control, runtime.parent, runtime, marker, executable.parent, executable):
        if is_link(path):
            raise ValueError("Unexpected link in active runtime.")
    state = json.loads(marker.read_text(encoding="utf-8"))
    if (
        state.get("managed_by") != "autotem.bootstrap"
        or state.get("state") != "ready"
        or state.get("manifest_sha256") != entry["manifest_sha256"]
        or ("profile" in entry and state.get("profile", "traditional-core-mcp") != entry["profile"])
        or Path(state.get("runtime", "")) != runtime
        or Path(state.get("project", "")) != control.parent
        or not executable.is_file()
    ):
        raise ValueError("The active runtime requires repair.")
    return executable


def is_link(path: Path) -> bool:
    data = path.lstat()
    return stat.S_ISLNK(data.st_mode) or bool(getattr(data, "st_file_attributes", 0) & 0x400)


def main():
    commands = ("capabilities", "inspect", "validate", "submit", "status", "cancel", "artifacts",
                "read", "export", "stdio", "mcp")
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Choose an AutoTEM command: " + ", ".join(commands), file=sys.stderr)
        return 2
    control = Path(__file__).absolute().parent
    try:
        executable = active_python(control, sys.argv[1])
        return subprocess.call(
            [
                str(executable),
                "-I",
                "-m",
                "autotem.harness.cli",
                "--root",
                str(control / "runs"),
                *sys.argv[1:],
            ]
        )
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        print(f"AutoTEM host setup required: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
