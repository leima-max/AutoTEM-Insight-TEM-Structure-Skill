"""Prepare or verify a host case on the original project with its release's frozen verifier."""

import argparse
import hashlib
import io
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path
from zipfile import BadZipFile, ZipFile


def check_expected_input(summary, expected_sha256):
    if summary["case_verdict"] != "passed":
        return
    runs = summary["runs"]
    if not runs:
        summary["checks"]["expected_input"] = "not_evaluated"
        summary["case_verdict"] = "not_evaluated"
        summary["reason"] = "no_registered_run_to_verify_input"
    elif all(run["source_sha256"] == expected_sha256 for run in runs):
        summary["checks"]["expected_input"] = "passed"
    else:
        summary["checks"]["expected_input"] = "failed"
        summary["case_verdict"] = "failed"
        summary["reason"] = "registered_source_differs_from_expected_input"
    summary["evidence"]["expected_input_sha256"] = expected_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument(
        "--release", required=True, type=Path, help="Unpacked release used to install the project"
    )
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("init", help="Create a new, unexecuted case template")
    prepare.add_argument("directory", type=Path)
    prepare.add_argument("--host", required=True, choices=("codex", "openclaw", "workbuddy"))
    prepare.add_argument("--case", required=True)
    prepare.add_argument("--variant", help="Variant supported by the installed release verifier")
    check = actions.add_parser("verify", help="Verify case originals and write a new summary")
    check.add_argument("record", type=Path)
    check.add_argument("--out", required=True, type=Path)
    check.add_argument(
        "--expected-input", type=Path, help="Original file for a single-input field card"
    )
    args = parser.parse_args(argv)

    project = args.project.resolve(strict=True)
    bundle = args.release.resolve(strict=True)
    control = project / ".autotem"
    if not sys.flags.isolated:
        raise ValueError("Run this tool with the installed project interpreter and -I.")
    from autotem.installation.host_entry import active_python
    from autotem.installation.release import unique_json, verify_release

    if Path(sys.executable) != active_python(control, "mcp"):
        raise ValueError("The interpreter does not belong to the active project installation.")
    verified = verify_release(bundle)
    activation = unique_json((control / "active-runtime.json").read_bytes())
    if activation["manifest_sha256"] != verified.manifest_sha256:
        raise ValueError("The unpacked release differs from the project's active release.")

    expected = None
    if args.action == "init":
        if args.host not in (activation.get("host_installations") or {}):
            raise ValueError(f"This project has no installation for {args.host}.")
        command = ["init", str(args.directory.absolute()), "--host", args.host, "--case", args.case]
        if args.variant is not None:
            command.extend(["--variant", args.variant])
    else:
        output = args.out.absolute()
        expected = args.expected_input.resolve(strict=True) if args.expected_input else None
        command = [
            "verify", str(args.record.absolute()), "--project", str(project),
            "--harness-root", str(control / "runs"), "--release", str(bundle),
            "--out", str(output),
        ]

    # Use the same bytes for checking and reading the archive. The verifier imports
    # the product from this installed interpreter and reads matrix.json beside itself.
    manifest_bytes = (bundle / "release-manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != verified.manifest_sha256:
        raise ValueError("The release manifest changed during verification.")
    source = unique_json(manifest_bytes)["source"]
    frozen = (bundle / source["archive"]).read_bytes()
    if hashlib.sha256(frozen).hexdigest() != source["archive_sha256"]:
        raise ValueError("The frozen source archive changed during verification.")
    with (
        tempfile.TemporaryDirectory(dir=control / "tmp") as folder,
        ZipFile(io.BytesIO(frozen)) as archive,
    ):
        for name in ("verify.py", "matrix.json"):
            (Path(folder) / name).write_bytes(archive.read("tools/host_acceptance/" + name))
        # Native host requests resolve relative inputs from the installed project root.
        previous = Path.cwd()
        try:
            os.chdir(project)
            verifier = runpy.run_path(str(Path(folder) / "verify.py"))
            if expected is None:
                return verifier["main"](command)
            summary_file = Path(folder) / "summary.json"
            command[-1] = str(summary_file)
            verifier["main"](command)
            summary = unique_json(summary_file.read_bytes())
            check_expected_input(summary, verifier["digest"](expected))
            payload = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
            with output.open("x", encoding="utf-8") as target:
                print(payload, file=target)
            return 0 if summary["case_verdict"] == "passed" else 1
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, OSError, ValueError, BadZipFile) as exc:
        print(f"Field capture: {exc}", file=sys.stderr)
        sys.exit(2)
