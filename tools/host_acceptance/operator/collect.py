"""Start an operator card or verify its originals using the installed edition."""

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path
from string import Formatter
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile, ZipFile

from autotem.installation.release import sha256 as sha

HERE = Path(__file__).resolve().parent
GPA = {"gpa_answered", "gpa_unanswered", "gpa_invalid_reply"}
MEASUREMENT_CARDS = {"J01", "J02", "J03", "J04", "J05", "J06", "J07", "J08", "J09"}


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def first_inputs(kit, card, case, variant):
    if case == "J08" and variant in GPA:
        return card["inputs"]
    for name in ("science", "operations"):
        for row in read(kit / "basis" / (name + ".zh-CN.json")):
            if (row["case_id"], row["variant"]) == (case, variant):
                names = {key for _, key, _, _ in Formatter().parse(row["turns"][0]["text"])}
                return [key for key in card["inputs"] if key in names]
    raise ValueError(f"Missing first prompt for {case}/{variant}")


def stage_inputs(project, kit, index, keys):
    """Publish only the selected fixtures, leaving existing participant files intact."""
    from autotem.installation.release import plain_path, portable_path

    pending = {}
    for key in keys:
        name = portable_path(index["inputs"][key]["path"])
        require(Path(name).parent == Path("inputs"), f"Unexpected field input: {name}")
        expected = index["files_sha256"][name]
        source = kit / name
        plain_path(source)
        contents = source.read_bytes()
        require(hashlib.sha256(contents).hexdigest() == expected, f"Operator input changed: {name}")
        target = project / kit.name / name
        for parent in (project / kit.name, target.parent):
            if parent.exists() or parent.is_symlink():
                plain_path(parent)
        if target.exists() or target.is_symlink():
            plain_path(target)
            require(
                target.is_file() and sha(target) == expected, f"Different project input: {target}"
            )
        else:
            pending[target] = contents
    for target, contents in pending.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        plain_path(target.parent)
        stream = NamedTemporaryFile(dir=target.parent, delete=False)
        staging = Path(stream.name)
        try:
            with stream:
                stream.write(contents)
            try:
                # Windows rename fails if the destination exists, including on FAT volumes.
                # POSIX link provides the same no-replace publication.
                if os.name == "nt":
                    os.rename(staging, target)
                else:
                    os.link(staging, target)
            except FileExistsError:
                plain_path(target)
                require(
                    target.is_file() and target.read_bytes() == contents,
                    f"Different project input: {target}",
                )
        finally:
            staging.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "verify"))
    for key in ("project", "release", "record"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--case", choices=[f"J{i:02}" for i in range(1, 16)])
    parser.add_argument("--host", choices=("codex", "openclaw", "workbuddy"))
    parser.add_argument("--variant")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    project = args.project.resolve(strict=True)
    release = args.release.resolve(strict=True)
    directory = args.record.resolve()
    control = project / ".autotem"
    active = read(control / "active-runtime.json")
    require(
        active["profile"] in ("traditional-core-mcp", "traditional-simulation"),
        "This card requires the core MCP or simulation edition.",
    )
    from autotem.installation.host_entry import active_python

    require(
        sys.flags.isolated and Path(sys.executable) == active_python(control, "mcp"),
        "Use the active project interpreter in isolated mode.",
    )
    kit = HERE.parent / "现场验收"
    index_path = kit / "index.json"
    binding = read(HERE / "binding.json")
    require(
        sha(index_path) == binding["kit_index_sha256"],
        "Field kit index differs from this operator package.",
    )
    index = read(index_path)
    require(
        index["release"]["profiles"].get(active["profile"])
        == active["manifest_sha256"]
        == sha(release / "release-manifest.json"),
        "The selected release, installed project and field kit must identify the same edition.",
    )
    for path, expected in index["files_sha256"].items():
        require(sha(kit / path) == expected, f"Field material changed: {path}")
    if args.action == "init":
        require(
            args.case and args.host and not args.summary,
            "init requires -HostName and -Case. Choose -Summary when verifying.",
        )
        case, variant, host = args.case, args.variant, args.host
    else:
        require(
            not (args.case or args.host or args.variant),
            "verify reads the host, case and variant from case.json.",
        )
        spec = read(directory / "case.json")
        case, variant, host = spec["case_id"], spec.get("variant"), spec["host"]
    require(
        host in active["host_installations"], f"No installation for {host} in the chosen project."
    )
    if case == "J08" and variant in GPA:
        card = {
            "path": "cards/J08-GPA.md",
            "profiles": ["traditional-core-mcp"],
            "inputs": ["j03"],
            "condition": "ready",
        }
    else:
        choices = [row for row in index["cards"] if row["case_id"] == case]
        card = next((row for row in choices if row["variant"] == variant), None)
        require(
            card is not None,
            f"Choose a variant for {case}: " + ", ".join(str(row["variant"]) for row in choices),
        )
    require(
        active["profile"] in card["profiles"],
        f"This card requires {' or '.join(card['profiles'])}.",
    )
    require(
        card["path"] in index["files_sha256"],
        "This edition has no operator card for the selected variant.",
    )
    print(
        f"{case} {variant or ''} | {host} | {active['profile']} | {card['condition']}\n"
        f"{kit / card['path']}",
        flush=True,
    )
    helper = HERE / "field_capture.py"
    manifest = read(release / "release-manifest.json")
    archive = release / manifest["source"]["archive"]
    require(
        sha(archive) == index["release"]["source_archive_sha256"],
        "Frozen release source archive differs.",
    )
    with ZipFile(archive) as z:
        require(
            helper.read_bytes() == z.read("tools/host_acceptance/field_capture.py"),
            "Capture helper differs from the selected release.",
        )
    command = ["--project", str(project), "--release", str(release), args.action]
    if args.action == "init":
        stage_inputs(project, kit, index, first_inputs(kit, card, case, variant))
        command.extend([str(directory), "--host", host, "--case", case])
        if variant:
            command.extend(["--variant", variant])
    else:
        command.extend(
            [
                str(directory / "case.json"),
                "--out",
                str(args.summary or (directory / "summary.json")),
            ]
        )
        # Rejection-only and fault cards can succeed without submitting a scientific run.
        if case in MEASUREMENT_CARDS:
            [key] = card["inputs"]
            command.extend(["--expected-input", str(kit / index["inputs"][key]["path"])])
    return runpy.run_path(str(helper))["main"](command)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, OSError, ValueError, BadZipFile) as exc:
        print(f"Field capture: {exc}", file=sys.stderr)
        sys.exit(2)
