"""Reconstitute a pinned historical Windows core release and qualify two code generations."""

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from autotem.installation.release import (  # noqa: E402
    PROFILE,
    portable_path,
    sha256,
    unique_json,
    verify_release,
    verify_snapshot,
)
from tools.release.qualify_ci import stage_verifiers  # noqa: E402

MANIFEST = "release-manifest.json"
BASELINE = ROOT / ".github/release-baselines/windows-core-rc8.zip"
BASELINE_MANIFEST = "28b37d7ec5da3347660590df1f60e6eb035005387837ca8eef0640016e49a8fe"
BASELINE_ARCHIVE = "b52d4826b566cbe1c3db97b80c2a6f81d761245841530f76850ff4812e71f923"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def external_files(manifest):
    profile = manifest["profiles"][PROFILE]
    wheels = {
        item["path"]
        for item in manifest["artifacts"]
        if item["path"].startswith(profile["wheelhouse"] + "/")
        and item["path"].endswith(".whl")
        and not Path(item["path"]).name.startswith("autotem_insight-")
    }
    return wheels | {manifest["runtime"]["archive"], manifest["runtime"]["license"]}


def capture(bundle, expected, output):
    verified = verify_release(bundle, profile=PROFILE)
    require(verified.manifest_sha256 == expected, "Historical manifest pin mismatch")
    data = unique_json((bundle / MANIFEST).read_bytes())
    names = {item["path"] for item in data["artifacts"]} - external_files(data)
    with ZipFile(output, "x") as archive:
        for name in sorted(names | {MANIFEST}):
            member = ZipInfo(name, (2000, 1, 1, 0, 0, 0))
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            member.compress_type = ZIP_DEFLATED
            archive.writestr(member, (bundle / name).read_bytes())
    return {"release_manifest_sha256": expected, "sparse_archive_sha256": sha256(output)}


def sparse_files(carrier, expected=BASELINE_MANIFEST, archive_sha=BASELINE_ARCHIVE):
    require(carrier.is_file(), "Pinned historical release fixture is missing")
    require(sha256(carrier) == archive_sha, "Historical fixture ZIP pin mismatch")
    with ZipFile(carrier) as archive:
        names = archive.namelist()
        require(names.count(MANIFEST) == 1, "Historical manifest missing or duplicated")
        require(archive.getinfo(MANIFEST).file_size <= 1024 * 1024, "Oversized historical manifest")
        raw = archive.read(MANIFEST)
        require(hashlib.sha256(raw).hexdigest() == expected, "Historical manifest pin mismatch")
        data = unique_json(raw)
        require(set(data["profiles"]) == {PROFILE}, "Historical fixture must be core/MCP")
        records = {portable_path(row["path"]): row for row in data["artifacts"]}
        included = records.keys() - external_files(data)
        require(
            set(names) == included | {MANIFEST} and len(set(names)) == len(names),
            "Historical fixture has missing, duplicate or unexpected members",
        )
        files = {MANIFEST: raw}
        for name in included:
            row = records[name]
            require(archive.getinfo(name).file_size == row["size"], f"Wrong fixture size: {name}")
            files[name] = archive.read(name)
            require(hashlib.sha256(files[name]).hexdigest() == row["sha256"], f"Wrong hash: {name}")
    return data, files


def inputs(carrier, lock):
    data, files = sparse_files(carrier)
    source = data["source"]
    with ZipFile(io.BytesIO(files[source["archive"]])) as archive:
        verify_snapshot(archive, source["inventory_member"])
        policy = unique_json(archive.read(f"configs/release/{PROFILE}.json"))
        locked = archive.read(portable_path(policy["build_inputs_lock"]))
    require(policy["python_nupkg"]["sha256"] == data["runtime"]["sha256"], "Historical PSF pin")
    with lock.open("xb") as stream:
        stream.write(locked)
    return {"build_inputs_sha256": sha256(lock), "python_nupkg": policy["python_nupkg"]}


def generation(bundle):
    verified = verify_release(bundle, profile=PROFILE)
    data = unique_json((bundle / MANIFEST).read_bytes())
    source = data["source"]
    with ZipFile(bundle / source["archive"]) as archive:
        index = verify_snapshot(archive, source["inventory_member"])
    code = {
        k: v for k, v in index["files"].items() if k.startswith("autotem/") and k.endswith(".py")
    }
    require(bool(code), "Missing application modules in release source")
    return {
        "release_id": verified.release_id,
        "manifest_sha256": verified.manifest_sha256,
        "source_sha256": source["archive_sha256"],
        "wheel_sha256": sha256(verified.wheel),
    }, code


def compare(previous, candidate, expected):
    old, old_code = generation(previous)
    new, new_code = generation(candidate)
    require(old["manifest_sha256"] == expected, "Selected historical manifest differs from its pin")
    changed = sorted(k for k in old_code.keys() | new_code if old_code.get(k) != new_code.get(k))
    require(
        changed
        and all(old[k] != new[k] for k in ("manifest_sha256", "source_sha256", "wheel_sha256")),
        "Upgrade requires different shipped AutoTEM application code in two verified releases",
    )
    return {
        "schema_version": "1.0.0",
        "case_verdict": "passed",
        "previous": old,
        "current": new,
        "changed_application_modules": changed,
    }


def restore(carrier, wheels, seed, output, candidate):
    data, files = sparse_files(carrier)
    runtime = data["runtime"]
    require(sha256(seed) == runtime["sha256"], "Historical PSF package hash mismatch")
    with ZipFile(seed) as archive:
        files[runtime["license"]] = archive.read("tools/LICENSE.txt")
    require(not output.exists(), "Previous release destination already exists")
    stage = output.with_name(".previous-" + uuid.uuid4().hex)
    stage.mkdir()
    try:
        for item in [*data["artifacts"], {"path": MANIFEST}]:
            name = item["path"]
            dst = stage / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if name in files:
                dst.write_bytes(files[name])
            else:
                source = seed if name == runtime["archive"] else wheels / dst.name
                require(source.is_file(), f"Missing historical input: {dst.name}")
                shutil.copyfile(source, dst)
        pair = compare(stage, candidate, BASELINE_MANIFEST)
        os.rename(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    stage = commands.add_parser("stage", help="Extract immutable current qualification drivers")
    capture_parser = commands.add_parser(
        "capture", help="Capture a maintainer-selected core release"
    )
    locks = commands.add_parser(
        "inputs", help="Write the original build-input lock of the baseline"
    )
    previous = commands.add_parser(
        "restore", help="Reproduce the original previous manifest and bytes"
    )
    pair = commands.add_parser("pair", help="Check two existing releases for a local replay")
    for command in (stage, capture_parser, previous, pair):
        command.add_argument("--bundle", type=Path, required=True)
    for command in (stage, capture_parser, locks, previous):
        command.add_argument("--out", type=Path, required=True)
    for command in (locks, previous):
        command.add_argument("--carrier", type=Path, default=BASELINE)
    for command in (pair, capture_parser):
        command.add_argument("--previous-manifest-sha256", required=True)
    previous.add_argument("--wheelhouse", type=Path, required=True)
    previous.add_argument("--python-nupkg", type=Path, required=True)
    pair.add_argument("--previous-bundle", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "stage":
        verify_release(args.bundle, profile=PROFILE)
        info = unique_json((args.bundle / MANIFEST).read_bytes())
        print(stage_verifiers(args.bundle, info["source"], args.out))
        return
    if args.action == "capture":
        result = capture(args.bundle, args.previous_manifest_sha256, args.out)
    elif args.action == "inputs":
        result = inputs(args.carrier, args.out)
    elif args.action == "restore":
        result = restore(args.carrier, args.wheelhouse, args.python_nupkg, args.out, args.bundle)
    else:
        result = compare(args.previous_bundle, args.bundle, args.previous_manifest_sha256)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
