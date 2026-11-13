"""Build one headless Windows release profile from archived sources and pinned offline wheels."""

from __future__ import annotations

import argparse
import ensurepip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from autotem.installation.release import (  # noqa: E402
    APP,
    GATES,
    PROFILE,
    PROFILE_EXTRAS,
    SCHEMA,
    artifact_paths,
    current_target,
    plain_path,
    portable_path,
    published_wheel_url,
    release_lock,
    root_extra,
    sha256,
    unique_json,
    verify_release,
    verify_snapshot,
    verify_source_wheel,
    wheel_identity,
    wheelhouse_target,
)

SNAPSHOT_INDEX = "autotem-source-index.json"
SOURCE_PATHS = [
    ".gitattributes", "README.md", "pyproject.toml", "LICENSE*", "NOTICE*", "COPYING*", "autotem",
    "apps/atom_detection_ui.py", "apps/requirements.txt",
    "skills/autotem", "configs/harness", "configs/release", "tools/release", "tools/acceptance",
    "requirements/release", "tools/setup", "tools/host_acceptance",
    "tests", "configs/*.yaml", "configs/benchmarks", ".github/workflows/windows-release.yml",
    "docs/HARNESS_RELEASE_TECHNICAL_DEVELOPMENT_PLAN.md", "docs/harness-windows-release.md",
    "docs/harness-report-compatibility.md", "docs/harness-host-adapters.md",
    "docs/harness-release-qualification.md",
]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
    path.write_bytes((content + "\n").encode())


def _member(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    info = zipfile.ZipInfo(portable_path(name), date_time=(2000, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, content)


def check_resource_mirror(hashes: dict) -> None:
    prefix = "autotem/harness/resources/skill/"
    packaged = {n.removeprefix(prefix): h for n, h in hashes.items() if n.startswith(prefix)}
    mirror = {n.removeprefix("skills/autotem/"): h for n, h in hashes.items()
              if n.startswith("skills/autotem/")}
    if "SKILL.md" not in packaged or packaged != mirror:
        raise ValueError("Packaged Skill/Profile/recipe resources differ from their mirror")


def snapshot(root: Path, destination: Path) -> dict:
    """Capture all current build inputs, including uncommitted/untracked package sources."""
    head = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"], cwd=root)
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "-co", "--exclude-standard", "--", *SOURCE_PATHS], cwd=root,
    ).decode("utf-8").split("\0")
    hashes = {}
    seen = set()
    with zipfile.ZipFile(destination, "x") as archive:
        for name in sorted(set(names) - {""}):
            portable_path(name)
            if name.casefold() in seen:
                raise ValueError("Case-insensitive duplicate in source")
            seen.add(name.casefold())
            source = root / name
            if not source.exists():  # A deletion in the working tree belongs in the snapshot.
                continue
            plain_path(source)
            data = source.read_bytes()
            hashes[name] = hashlib.sha256(data).hexdigest()
            _member(archive, name, data)
        check_resource_mirror(hashes)
        index = {"schema_version": SCHEMA, "git_commit": head.decode().strip(), "files": hashes}
        _member(archive, SNAPSHOT_INDEX, json.dumps(index, sort_keys=True).encode("ascii"))
    return index


def unpack_snapshot(archive_path: Path, root: Path) -> dict:
    """Materialize and hash only indexed files under an empty staging directory."""
    root.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        index = verify_snapshot(archive, SNAPSHOT_INDEX)
        check_resource_mirror(index["files"])
        for name in index["files"]:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
    return index


def run_pip(arguments: list[str], cwd: Path, scratch: Path) -> None:
    # Builder input is already local. Host/organization registry settings play no part here.
    env = {
        k: v for k, v in os.environ.items()
        if not k.upper().startswith("PIP_") and k.upper() not in {"PYTHONHOME", "PYTHONPATH"}
    }
    env.update(
        PIP_CONFIG_FILE=os.devnull, PIP_NO_INDEX="1", PIP_NO_CACHE_DIR="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1", SOURCE_DATE_EPOCH="946684800",
        TMP=str(scratch), TEMP=str(scratch), TMPDIR=str(scratch),
    )
    subprocess.run([sys.executable, "-I", "-m", "pip", *arguments], cwd=cwd, env=env, check=True)


def verify_pool(pool: Path, lock: Path, published: dict | None = None) -> dict:
    expected = {}
    for raw in lock.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([\w.-]+)==([\w.+!]+)\s+--hash=sha256:([a-f0-9]{64})", line)
        if not match:
            raise ValueError("Invalid upstream wheel pin in archived build inputs")
        name, version, digest = match.groups()
        name = re.sub(r"[-_.]+", "-", name.lower())
        if name in expected:
            raise ValueError("Duplicate upstream project pin")
        expected[name] = (version, digest)
    actual = {}
    origins = {}
    for name, path in artifact_paths(pool).items():
        if "/" in name or name != path.name or not name.endswith(".whl"):
            raise ValueError("The input pool must contain only the pinned local wheels")
        project, version = wheel_identity(path)
        if project in actual:
            raise ValueError(f"Multiple upstream wheels provided for {project}")
        actual[project] = (version, sha256(path))
        if published is not None:
            origins[name] = published_wheel_url(name, actual[project][1], published["wheels"][name])
    if not expected or actual != expected:
        raise ValueError("Input wheel pool differs from the archived upstream hash pins")
    if published is not None and (
        published["schema_version"] != SCHEMA or set(origins) != set(published["wheels"])
    ):
        raise ValueError("Wheel origins index differs from the verified input pool")
    return origins


def collect_wheels(
    pool: Path, bundle: Path, source: Path, scratch: Path, index: dict, profile: str,
) -> Path:
    """Resolve build requirements and runtime requirements independently on the actual target."""
    built = scratch / "built"
    built.mkdir()
    directory = bundle / "wheelhouse" / profile
    directory.mkdir(parents=True)
    common = ["--no-index", "--only-binary=:all:", "--find-links", str(pool)]
    run_pip(["wheel", "--use-pep517", "--no-deps", *common, "-w", str(built), "."], source, scratch)
    own, = built.glob("*.whl")
    wheel_identity(own)
    verify_source_wheel(own, index)
    # The complete dependency graph is resolved, including the selected root extra.
    run_pip(["download", *common, "-d", str(directory), str(own) + root_extra(profile)],
            scratch, scratch)
    return directory


def license_record(path: Path, name: str, version: str) -> dict:
    with zipfile.ZipFile(path) as wheel:
        meta, = [p for p in wheel.namelist() if re.fullmatch(r"[^/]+\.dist-info/METADATA", p)]
        metadata = BytesParser().parsebytes(wheel.read(meta), headersonly=True)
        texts = []
        for info in wheel.infolist():
            parts = info.filename.split("/")
            if info.is_dir() or not (
                (parts[0].endswith(".dist-info") and "licenses" in parts[1:-1])
                or re.match(r"^(?:licen[cs]e|copying|notice)(?:\b|[-_.])", parts[-1], re.I)
            ):
                continue
            texts.append({
                "wheel_member": info.filename,
                "sha256": hashlib.sha256(wheel.read(info)).hexdigest(),
            })
        declared = metadata.get("License-Expression") or metadata.get("License")
        classifiers = [
            str(v) for v in metadata.get_all("Classifier", []) if str(v).startswith("License ::")
        ]
    return {
        "name": name, "version": version, "wheel": path.name,
        "declaration": str(declared) if declared else None,
        "license_classifiers": classifiers, "texts_in_distributed_wheel": texts,
        "review_status": "requires_review" if declared or texts or classifiers else "missing",
    }


def write_profile(
    bundle: Path, directory: Path, policy: dict, source_hash: str, upstream: dict,
) -> tuple[dict, str]:
    profile = policy["profile_id"]
    versions = {}
    hashes = {}
    licenses = []
    origins = {}
    for path in sorted(directory.glob("*.whl")):
        name, version = wheel_identity(path)
        if name in versions:
            raise ValueError(f"Multiple wheels selected for {name}")
        digest = sha256(path)
        versions[name] = (version, digest)
        hashes[path.name] = digest
        origins[path.name] = (
            f"source/archive sha256:{source_hash}" if name == APP else upstream[path.name]
        )
        licenses.append(license_record(path, name, version))
    write_json(directory / "wheelhouse-manifest.json", {
        "schema_version": SCHEMA, "target": wheelhouse_target(), "wheels": hashes,
    })
    lock = bundle / "locks" / f"{profile}.txt"
    lock.parent.mkdir()
    lock.write_bytes(release_lock(versions, profile).encode("ascii"))
    write_json(bundle / "licenses/inventory.json", {
        "schema_version": SCHEMA, "review_status": "not_evaluated", "distributions": licenses,
    })
    return {
        "extras": list(PROFILE_EXTRAS[profile]), "lock": lock.relative_to(bundle).as_posix(),
        "lock_sha256": sha256(lock), "wheelhouse": directory.relative_to(bundle).as_posix(),
        "requested_scope": policy["requested_scope"], "wheel_origins": origins,
    }, versions[APP][0]


def copy_runtime(bundle: Path, policy: dict, nupkg: Path | None) -> dict:
    spec = policy["python_nupkg"]
    if nupkg is None:
        return {"status": "external_interpreter_required", "expected_source": spec["url"]}
    target = bundle / "runtime" / spec["filename"]
    target.parent.mkdir()
    shutil.copyfile(nupkg, target)
    if target.stat().st_size != spec["size"] or sha256(target) != spec["sha256"]:
        raise ValueError("CPython archive does not match the frozen release policy")
    required = {
        "tools/python.exe", "tools/Lib/venv/__init__.py", "tools/Lib/ensurepip/__init__.py",
        spec["license_member"],
    }
    with zipfile.ZipFile(target) as archive:
        if not required.issubset(archive.namelist()):
            raise ValueError("Private Python distribution lacks required runtime components")
        (bundle / "licenses/CPython-LICENSE.txt").write_bytes(archive.read(spec["license_member"]))
    return {
        "status": "nuget_archive_included", "archive": target.relative_to(bundle).as_posix(),
        "source": spec["url"], "sha256": spec["sha256"], "license": "licenses/CPython-LICENSE.txt",
        "venv_and_ensurepip": "archive_members_checked", "signature_status": "not_evaluated",
    }


def assemble(args, bundle: Path, scratch: Path) -> None:
    (bundle / "source").mkdir()
    archived = bundle / "source/autotem-source.zip"
    if args.source_archive:
        shutil.copyfile(args.source_archive, archived)
        if sha256(archived) != args.source_sha256:
            raise ValueError("Input source archive differs from the supplied SHA256")
    else:
        snapshot(ROOT, archived)
    source = scratch / "source"
    index = unpack_snapshot(archived, source)
    profile_id = args.profile
    if profile_id not in PROFILE_EXTRAS:
        raise ValueError("Unsupported release profile")
    policy = unique_json((source / f"configs/release/{profile_id}.json").read_bytes())
    if policy["profile_id"] != profile_id:
        raise ValueError("Archived release policy identifies a different profile")
    target = current_target()
    if (
        any(target[k] != policy[k] for k in ("python_version", "platform", "architecture",
                                              "implementation"))
        or ensurepip.version() != policy["seed_pip_version"]
    ):
        raise ValueError("Run the builder under the exact interpreter in the selected policy")
    pool = args.wheelhouse.resolve(strict=True)
    published = unique_json((source / portable_path(policy["wheel_sources"])).read_bytes())
    upstream = verify_pool(pool, source / portable_path(policy["build_inputs_lock"]), published)
    directory = collect_wheels(pool, bundle, source, scratch, index, profile_id)
    source_hash = sha256(archived)
    profile, pkg_version = write_profile(bundle, directory, policy, source_hash, upstream)
    runtime = copy_runtime(bundle, policy, args.python_nupkg)
    installation = {"launcher": None, "app": None}
    if args.launcher:
        installation = {"launcher": "AutoTEM-Setup.exe", "app": "AutoTEM-install.pyz"}
        shutil.copyfile(args.launcher, bundle / installation["launcher"])
        shutil.copyfile(args.installer_app, bundle / installation["app"])
    origins = {
        "source/autotem-source.zip": "working-tree source snapshot" if not args.source_archive
        else "supplied hash-pinned source snapshot",
        "AutoTEM-Setup.exe": "supplied Windows launcher", "AutoTEM-install.pyz": "supplied zipapp",
        f"runtime/{policy['python_nupkg']['filename']}": policy["python_nupkg"]["url"],
        **{f"wheelhouse/{profile_id}/{n}": v for n, v in profile.pop("wheel_origins").items()},
    }
    manifest = {
        "schema_version": SCHEMA, "release_id": args.release_id, "package_version": pkg_version,
        "built_at": datetime.now(timezone.utc).isoformat(), "target": target,
        "pip_version": ensurepip.version(), "profiles": {profile_id: profile},
        "source": {
            "git_commit": index["git_commit"], "archive": "source/autotem-source.zip",
            "archive_sha256": source_hash, "inventory_member": SNAPSHOT_INDEX,
            "file_count": len(index["files"]),
        },
        "artifacts": [
            {"path": name, "size": path.stat().st_size, "sha256": sha256(path),
             "origin": origins.get(name, "derived from bundled materials")}
            for name, path in sorted(artifact_paths(bundle).items())
        ],
        "runtime": runtime, "installation": installation,
        "licenses": {
            "inventory": "licenses/inventory.json",
            "sha256": sha256(bundle / "licenses/inventory.json"),
            "redistribution_review": "not_evaluated",
        },
        "publisher": {"source": "AutoTEM working tree", "signature_status": "not_evaluated"},
        "acceptance": {"gates": dict.fromkeys(GATES, "not_evaluated"), "release_ready": False},
    }
    write_json(bundle / "release-manifest.json", manifest)
    verify_release(bundle)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILE_EXTRAS, default=PROFILE)
    parser.add_argument("--output", required=True, type=Path, help="New, absent release directory.")
    parser.add_argument("--wheelhouse", required=True, type=Path, help="Offline build/runtime pool")
    parser.add_argument("--scratch-parent", type=Path, default=ROOT / ".autotem/tmp",
                        help="Short project-owned directory for pip staging.")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--python-nupkg", type=Path)
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--installer-app", type=Path)
    parser.add_argument("--source-archive", type=Path, help="Rebuild a previous source snapshot.")
    parser.add_argument("--source-sha256")
    args = parser.parse_args(argv)
    if bool(args.source_archive) != bool(args.source_sha256):
        parser.error("--source-archive and --source-sha256 must be used together")
    if bool(args.launcher) != bool(args.installer_app) or (args.launcher and not args.python_nupkg):
        parser.error("The launcher requires both --installer-app and --python-nupkg")
    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    plain_path(output.parent)
    if output.exists():
        parser.error("The output already exists. Choose a new release directory.")
    short_parent = args.scratch_parent.absolute()
    short_parent.mkdir(parents=True, exist_ok=True)
    plain_path(short_parent)
    scratch = short_parent / ("r" + uuid.uuid4().hex[:12])
    bundle = output.parent / (".staging-" + uuid.uuid4().hex)
    owned = []
    try:
        for path in (scratch, bundle):
            path.mkdir()
            owned.append((path, path.stat()))
        assemble(args, bundle, scratch)
        if output.exists():
            raise FileExistsError(output)
        bundle.rename(output)
    finally:
        for path, identity in owned:
            if path.exists():
                plain_path(path)
                if os.path.samestat(identity, path.stat()):
                    try:
                        shutil.rmtree(path)
                    except OSError as exc:
                        print(f"Could not clean owned staging directory {path}: {exc}",
                              file=sys.stderr)
    print(output / "release-manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
