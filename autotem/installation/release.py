"""Verify a complete offline release before bootstrap writes to a project."""

from __future__ import annotations

import email.parser
import ensurepip
import hashlib
import json
import platform
import re
import stat
import sys
import sysconfig
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import unquote, urlsplit

SCHEMA = "1.0.0"
PROFILE = "traditional-core-mcp"
PROFILE_EXTRAS = {
    PROFILE: ("mcp",),
    "traditional-cli": (),
    "traditional-simulation": ("mcp", "simulation"),
}
APP = "autotem-insight"
GATES = tuple(f"RG{i}" for i in range(6))


class ReleaseError(ValueError):
    """A release could not be used. The code is stable for installer diagnostics."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class VerifiedRelease:
    wheel: Path
    wheelhouse: Path
    lock: Path
    release_id: str
    manifest_sha256: str
    with_mcp: bool
    pip_version: str
    profile: str = PROFILE


def current_target() -> dict:
    return {
        "implementation": sys.implementation.name,
        "python_version": platform.python_version(),
        "abi": sysconfig.get_config_var("SOABI") or sys.implementation.cache_tag,
        "platform": sys.platform,
        "architecture": sysconfig.get_platform(),
    }


def wheelhouse_target() -> dict:
    return {
        "platform": sys.platform,
        "architecture": platform.machine(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_json(data: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    parsed = json.loads(data.decode("utf-8"), object_pairs_hook=unique)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def portable_path(name: str) -> str:
    """Only canonical relative POSIX names with an unambiguous Windows interpretation."""
    if not isinstance(name, str) or not name or PurePosixPath(name).as_posix() != name:
        raise ValueError("Invalid relative path")
    for part in name.split("/"):
        if (
            part in ("", ".", "..")
            or part.endswith((" ", "."))
            or PureWindowsPath(part).is_reserved()
            or re.search(r'[<>:"\\|?*\x00-\x1f]', part)
        ):
            raise ValueError("Unsafe bundle path")
    return name


def plain_path(path: Path) -> None:
    for part in (path, *path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Release paths must not traverse links or reparse points")


def artifact_paths(root: Path) -> dict[str, Path]:
    plain_path(root)
    result = {}
    seen = set()
    pending = [root]
    while pending:
        for path in pending.pop().iterdir():
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("Linked release artifact")
            name = portable_path(path.relative_to(root).as_posix())
            if name.casefold() in seen:
                raise ValueError("Duplicate bundle path")
            seen.add(name.casefold())
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
            elif stat.S_ISREG(info.st_mode):
                result[name] = path
            else:
                raise ValueError("Expected a regular release file")
    return result


def wheel_identity(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as wheel:
        records = [
            info
            for info in wheel.infolist()
            if re.fullmatch(r"[^/]+\.dist-info/METADATA", info.filename)
        ]
        if len(records) != 1 or records[0].file_size > 8 * 1024 * 1024:
            raise ValueError("Invalid wheel METADATA")
        metadata = email.parser.BytesParser().parsebytes(wheel.read(records[0]), headersonly=True)
    name = re.sub(r"[-_.]+", "-", str(metadata.get("Name", "")).lower())
    version = str(metadata.get("Version", ""))
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise ValueError("Invalid wheel project name")
    if not re.fullmatch(r"[0-9][0-9a-zA-Z.!+_]*", version):
        raise ValueError("Invalid pinned wheel version")
    parts = path.name.split("-")
    if (
        len(parts) not in (5, 6)
        or not path.name.endswith(".whl")
        or re.sub(r"[-_.]+", "-", parts[0].lower()) != name
        or parts[1] != version
    ):
        raise ValueError("Wheel filename and identity differ")
    # --no-index alone does not prevent pip from following a direct URL requirement.
    if any("@" in req.split(";", 1)[0] for req in metadata.get_all("Requires-Dist", [])):
        raise ValueError("Direct URL dependency is incompatible with offline installation")
    return name, version


def published_wheel_url(filename: str, digest: str, origin: dict) -> str:
    url = urlsplit(origin["url"])
    if (
        (url.scheme, url.netloc) != ("https", "files.pythonhosted.org")
        or url.query or url.fragment or unquote(url.path.split("/")[-1]) != filename
        or digest != origin["sha256"]
    ):
        raise ValueError(f"Wheel differs from its frozen upstream source: {filename}")
    return origin["url"]


def root_extra(profile: str) -> str:
    extras = PROFILE_EXTRAS[profile]
    return f"[{','.join(extras)}]" if extras else ""


def release_lock(wheels: dict[str, tuple[str, str]], profile: str = PROFILE) -> str:
    extras = PROFILE_EXTRAS[profile]
    if APP not in wheels or ("mcp" in wheels) != ("mcp" in extras):
        raise ValueError("The AutoTEM/MCP wheels do not match the selected profile")
    simulation = {"diffsims", "orix", "diffpy-structure"}
    if simulation.intersection(wheels) != (simulation if "simulation" in extras else set()):
        raise ValueError("The diffraction simulation wheels do not match the selected profile")
    if {"py4dstem", "pyxem"}.intersection(wheels):
        raise ValueError("4D-STEM adapters are outside the traditional release profiles")
    lines = []
    for name, (version, digest) in sorted(wheels.items()):
        extra = root_extra(profile) if name == APP else ""
        lines.append(f"{name}{extra}=={version} --hash=sha256:{digest}")
    return "\n".join(lines) + "\n"


def verify_snapshot(archive: zipfile.ZipFile, index_name: str) -> dict:
    members = archive.namelist()
    if len(members) != len({name.casefold() for name in members}):
        raise ValueError("Duplicate source archive member")
    index = unique_json(archive.read(index_name))
    if index["schema_version"] != SCHEMA or set(members) != {*index["files"], index_name}:
        raise ValueError("Source archive differs from its inventory")
    for name, digest in index["files"].items():
        portable_path(name)
        if hashlib.sha256(archive.read(name)).hexdigest() != digest:
            raise ValueError(f"Source archive hash mismatch: {name}")
    return index


def verify_source_wheel(wheel_path: Path, index: dict) -> None:
    expected = {name: h for name, h in index["files"].items() if name.startswith("autotem/")}
    with zipfile.ZipFile(wheel_path) as wheel:
        names = [name for name in wheel.namelist() if name.startswith("autotem/")]
        if not expected or len(names) != len(expected) or set(names) != set(expected):
            raise ValueError("AutoTEM wheel inventory differs from archived sources")
        for name in names:
            if hashlib.sha256(wheel.read(name)).hexdigest() != expected[name]:
                raise ValueError(f"Wheel/source byte mismatch: {name}")


def verify_installer_app(path: Path, index: dict) -> None:
    names = {
        "autotem/__init__.py", "autotem/bootstrap.py",
        "autotem/harness/__init__.py", "autotem/harness/hosts.py",
    }
    names.update(n for n in index["files"] if re.fullmatch(r"autotem/installation/[^/]+\.py", n))
    entry = b"from autotem.installation.managed import main\nraise SystemExit(main())\n"
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if len(members) != len(names) + 1 or set(members) != names | {"__main__.py"}:
            raise ValueError("Installer module inventory differs from the release source")
        if archive.read("__main__.py") != entry:
            raise ValueError("Unexpected installer entry point")
        for name in names:
            if hashlib.sha256(archive.read(name)).hexdigest() != index["files"][name]:
                raise ValueError(f"Installer/source byte mismatch: {name}")


def _verify(root: Path, profile: str | None) -> VerifiedRelease:
    files = artifact_paths(root)
    manifest_file = files.pop("release-manifest.json")
    manifest_bytes = manifest_file.read_bytes()
    manifest = unique_json(manifest_bytes)
    profiles = manifest["profiles"]
    if manifest["schema_version"] != SCHEMA or not isinstance(profiles, dict) or len(profiles) != 1:
        raise ReleaseError("RELEASE_UNSUPPORTED", "Unsupported release schema or profile.")
    issued = next(iter(profiles))
    if issued not in PROFILE_EXTRAS or (profile is not None and profile != issued):
        raise ReleaseError("RELEASE_UNSUPPORTED", "The requested profile is not in this release.")
    profile = issued
    if manifest["target"] != current_target() or manifest["pip_version"] != ensurepip.version():
        raise ReleaseError(
            "RELEASE_TARGET_MISMATCH", "Release requires its exact target interpreter and seed pip."
        )
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}", manifest["release_id"]):
        raise ValueError("Invalid release ID")
    selected = manifest["profiles"][profile]
    if selected["extras"] != list(PROFILE_EXTRAS[profile]):
        raise ValueError("Unsupported profile contents")
    records = {}
    for item in manifest["artifacts"]:
        name = portable_path(item["path"])
        if name in records or not isinstance(item["origin"], str) or not item["origin"]:
            raise ValueError("Invalid artifact descriptor")
        records[name] = item
    if set(files) != set(records):
        raise ReleaseError(
            "RELEASE_INVENTORY_MISMATCH", "Bundle has missing or unlisted files."
        )
    for name, path in files.items():
        item = records[name]
        if path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
            raise ReleaseError("RELEASE_HASH_MISMATCH", f"Artifact failed integrity check: {name}")
    source = manifest["source"]
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source["git_commit"]):
        raise ValueError("Missing source commit")
    if (
        source["archive_sha256"] != records[source["archive"]]["sha256"]
        or manifest["licenses"]["sha256"] != records[manifest["licenses"]["inventory"]]["sha256"]
        or selected["lock_sha256"] != records[selected["lock"]]["sha256"]
    ):
        raise ValueError("Inconsistent release index")
    acceptance = manifest["acceptance"]
    if (
        not manifest["publisher"]["source"]
        or not manifest["publisher"].get("signature_status")
        or type(acceptance["release_ready"]) is not bool
        or set(acceptance["gates"]) != set(GATES)
        or not set(acceptance["gates"].values()) <= {"not_evaluated", "passed", "failed", "blocked"}
        or (acceptance["release_ready"] and set(acceptance["gates"].values()) != {"passed"})
    ):
        raise ValueError("Incomplete publisher or acceptance status")
    archive_name = "source/autotem-source.zip"
    if (
        source["archive"] != archive_name
        or source["inventory_member"] != "autotem-source-index.json"
    ):
        raise ValueError("Invalid source archive layout")
    with zipfile.ZipFile(files[archive_name]) as archive:
        index = verify_snapshot(archive, source["inventory_member"])
        policy = unique_json(archive.read(f"configs/release/{profile}.json"))
        sources = unique_json(archive.read(portable_path(policy["wheel_sources"])))
    if sources["schema_version"] != SCHEMA:
        raise ValueError("Unsupported upstream sources schema")
    if index["git_commit"] != source["git_commit"] or len(index["files"]) != source["file_count"]:
        raise ValueError("Release source identity differs from the archived tree")
    target = manifest["target"]
    target_fields = ("python_version", "platform", "architecture", "implementation")
    if (
        policy["profile_id"] != profile
        or any(target[k] != policy[k] for k in target_fields)
        or manifest["pip_version"] != policy["seed_pip_version"]
        or selected["requested_scope"] != policy["requested_scope"]
    ):
        raise ValueError("Release target/scope differs from the archived release policy")
    runtime = manifest["runtime"]
    if runtime["status"] == "nuget_archive_included":
        spec = policy["python_nupkg"]
        if (
            runtime["archive"] != f"runtime/{spec['filename']}"
            or runtime["sha256"] != spec["sha256"]
            or runtime["source"] != spec["url"]
            or runtime["license"] not in files
            or runtime["sha256"] != records[runtime["archive"]]["sha256"]
        ):
            raise ValueError("Private CPython archive hash mismatch")
    elif runtime["status"] != "external_interpreter_required":
        raise ValueError("Unknown runtime provision mode")
    installation = manifest["installation"]
    if installation["launcher"] is not None or installation["app"] is not None:
        if (
            installation != {"launcher": "AutoTEM-Setup.exe", "app": "AutoTEM-install.pyz"}
            or not set(installation.values()).issubset(files)
            or runtime["status"] != "nuget_archive_included"
        ):
            raise ValueError("Incomplete user installation entry")
        verify_installer_app(files[installation["app"]], index)
    lock_name = f"locks/{profile}.txt"
    directory = f"wheelhouse/{profile}"
    if selected["lock"] != lock_name or selected["wheelhouse"] != directory:
        raise ValueError("Noncanonical profile layout")
    if any(
        (name.startswith("locks/") and name != lock_name)
        or (name.startswith("wheelhouse/") and path.parent != root / directory)
        or (name.endswith(".whl") and path.parent != root / directory)
        for name, path in files.items()
    ):
        raise ValueError("The bundle contains files for another profile")
    wheels = {}
    hashes = {}
    app_wheel = None
    for path in (root / directory).iterdir():
        if path.name == "wheelhouse-manifest.json":
            continue
        name, version = wheel_identity(path)
        if name in wheels:
            raise ValueError("Duplicate wheel project")
        digest = records[f"{directory}/{path.name}"]["sha256"]
        wheels[name] = (version, digest)
        hashes[path.name] = digest
        if name == APP:
            app_wheel = path
        else:
            origin = sources["wheels"][path.name]
            released_origin = records[f"{directory}/{path.name}"]["origin"]
            if published_wheel_url(path.name, digest, origin) != released_origin:
                raise ValueError("Released wheel differs from the archived upstream source record")
    if (root / lock_name).read_bytes() != release_lock(wheels, profile).encode("ascii"):
        raise ReleaseError("RELEASE_LOCK_MISMATCH", "Lock does not match pinned wheel metadata.")
    wheel_manifest = unique_json((root / directory / "wheelhouse-manifest.json").read_bytes())
    if wheel_manifest != {
        "schema_version": SCHEMA,
        "target": wheelhouse_target(),
        "wheels": hashes,
    }:
        raise ValueError("Wheelhouse index does not match the release")
    if app_wheel is None or wheels[APP][0] != manifest["package_version"]:
        raise ValueError("AutoTEM wheel does not match the release")
    verify_source_wheel(app_wheel, index)
    return VerifiedRelease(
        app_wheel, root / directory, root / lock_name, manifest["release_id"],
        hashlib.sha256(manifest_bytes).hexdigest(), "mcp" in PROFILE_EXTRAS[profile],
        manifest["pip_version"], profile,
    )


def verify_release(
    bundle_root: str | Path, profile: str | None = None,
) -> VerifiedRelease:
    """Verify file content/target/lock locally. Publisher authentication is independent."""
    try:
        return _verify(Path(bundle_root).expanduser().absolute(), profile)
    except ReleaseError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, zipfile.BadZipFile) as exc:
        raise ReleaseError(
            "RELEASE_MANIFEST_INVALID", f"Release is malformed or incomplete: {exc}"
        ) from exc
