"""Standard-library-only, project-local AutoTEM runtime provisioning.

This installer never changes global pip configuration, host permissions or gateway
services. A ready runtime is deliberately separate from host/LLM acceptance.
"""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import os
import platform
import re
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
import urllib.parse
import zipfile
from contextlib import closing
from pathlib import Path
from typing import Any

BOOTSTRAP_VERSION = "1.0.0"
MARKER_NAME = ".autotem-runtime.json"
HOSTS = ("codex", "workbuddy", "openclaw")


def host_install_next_action(host: str, *, mcp_available: bool, codex_configured: bool) -> str:
    """Describe the selected project's installed entry points after a successful apply."""
    adapter = f"autotem-host/{host}/adapter.json"
    if not mcp_available:
        action = (
            f"打开本项目的 {adapter}。按 cli_command 中的参数顺序运行，末尾加 capabilities "
            "可查看可用能力。也可运行 jsonl_command，逐行发送 UTF-8 JSON 请求。"
        )
        if host == "openclaw":
            action += "若要经 OpenClaw 网关调用，按本项目 autotem-host/openclaw/gateway-config.example.json 启用所需工具。"
        return action
    if host == "workbuddy":
        action = (
            "若尚未导入此项目的 Skill，请在 WorkBuddy 中导入 "
            "autotem-host/workbuddy/autotem-skill.zip。"
            f"按 WorkBuddy 支持的接入方式，使用 {adapter} 中的 "
            "mcp.mcpServers.autotem 配置连接。"
        )
    elif host == "openclaw":
        return (
            "在 OpenClaw 中将所选项目设为工作区，并确认网关运行正常。要经网关调用，请查看项目中的 "
            "autotem-host/openclaw/gateway-config.example.json。由操作者在该工作区对应的 "
            "OpenClaw 配置中启用插件及所需工具，授权后调用 autotem__list_capabilities。"
            f"使用会话级 MCP 的客户端可按 {adapter} 中的 mcp.mcpServers.autotem 配置连接。"
        )
    elif host == "codex":
        action = "在 Codex 中打开所选项目。确认项目可信并完成项目信任后，"
        if codex_configured:
            action += "重新加载项目配置 .codex/config.toml 中已有的 mcp_servers.autotem。"
        else:
            action += (
                f"用本项目 {adapter} 中的 mcp.mcpServers.autotem 配置项目连接并重新加载。"
            )
    else:
        raise ValueError("Unknown host.")
    return action + "待宿主授权使用 AutoTEM 工具后，调用 list_capabilities 验证连接。"


class BootstrapError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _no_links(path: Path) -> None:
    for candidate in (path, *path.parents):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or (
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise BootstrapError("UNSAFE_PATH", "Symlink/reparse installation paths are refused.")


def _project_path(project: str | Path) -> Path:
    raw = Path(project).expanduser().absolute()
    _no_links(raw)
    root = raw.resolve()
    home = Path.home().resolve()
    forbidden = {home, Path(root.anchor), Path(sys.base_prefix).resolve()}
    config_roots = {home / name for name in (".codex", ".openclaw", ".workbuddy")}
    forbidden.update(config_roots)
    for key in (
        "CODEX_HOME",
        "SystemRoot",
        "ProgramFiles",
        "ProgramData",
        "APPDATA",
        "LOCALAPPDATA",
    ):
        if os.environ.get(key):
            forbidden.add(Path(os.environ[key]).expanduser().resolve())
            if key in ("CODEX_HOME", "SystemRoot"):
                config_roots.add(Path(os.environ[key]).expanduser().resolve())
    if not root.is_dir() or any(root == item or item in root.parents for item in config_roots):
        raise BootstrapError(
            "UNSAFE_PROJECT", "Choose an existing project, outside global/configuration roots."
        )
    if root in forbidden:
        raise BootstrapError(
            "UNSAFE_PROJECT", "A home, filesystem or global runtime root is not a project."
        )
    return root


def _inside(root: Path, relative: str) -> Path:
    path = root / relative
    _no_links(path)
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise BootstrapError("UNSAFE_PATH", "Managed runtime path escapes this project.") from exc
    return path


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _no_links(path)
    descriptor, name = tempfile.mkstemp(prefix=".bootstrap-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=True, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _wheel_info(wheel: str | Path) -> tuple[Path, str, str]:
    path = Path(wheel).expanduser().absolute()
    if not path.is_file() or path.suffix.lower() != ".whl":
        raise BootstrapError("WHEEL_INVALID", "An existing AutoTEM wheel is required.")
    try:
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(names) != 1 or archive.getinfo(names[0]).file_size > 1024 * 1024:
                raise ValueError("metadata")
            metadata = email.parser.BytesParser().parsebytes(archive.read(names[0]))
            normalized = str(metadata.get("Name", "")).lower().replace("_", "-").replace(".", "-")
            if normalized != "autotem-insight" or not metadata.get("Version"):
                raise ValueError("distribution")
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
        raise BootstrapError(
            "WHEEL_INVALID", "Wheel metadata must identify autotem-insight."
        ) from exc
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return path.resolve(), digest.hexdigest(), str(metadata["Version"])


def _index_url(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        parsed = urllib.parse.urlsplit(value)
        valid = (
            parsed.scheme == "https"
            and parsed.hostname
            and not (parsed.username or parsed.password or parsed.query or parsed.fragment)
        )
        _port = parsed.port  # Validate malformed/non-numeric ports without echoing the URL.
    except ValueError:
        valid = False
    if not valid:
        raise BootstrapError(
            "INDEX_URL_INVALID", "Use an HTTPS index URL without credentials, query or fragment."
        )
    return value


def _child_environment(allow_network: bool) -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH"):
        environment.pop(name, None)
    # Preserve enterprise index, proxy and certificate settings. Destination
    # redirects are checked before installation instead of silently discarded.
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    environment["PIP_NO_INPUT"] = "1"
    environment["PIP_NO_INDEX"] = "0" if allow_network else "1"
    return environment


def _check_pip_configuration(text: str, *, allow_network: bool) -> None:
    for line in text.splitlines():
        match = re.match(r"^([^=]+)=(.*)$", line.strip())
        if not match:
            continue
        key, value = match.group(1).strip(), match.group(2).strip().strip("'\"")
        option = key.rsplit(".", 1)[-1].replace("_", "-")
        active = value.lower() not in ("", "0", "false", "no", "off", "none")
        if option in ("target", "prefix", "root", "user") and active:
            raise BootstrapError(
                "PIP_DESTINATION_UNSAFE",
                "Pip destination/user configuration redirects private installation; "
                "review that setting first.",
            )
        if not allow_network and option == "find-links" and re.search(r"(?i)https?://", value):
            raise BootstrapError(
                "OFFLINE_CONFIGURATION_NETWORK",
                "Offline mode refuses additional configured network find-links.",
            )


def _verify_wheelhouse(directory: Path) -> str:
    """Verify an optional publisher manifest; missing integrity stays explicit."""
    manifest = directory / "wheelhouse-manifest.json"
    if not manifest.is_file():
        return "not_verified"
    try:
        if manifest.stat().st_size > 1024 * 1024:
            raise ValueError("manifest size")

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate manifest key")
                result[key] = value
            return result

        data = json.loads(manifest.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
        if not isinstance(data, dict) or data.get("schema_version") != BOOTSTRAP_VERSION:
            raise ValueError("manifest version")
        expected = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": sys.platform,
            "architecture": platform.machine().lower(),
        }
        target = data.get("target", {})
        if any(
            str(target.get(key, "")).lower() != value.lower() for key, value in expected.items()
        ):
            raise BootstrapError(
                "WHEELHOUSE_TARGET_MISMATCH",
                "Wheelhouse target Python, platform or architecture "
                "does not match this interpreter.",
            )
        hashes = data.get("wheels")
        files = {item.name: item for item in directory.glob("*.whl") if item.is_file()}
        if not isinstance(hashes, dict) or not hashes or set(files) != set(hashes):
            raise ValueError("unlisted or missing wheels")
        for name, expected_hash in hashes.items():
            if Path(name).name != name or not re.fullmatch(r"[0-9a-f]{64}", str(expected_hash)):
                raise ValueError("invalid hash")
            digest = hashlib.sha256()
            with files[name].open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                raise BootstrapError(
                    "WHEELHOUSE_INTEGRITY_FAILED",
                    "Dependency wheel differs from the publisher's hash manifest.",
                )
        return "manifest_hashes_and_target_verified"
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        if isinstance(exc, BootstrapError):
            raise
        raise BootstrapError(
            "WHEELHOUSE_MANIFEST_INVALID",
            "Wheelhouse publisher manifest is malformed or incomplete.",
        ) from exc


def _run(command: list[str], *, environment: dict[str, str], cwd: Path, deadline: float,
         input_text: str | None = None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise BootstrapError("BOOTSTRAP_TIMEOUT", "Installation exceeded its total time budget.")
    try:
        return subprocess.run(
            command,
            cwd=str(cwd),
            env=environment,
            capture_output=True,
            input=input_text,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=remaining,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BootstrapError(
            "BOOTSTRAP_TIMEOUT", "Installation exceeded its total time budget."
        ) from exc
    except OSError as exc:
        raise BootstrapError(
            "PROCESS_UNAVAILABLE", "The required private Python process could not start."
        ) from exc


def _json_result(process: subprocess.CompletedProcess, code: str) -> dict[str, Any]:
    if process.returncode != 0:
        raise BootstrapError(code, "Private runtime verification or host preparation failed.")
    try:
        payload = json.loads(process.stdout)
    except (ValueError, TypeError) as exc:
        raise BootstrapError(code, "The runtime did not return a valid JSON result.") from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise BootstrapError(code, "The runtime reported an unsuccessful result.")
    return payload


def _verify_provenance(
    python: Path,
    runtime: Path,
    version: str,
    *,
    environment: dict[str, str],
    cwd: Path,
    deadline: float,
) -> None:
    script = (
        "import importlib.metadata, importlib.util, json, pathlib, sys; "
        "root=pathlib.Path(sys.argv[1]).resolve(); "
        "origin=pathlib.Path(importlib.util.find_spec('autotem').origin).resolve(); "
        "origin.relative_to(root); "
        "assert pathlib.Path(sys.prefix).resolve()==root; "
        "assert importlib.metadata.version('autotem-insight')==sys.argv[2]; "
        "print(json.dumps({'ok':True}))"
    )
    _json_result(
        _run(
            [str(python), "-I", "-c", script, str(runtime), version],
            environment=environment,
            cwd=cwd,
            deadline=deadline,
        ),
        "RUNTIME_PROVENANCE_FAILED",
    )


def _verify_release_files(runtime: Path, wheel: Path) -> None:
    """Check the installed first-party code and resources before invoking AutoTEM."""
    site = runtime / "Lib/site-packages"  # The managed release target is Windows x64.
    package = site / "autotem"
    _no_links(package)
    with zipfile.ZipFile(wheel) as archive:
        members = {
            p.filename
            for p in archive.infolist()
            if p.filename.startswith("autotem/") and not p.is_dir()
        }
        installed = set()
        for path in package.rglob("*"):
            _no_links(path)
            if path.is_file() and not (path.parent.name == "__pycache__" and path.suffix == ".pyc"):
                installed.add(path.relative_to(site).as_posix())
        if installed == members and all(
            (site / name).read_bytes() == archive.read(name) for name in members
        ):
            return
    raise BootstrapError(
        "INSTALLED_PACKAGE_CHANGED",
        "Installed AutoTEM files differ from the release. Roll back, then reinstall.",
    )


def bootstrap_runtime(
    project: str | Path,
    wheel: str | Path,
    *,
    host: str = "codex",
    wheelhouse: str | Path | None = None,
    allow_network: bool = False,
    index_url: str | None = None,
    with_mcp: bool = True,
    timeout: float = 600,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Plan or install an owned, fixed-path venv; return bounded JSON diagnostics.

    ``allow_network`` authorizes only this installer child's package downloads.
    Offline mode needs complete transitive wheels; pip validates their resolution.
    No captured pip output or inherited credential values enter the returned JSON.
    """
    return _provision(
        project,
        wheel,
        host=host,
        wheelhouse=wheelhouse,
        allow_network=allow_network,
        index_url=index_url,
        with_mcp=with_mcp,
        timeout=timeout,
        dry_run=dry_run,
    )


def _provision(
    project,
    wheel,
    *,
    host,
    wheelhouse,
    allow_network,
    index_url,
    with_mcp,
    timeout,
    dry_run,
    release=None,
) -> dict[str, Any]:
    """Use the legacy fixed directory or stage a verified release in its final directory."""
    result: dict[str, Any] = {
        "schema_version": BOOTSTRAP_VERSION,
        "ok": False,
        "dry_run": dry_run,
        "runtime_ready": False,
        "host": host,
        "host_install": None,
        "host_probe": "not_evaluated",
        "host_llm_e2e": "not_evaluated",
        "scientific_e2e": "not_evaluated",
        "error": None,
        "checks": [],
    }
    marker: Path | None = None
    state: dict[str, Any] | None = None
    lock: Path | None = None
    locked = False
    try:
        if sys.version_info < (3, 10):  # noqa: UP036 - runs before package installation.
            raise BootstrapError("PYTHON_UNSUPPORTED", "Bootstrap requires Python 3.10 or newer.")
        if host not in HOSTS or type(allow_network) is not bool or type(with_mcp) is not bool:
            raise BootstrapError("PARAMETER_INVALID", "Invalid host or boolean parameter.")
        if type(dry_run) is not bool:
            raise BootstrapError("PARAMETER_INVALID", "dry_run must be a boolean.")
        if type(timeout) not in (int, float) or not 0 < timeout <= 3600:
            raise BootstrapError("PARAMETER_INVALID", "Timeout must be between 0 and 3600 seconds.")
        if allow_network and wheelhouse is not None:
            raise BootstrapError(
                "PARAMETER_INVALID", "Select either offline wheelhouse or authorized network mode."
            )
        if index_url is not None and not allow_network:
            raise BootstrapError(
                "PARAMETER_INVALID", "An index URL requires explicit network authorization."
            )
        index_url = _index_url(index_url)
        root = _project_path(project)
        wheel_path, wheel_hash, version = _wheel_info(wheel)
        relative = ".autotem/runtime"
        if release:
            relative = f".autotem/runtimes/rt-{release.manifest_sha256[:20]}"
        runtime = _inside(root, relative)
        marker = _inside(root, f"{relative}/{MARKER_NAME}")
        executable = "Scripts/python.exe" if os.name == "nt" else "bin/python"
        python = _inside(root, f"{relative}/{executable}")
        run_root = _inside(root, ".autotem/runs")
        existing: dict[str, Any] | None = None
        if runtime.exists():
            try:
                existing = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise BootstrapError(
                    "UNOWNED_RUNTIME", "Existing runtime has no valid AutoTEM ownership marker."
                ) from exc
            if not isinstance(existing, dict) or any(
                existing.get(key) != value
                for key, value in {
                    "managed_by": "autotem.bootstrap",
                    "schema_version": BOOTSTRAP_VERSION,
                    "project": str(root),
                    "runtime": str(runtime),
                    **({"manifest_sha256": release.manifest_sha256} if release else {}),
                }.items()
            ):
                raise BootstrapError(
                    "UNOWNED_RUNTIME", "Existing environment belongs to a different owner/project."
                )
            if release and existing.get("profile", "traditional-core-mcp") != release.profile:
                raise BootstrapError("UNOWNED_RUNTIME", "Existing runtime has another profile.")
            if existing.get("state") == "ready" and (
                existing.get("wheel_sha256") != wheel_hash or existing.get("with_mcp") != with_mcp
            ):
                raise BootstrapError(
                    "UPGRADE_REQUIRES_REVIEW",
                    "A ready runtime has a different wheel/MCP mode. "
                    "Preserve it and review a separate rollback-capable upgrade first.",
                )
        current = bool(
            existing
            and existing.get("state") == "ready"
            and existing.get("wheel_sha256") == wheel_hash
            and existing.get("with_mcp") == with_mcp
            and python.is_file()
        )
        transport_retry = bool(
            release
            and existing
            and existing.get("state") == "failed"
            and python.is_file()
            and existing.get("wheel_sha256") == wheel_hash
            and (existing.get("error") or {}).get("code") == "MCP_HANDSHAKE_FAILED"
            and "pip_check_passed" in existing.get("checks", [])
        )
        wheelhouse_path = None
        if not allow_network and not current:
            if wheelhouse is None:
                raise BootstrapError(
                    "WHEELHOUSE_REQUIRED",
                    "Offline installation requires a complete dependency wheelhouse.",
                )
            wheelhouse_path = Path(wheelhouse).expanduser().resolve()
            if not wheelhouse_path.is_dir() or not any(wheelhouse_path.glob("*.whl")):
                raise BootstrapError(
                    "WHEELHOUSE_INCOMPLETE",
                    "No dependency wheels were found; package index access is disabled.",
                )
        elif wheelhouse is not None:
            wheelhouse_path = Path(wheelhouse).expanduser().resolve()
        integrity = (
            _verify_wheelhouse(wheelhouse_path) if wheelhouse_path is not None else "not_applicable"
        )
        result.update(
            {
                "project": str(root),
                "runtime": str(runtime),
                "python_executable": str(python),
                "wheel_sha256": wheel_hash,
                "package_version": version,
                "network_mode": "authorized_network" if allow_network else "offline",
                "reuse_owned_runtime": current,
                "with_mcp": with_mcp,
                "host_install_mode": "apply_project_files"
                if host == "codex"
                else "apply_project_assets_connector_proposal",
                "wheelhouse_integrity": integrity,
            }
        )
        if dry_run:
            result.update(
                ok=True,
                next_action="Review plan; apply provisioning to verify runtime and host files.",
            )
            return result
        deadline = time.monotonic() + timeout
        control = _inside(root, ".autotem")
        control.mkdir(exist_ok=True)
        if release is None:  # The managed installer holds a kernel-backed project lock.
            lock = _inside(root, ".autotem/runtime-bootstrap.lock")
            try:
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError as exc:
                raise BootstrapError(
                    "BOOTSTRAP_BUSY",
                    "Another bootstrap owns the lock; do not overwrite its environment.",
                ) from exc
            locked = True
            with os.fdopen(descriptor, "w", encoding="ascii") as stream:
                stream.write(str(os.getpid()))
        runtime.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": BOOTSTRAP_VERSION,
            "managed_by": "autotem.bootstrap",
            "project": str(root),
            "runtime": str(runtime),
            "wheel_sha256": wheel_hash,
            "package_version": version,
            "with_mcp": with_mcp,
            **({"manifest_sha256": release.manifest_sha256, "profile": release.profile}
               if release else {}),
            "state": "provisioning",
            "checks": [],
            "error": None,
        }
        if not current:
            _atomic_json(marker, state)
        environment = _child_environment(allow_network)
        scratch = _inside(root, ".autotem/tmp/installer")
        scratch.mkdir(parents=True, exist_ok=True)
        environment.update(TEMP=str(scratch), TMP=str(scratch), TMPDIR=str(scratch))
        environment["MPLCONFIGDIR"] = str(scratch / "matplotlib")
        if not current and not transport_retry:
            create = _run(
                [sys.executable, "-I", "-m", "venv", "--copies", str(runtime)],
                environment=environment,
                cwd=root,
                deadline=deadline,
            )
            if create.returncode != 0:
                raise BootstrapError(
                    "VENV_FAILED", "Private venv creation failed; no runtime was marked ready."
                )
            _no_links(python)
            if not python.is_file():
                raise BootstrapError(
                    "VENV_FAILED", "Private venv did not create its Python interpreter."
                )
            configured = _run(
                [str(python), "-I", "-m", "pip", "config", "list"],
                environment=environment,
                cwd=root,
                deadline=deadline,
            )
            if configured.returncode != 0:
                raise BootstrapError(
                    "PIP_CONFIGURATION_FAILED",
                    "Cannot verify private pip's effective configuration.",
                )
            _check_pip_configuration(configured.stdout, allow_network=allow_network)
            requirement = str(wheel_path) + ("[mcp]" if with_mcp else "")
            command = [
                str(python),
                "-I",
                "-m",
                "pip",
                "--require-virtualenv",
                "install",
                "--no-user",
                "--prefix",
                str(runtime),
                "--force-reinstall",
                "--no-cache-dir",
            ]
            if allow_network:
                if index_url:
                    command += ["--index-url", index_url]
            else:
                command += [
                    "--no-index",
                    "--only-binary=:all:",
                    "--find-links",
                    str(wheelhouse_path),
                ]
            if release:
                command.extend(["--no-compile", "--require-hashes", "-r", str(release.lock)])
            else:
                command.append(requirement)
            installed = _run(command, environment=environment, cwd=root, deadline=deadline)
            if installed.returncode != 0:
                code = (
                    "DEPENDENCIES_UNAVAILABLE" if not allow_network else "DEPENDENCY_INSTALL_FAILED"
                )
                raise BootstrapError(
                    code,
                    "Dependency installation failed; offline mode needs compatible wheels."
                    if not allow_network
                    else "Authorized install failed; check index access and compatible wheels.",
                )
            result["checks"].append("dependency_install_passed")
        elif transport_retry:
            result["checks"].append("retry_verified_installation_files")
        cfg = _inside(root, f"{relative}/pyvenv.cfg").read_text(encoding="utf-8")
        if not re.search(r"(?im)^include-system-site-packages\s*=\s*false\s*$", cfg):
            raise BootstrapError(
                "VENV_ISOLATION_FAILED", "Runtime must exclude global scientific packages."
            )
        if release:
            _verify_release_files(runtime, wheel_path)
            result["checks"].append("installed_autotem_wheel_bytes_passed")
        _verify_provenance(
            python, runtime, version, environment=environment, cwd=root, deadline=deadline
        )
        result["checks"].append("private_package_provenance_passed")
        checked = _run(
            [str(python), "-I", "-m", "pip", "check"],
            environment=environment,
            cwd=root,
            deadline=deadline,
        )
        if checked.returncode != 0:
            raise BootstrapError(
                "DEPENDENCY_CHECK_FAILED", "Private runtime dependencies are inconsistent."
            )
        result["checks"].append("pip_check_passed")
        if release:
            probe = """
import importlib.metadata as metadata, json, pathlib, re, sys
root = pathlib.Path(sys.prefix).resolve()
expected = {"pip": sys.argv[2]}
for line in pathlib.Path(sys.argv[1]).read_text(encoding="ascii").splitlines():
    package, version = line.split()[0].split("==")
    expected[package.split("[")[0]] = version
installed = {}
for dist in metadata.distributions():
    pathlib.Path(dist.locate_file("")).resolve().relative_to(root)
    name = re.sub(r"[-_.]+", "-", dist.metadata["Name"].lower())
    if name in installed:
        raise ValueError("Duplicate installed distribution")
    installed[name] = dist.version
print(json.dumps({"ok": installed == expected}))
"""
            _json_result(
                _run(
                    [str(python), "-I", "-c", probe, str(release.lock), release.pip_version],
                    environment=environment,
                    cwd=root,
                    deadline=deadline,
                ),
                "INSTALLED_LOCK_MISMATCH",
            )
            result["checks"].append("installed_profile_versions_and_origins_passed")
        probe_root = scratch / runtime.name / "stage-checks" if release else run_root
        command = [str(python), "-I", "-m", "autotem.harness.cli", "--root", str(probe_root)]
        capability = _json_result(
            _run(command + ["capabilities"], environment=environment, cwd=root, deadline=deadline),
            "CAPABILITY_CHECK_FAILED",
        )
        operations = capability.get("result", {}).get("operations")
        if not isinstance(operations, list) or not operations:
            raise BootstrapError(
                "CAPABILITY_CHECK_FAILED",
                "Runtime returned an empty or invalid operation registry.",
            )
        result["checks"].append("capabilities_passed")
        result["capability_count"] = len(operations)
        if release and not with_mcp:
            request = {"tool": "list_capabilities", "request_id": "install-jsonl"}
            response = _json_result(_run(
                command + ["stdio"], input_text=json.dumps(request) + "\n",
                environment=environment, cwd=root, deadline=deadline,
            ), "JSONL_HANDSHAKE_FAILED")
            if (response.get("request_id") != request["request_id"]
                    or response.get("tool") != request["tool"]
                    or response.get("result", {}).get("operations") != operations):
                raise BootstrapError("JSONL_HANDSHAKE_FAILED", "JSONL capability response differs.")
            result["checks"].append("local_jsonl_handshake_passed")
        if release:
            database = (probe_root / "harness.sqlite3").as_uri() + "?mode=ro"
            with closing(sqlite3.connect(database, uri=True)) as probe_db:
                result["store_schema_version"] = probe_db.execute("PRAGMA user_version").fetchone()[
                    0
                ]
        doctor_args = ["doctor", "--host", host, "--project", str(root), "--python", str(python)]
        if release:
            doctor_args += ["--timeout", "60"]
        doctor = _run(
            command + doctor_args,
            environment=environment,
            cwd=root,
            deadline=deadline,
        )
        try:
            readiness = json.loads(doctor.stdout)
        except (ValueError, TypeError) as exc:
            raise BootstrapError(
                "READINESS_CHECK_FAILED", "Installed doctor did not return valid readiness JSON."
            ) from exc
        if doctor.returncode not in (0, 2) or not isinstance(readiness, dict):
            raise BootstrapError("READINESS_CHECK_FAILED", "Installed doctor could not finish.")
        result["readiness"] = readiness
        runtime_status = readiness.get("runtime")
        if not isinstance(runtime_status, dict) or runtime_status.get("status") != "ready":
            raise BootstrapError("READINESS_CHECK_FAILED", "Installed runtime did not pass doctor.")
        local_mcp = runtime_status.get("local_mcp")
        if with_mcp and (not isinstance(local_mcp, dict) or local_mcp.get("status") != "ready"):
            raise BootstrapError(
                "MCP_HANDSHAKE_FAILED", "Local MCP handshake failed; runtime was not marked ready."
            )
        if with_mcp:
            result["checks"].append("local_mcp_handshake_passed")
        state.update(state="ready", checks=list(result["checks"]))
        _atomic_json(marker, state)
        result["runtime_ready"] = True
        if release:
            result.update(ok=True, next_action="Verified release runtime is staged for activation.")
            return result
        host_command = command + ["install-host", host, "--project", str(root), "--apply"]
        if host == "codex" and with_mcp:
            host_command.append("--with-mcp")
        result["host_install"] = _json_result(
            _run(host_command, environment=environment, cwd=root, deadline=deadline),
            "HOST_PREPARATION_FAILED",
        )
        result.update(
            ok=True,
            next_action=host_install_next_action(
                host, mcp_available=with_mcp, codex_configured=host == "codex" and with_mcp
            ),
        )
        return result
    except BootstrapError as exc:
        result["error"] = {"code": exc.code, "message": str(exc)}
    except (OSError, ValueError, TypeError, sqlite3.Error):
        result["error"] = {
            "code": "BOOTSTRAP_FAILED",
            "message": "Private runtime preparation failed; no global settings were changed.",
        }
    finally:
        corrupt = (result.get("error") or {}).get("code") in {
            "INSTALLED_PACKAGE_CHANGED",
            "INSTALLED_LOCK_MISMATCH",
            "VENV_ISOLATION_FAILED",
        }
        if (
            state is not None
            and marker is not None
            and not result["runtime_ready"]
            and (not current or corrupt)
        ):
            state.update(state="failed", error=result["error"], checks=list(result["checks"]))
            try:
                _atomic_json(marker, state)
            except (OSError, BootstrapError):
                pass
        if locked and lock is not None:
            try:
                lock.unlink()
            except OSError:
                pass
    result["next_action"] = (
        "Resolve the error and rerun the owned installer; host safety policy stays enforced."
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--host", choices=HOSTS, default="codex")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--wheelhouse")
    source.add_argument("--allow-network", action="store_true")
    parser.add_argument("--index-url", help="Explicit HTTPS index without embedded credentials.")
    parser.add_argument("--without-mcp", action="store_true")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--apply", action="store_true", help="Default is a read-only installation preview."
    )
    args = parser.parse_args(argv)
    result = bootstrap_runtime(
        args.project,
        args.wheel,
        host=args.host,
        wheelhouse=args.wheelhouse,
        allow_network=args.allow_network,
        index_url=args.index_url,
        with_mcp=not args.without_mcp,
        timeout=args.timeout,
        dry_run=not args.apply,
    )
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
