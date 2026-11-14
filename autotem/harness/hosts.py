"""Project-local host resources with dry-run, conflict checks and guarded rollback.

Profiles describe responsibilities. Installing a Skill does not prove native subagent
configuration or a successful host/LLM journey. This adapter ships sequential fallback.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from autotem.bootstrap import host_install_next_action
from autotem.installation import host_conflict

HOSTS = ("codex", "openclaw", "workbuddy")
INSTALL_VERSION = "1.0.0"


class _TomlReview(ValueError):
    def __init__(self, message, section):
        super().__init__(message)
        self.section = section


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _project_root(project: str | Path) -> Path:
    root = Path(project).expanduser().resolve()
    forbidden = {
        Path.home().resolve(),
        Path(root.anchor),
        *(Path.home().joinpath(name).resolve() for name in (".codex", ".openclaw", ".workbuddy")),
    }
    if root in forbidden or not root.is_dir():
        raise ValueError(
            "Choose an existing project directory, not a user/global configuration root."
        )
    return root


def _contained(root: Path, relative: str) -> Path:
    path = root / relative
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Host target escapes the project root (including symlink/reparse target)."
        ) from exc
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Host target must be relative to the project root.")
    from autotem.bootstrap import _no_links

    _no_links(path)
    return path


def _resource_files() -> dict[str, bytes]:
    directory = importlib.resources.files("autotem.harness").joinpath("resources", "skill")
    result: dict[str, bytes] = {}

    def visit(node, prefix: str = ""):
        for child in sorted(node.iterdir(), key=lambda item: item.name):
            name = f"{prefix}{child.name}"
            if child.is_dir():
                visit(child, name + "/")
            else:
                result[name] = child.read_bytes()

    visit(directory)
    if "SKILL.md" not in result:
        raise RuntimeError("Installed wheel is missing the bundled AutoTEM Skill resources.")
    return result


def _read_toml(text: str) -> dict:
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError as exc:
            raise RuntimeError(
                "Merging existing TOML on Python 3.10 needs autotem-insight[mcp]."
            ) from exc
    return tomllib.loads(text)


def _mcp_entry(root: Path, harness_root: str | Path, python_executable: str) -> dict:
    interpreter = Path(python_executable).expanduser()
    if not interpreter.is_absolute() or not interpreter.is_file():
        raise ValueError("Use an existing absolute Python interpreter path for host tools.")
    run_root = Path(harness_root)
    if not run_root.is_absolute():
        run_root = root / run_root
    run_root = run_root.resolve()
    try:
        run_root.relative_to(root)
    except ValueError as exc:
        raise ValueError("Installer Harness root must remain inside this project.") from exc
    return {
        # Keep a venv's symlink path: resolving it to the base executable can
        # discard the environment selected by the adjacent pyvenv.cfg.
        "command": str(interpreter),
        "args": ["-I", "-m", "autotem.harness.cli", "--root", str(run_root), "mcp"],
        "cwd": str(root),
    }


def _codex_merge(existing: bytes | None, entry: dict) -> bytes:
    text = existing.decode("utf-8-sig") if existing is not None else ""
    if text.strip():
        document = _read_toml(text)
        servers = document.get("mcp_servers", {})
        if not isinstance(servers, dict):
            raise ValueError("Existing mcp_servers is not a table; refusing unsafe merge.")
        if "autotem" in servers:
            if servers["autotem"] == entry:
                return existing
            raise ValueError(
                "Project already has a different autotem MCP entry; no overwrite performed."
            )
        # Inline tables cannot be extended safely by appending a dotted table.
        if re.search(r"(?m)^\s*mcp_servers\s*=", text):
            raise ValueError("Inline mcp_servers config requires a reviewed manual merge.")
    suffix = (
        "\n\n[mcp_servers.autotem]\n"
        + "\n".join(
            f"{key} = {json.dumps(value, ensure_ascii=False)}" for key, value in entry.items()
        )
        + "\n"
    )
    merged = text.rstrip() + suffix
    _read_toml(merged)
    return merged.encode("utf-8")


def _workbuddy_archive(files: dict[str, bytes]) -> bytes:
    # The Tencent Skill importer expects SKILL.md at the ZIP root.
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            if name == "SKILL.md":
                text = content.decode("utf-8")
                boundary = text.find("\n---", 4)
                metadata = (
                    "\ndisplay_name: AutoTEM\ndisplay_name_en: AutoTEM\n"
                    "description_zh: 使用本地传统算法分析电镜图像并生成可追溯证据。\n"
                    "description_en: Local traditional TEM tools with traceable evidence.\n"
                    "version: 1.0.0\nauthor: AutoTEM-Insight Contributors"
                )
                if boundary < 0:
                    raise ValueError("Bundled Skill frontmatter is malformed.")
                content = (text[:boundary] + metadata + text[boundary:]).encode("utf-8")
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 5, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, content)
    return stream.getvalue()


def _planned_files(
    root: Path, host: str, *, with_mcp: bool, harness_root: str | Path, python_executable: str,
    managed: bool = False, mcp_available: bool = True,
) -> dict[str, bytes]:
    skill = _resource_files()
    files: dict[str, bytes]
    if host == "codex":
        files = {f".agents/skills/autotem/{name}": data for name, data in skill.items()}
        files.update(
            {
                "autotem-host/codex/native-agent-templates/" + name.rsplit("/", 1)[-1]: data
                for name, data in skill.items()
                if name.startswith("references/hosts/codex/") and name.endswith(".toml")
            }
        )
    elif host == "openclaw":
        files = {f"skills/autotem/{name}": data for name, data in skill.items()}
    else:
        files = {"autotem-host/workbuddy/autotem-skill.zip": _workbuddy_archive(skill)}
        files.update({f"autotem-host/workbuddy/skill/{name}": data for name, data in skill.items()})
    entry = _mcp_entry(root, harness_root, python_executable)
    if managed:
        from autotem.installation import host_entry

        if Path(entry["args"][-2]) != root / ".autotem" / "runs":
            raise ValueError("Managed hosts require the project's release run directory.")
        files[".autotem/host-entry.py"] = Path(host_entry.__file__).read_bytes()
        entry = {
            "command": sys._base_executable,
            "args": ["-I", str(root / ".autotem" / "host-entry.py"), "mcp"],
            "cwd": str(root),
        }
        if host == "codex":
            entry["startup_timeout_sec"] = 60
    command = [entry["command"], *entry["args"][:-1]]
    host_note = {
        "schema_version": INSTALL_VERSION,
        "host": host,
        "execution_mode": "sequential",
        "native_subagents_verified": False,
        "native_template_status": "provided_not_activated" if host == "codex" else "not_applicable",
        "host_llm_e2e_verified": False,
        "preferred_transport": ("mcp" if with_mcp else "host_supported_bridge")
        if mcp_available else "jsonl",
        "runtime_readiness": "not_evaluated",
        "host_tool_permission": "not_evaluated",
        "python_executable": python_executable,
        "readiness_command": [
            python_executable,
            "-I",
            "-m",
            "autotem.harness.cli",
            "doctor",
            "--host",
            host,
            "--project",
            str(root),
        ],
        "cli_command": command,
        "jsonl_command": [*command, "stdio"],
        "instruction": (
            "WorkBuddy import/MCP GUI configuration requires explicit user action; GUI unverified."
            if host == "workbuddy"
            else "Load this project's Skill; use sequential Harness tools."
        ),
    }
    if mcp_available:
        host_note["mcp"] = {"mcpServers": {"autotem": {
            "command": entry["command"], "args": entry["args"],
        }}}
    else:
        host_note["instruction"] = (
            "Use cli_command with a Harness subcommand or send JSONL using jsonl_command."
        )
    if host == "openclaw":
        from .openclaw import PLUGIN_DIR, gateway_files

        files.update(gateway_files(root, host_note["jsonl_command"], python_executable,
                                   (root / harness_root).resolve(), managed=managed))
        host_note["gateway_plugin"] = {
            "directory": PLUGIN_DIR,
            "config_example": "autotem-host/openclaw/gateway-config.example.json",
            "activation": "operator_opt_in",
            "transport": "harness_jsonl",
        }
    files[f"autotem-host/{host}/adapter.json"] = (
        json.dumps(host_note, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if with_mcp:
        if host != "codex":
            raise ValueError(
                "Automatic MCP merge is implemented only for project-local Codex TOML."
            )
        target = _contained(root, ".codex/config.toml")
        if target.is_dir():
            raise _TomlReview("Project TOML path is a directory.", _codex_merge(None, entry))
        try:
            files[".codex/config.toml"] = _codex_merge(
                target.read_bytes() if target.exists() else None, entry
            )
        except (ValueError, IsADirectoryError) as exc:
            reason = (
                "Project already has a different autotem MCP entry."
                if "different autotem MCP entry" in str(exc)
                else "Project TOML needs manual review."
            )
            raise _TomlReview(reason, _codex_merge(None, entry)) from exc
    return files


def _codex_config_before_managed_mcp(root: Path, previous_installation: str) -> bytes | None:
    """Recover the original TOML from the active receipt chain when switching to CLI."""
    from autotem.installation import uninstall

    relative = ".codex/config.toml"
    path = _contained(root, relative)
    if not path.exists():
        return None
    content = path.read_bytes()
    if "autotem" not in _read_toml(content.decode("utf-8-sig")).get("mcp_servers", {}):
        return None
    control = root / ".autotem"
    active = json.loads(_contained(control, "active-runtime.json").read_bytes())
    if active["host_installations"].get("codex") != previous_installation:
        raise ValueError("Previous Codex receipt differs from the active installation.")
    for key in uninstall._chain(active, uninstall._records(control)):
        directory, receipt = uninstall._host_receipt(control, key)
        for row in receipt["files"]:
            if row["path"] != relative or row["action"] == "unchanged":
                continue
            if _hash(content) != row["after_sha256"]:
                raise ValueError("Review the changed .codex/config.toml before switching profile.")
            if row["before_sha256"] is None:
                return b""
            content = _contained(directory, row["backup"]).read_bytes()
            if _hash(content) != row["before_sha256"]:
                raise ValueError("The .codex/config.toml backup differs from its receipt.")
            if "autotem" not in _read_toml(content.decode("utf-8-sig")).get("mcp_servers", {}):
                return content
    return None  # A pre-existing, user-owned connector stays under the user's control.


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A basename independent of the destination avoids MAX_PATH inflation on Windows.
    # mkstemp owns an exclusively created file, so cleanup never removes a collision.
    descriptor, name = tempfile.mkstemp(prefix=".", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def install_host(
    project: str | Path,
    host: str,
    *,
    dry_run: bool = True,
    with_mcp: bool = False,
    harness_root: str | Path = "outputs/harness",
    python_executable: str | None = None,
    managed: bool = False,
    mcp_available: bool = True,
    previous_installation: str | None = None,
    installation_id: str | None = None,
) -> dict:
    """Copy only reviewed bundled assets into project paths; refuse user conflicts."""
    if host not in HOSTS:
        raise ValueError("Unknown host.")
    if with_mcp and not mcp_available:
        raise ValueError("MCP is outside the selected release profile.")
    root = _project_root(project)
    owned = {}
    if previous_installation:
        if not re.fullmatch(r"install-[0-9a-f]{32}", previous_installation):
            raise ValueError("Invalid previous host installation.")
        prior = _contained(
            root, f".autotem/host-installations/{previous_installation}/manifest.json"
        )
        previous = json.loads(prior.read_bytes())
        if (previous.get("state") != "applied" or previous.get("host") != host
                or previous.get("installation_id") != previous_installation):
            raise ValueError("Previous host installation does not match.")
        owned = {
            row["path"]: (previous_installation, index, row)
            for index, row in enumerate(previous["files"])
            if not row.get("preexisting_unmanaged")
        }
    rows = []
    originals: dict[str, bytes | None] = {}
    conflicts = []
    result = {
        "schema_version": INSTALL_VERSION, "ok": True, "dry_run": dry_run, "host": host,
        "execution_mode": "sequential", "native_subagents_verified": False,
        "host_llm_e2e_verified": False, "files": rows, "conflicts": [], "installation_id": None,
        "next_action": "Review dry-run, apply project files, then verify host discovery.",
    }
    operation = "upgrade" if previous_installation else "install"

    def issue(relative, condition, proposed_content=None):
        prior_row = owned.get(relative)
        expected = prior_row[2]["after_sha256"] if prior_row else (
            _hash(proposed_content) if condition == "available_path" else None
        )
        return host_conflict.describe(
            root, relative, "recorded_file" if prior_row else condition, expected,
            receipt=prior_row, proposed=proposed_content, save=managed and not dry_run,
        )

    def blocked(details):
        info, text = host_conflict.resolution(operation, details)
        return {**result, "ok": False, "conflicts": [v["path"] for v in details],
                "conflict_resolution": info, "next_action": text}

    try:
        proposed = _planned_files(
            root, host, with_mcp=with_mcp, harness_root=harness_root,
            python_executable=python_executable or sys.executable,
            managed=managed, mcp_available=mcp_available,
        )
    except _TomlReview as exc:
        details = blocked([issue(".codex/config.toml", "compatible_toml", exc.section)])
        if not managed:  # Preserve the existing Python API exception for invalid TOML merges.
            raise _TomlReview(str(exc) + "\n" + details["next_action"], exc.section) from exc
        return details
    config = ".codex/config.toml"
    if managed and host == "codex" and not mcp_available and config in owned:
        path = _contained(root, config)
        if not path.is_file() or _hash(path.read_bytes()) != owned[config][2]["after_sha256"]:
            return blocked([issue(config, "recorded_file")])
        original = _codex_config_before_managed_mcp(root, previous_installation)
        if original is not None:
            proposed[config] = original
    for relative, content in proposed.items():
        target = _contained(root, relative)
        before = target.read_bytes() if target.is_file() else None
        originals[relative] = before
        prior_row = owned.get(relative)
        before_sha = _hash(before) if before is not None else None
        if target.is_dir():
            action = "conflict"
        elif prior_row and before_sha != prior_row[2]["after_sha256"] and (
            before != content or relative == config
        ):
            action = "conflict"
        elif before == content:
            action = "unchanged"
        elif before is None:
            action = "create"
        elif prior_row:
            action = "update"
        elif relative == config and with_mcp:
            action = "merge"
        else:
            action = "conflict"
        if action == "conflict":
            conflicts.append(issue(relative, "available_path", content))
        rows.append(
            {
                "path": relative,
                "action": action,
                "before_sha256": before_sha,
                "after_sha256": _hash(content),
                "size_bytes": len(content),
            }
        )
    if conflicts:
        return blocked(conflicts)
    if dry_run:
        return result
    result["next_action"] = host_install_next_action(
        host, mcp_available=mcp_available, codex_configured=with_mcp
    )
    if all(row["action"] == "unchanged" for row in rows):
        return result
    installation_id = installation_id or "install-" + uuid.uuid4().hex
    if not re.fullmatch(r"install-[0-9a-f]{32}", installation_id):
        raise ValueError("Invalid host installation identifier.")
    directory = _contained(root, f".autotem/host-installations/{installation_id}")
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": INSTALL_VERSION,
        "installation_id": installation_id,
        "host": host,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": rows,
        "state": "prepared",
    }
    for index, row in enumerate(rows):
        relative = row["path"]
        before = originals[relative]
        if before is not None:
            backup = f"backups/{index:04d}.bin"
            _atomic_write(directory / backup, before)
            row["backup"] = backup
        if row["action"] != "unchanged" or relative in owned:
            row["after_copy"] = host_conflict.copy_name(index)
            _atomic_write(directory / row["after_copy"], proposed[relative])
        else:
            # The first install found identical bytes already here and never owned the file.
            row["preexisting_unmanaged"] = True
    _atomic_write(directory / "manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
    changed = []
    try:
        for row in rows:
            if row["action"] == "unchanged":
                continue
            target = _contained(root, row["path"])
            current = target.read_bytes() if target.exists() else None
            if current != originals[row["path"]]:
                raise ValueError("Project file changed since planning; install aborted.")
            _atomic_write(target, proposed[row["path"]])
            changed.append(row)
        manifest["state"] = "applied"
    except Exception:
        for row in reversed(changed):
            target = _contained(root, row["path"])
            if _hash(target.read_bytes()) != row["after_sha256"]:
                continue
            before = originals[row["path"]]
            if before is None:
                target.unlink()
            else:
                _atomic_write(target, before)
        manifest["state"] = "failed"
        _atomic_write(directory / "manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
        raise
    _atomic_write(directory / "manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
    result["installation_id"] = installation_id
    return result


def rollback_host(
    project: str | Path, installation_id: str, *, dry_run: bool = True, recover: bool = False
) -> dict:
    root = _project_root(project)
    if not re.fullmatch(r"install-[0-9a-f]{32}", installation_id):
        raise ValueError("Invalid installation_id.")
    directory = _contained(root, f".autotem/host-installations/{installation_id}")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    states = {"prepared", "applied", "failed", "rolled_back"} if recover else {"applied"}
    if manifest.get("installation_id") != installation_id or manifest.get("state") not in states:
        raise ValueError("Installation is not an applied, matching installation.")
    conflicts = []
    restores = []
    host = manifest["host"]
    for index, row in enumerate(manifest["files"]):
        if row["action"] == "unchanged":
            continue
        if host not in HOSTS:
            raise ValueError("Invalid host in installation receipt.")
        name = host_conflict.host_target(host, row["path"])
        target = _contained(root, name)
        current = target.read_bytes() if target.is_file() else None
        current_sha = _hash(current) if current is not None else None
        if not target.is_dir() and recover and current_sha == row["before_sha256"]:
            continue
        if current_sha != row["after_sha256"] or target.is_dir():
            conflicts.append(host_conflict.describe(
                root, name, "recorded_file", row["after_sha256"],
                receipt=(installation_id, index, row),
            ))
            continue
        before = None
        if row["before_sha256"] is not None and "backup" not in row:
            raise ValueError("Required host installation backup is missing.")
        if "backup" in row:
            backup_path = _contained(directory, row["backup"])
            before = backup_path.read_bytes()
            if _hash(before) != row["before_sha256"]:
                raise ValueError("Installation backup integrity check failed.")
        restores.append((target, before, current, index, row))
    result = {
        "schema_version": INSTALL_VERSION,
        "ok": not conflicts,
        "dry_run": dry_run,
        "installation_id": installation_id,
        "conflicts": [row["path"] for row in conflicts],
        "file_count": len(restores) + len(conflicts),
    }
    if conflicts:
        result["conflict_resolution"], result["next_action"] = host_conflict.resolution(
            "rollback", conflicts
        )
        return result
    if dry_run:
        return result
    changed = []
    try:
        for target, before, after, index, row in reversed(restores):
            if not target.is_file() or target.read_bytes() != after:
                issue = host_conflict.describe(
                    root, row["path"], "recorded_file", row["after_sha256"],
                    receipt=(installation_id, index, row),
                )
                info, text = host_conflict.resolution("rollback", [issue])
                raise host_conflict.HostFileConflict(
                    {"conflict_resolution": info, "next_action": text}
                )
            if before is None:
                target.unlink()
            else:
                _atomic_write(target, before)
            changed.append((target, before, after))
    except Exception:
        for target, before, after in reversed(changed):
            if (not target.exists() and before is None) or (
                target.is_file() and target.read_bytes() == before
            ):
                _atomic_write(target, after)
        raise
    manifest["state"] = "rolled_back"
    _atomic_write(directory / "manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
    return result


def probe_host(host: str, *, timeout: float = 15) -> dict:
    """Read only version/help; never read credentials/config or launch a host LLM."""
    if host not in HOSTS or not 0 < timeout <= 60:
        raise ValueError("Unknown host or invalid probe timeout.")
    executable = shutil.which(host)
    result = {
        "schema_version": INSTALL_VERSION,
        "host": host,
        "cli_available": bool(executable),
        "cli_version": None,
        "help_available": False,
        "execution_mode": "sequential",
        "native_subagents_verified": False,
        "host_llm_e2e_verified": False,
        "discovery_verified": False,
        "warnings": [],
    }
    if executable is None:
        result["warnings"].append("CLI not found; GUI import/discovery remains not_evaluated.")
        return result
    options = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout,
        "check": False,
    }
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        version = subprocess.run([executable, "--version"], **options)
        match = re.search(
            r"(?:codex-cli\s+|OpenClaw\s+)?(\d{1,4}\.\d{1,4}\.\d{1,3}(?:[-+][\w.-]+)?)",
            version.stdout,
        )
        result["cli_version"] = match.group(1) if match and version.returncode == 0 else None
        help_result = subprocess.run([executable, "--help"], **options)
        result["help_available"] = help_result.returncode == 0
        if version.returncode != 0 or help_result.returncode != 0:
            result["warnings"].append("Version/help probe failed; no compatibility verified.")
        if version.stderr.strip() or help_result.stderr.strip():
            result["warnings"].append(
                "Host emitted diagnostic stderr; no credentials or raw diagnostics retained."
            )
    except (OSError, subprocess.TimeoutExpired):
        result["warnings"].append(
            "CLI probe unavailable or timed out; no native/LLM capability verified."
        )
    return result
