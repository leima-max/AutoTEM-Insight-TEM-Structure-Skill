"""Bounded installation diagnostics with ephemeral project-owned scratch.

The selected interpreter must import the installed package in isolated mode. A
successful local MCP handshake does not prove host discovery, policy approval,
native delegation, or a real host/LLM analysis journey. No host config, service,
package index, installed files, or existing run data are changed by this module.
Inspection may create and remove its own temporary directory inside the selected
project; it does not rely on system-temp permissions or bypass host policy.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path

HOSTS = ("codex", "workbuddy", "openclaw")
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
CORE_MODULES = (
    "numpy",
    "PIL",
    "pydantic",
    "yaml",
    "scipy",
    "skimage",
    "matplotlib",
    "imageio",
    "tifffile",
    "ncempy",
)
# Fixed application-owned code. No caller-provided Python/script/shell is evaluated.
RUNTIME_PROBE = """
import contextlib, importlib, importlib.util, importlib.metadata, json, sys
modules = ('autotem', 'numpy', 'PIL', 'pydantic', 'yaml', 'scipy',
           'skimage', 'matplotlib', 'imageio', 'tifffile', 'ncempy', 'mcp')
present = {}
broken = []
for name in modules:
    try:
        present[name] = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        present[name] = False
    if present[name] and name not in ('autotem', 'mcp'):
        try:
            with contextlib.redirect_stdout(sys.stderr):
                importlib.import_module(name)
        except Exception:
            broken.append(name)
installation_type = 'unregistered_package'
try:
    distribution = importlib.metadata.distribution('autotem-insight')
    direct = json.loads(distribution.read_text('direct_url.json') or '{}')
    installation_type = 'editable' if direct.get('dir_info', {}).get('editable') else 'non_editable'
except (importlib.metadata.PackageNotFoundError, ValueError, AttributeError):
    pass
print(json.dumps({'python_version': list(sys.version_info[:3]), 'modules': present,
                  'installation_type': installation_type, 'broken_modules': broken}))
"""


def _options(timeout: float) -> dict:
    options = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "check": False,
        "timeout": timeout,
        "shell": False,
    }
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    return options


def _run(arguments: list[str], timeout: float) -> dict:
    try:
        process = subprocess.run(arguments, **_options(timeout))
    except subprocess.TimeoutExpired:
        return {"code": "PROBE_TIMEOUT", "returncode": None}
    except (OSError, ValueError):
        return {"code": "PROCESS_UNAVAILABLE", "returncode": None}
    if len(process.stdout.encode("utf-8")) > MAX_OUTPUT_BYTES:
        return {"code": "PROBE_OUTPUT_TOO_LARGE", "returncode": process.returncode}
    return {
        "code": "OK" if process.returncode == 0 else "PROCESS_FAILED",
        "returncode": process.returncode,
        "stdout": process.stdout,
        # stderr is never included in the public projection.
        "stderr": process.stderr,
    }


def _json_object(probe: dict) -> dict | None:
    def reject_constant(_value):
        raise ValueError("Invalid non-finite JSON")

    try:
        payload = json.loads(probe.get("stdout", ""), parse_constant=reject_constant)
    except (ValueError, TypeError, RecursionError):
        return None
    return payload if isinstance(payload, dict) else None


def _missing_module(probe: dict) -> str | None:
    match = re.search(
        r"No module named ['\"]([A-Za-z_][A-Za-z0-9_.]*)['\"]", probe.get("stderr", "")
    )
    if not match:
        return None
    # Only known package identifiers may escape the diagnostic boundary.
    name = match.group(1).split(".")[0]
    return name if name in {*CORE_MODULES, "autotem", "mcp"} else None


def _nonblocking_stdout(stream):
    """Read an anonymous child pipe without a reader thread or a wait on another writer."""
    fd = stream.fileno()
    if os.name == "nt" and sys.version_info < (3, 12):
        # os.set_blocking gained Windows pipe support in Python 3.12.
        import ctypes
        import msvcrt
        from ctypes import wintypes

        peek = ctypes.WinDLL("kernel32", use_last_error=True).PeekNamedPipe
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(fd))
        available = wintypes.DWORD()

        def read_available(size):
            # We own the pipe's only reader, so another reader cannot drain this byte count.
            if not peek(handle, None, 0, None, ctypes.byref(available), None):
                error = ctypes.get_last_error()
                if error in (109, 233):  # ERROR_BROKEN_PIPE, ERROR_PIPE_NOT_CONNECTED
                    return b""
                raise ctypes.WinError(error)
            return os.read(fd, min(size, available.value)) if available.value else None

        return read_available

    os.set_blocking(fd, False)

    def read(size):
        try:
            return os.read(fd, size)
        except BlockingIOError:
            return None

    return read


def _mcp_line(read, buffered: bytearray, deadline: float) -> bytes | None:
    """Return one bounded JSON line (or EOF), retaining any following partial reply."""
    scanned = 0
    while True:
        newline = buffered.find(b"\n", scanned)
        if newline >= 0:
            if newline + 1 > MAX_OUTPUT_BYTES:
                return None
            line = bytes(buffered[:newline])
            del buffered[:newline + 1]
            return line
        if len(buffered) >= MAX_OUTPUT_BYTES:
            return None
        scanned = len(buffered)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        chunk = read(min(65536, MAX_OUTPUT_BYTES - len(buffered)))
        if chunk is None:
            time.sleep(min(0.01, remaining))
        elif chunk:
            buffered.extend(chunk)
        else:
            # Text streams also accept a final line that ends at EOF without a newline.
            line = bytes(buffered)
            buffered.clear()
            return line or None


def _mcp_probe(python: str, root: str, timeout: float) -> dict:
    """Real initialize, tool discovery and capability call; no host/LLM launch."""
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "autotem-readiness", "version": "1.0.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    options = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.DEVNULL,
        "shell": False,
        "bufsize": 0,
    }
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    process = None
    timeout_detail = None
    stage = "initialize"
    last_response = None
    started = time.monotonic()
    stage_started = started
    cleanup = "closed"

    try:
        process = subprocess.Popen(
            [python, "-I", "-m", "autotem.harness.cli", "--root", root, "mcp"], **options
        )
        read = _nonblocking_stdout(process.stdout)
        buffered = bytearray()
        started = time.monotonic()
        deadline = started + timeout
        responses = {}
        for message in messages:
            stage = message["method"]
            stage_started = time.monotonic()
            process.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
            process.stdin.flush()
            if "id" not in message:
                continue
            line = _mcp_line(read, buffered, deadline)
            response = _json_object({"stdout": line}) if line is not None else None
            if response is None or response.get("id") != message["id"] or "error" in response:
                return {"status": "blocked", "code": "MCP_PROTOCOL_FAILED"}
            responses[message["id"]] = response.get("result")
            last_response = stage
        initialized = responses.get(1)
        listed = responses.get(2)
        tools = listed.get("tools") if isinstance(listed, dict) else None
        expected = {
            "list_capabilities",
            "inspect_input",
            "validate_plan",
            "submit_analysis",
            "get_run_status",
            "cancel_run",
            "list_artifacts",
            "read_artifact",
            "export_report",
        }
        names = (
            {
                tool.get("name")
                for tool in tools
                if isinstance(tool, dict) and isinstance(tool.get("name"), str)
            }
            if isinstance(tools, list)
            else set()
        )
        if not isinstance(initialized, dict) or names != expected:
            return {"status": "blocked", "code": "MCP_TOOL_CONTRACT_FAILED"}
        # Tool registration is insufficient: the SDK handler may fail or hang
        # during its first dispatch/import. Execute a read-only registered tool.
        call = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "list_capabilities", "arguments": {}},
        }
        stage = call["method"]
        stage_started = time.monotonic()
        process.stdin.write((json.dumps(call) + "\n").encode("utf-8"))
        process.stdin.flush()
        line = _mcp_line(read, buffered, deadline)
        response = _json_object({"stdout": line}) if line is not None else None
        if response is None or response.get("id") != 3 or "error" in response:
            return {"status": "blocked", "code": "MCP_TOOL_CALL_FAILED"}
        called = response.get("result")
        if not isinstance(called, dict) or called.get("isError") is True:
            return {"status": "blocked", "code": "MCP_TOOL_CALL_FAILED"}
        structured = called.get("structuredContent")
        content = called.get("content")
        text_envelope = None
        if isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict):
            if content[0].get("type") == "text" and isinstance(content[0].get("text"), str):
                text_envelope = _json_object({"stdout": content[0]["text"]})
        if structured is not None and not isinstance(structured, dict):
            return {"status": "blocked", "code": "MCP_TOOL_CALL_CONTRACT_FAILED"}
        if structured is not None and content is not None and structured != text_envelope:
            return {"status": "blocked", "code": "MCP_TOOL_CALL_CONTRACT_FAILED"}
        envelope = structured if isinstance(structured, dict) else text_envelope
        if not isinstance(envelope, dict) or type(envelope.get("ok")) is not bool:
            return {"status": "blocked", "code": "MCP_TOOL_CALL_CONTRACT_FAILED"}
        if envelope.get("ok") is False:
            return {"status": "blocked", "code": "MCP_TOOL_CALL_FAILED"}
        if envelope.get("schema_version") != "1.0.0":
            return {"status": "blocked", "code": "MCP_TOOL_CALL_CONTRACT_FAILED"}
        data = envelope.get("result")
        operations = data.get("operations") if isinstance(data, dict) else None
        if (
            not isinstance(operations, list)
            or not operations
            or any(
                not isinstance(operation, dict)
                or not isinstance(operation.get("operation_id"), str)
                or not re.fullmatch(r"[a-z][a-z0-9_.-]*", operation["operation_id"])
                for operation in operations
            )
        ):
            return {"status": "blocked", "code": "MCP_TOOL_CALL_CONTRACT_FAILED"}
        return {
            "status": "ready",
            "code": "LOCAL_MCP_TOOL_CALL_OK",
            "tool_count": len(names),
            "operation_count": len(operations),
        }
    except TimeoutError:
        ended = time.monotonic()
        timeout_detail = {
            "stage": stage,
            "last_response": last_response,
            "budget_seconds": timeout,
            "elapsed_seconds": round(ended - started, 3),
            "stage_seconds": round(ended - stage_started, 3),
        }
        return {"status": "blocked", "code": "MCP_PROBE_TIMEOUT", "diagnostic": timeout_detail}
    except (OSError, ValueError, TypeError):
        return {"status": "blocked", "code": "MCP_PROCESS_FAILED"}
    finally:
        if process is not None:
            cleanup_started = time.monotonic()
            if process.stdin:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=min(2.0, timeout))
            except subprocess.TimeoutExpired:
                cleanup = "direct_child_killed"
                process.kill()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    cleanup = "direct_child_exit_unconfirmed"
            if process.stdout:
                process.stdout.close()
            if timeout_detail is not None:
                timeout_detail.update(
                    cleanup=cleanup, cleanup_seconds=round(time.monotonic() - cleanup_started, 3)
                )


def _runtime(python: str | None, timeout: float, project_root: Path | None = None) -> dict:
    result = {
        "status": "blocked",
        "code": "PYTHON_NOT_FOUND",
        "capabilities": {"status": "not_evaluated"},
        "local_mcp": {"status": "not_evaluated"},
    }
    if python is None:
        return result
    probe = _run([python, "-I", "-c", RUNTIME_PROBE], timeout)
    if probe["code"] != "OK":
        result["code"] = probe["code"]
        return result
    payload = _json_object(probe)
    if payload is None or not isinstance(payload.get("modules"), dict):
        result["code"] = "RUNTIME_PROBE_INVALID_JSON"
        return result
    version = payload.get("python_version")
    modules = payload["modules"]
    if (
        not isinstance(version, list)
        or len(version) != 3
        or any(type(value) is not int or not 0 <= value < 1000 for value in version)
    ):
        result["code"] = "RUNTIME_PROBE_INVALID_JSON"
        return result
    result["python_version"] = ".".join(map(str, version))
    installation_type = payload.get("installation_type")
    result["installation_type"] = (
        installation_type
        if isinstance(installation_type, str)
        and installation_type in {"editable", "non_editable", "unregistered_package"}
        else "unknown"
    )
    if tuple(version[:2]) < (3, 10):
        result["code"] = "PYTHON_VERSION_UNSUPPORTED"
        return result
    if modules.get("autotem") is not True:
        result["code"] = "AUTOTEM_NOT_INSTALLED"
        return result
    missing = [name for name in CORE_MODULES if modules.get(name) is not True]
    if missing:
        result.update(code="DEPENDENCY_UNAVAILABLE", missing_modules=missing)
        return result
    broken = payload.get("broken_modules", [])
    if not isinstance(broken, list) or any(
        not isinstance(name, str) or name not in CORE_MODULES for name in broken
    ):
        result["code"] = "RUNTIME_PROBE_INVALID_JSON"
        return result
    broken = [name for name in broken if isinstance(name, str) and name in CORE_MODULES]
    if broken:
        result.update(code="DEPENDENCY_BROKEN", broken_modules=broken)
        return result
    dependency_check = _run([python, "-I", "-m", "pip", "check"], timeout)
    result["dependency_check"] = {
        "status": "ready" if dependency_check["code"] == "OK" else "blocked",
        "code": "PIP_CHECK_OK" if dependency_check["code"] == "OK" else "PIP_CHECK_FAILED",
    }
    if dependency_check["code"] != "OK":
        result["code"] = "DEPENDENCY_CONFLICT_OR_CHECK_FAILED"
        return result
    # Some Windows policies report the system temp directory as writable but
    # reject actual writes; tempfile can then retry PermissionError indefinitely.
    # Make exactly one creation attempt in the authorized project directory.
    scratch_root = Path(project_root if project_root is not None else Path.cwd()).resolve()
    directory = scratch_root / f".autotem-readiness-{uuid.uuid4().hex}"
    try:
        directory.mkdir(mode=0o700)
    except OSError:
        result["code"] = "SCRATCH_UNAVAILABLE"
        return result
    identity = None
    try:
        owned = directory.lstat()
        identity = (owned.st_dev, owned.st_ino)
        directory.resolve().relative_to(scratch_root)
        capability = _run(
            [python, "-I", "-m", "autotem.harness.cli", "--root", str(directory), "capabilities"],
            timeout,
        )
        envelope = _json_object(capability)
        if capability["code"] != "OK":
            missing_name = _missing_module(capability)
            code = "DEPENDENCY_UNAVAILABLE" if missing_name else capability["code"]
            if code == "PROCESS_FAILED":
                code = "RUNTIME_IMPORT_OR_EXECUTION_FAILED"
            result["code"] = code
            result["capabilities"] = {"status": "blocked", "code": code}
            return result
        data = envelope.get("result") if envelope and envelope.get("ok") is True else None
        operations = data.get("operations") if isinstance(data, dict) else None
        if (
            not isinstance(operations, list)
            or not operations
            or any(
                not isinstance(operation, dict)
                or not isinstance(operation.get("operation_id"), str)
                or not re.fullmatch(r"[a-z][a-z0-9_.-]*", operation["operation_id"])
                for operation in operations
            )
        ):
            result["code"] = "CAPABILITY_CONTRACT_INVALID"
            result["capabilities"] = {"status": "blocked", "code": result["code"]}
            return result
        result.update(status="ready", code="INSTALLED_RUNTIME_READY")
        result["capabilities"] = {
            "status": "ready",
            "code": "CAPABILITIES_EXECUTED",
            "operation_count": len(operations),
        }
        result["local_mcp"] = (
            _mcp_probe(python, str(directory), timeout)
            if modules.get("mcp") is True
            else {
                "status": "dependency_unavailable",
                "code": "MCP_EXTRA_NOT_INSTALLED",
                "alternative": "json_cli",
            }
        )
    except (OSError, ValueError):
        result.update(status="blocked", code="SCRATCH_UNAVAILABLE")
    finally:
        try:
            resolved = directory.resolve()
            resolved.relative_to(scratch_root)
            current = directory.lstat()
            if (
                resolved != directory
                or directory.is_symlink()
                or getattr(current, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                or (current.st_dev, current.st_ino) != identity
            ):
                raise OSError("Owned scratch directory was replaced")
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            result.update(status="blocked", code="SCRATCH_CLEANUP_FAILED")
    return result


def _host_cli(host: str, timeout: float) -> tuple[dict, str | None]:
    executable = shutil.which(host)
    result = {"status": "not_found", "code": "HOST_CLI_NOT_FOUND", "version": None}
    if executable is None:
        # WorkBuddy is commonly a GUI-only installation; absence of CLI is not
        # proof that its connector/import functionality is unavailable.
        if host == "workbuddy":
            result["status"] = "gui_not_evaluated"
        return result, None
    probe = _run([executable, "--version"], timeout)
    match = re.search(r"\b(\d{1,4}\.\d{1,4}\.\d{1,3}(?:[-+][\w.-]+)?)", probe.get("stdout", ""))
    if probe["code"] != "OK":
        result.update(status="blocked", code=probe["code"])
    elif match:
        result.update(status="ready", code="HOST_CLI_RESPONDS", version=match.group(1))
    else:
        result.update(status="blocked", code="HOST_VERSION_UNRECOGNIZED")
    return result, executable


def _gateway(executable: str | None, timeout: float) -> dict:
    if executable is None:
        return {"status": "not_evaluated", "code": "HOST_CLI_NOT_FOUND"}
    status = _run([executable, "gateway", "status", "--json"], timeout)
    payload = _json_object(status)
    status_failure = (
        (status["code"] if status["code"] != "OK" else "GATEWAY_STATUS_INVALID_JSON")
        if payload is None
        else None
    )
    service = payload.get("service") if payload is not None else None
    service = service if isinstance(service, dict) else {}
    runtime = service.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    state = runtime.get("status")
    state = (
        state
        if isinstance(state, str) and state in {"running", "stopped", "unknown"}
        else "unknown"
    )
    result = {
        "status": "blocked",
        "code": "GATEWAY_UNREACHABLE",
        "service_registered": service.get("loaded")
        if type(service.get("loaded")) is bool
        else None,
        "runtime_state": state,
    }
    # A foreground gateway can be healthy while the registered service is
    # stopped. Reachability is authoritative; service state provides guidance.
    health = _run([executable, "health", "--json"], timeout)
    health_payload = _json_object(health)
    # Exit 0 alone or a running listener alone is insufficient evidence.
    if health["code"] == "OK" and health_payload and health_payload.get("ok") is True:
        result.update(status="ready", code="GATEWAY_HEALTHY")
    else:
        error = health_payload.get("error") if health_payload is not None else None
        error = error if isinstance(error, dict) else {}
        error_type = error.get("type")
        if isinstance(error_type, str) and error_type in {
            "authentication_error",
            "auth_error",
            "unauthorized",
            "pairing_required",
        }:
            result["code"] = "GATEWAY_AUTH_REQUIRED"
        elif state == "stopped":
            result["code"] = "GATEWAY_STOPPED"
        elif status_failure:
            result["code"] = status_failure
        elif health_payload is None:
            result["code"] = (
                health["code"] if health["code"] != "OK" else "GATEWAY_HEALTH_INVALID_JSON"
            )
        else:
            result["code"] = "GATEWAY_HEALTH_FAILED"
    return result


def inspect_readiness(
    project: str | Path,
    host: str,
    *,
    python_executable: str | Path | None = None,
    timeout: float = 15,
) -> dict:
    """Inspect installed local execution; ``timeout`` bounds each subprocess.

    Do not infer host permission from a successful subprocess in this caller.
    Neither gateway start nor package installation is performed. Caller-provided
    project owns an ephemeral diagnostic directory, removed after the probe.
    """
    if (
        host not in HOSTS
        or isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or not 0 < timeout <= 60
    ):
        raise ValueError("Unknown host or invalid readiness timeout.")
    root = Path(project).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Readiness requires an existing project directory.")
    selected = str(python_executable) if python_executable is not None else sys.executable
    # Resolving a POSIX venv symlink would run the base interpreter and lose isolation.
    python = str(Path(selected).absolute()) if Path(selected).is_file() else shutil.which(selected)
    runtime = _runtime(python, timeout, root)
    host_cli, host_executable = _host_cli(host, timeout)
    gateway = (
        _gateway(host_executable, timeout)
        if host == "openclaw"
        else {
            "status": "not_applicable",
        }
    )
    no_index = os.environ.get("PIP_NO_INDEX", "").strip().lower() in {"1", "true", "yes", "on"}
    actions = []
    if runtime["code"] in {"SCRATCH_UNAVAILABLE", "SCRATCH_CLEANUP_FAILED"}:
        actions.append(
            {
                "code": runtime["code"],
                "owner": "installer",
                "instruction": (
                    "Use an authorized writable project directory for diagnostic scratch. "
                    "Do not change system temp permissions or bypass host policy."
                ),
            }
        )
    elif runtime["status"] != "ready":
        actions.append(
            {
                "code": "INSTALL_COMPLETE_RUNTIME",
                "owner": "installer",
                "instruction": (
                    "Install the complete locked dependency bundle in the selected runtime."
                ),
            }
        )
    if (
        no_index
        and runtime["status"] != "ready"
        and runtime["code"] not in {"SCRATCH_UNAVAILABLE", "SCRATCH_CLEANUP_FAILED"}
    ):
        actions.append(
            {
                "code": "OFFLINE_BUNDLE_REQUIRED",
                "owner": "installer",
                "instruction": (
                    "Use a complete local wheelhouse; preserve the configured offline policy."
                ),
            }
        )
    if host_cli["status"] not in {"ready", "gui_not_evaluated"}:
        actions.append(
            {
                "code": "HOST_INSTALLATION_CHECK",
                "owner": "user",
                "instruction": (
                    "Use the host's supported installation and verify its CLI or connector."
                ),
            }
        )
    if gateway["status"] == "blocked":
        actions.append(
            {
                "code": gateway["code"],
                "owner": "user_or_host_admin",
                "instruction": (
                    "Check the existing OpenClaw gateway service; "
                    "start or repair only with authorization."
                ),
            }
        )
    if runtime["status"] == "ready" and runtime["local_mcp"]["status"] != "ready":
        actions.append(
            {
                "code": "MCP_TRANSPORT_CHECK",
                "owner": "installer",
                "instruction": (
                    "Install/test the MCP extra for a connector, "
                    "or use the verified JSON CLI transport."
                ),
            }
        )
    actions.append(
        {
            "code": "HOST_TOOL_APPROVAL_AND_E2E_REQUIRED",
            "owner": "host",
            "instruction": (
                "Authorize the configured AutoTEM connector and execute a real host analysis; "
                "retain its run evidence."
            ),
        }
    )
    blocked = (
        runtime["status"] != "ready"
        or host_cli["status"] not in {"ready", "gui_not_evaluated"}
        or gateway["status"] == "blocked"
        or (host == "openclaw" and gateway["status"] != "ready")
    )
    return {
        "schema_version": "1.0.0",
        "host": host,
        "status": "blocked" if blocked else "ready_for_host_verification",
        "ok": not blocked,
        "scope": "local_installation_probes_only",
        "runtime": runtime,
        "host_cli": host_cli,
        "gateway": gateway,
        "package_index": {"pip_no_index_active": no_index, "configuration_changed": False},
        "host_policy": {"status": "not_evaluated", "code": "HOST_APPROVAL_NOT_PROBED"},
        "host_mcp_connection": "not_evaluated",
        "native_subagents": "not_evaluated",
        "host_llm_e2e": "not_evaluated",
        "scientific_accuracy": "not_evaluated",
        "next_actions": actions,
    }
