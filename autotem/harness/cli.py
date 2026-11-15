"""Bounded JSON/JSONL host transport over the shared Harness service."""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path
from typing import Any, TextIO

PROTOCOL_VERSION = "1.0.0"
MAX_REQUEST_BYTES = 1024 * 1024


def _object_schema(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


STRING = {"type": "string", "minLength": 1}
OBJECT = {"type": "object"}
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "list_capabilities": _object_schema(
        {
            "enabled_plugins": {"type": "array"},
            "view": {"type": "string", "enum": ["full", "summary", "parameters"]},
            "operation_id": STRING,
        }
    ),
    "inspect_input": _object_schema({"path": STRING, "selection": OBJECT}, ("path",)),
    "validate_plan": _object_schema({"plan": OBJECT, "run_id": STRING}, ("plan",)),
    "submit_analysis": _object_schema(
        {
            "plan": OBJECT,
            "run_id": STRING,
            "idempotency_key": STRING,
            "parent_run_id": STRING,
            "execution_mode": {"type": "string", "enum": ["sequential"]},
            "user_request": {"type": "string", "maxLength": 10000},
            "asynchronous": {"type": "boolean"},
        },
        ("plan",),
    ),
    "get_run_status": _object_schema({"run_id": STRING}, ("run_id",)),
    "cancel_run": _object_schema({"run_id": STRING}, ("run_id",)),
    "list_artifacts": _object_schema({"run_id": STRING}, ("run_id",)),
    "read_artifact": _object_schema(
        {
            "artifact_id": STRING,
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
        },
        ("artifact_id",),
    ),
    "export_report": _object_schema(
        {
            "run_id": STRING,
            "formats": {
                "type": "array",
                "items": {"type": "string", "enum": ["json", "html", "pdf"]},
                "minItems": 1,
                "maxItems": 3,
                "uniqueItems": True,
            },
        },
        ("run_id",),
    ),
}
TOOL_DESCRIPTIONS = {
    "list_capabilities": (
        "Discover operations. Use view=summary for availability, then view=parameters with "
        "one operation_id for its exact parameters and scientific limits. Default: full catalog."
    ),
    "inspect_input": "Inspect authorized input with explicit dataset/frame/channel selection.",
    "validate_plan": "Validate operations, parameters, dependencies and budgets without analysis.",
    "submit_analysis": "Execute a validated traditional plan; returns a run ID and real status.",
    "get_run_status": "Read persisted execution, science, warnings and human input requirements.",
    "cancel_run": "Request cancellation of this run and its owned workers.",
    "list_artifacts": "List committed evidence for this run; does not scan arbitrary disk paths.",
    "read_artifact": "Read a bounded registered artifact preview or return its format limitation.",
    "export_report": "Export JSON/HTML/PDF from finalized evidence and scientific status.",
}


class ProtocolError(ValueError):
    """Invalid host request; never dispatch unvalidated tools or arguments."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("Duplicate JSON object key.")
        result[key] = value
    return result


def strict_json(text: str) -> Any:
    def invalid_constant(_value: str) -> None:
        raise ProtocolError("Non-finite JSON number is not allowed.")

    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=invalid_constant)


def _validate_arguments(tool: str, arguments: object) -> dict[str, Any]:
    if tool not in TOOL_SCHEMAS:
        raise ProtocolError("Unknown tool; use one of the nine published Harness tools.")
    if not isinstance(arguments, dict):
        raise ProtocolError("arguments must be an object.")
    schema = TOOL_SCHEMAS[tool]
    if set(arguments) - set(schema["properties"]):
        raise ProtocolError("Unknown tool argument.")
    if set(schema["required"]) - set(arguments):
        raise ProtocolError("Missing required tool argument.")
    for key, value in arguments.items():
        rule = schema["properties"][key]
        expected = rule["type"]
        valid = {
            "string": isinstance(value, str),
            "object": isinstance(value, dict),
            "boolean": isinstance(value, bool),
            "integer": type(value) is int,
            "array": isinstance(value, list),
        }[expected]
        if not valid:
            raise ProtocolError(f"{key} must have type {expected}.")
        if expected == "string":
            if len(value) < rule.get("minLength", 0) or len(value) > rule.get("maxLength", 10000):
                raise ProtocolError(f"{key} has an invalid length.")
        if "enum" in rule and value not in rule["enum"]:
            raise ProtocolError(f"{key} has an unsupported value.")
        if expected == "integer":
            if value < rule.get("minimum", 0) or value > rule.get("maximum", 10**12):
                raise ProtocolError(f"{key} is outside the allowed range.")
        if key == "formats":
            if not 1 <= len(value) <= 3 or any(
                item not in ("json", "html", "pdf") for item in value
            ):
                raise ProtocolError("formats must contain json, html and/or pdf.")
            if len(set(value)) != len(value):
                raise ProtocolError("formats must be unique.")
        if key == "enabled_plugins" and (
            len(value) > 16 or any(not isinstance(item, str) or not item for item in value)
        ):
            raise ProtocolError("enabled_plugins must contain at most 16 installed plugin names.")
    return dict(arguments)


def _safe_message(exc: Exception) -> str:
    message = str(exc)[:1000]
    message = re.sub(
        r"(?i)(api[_-]?key|token|secret|password)(\s*[=:]\s*)[^\s,;]+",
        r"\1\2[REDACTED]",
        message,
    )
    return re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", message)


def _safe_details(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]"
            if re.search(r"(?i)(api[_-]?key|token|secret|password|authorization)", str(key))
            else _safe_details(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_safe_details(item) for item in value[:100]]
    if isinstance(value, str):
        return _safe_message(ValueError(value))
    return (
        value if value is None or type(value) in (bool, int, float) else str(type(value).__name__)
    )


def dispatch_request(harness: Any, request: object) -> dict[str, Any]:
    """Shared MCP/CLI dispatch. Explicit branches keep the tool boundary closed."""
    tool: str | None = None
    request_id: str | int | None = None
    try:
        if not isinstance(request, dict):
            raise ProtocolError("Request must be an object.")
        if set(request) - {"tool", "arguments", "request_id", "schema_version"}:
            raise ProtocolError("Unknown request field.")
        request_id = request.get("request_id")
        if request_id is not None and (
            type(request_id) not in (str, int) or len(str(request_id)) > 128
        ):
            request_id = None
            raise ProtocolError("Invalid request_id.")
        if request.get("schema_version", PROTOCOL_VERSION) != PROTOCOL_VERSION:
            raise ProtocolError("Unsupported protocol schema_version.")
        tool = request.get("tool")
        if not isinstance(tool, str):
            raise ProtocolError("tool must be a string.")
        args = _validate_arguments(tool, request.get("arguments", {}))
        with contextlib.redirect_stdout(sys.stderr):
            if tool == "list_capabilities":
                result = harness.list_capabilities(**args) if args else harness.list_capabilities()
            elif tool == "inspect_input":
                result = harness.inspect_input(args["path"], **args.get("selection", {}))
            elif tool == "validate_plan":
                result = harness.validate_plan(**args)
            elif tool == "submit_analysis":
                result = harness.submit_analysis(**args)
            elif tool == "get_run_status":
                result = harness.get_run_status(args["run_id"])
            elif tool == "cancel_run":
                result = harness.cancel_run(args["run_id"])
            elif tool == "list_artifacts":
                result = harness.list_artifacts(args["run_id"])
            elif tool == "read_artifact":
                result = harness.read_artifact(**args)
            else:
                result = harness.export_report(
                    args["run_id"], formats=args.get("formats", ["json", "html"])
                )
        envelope = {
            "schema_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "tool": tool,
            "ok": True,
            "result": result,
            "error": None,
        }
        # Verify the actual transport result, rather than failing after printing a half-line.
        json.dumps(envelope, ensure_ascii=False, allow_nan=False)
        return envelope
    except Exception as exc:  # noqa: BLE001 - host transport must return a structured failure.
        from .contracts import HarnessError

        code = (
            "REQUEST_INVALID" if isinstance(exc, (ProtocolError, TypeError)) else "EXECUTION_ERROR"
        )
        error = {"code": code, "message": _safe_message(exc), "retryable": False}
        if isinstance(exc, HarnessError):
            error["code"] = exc.code
            error["details"] = _safe_details(exc.details)
        return {
            "schema_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "tool": tool,
            "ok": False,
            "result": None,
            "error": error,
        }


def serve_stdio(harness: Any, source: TextIO | None = None, sink: TextIO | None = None) -> int:
    """One JSON response per line, including malformed requests; JSONL is not MCP."""
    # A redirected Windows standard stream uses the ANSI code page even under Python -I.
    # Read wire bytes before decoding so a malformed line cannot consume the next request.
    stream = source if source is not None else getattr(sys.stdin, "buffer", sys.stdin)
    binary_sink = getattr(sys.stdout, "buffer", None) if sink is None else None
    writer = sink if sink is not None else sys.stdout
    while True:
        line = stream.readline(MAX_REQUEST_BYTES + 1)
        if not line:
            break
        try:
            wire = line if isinstance(line, bytes) else line.encode("utf-8")
            if len(wire) > MAX_REQUEST_BYTES:
                newline = b"\n" if isinstance(line, bytes) else "\n"
                while line and not line.endswith(newline):
                    line = stream.readline(MAX_REQUEST_BYTES + 1)
                raise ProtocolError("Request exceeds the JSONL size limit.")
            request = strict_json(wire.decode("utf-8"))
            response = dispatch_request(harness, request)
        except (ValueError, RecursionError) as exc:
            response = dispatch_request(harness, None)
            response["error"]["message"] = _safe_message(exc)
        rendered = json.dumps(response, ensure_ascii=True, allow_nan=False) + "\n"
        if binary_sink is not None:
            binary_sink.write(rendered.encode("utf-8"))
            binary_sink.flush()
        else:
            writer.write(rendered)
            writer.flush()
    return 0


def load_plan(path: str, input_path: str | None = None) -> dict:
    plan_path = Path(path)
    if plan_path.stat().st_size > MAX_REQUEST_BYTES:
        raise ProtocolError("Plan exceeds the size limit.")
    text = plan_path.read_text(encoding="utf-8-sig")
    if plan_path.suffix.lower() == ".json":
        payload = strict_json(text)
    else:
        import yaml

        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ProtocolError("Plan must be a mapping.")
    if input_path is not None:
        payload["input_path"] = str(Path(input_path).resolve())
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoTEM traditional Harness and host bridge.")
    parser.add_argument("--root", default="outputs/harness", help="Persistent Harness root.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("capabilities")
    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("path")
    inspect_parser.add_argument("--selection-json", default="{}")
    for name in ("validate", "submit"):
        child = sub.add_parser(name)
        child.add_argument("--plan", required=True, help="JSON/YAML operation plan.")
        child.add_argument("--input", help="Override recipe input_path with an authorized file.")
        child.add_argument(
            "--run-id", help="Append new work to a completed run; never edit old steps."
        )
        if name == "submit":
            child.add_argument("--idempotency-key")
            child.add_argument("--parent-run-id")
            child.add_argument("--user-request")
            child.add_argument("--wait", action="store_true", help="Execute synchronously.")
    for name in ("status", "cancel", "artifacts", "export"):
        child = sub.add_parser(name)
        child.add_argument("run_id")
        if name == "export":
            child.add_argument(
                "--formats", nargs="+", choices=("json", "html", "pdf"), default=["json", "html"]
            )
    read = sub.add_parser("read")
    read.add_argument("artifact_id")
    read.add_argument("--offset", type=int, default=0)
    read.add_argument("--limit", type=int, default=100)
    stdio = sub.add_parser("stdio", help="JSONL bridge (not MCP protocol).")
    stdio.add_argument(
        "--workspace-only", type=Path, metavar="PROJECT",
        help="Restrict analysis inputs and container reads to this project.",
    )
    sub.add_parser("mcp", help="Optional real MCP stdio server; needs the mcp extra.")
    probe = sub.add_parser("host-probe")
    probe.add_argument("host", choices=("codex", "openclaw", "workbuddy"))
    probe.add_argument("--timeout", type=float, default=15)
    doctor = sub.add_parser(
        "doctor", help="Check runtime/transport/host readiness without repairs."
    )
    doctor.add_argument("--host", choices=("codex", "openclaw", "workbuddy"), required=True)
    doctor.add_argument("--project", required=True)
    doctor.add_argument("--python", dest="python_executable")
    doctor.add_argument("--timeout", type=float, default=15)
    install = sub.add_parser("install-host")
    install.add_argument("host", choices=("codex", "openclaw", "workbuddy"))
    install.add_argument("--managed", action="store_true", help=argparse.SUPPRESS)
    install.add_argument("--without-mcp", action="store_true", help=argparse.SUPPRESS)
    install.add_argument("--previous-installation", help=argparse.SUPPRESS)
    install.add_argument("--installation-id", help=argparse.SUPPRESS)
    install.add_argument("--project", required=True)
    install.add_argument(
        "--apply", action="store_true", help="Default is dry run; apply project-local files."
    )
    install.add_argument(
        "--with-mcp", action="store_true", help="Merge only the project's Codex MCP entry."
    )
    rollback = sub.add_parser("rollback-host")
    rollback.add_argument("--project", required=True)
    rollback.add_argument("installation_id")
    rollback.add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            from .readiness import inspect_readiness

            result = inspect_readiness(
                args.project,
                args.host,
                python_executable=args.python_executable,
                timeout=args.timeout,
            )
            print(json.dumps(result, ensure_ascii=True, allow_nan=False))
            return 0 if result["ok"] else 2
        if args.command in ("host-probe", "install-host", "rollback-host"):
            from .hosts import install_host, probe_host, rollback_host

            if args.command == "host-probe":
                result = probe_host(args.host, timeout=args.timeout)
            elif args.command == "install-host":
                result = install_host(
                    args.project,
                    args.host,
                    dry_run=not args.apply,
                    with_mcp=args.with_mcp,
                    harness_root=args.root,
                    managed=args.managed,
                    mcp_available=not args.without_mcp,
                    previous_installation=args.previous_installation,
                    installation_id=args.installation_id,
                )
            else:
                result = rollback_host(args.project, args.installation_id, dry_run=not args.apply)
            print(json.dumps(result, ensure_ascii=True, allow_nan=False))
            return 0 if result.get("ok", True) else 2
        if args.command == "mcp":
            from .mcp import run_server

            run_server(Path(args.root))
            return 0
        from autotem.harness import Harness

        with contextlib.redirect_stdout(sys.stderr):
            harness = Harness(Path(args.root), workspace=getattr(args, "workspace_only", None))
        if args.command == "stdio":
            return serve_stdio(harness)
        aliases = {
            "capabilities": "list_capabilities",
            "inspect": "inspect_input",
            "validate": "validate_plan",
            "submit": "submit_analysis",
            "status": "get_run_status",
            "cancel": "cancel_run",
            "artifacts": "list_artifacts",
            "read": "read_artifact",
            "export": "export_report",
        }
        arguments: dict[str, Any] = {}
        if args.command == "inspect":
            arguments = {"path": args.path, "selection": strict_json(args.selection_json)}
        elif args.command in ("validate", "submit"):
            arguments = {"plan": load_plan(args.plan, args.input)}
            if args.run_id is not None:
                arguments["run_id"] = args.run_id
            if args.command == "submit":
                arguments["asynchronous"] = not args.wait
                for key in ("idempotency_key", "parent_run_id", "user_request"):
                    if getattr(args, key) is not None:
                        arguments[key] = getattr(args, key)
        elif args.command == "read":
            arguments = {
                "artifact_id": args.artifact_id,
                "offset": args.offset,
                "limit": args.limit,
            }
        elif hasattr(args, "run_id"):
            arguments = {"run_id": args.run_id}
            if args.command == "export":
                arguments["formats"] = args.formats
        response = dispatch_request(
            harness, {"tool": aliases[args.command], "arguments": arguments}
        )
        print(json.dumps(response, ensure_ascii=True, allow_nan=False))
        # ok is transport success. Callers still inspect execution/scientific/review status.
        return 0 if response["ok"] else 2
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "schema_version": PROTOCOL_VERSION,
                    "ok": False,
                    "result": None,
                    "error": {
                        "code": "CLI_ERROR",
                        "message": _safe_message(exc),
                        "retryable": False,
                    },
                },
                ensure_ascii=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
