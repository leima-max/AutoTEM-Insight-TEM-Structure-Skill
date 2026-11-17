"""Read operator-observed reconnect failures and check the host files to be restored."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from autotem import bootstrap as boot
from autotem.installation import host_conflict, uninstall
from autotem.installation.release import sha256, unique_json

REPORT_FIELDS = {
    "schema_version",
    "installation_id",
    "release_id",
    "host",
    "stage",
    "error_code",
    "message",
}
NON_FAILURE_TERMS = frozenset(
    "AUTH AUTHORIZATION AUTHORISATION AUTHENTICATION PERMISSION APPROVAL CONSENT TRUST POLICY "
    "UNAUTHORIZED UNAUTHORISED UNAUTHENTICATED UNTRUSTED PENDING WAITING DENIED FORBIDDEN "
    "DECLINED CANCEL CANCELED CANCELLED CANCELLATION STALE".split()
)
NON_FAILURE_CODES = {
    "HOST_RECONNECT_REQUIRED",
    "HOST_SETUP_REQUIRED",
    "NEEDS_INPUT",
    "INVALID_PARAMETERS",
    "NUMERICAL_FAILURE",
    "NOT_EVALUATED",
    "OK",
    "SUCCESS",
}
# A successful tool response can carry a scientific/domain error. Require evidence of a failed
# connector exchange for call_tool; an arbitrary runtime error code cannot authorize rollback.
CALL_CONNECTION_FAILURE = re.compile(
    r"(?:[A-Z0-9]+_)*(?:CONNECTION|CONNECTOR|TRANSPORT|SESSION|SOCKET|CHANNEL|STREAM|PIPE|"
    r"HANDSHAKE|PROTOCOL|STDIO|NETWORK|SERVER|PEER)_"
    r"(?:ABORTED|BROKEN|CLOSED|DISCONNECTED|DISCONNECT|EOF|ERROR|FAILED|GONE|LOST|"
    r"REFUSED|RESET|TIMEOUT|TIMED_OUT|UNAVAILABLE|UNREACHABLE|EXITED|TERMINATED)"
)
CALL_EXCHANGE_CODES = {
    "ECONNABORTED",
    "ECONNREFUSED",
    "ECONNRESET",
    "ENOTCONN",
    "EPIPE",
    "ETIMEDOUT",
    "BROKEN_PIPE",
    "UNEXPECTED_EOF",
    "MCP_TIMEOUT",
    "JSONRPC_TIMEOUT",
    "TOOL_CALL_FAILED",
}


def read_report(source):
    # Read a single bounded snapshot. Never copy the operator's diagnostic file or persist its path.
    try:
        path = Path(source)
        if not path.is_file():
            raise ValueError("Select a regular local JSON file.")
        with path.open("rb") as stream:
            raw = stream.read(16_385)
        if len(raw) > 16_384:
            raise ValueError("The diagnostic must fit in 16 KiB.")
        report = unique_json(raw.removeprefix(b"\xef\xbb\xbf"))
        if (
            set(report) != REPORT_FIELDS
            or type(report["schema_version"]) is not int
            or report["schema_version"] != 1
        ):
            raise ValueError("Expected the seven fields of reconnect report version 1.")
        if (
            not re.fullmatch(r"[a-f0-9]{32}", report["installation_id"])
            or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}", report["release_id"])
            or report["host"] not in boot.HOSTS
            or report["stage"] not in ("connect", "discover_tools", "call_tool")
            or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", report["error_code"])
        ):
            raise ValueError("Invalid installation, release, host, stage or error code.")
        message = report["message"]
        if not isinstance(message, str) or not message.strip() or len(message) > 2048:
            raise ValueError("Supply a brief diagnostic message (at most 2048 characters).")
        if any(
            (ord(ch) < 32 and ch not in "\r\n\t") or 0xD800 <= ord(ch) <= 0xDFFF for ch in message
        ):
            raise ValueError("The diagnostic message contains an invalid control character.")
    except (OSError, TypeError, ValueError, RecursionError) as exc:
        raise boot.BootstrapError("INVALID_HOST_RECONNECT_REPORT", str(exc)) from exc

    code = report["error_code"]
    if (
        NON_FAILURE_TERMS.intersection(code.split("_"))
        or code in NON_FAILURE_CODES
        or report["stage"] == "call_tool"
        and code not in CALL_EXCHANGE_CODES
        and not CALL_CONNECTION_FAILURE.fullmatch(code)
    ):
        raise boot.BootstrapError(
            "HOST_RECONNECT_NOT_FAILED",
            "Report an observed failure of a new, authorized host connection.",
        )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return report, hashlib.sha256(canonical).hexdigest()


def verify_restored_assets(control, transaction, current, host):
    """Verify the previous adapter, including files that this upgrade did not modify."""
    current_owner = current["host_installations"][host]
    _, installed = uninstall._host_receipt(control, current_owner)
    if installed["host"] != host or (
        installed["state"] != "applied" and transaction["state"] != "reverting"
    ):
        raise boot.BootstrapError("INVALID_HOST_RECEIPT", "Review the upgraded host's receipt.")
    previous_host_receipt = transaction["previous"]["host_installations"][host]
    _, receipt = uninstall._host_receipt(control, previous_host_receipt)
    if receipt["host"] != host or receipt["state"] != "applied":
        raise boot.BootstrapError("INVALID_HOST_RECEIPT", "Review the predecessor's host receipt.")
    # The earliest backup of each changed file supplies its bytes after complete rollback.
    # rollback_host preflight already checks backup hashes and user edits to those files.
    restored = {}
    for owner in transaction["applied_assets"]:
        _, change = uninstall._host_receipt(control, owner)
        for row in change["files"]:
            if row["action"] != "unchanged":
                restored.setdefault(row["path"], row["before_sha256"])
    adapter = f"autotem-host/{host}/adapter.json"
    if not {adapter, ".autotem/host-entry.py"} <= {row["path"] for row in receipt["files"]}:
        raise boot.BootstrapError(
            "INVALID_HOST_RECEIPT", "The predecessor has no recorded managed adapter."
        )
    conflicts = []
    for index, row in enumerate(receipt["files"]):
        name = host_conflict.host_target(host, row["path"])
        expected = row["after_sha256"]
        if name in restored:
            if restored[name] != expected:
                raise boot.BootstrapError(
                    "INVALID_HOST_RECEIPT",
                    "The recorded upgrade would restore different host bytes.",
                )
            continue
        path = boot._inside(control.parent, name)
        if not path.is_file() or sha256(path) != expected:
            conflicts.append(
                host_conflict.describe(
                    control.parent,
                    name,
                    "recorded_file",
                    expected,
                    receipt=(previous_host_receipt, index, row),
                )
            )
    if conflicts:
        info, instruction = host_conflict.resolution("rollback", conflicts)
        raise host_conflict.HostFileConflict(
            {"conflict_resolution": info, "next_action": instruction}
        )
