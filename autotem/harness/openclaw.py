"""Build an opt-in, project-bound OpenClaw tool plugin from the published Harness contract."""

import json
from importlib.resources import files
from pathlib import Path

from .cli import TOOL_DESCRIPTIONS, TOOL_SCHEMAS

PLUGIN_ID = "autotem-harness"
PLUGIN_DIR = "autotem-host/openclaw/gateway-plugin"


def _json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def gateway_files(
    project: Path, jsonl_command: list[str], python_executable: str, harness_root: Path,
    *, managed: bool,
) -> dict[str, bytes]:
    tools = [
        {"name": name, "hostName": f"autotem__{name}", "description": TOOL_DESCRIPTIONS[name],
         "parameters": schema}
        for name, schema in TOOL_SCHEMAS.items()
    ]
    names = [tool["hostName"] for tool in tools]
    manifest = {
        "id": PLUGIN_ID, "name": "AutoTEM Harness",
        "description": "Project-local, opt-in TEM analysis tools.",
        "contracts": {"tools": names},
        "toolMetadata": {name: {"optional": True} for name in names},
        "configSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    }
    source = files("autotem.harness").joinpath("resources", "openclaw", "index.cjs")
    return {
        f"{PLUGIN_DIR}/index.cjs": source.read_bytes(),
        f"{PLUGIN_DIR}/package.json": _json({
            "name": PLUGIN_ID, "version": "1.0.0", "private": True,
            "openclaw": {"extensions": ["./index.cjs"]},
        }),
        f"{PLUGIN_DIR}/openclaw.plugin.json": _json(manifest),
        f"{PLUGIN_DIR}/project.json": _json({
            "project": str(project), "store": str(harness_root), "python": python_executable,
            "windows_base_python": jsonl_command[0] if managed else None,
            "runtime_id": Path(python_executable).parent.parent.name if managed else None,
            "tools": tools,
        }),
        "autotem-host/openclaw/gateway-config.example.json": _json({
            "plugins": {"load": {"paths": [str(project / PLUGIN_DIR)]},
                        "entries": {PLUGIN_ID: {"enabled": True}}},
            "tools": {"alsoAllow": names},
        }),
    }
