"""Optional real MCP stdio transport; JSONL bridge remains dependency-free."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

from .cli import TOOL_DESCRIPTIONS, TOOL_SCHEMAS, dispatch_request


def build_server(harness):
    try:
        from mcp import types
        from mcp.server.lowlevel import Server
    except ImportError as exc:
        raise RuntimeError(
            "MCP SDK is unavailable; install autotem-insight[mcp] or use stdio JSONL."
        ) from exc

    server = Server("autotem-harness")

    @server.list_tools()
    async def list_tools():
        return [
            types.Tool(name=name, description=TOOL_DESCRIPTIONS[name], inputSchema=schema)
            for name, schema in TOOL_SCHEMAS.items()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None):
        import anyio

        response = await anyio.to_thread.run_sync(
            lambda: dispatch_request(harness, {"tool": name, "arguments": arguments or {}})
        )
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text", text=json.dumps(response, ensure_ascii=False, allow_nan=False)
                )
            ],
            structuredContent=response,
            isError=not response["ok"],
        )

    return server


def run_server(root: Path) -> None:
    try:
        import anyio
        from mcp.server.stdio import stdio_server
    except ImportError as exc:
        raise RuntimeError(
            "Install autotem-insight[mcp] for real MCP; JSONL stdio is also available."
        ) from exc
    from autotem.harness import Harness
    from autotem.harness.registry import default_registry

    # Load native scientific extensions on the main thread before requests move
    # into anyio workers. Cold Windows SciPy imports stalled when the first tool
    # call loaded these libraries in a worker. This registers baseline handlers;
    # it neither runs an analysis nor enables optional plugins/models.
    with contextlib.redirect_stdout(sys.stderr):
        default_registry()

    server = build_server(Harness(root))

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(run)
