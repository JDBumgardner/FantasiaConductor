"""MCP server for Fantasia Conductor — lets Claude Code, Cursor, or any MCP
client drive the running DAW.

Forwards to the app's local control bridge (``fantasia_core/bridge.py``), so the
app must be running. Tool schemas are fetched live from the app — the MCP tool
list always matches the in-app agent's tools, with no duplication here.

Register once (``.mcp.json`` for Claude Code, ``.cursor/mcp.json`` for Cursor),
then an agent session in this repo can call fantasia tools directly — billed to
the host subscription instead of a separate pay-per-token API key.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request

BRIDGE = os.environ.get("FANTASIA_BRIDGE_URL", "http://127.0.0.1:8765")
_TIMEOUT = 600  # generation/separation can take minutes on CPU


# ---- bridge client (plain functions so they're testable) -----------------
def fetch_tools() -> list:
    with urllib.request.urlopen(f"{BRIDGE}/tools", timeout=10) as r:
        return json.loads(r.read())


def call_tool_http(name: str, args: dict) -> dict:
    req = urllib.request.Request(
        f"{BRIDGE}/call",
        data=json.dumps({"name": name, "args": args or {}}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read())


def _not_running_hint(exc: Exception) -> str:
    return (f"Fantasia Conductor doesn't appear to be running (bridge {BRIDGE} "
            f"unreachable: {exc}). Launch the app first: .venv/bin/python app.py")


# ---- MCP wiring ----------------------------------------------------------
def build_server():
    import mcp.types as types
    from mcp.server import Server

    server = Server("fantasia-conductor")

    @server.list_tools()
    async def list_tools() -> list:
        try:
            defs = await asyncio.to_thread(fetch_tools)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(_not_running_hint(exc)) from exc
        return [
            types.Tool(name=d["name"], description=d.get("description", ""),
                       inputSchema=d.get("input_schema", {"type": "object"}))
            for d in defs
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list:
        try:
            out = await asyncio.to_thread(call_tool_http, name, arguments or {})
        except Exception as exc:  # noqa: BLE001
            out = {"error": _not_running_hint(exc)}
        return [types.TextContent(type="text", text=json.dumps(out, default=str))]

    return server


async def _amain() -> None:
    from mcp.server.stdio import stdio_server

    server = build_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_amain())
