"""Back the agent panel with a Claude Code session instead of an API key.

The panel's default backend (:mod:`fantasia_core.agent.session`) talks to the
Anthropic API and bills per token against a key in ``.fantasia_cache``. This
backend runs a headless Claude Code session instead, so the work goes against
the host subscription — the same billing the MCP path already gets when Claude
Code drives the app from outside.

The DAW tools are not re-declared here. Claude Code reaches them through
``tools/mcp_server.py``, which forwards to the running app's control bridge, so
there is exactly one definition of what the agent can do. That also means tool
calls do not come back through ``execute_tool``: they arrive at the bridge as
ordinary HTTP requests and are marshalled onto the UI thread there, the same as
any other MCP client.

Two consequences worth knowing:

* The app must be running with its bridge up, because the tools are reached
  through it. Driving the app from inside the app is a loop, but a legal one —
  the panel's worker thread never blocks the UI thread, so the bridge can still
  service the calls.
* Each request is a fresh session. It shares authentication with Claude Code
  but not the conversation of any session you have open elsewhere.
"""

from __future__ import annotations

import os
import pathlib
import shutil
from typing import Callable, Iterable, Optional

# Where the CLI usually lives when it is not simply on PATH.
_CLI_CANDIDATES = (
    "~/.claude/local/claude",
    "~/.local/bin/claude",
    "/usr/local/bin/claude",
    "/opt/homebrew/bin/claude",
)


def find_cli() -> Optional[str]:
    """The Claude Code executable, or None."""
    found = shutil.which("claude")
    if found:
        return found
    for cand in _CLI_CANDIDATES:
        p = pathlib.Path(cand).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def sdk_available() -> bool:
    try:
        import claude_agent_sdk  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def available() -> bool:
    return sdk_available() and find_cli() is not None


def why_unavailable() -> str:
    """A sentence that says what to install, not merely that something failed."""
    if not sdk_available() and find_cli() is None:
        return ("Claude Code backend needs both pieces — install the CLI with "
                "`npm i -g @anthropic-ai/claude-code` and the bridge with "
                "`pip install claude-agent-sdk`.")
    if not sdk_available():
        return "Claude Code backend needs the SDK — run: pip install claude-agent-sdk"
    return ("Claude Code backend needs the CLI — run: "
            "npm i -g @anthropic-ai/claude-code")


def mcp_config(repo_root: Optional[str] = None) -> dict:
    """How to spawn the DAW's MCP server, as the SDK wants it.

    Deliberately the same server the repo registers in ``.mcp.json`` rather than
    a second copy of the tool list.
    """
    root = pathlib.Path(repo_root or pathlib.Path(__file__).resolve().parents[2])
    venv_py = root / ".venv" / "bin" / "python"
    return {
        "command": str(venv_py if venv_py.exists() else "python3"),
        "args": [str(root / "tools" / "mcp_server.py")],
        "env": {"FANTASIA_BRIDGE_URL":
                os.environ.get("FANTASIA_BRIDGE_URL", "http://127.0.0.1:8765")},
    }


SYSTEM = (
    "You are composing inside Fantasia Conductor, a running DAW. Use the "
    "fantasia tools to make changes; they go through the app's command bus, so "
    "everything you do is undoable. Prefer small, checkable steps, and say what "
    "you changed."
)


def _text_of(message) -> Iterable[str]:
    """Assistant text out of one SDK message, whatever shape it arrives in."""
    blocks = getattr(message, "content", None)
    if isinstance(blocks, str):
        yield blocks
        return
    for block in blocks or ():
        text = getattr(block, "text", None)
        if text:
            yield text


def _tool_name_of(message) -> Optional[str]:
    for block in getattr(message, "content", None) or ():
        if getattr(block, "name", None) and hasattr(block, "input"):
            return str(block.name)
    return None


class ClaudeCodeSession:
    """Same surface as :class:`AgentSession` so the panel does not care which."""

    def __init__(self, repo_root: Optional[str] = None,
                 model: Optional[str] = None, query=None) -> None:
        self.repo_root = repo_root
        self.model = model
        self._query = query          # injectable, so this is testable without the CLI
        self.messages: list = []

    def available(self) -> bool:
        return self._query is not None or available()

    def run(self, user_message: str,
            on_text: Callable[[str], None],
            execute_tool: Optional[Callable[[str, dict], object]] = None,
            on_usage: Optional[Callable[[dict], None]] = None,
            on_note: Optional[Callable[[str], None]] = None) -> str:
        """Run one request. ``execute_tool`` is unused: tools arrive over MCP.

        Kept in the signature so the panel can call either backend the same way.
        """
        if not self.available():
            raise RuntimeError(why_unavailable())
        query = self._query
        if query is None:
            from claude_agent_sdk import query as sdk_query  # noqa: PLC0415

            query = sdk_query

        options = {
            "system_prompt": SYSTEM,
            "mcp_servers": {"fantasia": mcp_config(self.repo_root)},
            "allowed_tools": ["mcp__fantasia"],
            "permission_mode": "acceptEdits",
        }
        if self.model:
            options["model"] = self.model

        self.messages.append({"role": "user", "content": user_message})
        collected: list = []
        seen_tools: set = set()
        for message in _iterate(query, user_message, options):
            for chunk in _text_of(message):
                collected.append(chunk)
                on_text(chunk)
            name = _tool_name_of(message)
            if name and name not in seen_tools and on_note is not None:
                seen_tools.add(name)
                on_note(f"calling {name.replace('mcp__fantasia__', '')}…")
        final = "".join(collected)
        self.messages.append({"role": "assistant", "content": final})
        return final


def _iterate(query, prompt: str, options: dict):
    """Drive the SDK's async generator from this synchronous worker thread."""
    import asyncio
    import inspect

    result = query(prompt=prompt, options=options)
    if not inspect.isasyncgen(result):
        yield from result
        return

    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                yield loop.run_until_complete(result.__anext__())
            except StopAsyncIteration:
                return
    finally:
        loop.close()
