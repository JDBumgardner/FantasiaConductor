"""Backing the agent panel with Claude Code instead of an API key.

The CLI is not installed here, so the SDK is faked. What that can still pin
down is everything except the live call: availability reporting, how the DAW's
tools are reached, and how the SDK's messages become panel text.
"""

from __future__ import annotations

import pathlib

import pytest

from fantasia_core.agent import claude_code as cc


# ---- availability -------------------------------------------------------
def test_unavailable_message_names_what_to_install(monkeypatch):
    """'It failed' is not actionable; the message has to say the command."""
    monkeypatch.setattr(cc, "sdk_available", lambda: False)
    monkeypatch.setattr(cc, "find_cli", lambda: None)
    both = cc.why_unavailable()
    assert "claude-agent-sdk" in both and "@anthropic-ai/claude-code" in both

    monkeypatch.setattr(cc, "sdk_available", lambda: True)
    assert "@anthropic-ai/claude-code" in cc.why_unavailable()

    monkeypatch.setattr(cc, "sdk_available", lambda: False)
    monkeypatch.setattr(cc, "find_cli", lambda: "/usr/local/bin/claude")
    assert "claude-agent-sdk" in cc.why_unavailable()


def test_cli_is_found_off_path_too(tmp_path, monkeypatch):
    """Claude Code is often installed under ~/.claude rather than on PATH."""
    fake = tmp_path / "claude"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setattr(cc.shutil, "which", lambda _n: None)
    monkeypatch.setattr(cc, "_CLI_CANDIDATES", (str(fake),))
    assert cc.find_cli() == str(fake)


def test_running_without_a_backend_says_why_rather_than_crashing():
    s = cc.ClaudeCodeSession()
    if s.available():
        pytest.skip("Claude Code really is installed here")
    with pytest.raises(RuntimeError) as e:
        s.run("make a track", on_text=lambda _t: None)
    assert "claude" in str(e.value).lower()


# ---- the DAW's tools are reached through the existing MCP server ---------
def test_mcp_config_points_at_the_repo_server_not_a_second_tool_list():
    cfg = cc.mcp_config()
    assert cfg["args"][0].endswith("tools/mcp_server.py")
    assert pathlib.Path(cfg["args"][0]).exists()
    assert cfg["env"]["FANTASIA_BRIDGE_URL"].startswith("http://127.0.0.1")


def test_mcp_config_honours_a_relocated_bridge(monkeypatch):
    monkeypatch.setenv("FANTASIA_BRIDGE_URL", "http://127.0.0.1:9999")
    assert cc.mcp_config()["env"]["FANTASIA_BRIDGE_URL"] == "http://127.0.0.1:9999"


# ---- turning SDK messages into panel output -----------------------------
class _Block:
    def __init__(self, text=None, name=None, tool_input=None):
        if text is not None:
            self.text = text
        if name is not None:
            self.name, self.input = name, (tool_input or {})


class _Msg:
    def __init__(self, *blocks):
        self.content = list(blocks)


def _session(messages):
    return cc.ClaudeCodeSession(query=lambda prompt, options: iter(messages))


def test_assistant_text_reaches_the_panel_in_order():
    out = []
    s = _session([_Msg(_Block(text="Adding ")), _Msg(_Block(text="a bass track."))])
    final = s.run("add a bass", on_text=out.append)
    assert out == ["Adding ", "a bass track."]
    assert final == "Adding a bass track."


def test_tool_calls_are_reported_once_each():
    """The panel shows what the agent is doing; repeating a name each time it
    is used turns the transcript into noise."""
    notes = []
    s = _session([
        _Msg(_Block(name="mcp__fantasia__add_track", tool_input={})),
        _Msg(_Block(name="mcp__fantasia__add_track", tool_input={})),
        _Msg(_Block(name="mcp__fantasia__write_midi", tool_input={})),
        _Msg(_Block(text="done")),
    ])
    s.run("go", on_text=lambda _t: None, on_note=notes.append)
    assert notes == ["calling add_track…", "calling write_midi…"]


def test_a_plain_string_message_is_handled():
    out = []
    s = cc.ClaudeCodeSession(query=lambda prompt, options: iter([
        type("M", (), {"content": "hello"})()]))
    s.run("hi", on_text=out.append)
    assert out == ["hello"]


def test_the_conversation_is_kept():
    s = _session([_Msg(_Block(text="ok"))])
    s.run("first", on_text=lambda _t: None)
    assert s.messages == [{"role": "user", "content": "first"},
                          {"role": "assistant", "content": "ok"}]


def test_an_async_generator_from_the_sdk_is_driven_from_this_thread():
    """The real SDK is async; the panel's worker is an ordinary thread."""
    async def agen(prompt, options):
        yield _Msg(_Block(text="from "))
        yield _Msg(_Block(text="async"))

    out = []
    cc.ClaudeCodeSession(query=agen).run("x", on_text=out.append)
    assert out == ["from ", "async"]


def test_the_run_signature_matches_the_api_key_backend():
    """The panel calls whichever backend the same way."""
    import inspect

    from fantasia_core.agent.session import AgentSession

    a = set(inspect.signature(AgentSession.run).parameters)
    b = set(inspect.signature(cc.ClaudeCodeSession.run).parameters)
    assert a <= b, f"Claude Code backend is missing {a - b}"
