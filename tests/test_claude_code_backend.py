"""Backing the agent panel with Claude Code instead of an API key.

The CLI is not installed here, so the SDK is faked. What that can still pin
down is everything except the live call: availability reporting, how the DAW's
tools are reached, and how the SDK's messages become panel text.
"""

from __future__ import annotations

import os
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


# ---- the plugin notes reach the agent ------------------------------------
def test_the_measured_parameter_rules_are_in_the_system_prompt():
    """This session is restricted to the DAW's tools, so it cannot read the
    notes itself. Without them it repeats the mistakes they were written to
    prevent — a plugin accepts a value, reports success, stores something else."""
    brief = cc.plugin_notes_brief()
    assert brief, "no rules folded in"
    for rule in ("envelope_times", "percent_params", "filter_cutoff_ceiling",
                 "choice_and_switch"):
        assert rule in brief, f"{rule} missing from the prompt"
    assert "32 * raw^4" in brief          # the one that made every note a click
    assert "sqrt" in brief                # levels and detune read back squared


def test_the_prompt_names_where_the_full_catalogue_lives():
    brief = cc.plugin_notes_brief()
    assert "plugin_notes" in brief
    assert "vital_params.json" in brief


def test_the_brief_stays_small_enough_to_prepend_every_turn():
    """It is sent with every request, so it cannot be the 170KB catalogue."""
    assert len(cc.plugin_notes_brief()) < 6000


def test_missing_notes_degrade_to_no_brief_rather_than_failing(tmp_path):
    assert cc.plugin_notes_brief(repo_root=tmp_path) == ""


def test_the_session_cannot_edit_files_or_run_shell():
    """A session spawned from inside the DAW should reach the DAW's tools and
    nothing else."""
    opts = cc.ClaudeCodeSession()._options()
    allowed = getattr(opts, "allowed_tools", None) or opts.get("allowed_tools")
    denied = getattr(opts, "disallowed_tools", None) or opts.get("disallowed_tools")
    assert allowed == ["mcp__fantasia"]
    for t in ("Bash", "Write", "Edit"):
        assert t in denied


# ---- the API key must not hijack the subscription ------------------------
def test_api_auth_is_removed_while_a_session_runs(monkeypatch):
    """The app loads a saved ANTHROPIC_API_KEY into its own environment for the
    other backend. A child inherits it, and Claude Code then bills the key
    instead of the subscription — reporting "Credit balance is too low", which
    is the exact opposite of why this backend exists.

    Blanking is not enough: the SDK merges its env over the inherited one and
    the CLI treats an empty value as still set, so the variable has to leave
    the parent for the duration of the call.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-something")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")

    seen = {}

    def fake_query(prompt, options):
        seen["key"] = os.environ.get("ANTHROPIC_API_KEY")
        seen["base"] = os.environ.get("ANTHROPIC_BASE_URL")
        return iter([type("M", (), {"content": "ok"})()])

    cc.ClaudeCodeSession(query=fake_query).run("hi", on_text=lambda _t: None)
    assert seen["key"] is None, "the API key reached the spawned session"
    assert seen["base"] is None
    # and it is put back, because the other backend needs it
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-something"
    assert os.environ["ANTHROPIC_BASE_URL"] == "https://example.invalid"


def test_the_key_is_restored_even_when_the_session_fails(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-something")

    def boom(prompt, options):
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        cc.ClaudeCodeSession(query=boom).run("hi", on_text=lambda _t: None)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-something"


def test_credit_balance_error_is_explained_as_the_key_taking_over():
    """The raw message blames a balance; the cause is an API key overriding the
    subscription, so the explanation should point there rather than echo it."""
    msg = cc.explain(RuntimeError(
        "Claude Code returned an error result: Credit balance is too low")).lower()
    assert "api key" in msg and "subscription" in msg
    assert "anthropic_" in msg, "should name the kind of variable to look for"
