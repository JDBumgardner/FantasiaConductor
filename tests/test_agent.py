"""Agent tools dispatch + the tool-calling loop with a fake Claude client (M6)."""

from __future__ import annotations

import json
import types

from fantasia_core.agent import AgentSession, AgentTools
from fantasia_core.commands import AddClipCommand, AddTrackCommand, CommandBus
from fantasia_core.document import Project


def _tools() -> AgentTools:
    bus = CommandBus(Project(name="T"))
    return AgentTools(bus)


def test_query_tools():
    t = _tools()
    tr = t.bus.dispatch(AddTrackCommand("Drums")).created_track
    t.bus.dispatch(AddClipCommand(tr.id, 0.0, 2.0, "c"))
    assert t.execute("get_project", {})["num_tracks"] == 1
    tracks = t.execute("list_tracks", {})
    assert tracks[0]["name"] == "Drums"
    clips = t.execute("list_clips", {"track_id": tr.id})
    assert clips[0]["content_type"] == "empty"


def test_mutation_tools_go_through_bus_and_undo():
    t = _tools()
    r = t.execute("add_track", {"name": "Bass"})
    tid = r["track_id"]
    t.execute("set_track", {"track_id": tid, "is_synth": True, "gain_db": -4.0})
    assert t.bus.project.track_by_id(tid).is_synth is True
    assert t.bus.project.track_by_id(tid).gain_db == -4.0
    c = t.execute("add_clip", {"track_id": tid, "start": 0.0, "duration": 1.0})
    cid = c["clip_id"]
    t.execute("write_midi", {"clip_id": cid, "notes": [{"pitch": 60, "start": 0.0, "duration": 0.5}]})
    _, clip = t.bus.project.find_clip(cid)
    assert clip.is_midi and len(clip.notes) == 1
    # everything is undoable via the bus
    t.execute("undo", {})
    _, clip = t.bus.project.find_clip(cid)
    assert not clip.is_midi


def test_add_fx_and_synth_param():
    t = _tools()
    tid = t.execute("add_track", {})["track_id"]
    t.execute("add_fx", {"track_id": tid, "type": "reverb", "params": {"wet": 0.5}})
    assert t.bus.project.track_by_id(tid).fx[0]["type"] == "reverb"
    t.execute("set_track", {"track_id": tid, "is_synth": True})
    t.execute("set_synth_param", {"track_id": tid, "key": "cutoff", "value": 800})
    assert t.bus.project.track_by_id(tid).synth["cutoff"] == 800


def test_design_synth_patch():
    t = _tools()
    tid = t.execute("add_track", {})["track_id"]
    t.execute("set_track", {"track_id": tid, "is_synth": True})
    # design a warm pad; junk keys and bad values are dropped/coerced
    r = t.execute("set_synth_patch", {"track_id": tid, "patch": {
        "osc1": "saw", "osc2": "triangle", "attack": "0.5", "cutoff": 1500,
        "resonance": 0.2, "bogus": 9, "osc1_bad": "wobble"}})
    patch = t.bus.project.track_by_id(tid).synth
    assert patch["osc1"] == "saw" and patch["osc2"] == "triangle"
    assert patch["attack"] == 0.5 and patch["cutoff"] == 1500.0
    assert "bogus" not in patch and "osc1_bad" not in patch
    # returned patch is the full effective patch (defaults merged in)
    assert "sustain" in r["patch"]
    # undoable
    t.execute("undo", {})
    assert "osc1" not in t.bus.project.track_by_id(tid).synth


# ---- fake Claude client for the loop --------------------------------------
def _block(**kw):
    return types.SimpleNamespace(**kw)


class _FakeClient:
    """Scripts a sequence of responses; records tool_result ids sent back."""

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self._i = 0
        self.messages = self

    def create(self, **kwargs):
        resp = self._scripted[self._i]
        self._i += 1
        return resp


def _resp(stop_reason, content):
    return types.SimpleNamespace(stop_reason=stop_reason, content=content)


def test_loop_executes_tools_and_returns():
    t = _tools()
    # Turn 1: model calls add_track. Turn 2: it reports done.
    scripted = [
        _resp("tool_use", [
            _block(type="text", text="Adding a track."),
            _block(type="tool_use", id="tu1", name="add_track", input={"name": "Lead"}),
        ]),
        _resp("end_turn", [_block(type="text", text="Done — added a Lead track.")]),
    ]
    session = AgentSession(t, client=_FakeClient(scripted))
    texts = []
    final = session.run("add a lead track", on_text=texts.append, execute_tool=t.execute)
    assert len(t.bus.project.tracks) == 1
    assert t.bus.project.tracks[0].name == "Lead"
    assert final == "Done — added a Lead track."
    assert "Adding a track." in texts

    # The assistant + tool_result turns were recorded in history.
    roles = [m["role"] for m in session.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    tool_result = session.messages[2]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "tu1"
    assert json.loads(tool_result["content"])["name"] == "Lead"


def test_loop_stops_on_refusal():
    t = _tools()
    scripted = [_resp("refusal", [])]
    session = AgentSession(t, client=_FakeClient(scripted))
    out = session.run("...", on_text=lambda s: None, execute_tool=t.execute)
    assert out == "(declined)"


def _marker_positions(messages):
    """Indices of messages carrying a cache_control marker at call time."""
    hits = []
    for i, m in enumerate(messages):
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list) and any(
            isinstance(b, dict) and "cache_control" in b for b in content
        ):
            hits.append(i)
    return hits


class _RecordingClient(_FakeClient):
    def __init__(self, scripted):
        super().__init__(scripted)
        self.marker_snapshots = []

    def create(self, **kwargs):
        self.marker_snapshots.append(_marker_positions(kwargs["messages"]))
        return super().create(**kwargs)


def test_history_cache_marker_moves_to_newest_message():
    t = _tools()
    scripted = [
        _resp("tool_use", [
            _block(type="tool_use", id="tu1", name="add_track", input={}),
        ]),
        _resp("end_turn", [_block(type="text", text="done")]),
    ]
    client = _RecordingClient(scripted)
    session = AgentSession(t, client=client)
    session.run("add a track", on_text=lambda s: None, execute_tool=t.execute)
    # Call 1: marker on the user message (idx 0). Call 2: only on the newest
    # message (the tool_result at idx 2) — exactly one marker per call.
    assert client.marker_snapshots == [[0], [2]]
