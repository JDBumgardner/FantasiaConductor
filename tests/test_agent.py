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
    ins = t.bus.project.track_by_id(tid).fx[0]
    assert ins.type == "reverb" and ins.id
    t.execute("set_track", {"track_id": tid, "is_synth": True})
    t.execute("set_synth_param", {"track_id": tid, "key": "cutoff", "value": 800})
    assert t.bus.project.track_by_id(tid).synth["cutoff"] == 800


def test_stock_eq_band_and_master_channel():
    from fantasia_core.document import MASTER_ID

    t = _tools()
    tid = t.execute("add_track", {})["track_id"]
    r = t.execute("set_eq_band", {"track_id": tid, "band": 3, "gain": 4.5, "freq": 800})
    assert r["ok"] is True
    eq = t.execute("get_eq", {"track_id": tid})
    assert eq["bands"][2]["gain"] == 4.5
    assert eq["bands"][2]["freq"] == 800
    tracks = t.execute("list_tracks", {})
    assert tracks[-1]["id"] == MASTER_ID and tracks[-1]["is_master"] is True
    t.execute("add_fx", {"track_id": MASTER_ID, "type": "eq"})
    assert t.bus.project.master.fx[0].type == "eq"
    assert t.bus.project.master.fx[0].id
    assert t.execute("remove_track", {"track_id": MASTER_ID})["error"]
    assert t.execute("add_clip", {"track_id": MASTER_ID, "start": 0, "duration": 1})["error"]


def test_insert_graph_tools_address_by_id():
    t = _tools()
    tid = t.execute("add_track", {})["track_id"]
    a = t.execute("add_fx", {"track_id": tid, "type": "reverb", "params": {"wet": 0.4}})
    b = t.execute("add_fx", {"track_id": tid, "type": "delay"})
    assert a["insert_id"] and b["insert_id"]
    listed = t.execute("list_fx", {"track_id": tid})
    assert [row["type"] for row in listed] == ["reverb", "delay"]
    tracks = t.execute("list_tracks", {})
    assert tracks[0]["fx"][0] == {
        "id": a["insert_id"], "type": "reverb", "bypassed": False}
    t.execute("move_fx", {"track_id": tid, "insert_id": b["insert_id"], "index": 0})
    assert [row["type"] for row in t.execute("list_fx", {"track_id": tid})] == [
        "delay", "reverb"]
    t.execute("bypass_fx", {
        "track_id": tid, "insert_id": a["insert_id"], "bypassed": True})
    assert t.execute("list_fx", {"track_id": tid})[1]["bypassed"] is True
    t.execute("remove_fx", {"track_id": tid, "insert_id": a["insert_id"]})
    remaining = t.execute("list_fx", {"track_id": tid})
    assert [row["id"] for row in remaining] == [b["insert_id"]]
    names = {d["name"] for d in t.definitions()}
    assert {"list_fx", "bypass_fx", "move_fx", "remove_fx"} <= names


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


def test_bar_beat_notes_are_rebased_to_the_clip():
    """bar/beat are absolute song positions; a clip starting at bar 5 must map
    'bar 5 beat 1' to its own time 0 (this is what silently ate whole sections)."""
    t = _tools()
    t.bus.project.tempo = 120.0          # 0.5 s/beat, 4/4 -> 2 s per bar
    tid = t.execute("add_track", {})["track_id"]
    r = t.execute("add_clip", {"track_id": tid, "bar": 5, "bars": 2})
    assert r["bar"] == 5 and r["bars"] == 2.0
    cid = r["clip_id"]
    _, clip = t.bus.project.find_clip(cid)
    assert clip.start == 8.0 and clip.duration == 4.0

    t.execute("write_midi", {"clip_id": cid, "notes": [
        {"pitch": 60, "bar": 5, "beat": 1, "beats": 1},       # clip-relative 0.0
        {"pitch": 62, "bar": 5, "beat": 2.5, "beats": 0.5},   # 'and' of 2 -> 0.75
        {"pitch": 64, "bar": 6, "beat": 1, "beats": 2},       # next bar -> 2.0
    ]})
    _, clip = t.bus.project.find_clip(cid)
    assert [round(n.start, 3) for n in clip.notes] == [0.0, 0.75, 2.0]
    assert [round(n.duration, 3) for n in clip.notes] == [0.5, 0.25, 1.0]
    # every note sits inside the clip, so nothing gets dropped at render time
    assert all(0 <= n.start < clip.duration for n in clip.notes)


def test_notes_outside_the_clip_are_reported_not_dropped():
    t = _tools()
    t.bus.project.tempo = 120.0
    tid = t.execute("add_track", {})["track_id"]
    cid = t.execute("add_clip", {"track_id": tid, "bar": 1, "bars": 1})["clip_id"]
    out = t.execute("write_midi", {"clip_id": cid, "notes": [
        {"pitch": 60, "bar": 1, "beat": 1, "beats": 1},
        {"pitch": 62, "bar": 9, "beat": 1, "beats": 1},   # way past this 1-bar clip
    ]})
    assert "error" in out and "outside" in out["error"]
    assert "bars 1-1" in out["error"]


def test_seconds_form_still_works():
    t = _tools()
    tid = t.execute("add_track", {})["track_id"]
    cid = t.execute("add_clip", {"track_id": tid, "start": 0.0, "duration": 2.0})["clip_id"]
    t.execute("write_midi", {"clip_id": cid, "notes": [
        {"pitch": 60, "start": 0.5, "duration": 0.25}]})
    _, clip = t.bus.project.find_clip(cid)
    assert clip.notes[0].start == 0.5 and clip.notes[0].duration == 0.25


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


# ---- FX routing ---------------------------------------------------------
def _fx_track(t):
    """A track with three inserts, so there is something to wire."""
    tr = t.bus.dispatch(AddTrackCommand("routing")).created_track
    ids = [t.execute("add_fx", {"track_id": tr.id, "type": k})["insert_id"]
           for k in ("reverb", "delay", "chorus")]
    return tr.id, ids


def test_routing_starts_serial_and_reports_its_nodes():
    t = _tools()
    tid, ids = _fx_track(t)
    got = t.execute("get_fx_routing", {"track_id": tid})
    assert got["serial"] is True
    assert got["wires"] == []            # empty means the implicit serial chain
    assert got["nodes"] == ["in", *ids, "out"]


def test_parallel_wet_dry_routing_is_not_serial():
    """The point of the tool: a dry path alongside a wet one."""
    t = _tools()
    tid, (rev, _dly, _cho) = _fx_track(t)
    res = t.execute("set_fx_routing", {"track_id": tid, "wires": [
        {"src": "in", "dst": rev}, {"src": "in", "dst": "out"},
        {"src": rev, "dst": "out"}]})
    assert res["ok"] and res["serial"] is False
    assert {(w["src"], w["dst"]) for w in res["wires"]} == {
        ("in", rev), ("in", "out"), (rev, "out")}


def test_a_cycle_is_refused_rather_than_stored():
    t = _tools()
    tid, (a, b, _c) = _fx_track(t)
    res = t.execute("set_fx_routing", {"track_id": tid, "wires": [
        {"src": "in", "dst": a}, {"src": a, "dst": b}, {"src": b, "dst": a}]})
    assert "error" in res and "cycle" in res["error"]
    assert t.execute("get_fx_routing", {"track_id": tid})["wires"] == []


def test_wires_naming_a_missing_insert_are_dropped_not_rejected():
    t = _tools()
    tid, (rev, *_rest) = _fx_track(t)
    res = t.execute("set_fx_routing", {"track_id": tid, "wires": [
        {"src": "in", "dst": rev}, {"src": "fx_gone", "dst": "out"},
        {"src": rev, "dst": "out"}]})
    assert res["ok"] and res["dropped"] == 1
    assert len(res["wires"]) == 2


def test_empty_wires_restores_the_serial_chain():
    t = _tools()
    tid, (rev, *_rest) = _fx_track(t)
    t.execute("set_fx_routing", {"track_id": tid, "wires": [
        {"src": "in", "dst": rev}, {"src": "in", "dst": "out"},
        {"src": rev, "dst": "out"}]})
    res = t.execute("set_fx_routing", {"track_id": tid, "wires": []})
    assert res["serial"] is True and res["wires"] == []


def test_routing_is_undoable():
    t = _tools()
    tid, (rev, *_rest) = _fx_track(t)
    t.execute("set_fx_routing", {"track_id": tid, "wires": [
        {"src": "in", "dst": rev}, {"src": "in", "dst": "out"},
        {"src": rev, "dst": "out"}]})
    assert t.execute("get_fx_routing", {"track_id": tid})["serial"] is False
    t.execute("undo", {})
    assert t.execute("get_fx_routing", {"track_id": tid})["serial"] is True
