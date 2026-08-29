"""Command bus: undo/redo correctness and coalescing (M2)."""

from __future__ import annotations

from fantasia_core.commands import (
    AddClipCommand,
    AddTrackCommand,
    CommandBus,
    DuplicateClipsCommand,
    RemoveClipCommand,
    RemoveTrackCommand,
    SetClipAttrCommand,
    SetClipGeometryCommand,
    SetClipSourceCommand,
    SetTrackAttrCommand,
    SetTrackFxCommand,
    SplitClipCommand,
)
from fantasia_core.document import Project


def _bus() -> CommandBus:
    return CommandBus(Project(name="T"))


def test_add_track_undo_redo_keeps_id():
    bus = _bus()
    cmd = bus.dispatch(AddTrackCommand("Drums"))
    tid = cmd.created_track.id
    assert len(bus.project.tracks) == 1
    bus.undo()
    assert bus.project.tracks == []
    bus.redo()
    assert len(bus.project.tracks) == 1
    assert bus.project.tracks[0].id == tid  # id stable across redo


def test_remove_track_restores_contents():
    bus = _bus()
    t = bus.dispatch(AddTrackCommand("Bass")).created_track
    bus.dispatch(AddClipCommand(t.id, 0.0, 2.0, "c"))
    bus.dispatch(RemoveTrackCommand(t.id))
    assert bus.project.tracks == []
    bus.undo()  # undo remove
    assert len(bus.project.tracks) == 1
    assert len(bus.project.tracks[0].clips) == 1


def test_add_clip_undo_redo():
    bus = _bus()
    t = bus.dispatch(AddTrackCommand()).created_track
    cmd = bus.dispatch(AddClipCommand(t.id, 1.0, 3.0, "loop"))
    cid = cmd.created_clip.id
    assert len(t.clips) == 1
    bus.undo()
    assert t.clips == []
    bus.redo()
    assert t.clips[0].id == cid  # same clip id after redo


def test_remove_clip_restores_position():
    bus = _bus()
    t = bus.dispatch(AddTrackCommand()).created_track
    bus.dispatch(AddClipCommand(t.id, 0.0, 1.0, "a"))
    c2 = bus.dispatch(AddClipCommand(t.id, 1.0, 1.0, "b")).created_clip
    bus.dispatch(AddClipCommand(t.id, 2.0, 1.0, "c"))
    bus.dispatch(RemoveClipCommand(c2.id))
    assert [c.name for c in t.clips] == ["a", "c"]
    bus.undo()
    assert [c.name for c in t.clips] == ["a", "b", "c"]  # restored at index 1


def test_set_clip_geometry_undo():
    bus = _bus()
    t = bus.dispatch(AddTrackCommand()).created_track
    c = bus.dispatch(AddClipCommand(t.id, 0.0, 2.0, "x")).created_clip
    bus.dispatch(SetClipGeometryCommand(c.id, 5.0, 3.5))
    assert (c.start, c.duration) == (5.0, 3.5)
    bus.undo()
    assert (c.start, c.duration) == (0.0, 2.0)


def test_resize_audio_freezes_source_span():
    bus = _bus()
    t = bus.dispatch(AddTrackCommand()).created_track
    c = bus.dispatch(AddClipCommand(t.id, 0.0, 4.0, "loop", source_path="/x.wav")).created_clip
    assert c.source_duration == 0.0
    bus.dispatch(SetClipGeometryCommand(c.id, 0.0, 2.0))
    assert c.duration == 2.0
    assert c.source_duration == 4.0
    bus.undo()
    assert c.duration == 4.0
    assert c.source_duration == 0.0


def test_fill_empty_clip_and_undo():
    bus = _bus()
    t = bus.dispatch(AddTrackCommand()).created_track
    c = bus.dispatch(AddClipCommand(t.id, 0.0, 2.0, "slot")).created_clip
    assert c.source_path is None  # empty container
    bus.dispatch(SetClipSourceCommand(c.id, "/x/loop.wav", 0.0, 4.0))
    assert c.source_path == "/x/loop.wav" and c.duration == 4.0
    bus.undo()  # back to empty, original duration restored
    assert c.source_path is None and c.duration == 2.0


def test_split_clip_and_undo():
    bus = _bus()
    t = bus.dispatch(AddTrackCommand()).created_track
    c = bus.dispatch(
        AddClipCommand(t.id, 1.0, 4.0, "loop", source_path="/x.wav", source_offset=0.5)
    ).created_clip
    bus.dispatch(SplitClipCommand(c.id, 3.0))  # split at t=3 (2s into the clip)
    assert len(t.clips) == 2
    left, right = t.clips
    assert left is c
    assert (left.start, left.duration) == (1.0, 2.0)
    assert (right.start, right.duration) == (3.0, 2.0)
    # Right clip reads 2s further into the source.
    assert abs(right.source_offset - (0.5 + 2.0)) < 1e-9
    bus.undo()
    assert len(t.clips) == 1
    assert t.clips[0].duration == 4.0


def test_split_outside_clip_is_noop():
    bus = _bus()
    t = bus.dispatch(AddTrackCommand()).created_track
    c = bus.dispatch(AddClipCommand(t.id, 0.0, 2.0, "c")).created_clip
    bus.dispatch(SplitClipCommand(c.id, 5.0))  # playhead past the clip
    assert len(t.clips) == 1


def test_clip_attr_gain_undo():
    bus = _bus()
    t = bus.dispatch(AddTrackCommand()).created_track
    c = bus.dispatch(AddClipCommand(t.id, 0.0, 2.0, "c")).created_clip
    bus.dispatch(SetClipAttrCommand(c.id, "reversed", True))
    assert c.reversed is True
    bus.dispatch(SetClipAttrCommand(c.id, "fade_in", 0.25))
    assert c.fade_in == 0.25
    bus.undo()
    assert c.fade_in == 0.0
    bus.undo()
    assert c.reversed is False


def test_track_fx_set_and_undo():
    bus = _bus()
    t = bus.dispatch(AddTrackCommand()).created_track
    assert t.fx == []
    bus.dispatch(SetTrackFxCommand(t.id, [{"type": "reverb", "params": {"wet": 0.4}}]))
    assert len(t.fx) == 1 and t.fx[0].type == "reverb" and t.fx[0].id
    bus.dispatch(SetTrackFxCommand(t.id, t.fx + [{"type": "delay", "params": {}}]))
    assert len(t.fx) == 2
    bus.undo()
    assert len(t.fx) == 1
    bus.undo()
    assert t.fx == []  # restored to empty (stored copy, not aliased)


def test_pitch_via_clip_attr():
    bus = _bus()
    t = bus.dispatch(AddTrackCommand()).created_track
    c = bus.dispatch(AddClipCommand(t.id, 0.0, 2.0, "c")).created_clip
    bus.dispatch(SetClipAttrCommand(c.id, "pitch_semitones", 7.0))
    assert c.pitch_semitones == 7.0
    bus.undo()
    assert c.pitch_semitones == 0.0


def test_set_attr_and_toggle():
    bus = _bus()
    t = bus.dispatch(AddTrackCommand()).created_track
    bus.dispatch(SetTrackAttrCommand(t.id, "mute", True))
    assert t.mute is True
    bus.undo()
    assert t.mute is False


def test_mergeable_slider_is_one_undo():
    bus = _bus()
    t = bus.dispatch(AddTrackCommand()).created_track
    # Simulate a volume slider drag: many changes, all mergeable.
    for v in (-1.0, -2.0, -3.0, -6.0):
        bus.dispatch(SetTrackAttrCommand(t.id, "gain_db", v, mergeable=True))
    assert t.gain_db == -6.0
    bus.undo()  # single undo returns to the pre-drag value
    assert t.gain_db == 0.0


def test_eq_drag_is_one_undo():
    bus = _bus()
    t = bus.dispatch(AddTrackCommand()).created_track
    from fantasia_core.engine.eq import default_bands, fx_with_eq

    bands = default_bands()
    bus.dispatch(SetTrackFxCommand(t.id, fx_with_eq([], bands), label="EQ"))
    for g in (1.0, 2.0, 3.5):
        bands[2]["gain"] = g
        bus.dispatch(SetTrackFxCommand(
            t.id, fx_with_eq(t.fx, bands), label="EQ", mergeable=True))
    assert t.fx[0].params["bands"][2]["gain"] == 3.5
    bus.undo()  # coalesced drag
    assert t.fx[0].params["bands"][2]["gain"] == 0.0


def test_non_mergeable_toggles_are_separate():
    bus = _bus()
    t = bus.dispatch(AddTrackCommand()).created_track
    bus.dispatch(SetTrackAttrCommand(t.id, "mute", True))
    bus.dispatch(SetTrackAttrCommand(t.id, "mute", False))
    bus.undo()
    assert t.mute is True  # only the last toggle undone
    bus.undo()
    assert t.mute is False


def test_dispatch_clears_redo():
    bus = _bus()
    bus.dispatch(AddTrackCommand("A"))
    bus.undo()
    assert bus.can_redo
    bus.dispatch(AddTrackCommand("B"))  # new edit invalidates redo
    assert not bus.can_redo


def test_set_project_clears_history():
    bus = _bus()
    bus.dispatch(AddTrackCommand("A"))
    bus.set_project(Project(name="fresh"))
    assert not bus.can_undo and not bus.can_redo
    assert bus.project.tracks == []


def test_duplicate_clips_places_after_selection_span():
    bus = _bus()
    t1 = bus.dispatch(AddTrackCommand("A")).created_track
    t2 = bus.dispatch(AddTrackCommand("B")).created_track
    a = bus.dispatch(AddClipCommand(t1.id, 0.0, 2.0, "a", content_type="midi")).created_clip
    b = bus.dispatch(AddClipCommand(t2.id, 1.0, 2.0, "b", source_path="/x.wav")).created_clip
    cmd = bus.dispatch(DuplicateClipsCommand([a.id, b.id]))
    assert len(cmd.created_ids) == 2
    copies = [bus.project.find_clip(cid)[1] for cid in cmd.created_ids]
    # Selection spans 0..3, so copies start 3 seconds later on the same tracks.
    assert copies[0].start == 3.0 and copies[0].duration == 2.0
    assert copies[1].start == 4.0 and copies[1].duration == 2.0
    assert copies[0].content_type == "midi"
    assert copies[1].source_path == "/x.wav"
    bus.undo()
    assert all(bus.project.find_clip(cid)[1] is None for cid in cmd.created_ids)
    bus.redo()
    assert all(bus.project.find_clip(cid)[1] is not None for cid in cmd.created_ids)


def test_add_bypass_move_remove_fx():
    from fantasia_core.commands import (
        AddFxCommand, BypassFxCommand, MoveFxCommand, RemoveFxCommand,
    )

    bus = _bus()
    t = bus.dispatch(AddTrackCommand()).created_track
    a = bus.dispatch(AddFxCommand(t.id, "reverb", {"wet": 0.4}))
    b = bus.dispatch(AddFxCommand(t.id, "delay", {}))
    assert a.insert_id and b.insert_id and a.insert_id != b.insert_id
    assert [e.type for e in t.fx] == ["reverb", "delay"]
    bus.dispatch(MoveFxCommand(t.id, b.insert_id, 0))
    assert [e.type for e in t.fx] == ["delay", "reverb"]
    bus.dispatch(BypassFxCommand(t.id, a.insert_id, True))
    assert t.fx_by_id(a.insert_id).bypassed is True
    bus.dispatch(RemoveFxCommand(t.id, a.insert_id))
    assert [e.type for e in t.fx] == ["delay"]
    bus.undo()
    assert [e.type for e in t.fx] == ["delay", "reverb"]
    bus.undo()
    assert t.fx_by_id(a.insert_id).bypassed is False
    bus.undo()
    assert [e.type for e in t.fx] == ["reverb", "delay"]
    bus.undo()
    assert [e.type for e in t.fx] == ["reverb"]
    bus.undo()
    assert t.fx == []
    bus.redo()
    assert t.fx[0].id == a.insert_id


def test_remove_fx_reconnects_explicit_wires():
    from fantasia_core.commands import AddFxCommand, ConnectFxCommand, RemoveFxCommand
    from fantasia_core.document.fx_insert import SOURCE, OUT, serial_wires

    bus = _bus()
    t = bus.dispatch(AddTrackCommand()).created_track
    a = bus.dispatch(AddFxCommand(t.id, "reverb"))
    b = bus.dispatch(AddFxCommand(t.id, "delay"))
    c = bus.dispatch(AddFxCommand(t.id, "gain"))
    t.fx_wires = serial_wires(t.fx)
    bus.dispatch(RemoveFxCommand(t.id, b.insert_id))
    keys = {(w.src, w.dst) for w in t.fx_wires}
    assert (a.insert_id, c.insert_id) in keys
    assert not any(b.insert_id in (w.src, w.dst) for w in t.fx_wires)
    bus.dispatch(ConnectFxCommand(t.id, SOURCE, a.insert_id, True))  # already present
    bus.dispatch(ConnectFxCommand(t.id, a.insert_id, OUT, False))
    # disconnect a→out (if present); graph still has a→c→out
    assert any(w.dst == OUT for w in t.fx_wires)
