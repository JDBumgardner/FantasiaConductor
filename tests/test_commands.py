"""Command bus: undo/redo correctness and coalescing (M2)."""

from __future__ import annotations

from fantasia_core.commands import (
    AddClipCommand,
    AddTrackCommand,
    CommandBus,
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
    assert len(t.fx) == 1 and t.fx[0]["type"] == "reverb"
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
