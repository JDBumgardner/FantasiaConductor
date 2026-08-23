"""Tempo follow: BPM changes rescale MIDI and audio clip times."""

from __future__ import annotations

import numpy as np

from fantasia_core.commands import AddClipCommand, AddTrackCommand, CommandBus, SetTempoCommand
from fantasia_core.document import Note, Project, scale_timeline, source_span
from fantasia_core.engine.buffers import resample_to_length


def test_scale_timeline_midi_and_clips():
    p = Project(tempo=120.0)
    t = p.add_track("Pad")
    clip = p.add_clip(t.id, start=2.0, duration=4.0, name="c", content_type="midi")
    clip.notes = [
        Note(pitch=60, start=0.0, duration=1.0, velocity=100),
        Note(pitch=64, start=1.0, duration=0.5, velocity=80),
    ]
    clip.fade_in = 0.2
    scale_timeline(p, 0.5)  # 120 → 240
    assert clip.start == 1.0
    assert clip.duration == 2.0
    assert clip.fade_in == 0.1
    assert clip.notes[0].duration == 0.5
    assert clip.notes[1].start == 0.5


def test_scale_timeline_scales_loop_brace():
    p = Project(tempo=120.0, loop_start=2.0, loop_end=6.0)
    scale_timeline(p, 0.5)
    assert p.loop_start == 1.0
    assert p.loop_end == 3.0


def test_scale_preserves_audio_source_span():
    p = Project(tempo=120.0)
    t = p.add_track("A")
    clip = p.add_clip(t.id, start=0.0, duration=4.0, name="loop", source_path="/x.wav")
    assert clip.source_duration == 0.0
    scale_timeline(p, 0.5)
    assert clip.duration == 2.0
    assert clip.source_duration == 4.0
    assert source_span(clip) == 4.0
    scale_timeline(p, 2.0)  # back
    assert abs(clip.duration - 4.0) < 1e-9
    assert clip.source_duration == 4.0


def test_set_tempo_command_undo_redo():
    bus = CommandBus(Project(tempo=120.0))
    t = bus.dispatch(AddTrackCommand("M")).created_track
    c = bus.dispatch(AddClipCommand(t.id, 0.0, 2.0, "c", content_type="midi")).created_clip
    c.notes = [Note(60, 0.0, 2.0, 100)]
    bus.dispatch(SetTempoCommand(60.0))
    assert bus.project.tempo == 60.0
    assert abs(c.duration - 4.0) < 1e-9
    assert abs(c.notes[0].duration - 4.0) < 1e-9
    bus.undo()
    assert bus.project.tempo == 120.0
    assert abs(c.duration - 2.0) < 1e-9
    assert abs(c.notes[0].duration - 2.0) < 1e-9
    bus.redo()
    assert bus.project.tempo == 60.0
    assert abs(c.duration - 4.0) < 1e-9


def test_set_tempo_coalesce_undo_returns_to_original():
    bus = CommandBus(Project(tempo=120.0))
    t = bus.dispatch(AddTrackCommand()).created_track
    c = bus.dispatch(AddClipCommand(t.id, 4.0, 4.0, "c")).created_clip
    bus.dispatch(SetTempoCommand(130.0))
    bus.dispatch(SetTempoCommand(140.0))
    bus.dispatch(SetTempoCommand(240.0))
    assert len([x for x in bus._undo if x.merge_key() == ("set_tempo",)]) == 1
    assert abs(c.start - 2.0) < 1e-9
    assert abs(c.duration - 2.0) < 1e-9
    bus.undo()
    assert bus.project.tempo == 120.0
    assert abs(c.start - 4.0) < 1e-9
    assert abs(c.duration - 4.0) < 1e-9


def test_resample_to_length_changes_frame_count():
    x = np.ones((1000, 2), dtype=np.float32)
    y = resample_to_length(x, 500)
    assert y.shape == (500, 2)
    assert y.dtype == np.float32
    z = resample_to_length(y, 1000)
    assert z.shape == (1000, 2)
