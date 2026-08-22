"""Ableton-style MIDI note operations used by the piano roll (no Qt)."""

from __future__ import annotations

from fantasia_core.document.model import Note
from ui import midi_ops


def _notes(*specs):
    return [Note(pitch, start, dur, vel) for pitch, start, dur, vel in specs]


def test_clone_is_independent():
    src = _notes((60, 0.0, 0.5, 100))
    copies = midi_ops.clone_notes(src)
    copies[0].pitch = 72
    copies[0].start = 1.0
    assert src[0].pitch == 60
    assert src[0].start == 0.0


def test_selection_span():
    notes = _notes((60, 1.0, 0.5, 100), (64, 1.25, 1.0, 90))
    assert midi_ops.selection_span(notes) == 1.25
    assert midi_ops.selection_span([]) == 0.0


def test_transpose_clamps():
    notes = _notes((60, 0.0, 0.5, 100), (126, 0.0, 0.5, 100))
    midi_ops.transpose(notes, 5)
    assert notes[0].pitch == 65
    assert notes[1].pitch == 127
    midi_ops.transpose(notes, -80, lo=0, hi=127)
    assert notes[0].pitch == 0


def test_nudge_time_does_not_go_negative():
    notes = _notes((60, 0.2, 0.5, 100), (64, 0.4, 0.5, 100))
    midi_ops.nudge_time(notes, -0.5)
    assert notes[0].start == 0.0
    assert abs(notes[1].start - 0.2) < 1e-9


def test_change_velocity_clamps():
    notes = _notes((60, 0.0, 0.5, 120), (61, 0.0, 0.5, 2))
    midi_ops.change_velocity(notes, 20)
    assert notes[0].velocity == 127
    midi_ops.change_velocity(notes, -10)
    assert notes[0].velocity == 117
    assert notes[1].velocity == 12
    midi_ops.change_velocity(notes, -80)
    assert notes[1].velocity == 1


def test_quantize_start_and_duration():
    notes = _notes((60, 0.13, 0.4, 100))
    midi_ops.quantize(notes, 0.25)
    assert notes[0].start == 0.25
    assert notes[0].duration == 0.5
    midi_ops.quantize(notes, None)
    assert notes[0].start == 0.25


def test_quantize_zero_duration_falls_back_to_grid():
    notes = _notes((60, 0.0, 0.04, 100))
    midi_ops.quantize(notes, 0.25)
    assert notes[0].duration == 0.25


def test_legato_per_pitch():
    notes = _notes(
        (60, 0.0, 0.2, 100),
        (60, 1.0, 0.2, 100),
        (64, 0.0, 0.3, 100),
        (64, 0.5, 0.3, 100),
    )
    midi_ops.legato(notes)
    assert abs(notes[0].duration - 1.0) < 1e-9
    assert notes[1].duration == 0.2
    assert abs(notes[2].duration - 0.5) < 1e-9
    assert notes[3].duration == 0.3


def test_duplicate_after_selection_span():
    notes = _notes((60, 0.0, 0.5, 100), (64, 0.25, 0.5, 90))
    copies = midi_ops.duplicate_after(notes)
    assert len(copies) == 2
    assert copies[0].start == 0.75
    assert copies[1].start == 1.0
    assert copies[0].pitch == 60
    assert notes[0].start == 0.0


def test_used_pitches_and_fold_rows():
    notes = _notes((60, 0.0, 0.5, 100), (64, 0.0, 0.5, 100), (60, 1.0, 0.5, 80))
    assert midi_ops.used_pitches(notes) == [60, 64]
    rows = midi_ops.fold_rows([60, 64], pad=1, lo=21, hi=108)
    assert rows == [65, 64, 63, 61, 60, 59]
    assert midi_ops.fold_rows([]) == list(range(60, 73))


def test_constrain_delta_picks_dominant_axis():
    assert midi_ops.constrain_delta(10.0, 3.0) == (10.0, 0.0)
    assert midi_ops.constrain_delta(2.0, -8.0) == (0.0, -8.0)
    assert midi_ops.constrain_delta(5.0, 5.0) == (5.0, 0.0)


def test_split_at_creates_right_hand_piece():
    notes = _notes((60, 0.0, 1.0, 100), (64, 0.8, 0.1, 90))
    created = midi_ops.split_at(notes, 0.4)
    assert len(created) == 1
    assert notes[0].duration == 0.4
    assert created[0].start == 0.4
    assert abs(created[0].duration - 0.6) < 1e-9
    assert created[0].pitch == 60
    assert notes[1].duration == 0.1
