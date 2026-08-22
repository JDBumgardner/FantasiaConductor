"""MIDI file import + keyswitch-pattern -> strum translation."""

from __future__ import annotations

import mido
import pytest

from fantasia_core.midi_io import (available, has_keyswitches, import_notes,
                                   read_events)
from fantasia_core.strum import chord_pitches, extract_events, import_strum

pytestmark = pytest.mark.skipif(not available(), reason="mido not installed")
TPB = 480


def _write(path, events, tpb=TPB):
    """events: (tick, pitch, velocity, dur_ticks)"""
    mf = mido.MidiFile(ticks_per_beat=tpb)
    tr = mido.MidiTrack()
    mf.tracks.append(tr)
    msgs = []
    for tick, pitch, vel, dur in events:
        msgs.append((tick, mido.Message("note_on", note=pitch, velocity=vel)))
        msgs.append((tick + dur, mido.Message("note_off", note=pitch, velocity=0)))
    msgs.sort(key=lambda m: m[0])
    prev = 0
    for tick, msg in msgs:
        msg.time = tick - prev
        prev = tick
        tr.append(msg)
    mf.save(str(path))
    return str(path)


def _keyswitch_file(tmp_path):
    """A held chord plus alternating down/up keyswitches — the library shape."""
    ev = [(0, 40, 100, TPB * 4), (0, 44, 100, TPB * 4), (0, 47, 100, TPB * 4)]
    for i in range(8):                      # 8th-note down/up strokes
        tick = i * (TPB // 2)
        ev.append((tick, 89 if i % 2 == 0 else 90, 120 if i % 2 == 0 else 85, 10))
    return _write(tmp_path / "ks.mid", ev)


def test_times_are_read_in_beats_so_they_follow_project_tempo(tmp_path):
    f = _write(tmp_path / "a.mid", [(0, 60, 100, TPB), (TPB, 62, 100, TPB)])
    slow = import_notes(f, spb=1.0)         # 60 BPM
    fast = import_notes(f, spb=0.25)        # 240 BPM
    assert [n.start for n in slow] == [0.0, 1.0]
    assert [n.start for n in fast] == [0.0, 0.25]
    assert slow[0].duration == 1.0 and fast[0].duration == 0.25


def test_keyswitch_pattern_is_detected(tmp_path):
    assert has_keyswitches(_keyswitch_file(tmp_path))
    plain = _write(tmp_path / "plain.mid",
                   [(0, 60, 100, TPB), (TPB, 64, 100, TPB), (TPB * 2, 67, 100, TPB)])
    assert not has_keyswitches(plain)


def test_strum_uses_keyswitch_rhythm_on_the_chosen_chord(tmp_path):
    f = _keyswitch_file(tmp_path)
    events = extract_events(f)
    assert len(events) == 8                       # one per keyswitch, chord ignored
    assert events[0]["down"] and not events[1]["down"]

    notes = import_strum(f, spb=0.5, chord="G")
    g = set(chord_pitches("G"))
    assert {n.pitch for n in notes} <= g           # plays G, not the file's E chord
    assert all(n.pitch < 72 for n in notes)        # no keyswitch beeps survive

    # First stroke is a downstroke: strings enter low -> high, spread over time.
    first = sorted([n for n in notes if n.start < 0.05], key=lambda n: n.start)
    assert len(first) > 1
    assert first[0].pitch == min(n.pitch for n in first)
    assert first[-1].start > first[0].start


def test_upstroke_runs_high_to_low(tmp_path):
    f = _keyswitch_file(tmp_path)
    notes = import_strum(f, spb=0.5, chord="G", strum_ms=30)
    up_start = 0.25                                # second event, an upstroke
    up = sorted([n for n in notes if abs(n.start - up_start) < 0.04],
                key=lambda n: n.start)
    assert len(up) > 1
    assert up[0].pitch == max(n.pitch for n in up)  # highest string first


def test_raw_import_can_drop_keyswitches(tmp_path):
    f = _keyswitch_file(tmp_path)
    raw = import_notes(f, spb=0.5)
    clean = import_notes(f, spb=0.5, drop_keyswitches=True)
    assert any(n.pitch >= 72 for n in raw)
    assert all(n.pitch < 72 for n in clean)
    assert len(clean) == 3                          # just the held chord


def test_length_is_reported_in_beats(tmp_path):
    f = _write(tmp_path / "len.mid", [(0, 60, 100, TPB * 2)])
    assert read_events(f)["length_beats"] == pytest.approx(2.0)
