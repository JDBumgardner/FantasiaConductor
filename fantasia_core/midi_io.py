"""Read standard MIDI files into the document model.

Note times are taken in **beats**, not seconds. A MIDI file carries its own
tempo, but when you drop a pattern into a project what you want is for it to
lock to *this* project's grid — so the caller supplies seconds-per-beat and the
pattern lands in time regardless of what tempo it was written at.

Headless (no Qt).
"""

from __future__ import annotations

from typing import List, Optional

from fantasia_core.document.model import Note

# Notes at or above this are treated as articulation keyswitches rather than
# music by the strum tools. C5=72 is well above any guitar/keyboard voicing a
# construction-kit pattern would use for harmony.
KEYSWITCH_MIN = 72


def available() -> bool:
    try:
        import mido  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def read_events(path: str) -> dict:
    """Parse a .mid into {'events': [(beat, pitch, velocity, dur_beats)], ...}."""
    import mido

    mf = mido.MidiFile(path)
    tpb = mf.ticks_per_beat or 480
    tempo_us = 500000  # default 120 BPM, only used to report the source tempo
    open_notes: dict = {}
    events: List[tuple] = []
    t = 0
    for msg in mido.merge_tracks(mf.tracks):
        t += msg.time
        if msg.type == "set_tempo":
            tempo_us = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            open_notes.setdefault(msg.note, []).append((t, msg.velocity))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            stack = open_notes.get(msg.note)
            if stack:
                on_t, vel = stack.pop(0)
                events.append((on_t / tpb, msg.note, vel, max((t - on_t) / tpb, 0.05)))
    for pitch, stack in open_notes.items():          # notes never released
        for on_t, vel in stack:
            events.append((on_t / tpb, pitch, vel, 1.0))
    events.sort(key=lambda e: (e[0], e[1]))
    return {
        "events": events,
        "ticks_per_beat": tpb,
        "source_bpm": round(60_000_000 / tempo_us, 2),
        "length_beats": max((e[0] + e[3] for e in events), default=0.0),
    }


def has_keyswitches(path: str) -> bool:
    """True if the file looks like a keyswitch-driven pattern (a high-register
    trigger layer over a held chord), rather than plain playable notes."""
    try:
        ev = read_events(path)["events"]
    except Exception:  # noqa: BLE001
        return False
    if not ev:
        return False
    high = sum(1 for _, p, _, _ in ev if p >= KEYSWITCH_MIN)
    low = len(ev) - high
    # Trigger layers dominate the event count and sit above any real voicing.
    return high >= 4 and high > low


def import_notes(path: str, spb: float, *, drop_keyswitches: bool = False) -> List[Note]:
    """Read a .mid as Notes timed against ``spb`` seconds-per-beat."""
    data = read_events(path)
    out: List[Note] = []
    for beat, pitch, vel, dur_beats in data["events"]:
        if drop_keyswitches and pitch >= KEYSWITCH_MIN:
            continue
        out.append(Note(pitch=int(pitch), start=round(beat * spb, 6),
                        duration=round(max(dur_beats * spb, 0.02), 6),
                        velocity=int(vel)))
    return out


def length_seconds(path: str, spb: float) -> float:
    return read_events(path)["length_beats"] * spb
