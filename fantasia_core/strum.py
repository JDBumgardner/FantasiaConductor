"""Turn keyswitch-driven guitar patterns into real, playable strums.

Construction-kit libraries (Big Fish Audio's Acou6tics and friends) don't store
strums as notes. They store a held chord in the low register plus a stream of
high-register **keyswitches** that tell the sample engine which stroke to fire.
Played on any ordinary instrument that reads as a static chord plus piercing
beeps — the rhythm never sounds.

This module keeps what's actually musical in those files — the *rhythm*, the
*velocity accents* and the *stroke direction* — and re-renders it as a genuine
strum on whatever chord you want: chord tones fanned out by a few milliseconds,
low->high for a downstroke and high->low for an up.

Headless (no Qt).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from fantasia_core.document.model import Note
from fantasia_core.midi_io import KEYSWITCH_MIN, read_events

# Guitar strings, low to high, for a standard-tuned open voicing.
_STANDARD = [40, 45, 50, 55, 59, 64]          # E2 A2 D3 G3 B3 E4

# Common shapes as fret offsets per string; None = string not played.
CHORDS: Dict[str, List[Optional[int]]] = {
    "E":  [0, 2, 2, 1, 0, 0],
    "Em": [0, 2, 2, 0, 0, 0],
    "A":  [None, 0, 2, 2, 2, 0],
    "Am": [None, 0, 2, 2, 1, 0],
    "D":  [None, None, 0, 2, 3, 2],
    "Dm": [None, None, 0, 2, 3, 1],
    "G":  [3, 2, 0, 0, 0, 3],
    "C":  [None, 3, 2, 0, 1, 0],
    "F":  [1, 3, 3, 2, 1, 1],
    "Bm": [None, 2, 4, 4, 3, 2],
}


def chord_pitches(name: str) -> List[int]:
    """MIDI pitches for a named open-position chord, low string first."""
    shape = CHORDS.get(name)
    if shape is None:
        return list(_STANDARD)
    return [s + f for s, f in zip(_STANDARD, shape) if f is not None]


def _is_downstroke(pitch: int) -> bool:
    """Keyswitch pitch -> stroke direction.

    In these libraries the naturals and their sharps alternate to mean
    down/up (F=down, F#=up, G=down, G#=up), which is exactly what the observed
    strong-weak velocity alternation implies.
    """
    return (pitch % 12) in (5, 7, 9, 11, 0, 2, 4)   # naturals -> downstroke


def extract_events(path: str) -> List[dict]:
    """Pull the strum events (rhythm + accent + direction) out of a pattern."""
    data = read_events(path)
    out = []
    for beat, pitch, vel, _dur in data["events"]:
        if pitch < KEYSWITCH_MIN:
            continue                       # the held chord, not a stroke
        out.append({"beat": beat, "velocity": vel,
                    "down": _is_downstroke(pitch), "keyswitch": pitch})
    return out


def render_strum(events: Sequence[dict], chord: Sequence[int], spb: float, *,
                 strum_ms: float = 22.0, ring: float = 1.0,
                 humanize: float = 0.35) -> List[Note]:
    """Render strum events onto a chord.

    ``strum_ms`` is how long the pick takes to cross the strings — the thing
    that makes a strum sound like a strum instead of a block chord.
    """
    pitches = sorted(chord)
    if not pitches:
        return []
    notes: List[Note] = []
    for i, ev in enumerate(events):
        base = ev["beat"] * spb
        # Upstrokes usually catch fewer strings, and the top ones first.
        order = pitches if ev["down"] else list(reversed(pitches))
        if not ev["down"] and len(order) > 4:
            order = order[:4]
        span = (strum_ms / 1000.0) * (1.0 if ev["down"] else 0.75)
        step = span / max(len(order) - 1, 1)
        # How long the chord rings: until the next stroke, plus a little.
        if i + 1 < len(events):
            ring_s = max((events[i + 1]["beat"] - ev["beat"]) * spb * ring, 0.05)
        else:
            ring_s = spb * ring
        for k, p in enumerate(order):
            offset = k * step
            # Strings nearer the end of the stroke are slightly quieter.
            vel = int(max(1, min(127, ev["velocity"] - k * humanize * 4)))
            notes.append(Note(pitch=int(p), start=round(base + offset, 6),
                              duration=round(max(ring_s - offset, 0.05), 6),
                              velocity=vel))
    return notes


def import_strum(path: str, spb: float, chord: str = "E", *,
                 strum_ms: float = 22.0) -> List[Note]:
    """Read a keyswitch pattern file and render it as a playable strum."""
    return render_strum(extract_events(path), chord_pitches(chord), spb,
                        strum_ms=strum_ms)
