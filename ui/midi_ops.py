"""Pure MIDI-note edits used by the piano roll (Ableton-style)."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from fantasia_core.document.model import Note

MIN_NOTE = 0.05


def clone_notes(notes: Iterable[Note]) -> List[Note]:
    return [Note(n.pitch, n.start, n.duration, n.velocity) for n in notes]


def selection_span(notes: Sequence[Note]) -> float:
    if not notes:
        return 0.0
    return max(n.start + n.duration for n in notes) - min(n.start for n in notes)


def transpose(notes: Iterable[Note], semis: int, lo: int = 0, hi: int = 127) -> None:
    for n in notes:
        n.pitch = int(max(lo, min(hi, n.pitch + semis)))


def nudge_time(notes: Iterable[Note], delta: float) -> None:
    if not notes:
        return
    earliest = min(n.start for n in notes)
    if earliest + delta < 0:
        delta = -earliest
    for n in notes:
        n.start = max(0.0, n.start + delta)


def change_velocity(notes: Iterable[Note], delta: int) -> None:
    for n in notes:
        n.velocity = int(max(1, min(127, n.velocity + delta)))


def quantize(notes: Iterable[Note], grid: Optional[float]) -> None:
    if not grid or grid <= 0:
        return
    for n in notes:
        n.start = max(0.0, round(n.start / grid) * grid)
        n.duration = max(MIN_NOTE, round(n.duration / grid) * grid or grid)


def legato(notes: Sequence[Note]) -> None:
    """Extend each note to the next note of the same pitch (Ableton legato)."""
    by_pitch: dict[int, List[Note]] = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)
    for group in by_pitch.values():
        group.sort(key=lambda n: n.start)
        for i, n in enumerate(group[:-1]):
            nxt = group[i + 1].start
            n.duration = max(MIN_NOTE, nxt - n.start)


def duplicate_after(notes: Sequence[Note]) -> List[Note]:
    """Copies placed immediately after the selection block (Ctrl+D)."""
    span = selection_span(notes)
    if span <= 0:
        span = max((n.duration for n in notes), default=0.0)
    return [Note(n.pitch, n.start + span, n.duration, n.velocity) for n in notes]


def used_pitches(notes: Iterable[Note]) -> List[int]:
    return sorted({n.pitch for n in notes})


def fold_rows(pitches: Sequence[int], pad: int = 1, lo: int = 21, hi: int = 108) -> List[int]:
    """Pitches to show when Fold is on: used notes plus ``pad`` neighbors."""
    if not pitches:
        return list(range(60, 73))
    shown = set()
    for p in pitches:
        for q in range(p - pad, p + pad + 1):
            if lo <= q <= hi:
                shown.add(q)
    return sorted(shown, reverse=True)


def constrain_delta(dx: float, dy: float) -> Tuple[float, float]:
    """Shift-constrain a drag to the dominant axis (Ableton)."""
    if abs(dx) >= abs(dy):
        return dx, 0.0
    return 0.0, dy


def split_at(notes: Sequence[Note], at: float) -> List[Note]:
    """Split notes that straddle ``at``. Returns the new right-hand pieces."""
    created: List[Note] = []
    for n in notes:
        end = n.start + n.duration
        if n.start < at < end:
            right = end - at
            n.duration = max(MIN_NOTE, at - n.start)
            created.append(Note(n.pitch, at, max(MIN_NOTE, right), n.velocity))
    return created


# Interval patterns from the root (0 = root). Chromatic is the unfiltered roll.
SCALES: dict[str, List[int]] = {
    "Chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "Major": [0, 2, 4, 5, 7, 9, 11],
    "Natural Minor": [0, 2, 3, 5, 7, 8, 10],
    "Harmonic Minor": [0, 2, 3, 5, 7, 8, 11],
    "Melodic Minor": [0, 2, 3, 5, 7, 9, 11],
    "Dorian": [0, 2, 3, 5, 7, 9, 10],
    "Phrygian": [0, 1, 3, 5, 7, 8, 10],
    "Lydian": [0, 2, 4, 6, 7, 9, 11],
    "Mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "Locrian": [0, 1, 3, 5, 6, 8, 10],
    "Major Pentatonic": [0, 2, 4, 7, 9],
    "Minor Pentatonic": [0, 3, 5, 7, 10],
    "Blues": [0, 3, 5, 6, 7, 10],
    "Whole Tone": [0, 2, 4, 6, 8, 10],
    "Diminished (H-W)": [0, 1, 3, 4, 6, 7, 9, 10],
    "Diminished (W-H)": [0, 2, 3, 5, 6, 8, 9, 11],
    "Hungarian Minor": [0, 2, 3, 6, 7, 8, 11],
    "Phrygian Dominant": [0, 1, 4, 5, 7, 8, 10],
    "Lydian Dominant": [0, 2, 4, 6, 7, 9, 10],
    "Persian": [0, 1, 4, 5, 6, 8, 11],
    "Hirajoshi": [0, 2, 3, 7, 8],
    "In Sen": [0, 1, 5, 7, 10],
    "Altered": [0, 1, 3, 4, 6, 8, 10],
    "Enigmatic": [0, 1, 4, 6, 8, 10, 11],
}

SCALE_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def scale_pitch_classes(name: str, root: int = 0) -> set:
    intervals = SCALES.get(name, SCALES["Chromatic"])
    return {(int(root) + i) % 12 for i in intervals}


def is_chromatic(name: str) -> bool:
    return name == "Chromatic" or name not in SCALES


def scale_rows(name: str, root: int = 0, lo: int = 21, hi: int = 108) -> List[int]:
    """Pitches in ``[lo, hi]`` that belong to the scale, high→low (piano order)."""
    if is_chromatic(name):
        return list(range(hi, lo - 1, -1))
    pcs = scale_pitch_classes(name, root)
    return [p for p in range(hi, lo - 1, -1) if (p % 12) in pcs]


def step_in_scale(pitch: int, steps: int, pcs, lo: int = 0, hi: int = 127) -> int:
    """Move ``steps`` scale degrees (negative = down). Octave = 12 semitones stays
    on the same degree when the destination is in range."""
    if not steps:
        return int(max(lo, min(hi, pitch)))
    if not pcs:
        return int(max(lo, min(hi, pitch + steps)))
    if abs(steps) >= 12 and steps % 12 == 0:
        return int(max(lo, min(hi, pitch + steps)))
    direction = 1 if steps > 0 else -1
    remaining = abs(int(steps))
    p = int(pitch)
    while remaining:
        p += direction
        if p < lo or p > hi:
            return int(max(lo, min(hi, p)))
        if (p % 12) in pcs:
            remaining -= 1
    return p


def transpose_in_scale(notes: Iterable[Note], steps: int, pcs, lo: int = 0, hi: int = 127) -> None:
    if not notes:
        return
    if not pcs:
        transpose(notes, steps, lo, hi)
        return
    # Keep the chord shape: shift every note by the same semitone distance
    # as the first note's scale-step, so a triad stays a triad.
    first = next(iter(notes))
    target = step_in_scale(first.pitch, steps, pcs, lo, hi)
    delta = target - first.pitch
    transpose(notes, delta, lo, hi)
