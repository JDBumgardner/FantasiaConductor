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
