"""Pure MIDI-note edits used by the piano roll (Ableton-style)."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from fantasia_core.document.model import Note

MIN_NOTE = 0.05
_SLOT_EPS = 1e-4


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


def set_velocity(notes: Iterable[Note], velocity: int) -> None:
    vel = int(max(1, min(127, velocity)))
    for n in notes:
        n.velocity = vel


def change_duration(notes: Iterable[Note], delta: float, min_len: float = MIN_NOTE) -> None:
    for n in notes:
        n.duration = max(min_len, n.duration + delta)


def nudge_duration_grid(notes: Iterable[Note], direction: int, grid: float,
                        min_len: float = MIN_NOTE) -> None:
    """Move each note's end to the next (+) or previous (−) grid line."""
    if not notes or grid is None or grid <= 0 or not direction:
        return
    sign = 1 if direction > 0 else -1
    for n in notes:
        end = n.start + n.duration
        if sign > 0:
            k = math.floor(end / grid + 1e-7) + 1
            new_end = k * grid
            if new_end <= end + 1e-9:
                new_end = (k + 1) * grid
        else:
            k = math.ceil(end / grid - 1e-7) - 1
            new_end = k * grid
            if new_end >= end - 1e-9:
                new_end = (k - 1) * grid
        n.duration = max(min_len, new_end - n.start)


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


def used_rows(pitches: Sequence[int], lo: int = 21, hi: int = 108) -> List[int]:
    """Pitches actually present in the clip, high→low. Empty → one C4 octave."""
    shown = sorted({int(p) for p in pitches if lo <= int(p) <= hi}, reverse=True)
    return shown or list(range(60, 73))


def same_slot(a: Note, b: Note, eps: float = _SLOT_EPS) -> bool:
    return (a.pitch == b.pitch
            and abs(a.start - b.start) < eps
            and abs(a.duration - b.duration) < eps)


def slot_key(note: Note) -> Tuple[int, float, float]:
    return (int(note.pitch), round(note.start, 4), round(note.duration, 4))


def collapse_exact_duplicates(notes: Sequence[Note],
                              keep: Optional[Sequence[Note]] = None) -> List[Note]:
    """One note per identical pitch/start/duration. ``keep`` wins when present."""
    keep_slots = {slot_key(n) for n in (keep or [])}
    chosen: Dict[Tuple[int, float, float], Note] = {}
    order: List[Tuple[int, float, float]] = []
    for n in notes:
        key = slot_key(n)
        prev = chosen.get(key)
        if prev is None:
            chosen[key] = n
            order.append(key)
            continue
        if key in keep_slots:
            chosen[key] = n
    return [Note(chosen[k].pitch, chosen[k].start, chosen[k].duration, chosen[k].velocity)
            for k in order]


def subtract_range(note: Note, cut_start: float, cut_end: float,
                   min_len: float = MIN_NOTE) -> List[Note]:
    """Pieces of ``note`` that lie outside ``[cut_start, cut_end)``."""
    ns, ne = note.start, note.start + note.duration
    if cut_end <= ns + 1e-9 or cut_start >= ne - 1e-9:
        return [Note(note.pitch, note.start, note.duration, note.velocity)]
    out: List[Note] = []
    if ns < cut_start - 1e-9:
        dur = cut_start - ns
        if dur >= min_len:
            out.append(Note(note.pitch, ns, dur, note.velocity))
    if ne > cut_end + 1e-9:
        dur = ne - cut_end
        if dur >= min_len:
            out.append(Note(note.pitch, cut_end, dur, note.velocity))
    return out


def resolve_overlaps(notes: Sequence[Note],
                     winners: Optional[Sequence[Note]] = None) -> List[Note]:
    """Same-pitch notes may not share time. ``winners`` punch holes in others."""
    collapsed = collapse_exact_duplicates(notes, keep=winners)
    if winners:
        win_keys = {slot_key(w) for w in winners}
        punched: List[Note] = []
        for n in collapsed:
            if slot_key(n) in win_keys:
                punched.append(n)
                continue
            pieces = [n]
            for w in winners:
                if w.pitch != n.pitch:
                    continue
                nxt: List[Note] = []
                for piece in pieces:
                    nxt.extend(subtract_range(piece, w.start, w.start + w.duration))
                pieces = nxt
            punched.extend(pieces)
        collapsed = punched
    by_pitch: Dict[int, List[Note]] = {}
    for n in collapsed:
        by_pitch.setdefault(n.pitch, []).append(n)
    out: List[Note] = []
    for group in by_pitch.values():
        group.sort(key=lambda n: (n.start, n.duration))
        kept: List[Note] = []
        for n in group:
            if kept and n.start < kept[-1].start + kept[-1].duration - 1e-9:
                kept[-1].duration = max(0.0, n.start - kept[-1].start)
                if kept[-1].duration < MIN_NOTE:
                    kept.pop()
            kept.append(n)
        out.extend(kept)
    out.sort(key=lambda n: (n.start, n.pitch))
    return out


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
    # Freygish is the klezmer scale proper — the same intervals as Phrygian
    # Dominant, listed under the name it is actually asked for. Misheberakh
    # (Ukrainian Dorian, a raised 4th) is the other one the repertoire uses.
    "Klezmer (Freygish)": [0, 1, 4, 5, 7, 8, 10],
    "Klezmer (Misheberakh)": [0, 2, 3, 6, 7, 9, 10],
    "Hirajoshi": [0, 2, 3, 7, 8],
    "In Sen": [0, 1, 5, 7, 10],
    "Altered": [0, 1, 3, 4, 6, 8, 10],
    "Enigmatic": [0, 1, 4, 6, 8, 10, 11],
}

SCALE_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Enharmonic pairs so the user can pick Bb vs A# (spelling follows the chosen name).
SCALE_ROOT_CHOICES: List[Tuple[int, str]] = [
    (0, "C"), (1, "C#"), (1, "Db"), (2, "D"), (3, "D#"), (3, "Eb"),
    (4, "E"), (5, "F"), (6, "F#"), (6, "Gb"), (7, "G"), (8, "G#"), (8, "Ab"),
    (9, "A"), (10, "A#"), (10, "Bb"), (11, "B"),
]

_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_LETTERS = "CDEFGAB"
_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
# Sharps (positive) / flats (negative) in the major key of each pitch class.
_MAJOR_FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: 7,
                 5: -1, 10: -2, 3: -3, 8: -4}
_MODE_FIFTHS = {
    "Major": 0, "Lydian": 1, "Mixolydian": -1, "Dorian": -2,
    "Natural Minor": -3, "Harmonic Minor": -3, "Melodic Minor": -3,
    "Phrygian": -4, "Locrian": -5, "Major Pentatonic": 0, "Minor Pentatonic": -3,
    "Blues": -3, "Hungarian Minor": -3, "Phrygian Dominant": -4,
    "Lydian Dominant": -1, "Altered": -5,
}


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


def nearest_in_scale(pitch: int, pcs, lo: int = 0, hi: int = 127) -> int:
    """The closest pitch in the scale, preferring upward on a tie.

    Used when a pitch comes from a mouse position rather than from an existing
    note: without it, choosing a scale colours the rows but a drawn note can
    still land outside it, which makes the setting look decorative.
    """
    if not pcs:
        return int(max(lo, min(hi, pitch)))
    pitch = int(max(lo, min(hi, pitch)))
    if (pitch % 12) in pcs:
        return pitch
    for delta in range(1, 13):
        for cand in (pitch + delta, pitch - delta):
            if lo <= cand <= hi and (cand % 12) in pcs:
                return cand
    return pitch


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


def parse_pitch_class_name(name: str) -> Tuple[str, int, int]:
    """``'Bb'`` → ``('B', -1, 10)``."""
    text = (name or "C").strip().replace("♯", "#").replace("♭", "b").replace("×", "x")
    letter = text[0].upper() if text else "C"
    if letter not in _LETTER_PC:
        letter = "C"
    rest = text[1:]
    acc = 2 * rest.count("x") + rest.count("#") - rest.count("b")
    return letter, acc, (_LETTER_PC[letter] + acc) % 12


def _format_accidental(acc: int) -> str:
    if acc == 0:
        return ""
    if acc == 1:
        return "#"
    if acc == 2:
        return "x"
    if acc == -1:
        return "b"
    if acc == -2:
        return "bb"
    return ("#" * acc) if acc > 0 else ("b" * (-acc))


def _spell_letter(letter: str, target_pc: int) -> str:
    acc = (int(target_pc) - _LETTER_PC[letter] + 6) % 12 - 6
    return letter + _format_accidental(acc)


def _prefer_flats(scale_name: str, root: int, root_name: Optional[str]) -> bool:
    if root_name and "b" in root_name[1:]:
        return True
    if root_name and "#" in root_name:
        return False
    fifths = _MAJOR_FIFTHS.get(int(root) % 12, 0) + _MODE_FIFTHS.get(scale_name, 0)
    return fifths < 0


def _letter_steps(scale_name: str, n: int) -> Optional[List[int]]:
    if n == 7:
        return [1] * 7
    if n == 5:
        if "Minor" in scale_name or scale_name in ("Hirajoshi", "In Sen"):
            return [2, 1, 1, 2, 1]
        return [1, 1, 2, 1, 2]
    if n == 6 and scale_name == "Whole Tone":
        return [1, 1, 1, 1, 1, 2]
    return None


def _letter_cycle_spellings(scale_name: str, root: int,
                            root_name: Optional[str]) -> Dict[int, str]:
    intervals = SCALES.get(scale_name, SCALES["Chromatic"])
    if len(intervals) == 12:
        return {}
    name = root_name or SCALE_ROOTS[int(root) % 12]
    letter, _acc, rpc = parse_pitch_class_name(name)
    if scale_name == "Blues":
        # Minor pentatonic plus a raised 4th on the same letter (C Eb F F# G Bb).
        pent = _letter_cycle_spellings("Minor Pentatonic", root, root_name)
        fourth_pc = (rpc + 5) % 12
        sharp_fourth = (rpc + 6) % 12
        fourth_name = pent.get(fourth_pc, _SHARP_NAMES[fourth_pc])
        pent[sharp_fourth] = fourth_name[0] + _format_accidental(
            (sharp_fourth - _LETTER_PC[fourth_name[0]] + 6) % 12 - 6)
        return pent
    steps = _letter_steps(scale_name, len(intervals))
    if not steps:
        return {}
    result: Dict[int, str] = {}
    idx = _LETTERS.index(letter)
    for i, iv in enumerate(intervals):
        L = _LETTERS[idx % 7]
        pc = (rpc + iv) % 12
        result[pc] = _spell_letter(L, pc)
        idx += steps[i]
    return result


def scale_pc_names(scale_name: str, root: int = 0,
                   root_name: Optional[str] = None) -> List[str]:
    """12 pitch-class names spelled for ``scale_name`` + root."""
    if is_chromatic(scale_name):
        return list(_SHARP_NAMES)
    names = list(_FLAT_NAMES if _prefer_flats(scale_name, root, root_name) else _SHARP_NAMES)
    for pc, spelled in _letter_cycle_spellings(scale_name, root, root_name).items():
        names[pc] = spelled
    return names


def pc_name(pc: int, scale_name: str = "Chromatic", root: int = 0,
            root_name: Optional[str] = None) -> str:
    return scale_pc_names(scale_name, root, root_name)[int(pc) % 12]


def pitch_name(pitch: int, scale_name: str = "Chromatic", root: int = 0,
               root_name: Optional[str] = None, with_octave: bool = True) -> str:
    """MIDI number → ``Bb4`` (or ``Bb``) using the scale's spelling."""
    name = pc_name(int(pitch), scale_name, root, root_name)
    if not with_octave:
        return name
    return f"{name}{int(pitch) // 12 - 1}"
