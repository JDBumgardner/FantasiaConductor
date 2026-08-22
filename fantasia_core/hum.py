"""Hum → melody: monophonic pitch tracking tuned for singing/humming.

``transcribe.py`` (basic-pitch) is a polyphonic instrument transcriber — it can
emit overlapping notes and short fragments, which is wrong for a hum. A hum is
always one note at a time, so tracking a single f0 contour and segmenting it is
both more accurate and easier to clean up:

  pitch track (pyin) → smooth → split where the pitch settles somewhere new →
  drop fragments → snap each segment to a semitone → optional grid/key quantise

Headless (no Qt). Slow enough (~1-2s) that callers should run it off the UI thread.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from fantasia_core.document.model import Note

FMIN, FMAX = 65.0, 1200.0        # C2..D6 — comfortable hum/sing range
_HOP = 256

_SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "chromatic": list(range(12)),
}
_PC = {"c": 0, "c#": 1, "db": 1, "d": 2, "d#": 3, "eb": 3, "e": 4, "f": 5,
       "f#": 6, "gb": 6, "g": 7, "g#": 8, "ab": 8, "a": 9, "a#": 10, "bb": 10, "b": 11}


def available() -> bool:
    try:
        import librosa  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _snap_to_key(midi: int, key: str, scale: str) -> int:
    allowed = _SCALES.get(scale, _SCALES["major"])
    root = _PC.get(str(key).strip().lower(), 0)
    for d in range(0, 7):
        for cand in (midi - d, midi + d):
            if (cand - root) % 12 in allowed:
                return cand
    return midi


def track_pitch(samples: np.ndarray, sr: int):
    """Return (times, midi_float, voiced) for the monophonic pitch contour."""
    import librosa

    y = samples if samples.ndim == 1 else samples.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak                      # hums are usually quiet; normalise first
    f0, voiced, _ = librosa.pyin(y.astype(float), fmin=FMIN, fmax=FMAX, sr=sr,
                                 hop_length=_HOP, fill_na=np.nan)
    times = librosa.times_like(f0, sr=sr, hop_length=_HOP)
    midi = np.full(len(f0), np.nan)
    ok = np.isfinite(f0) & (f0 > 0)
    midi[ok] = librosa.hz_to_midi(f0[ok])
    # Light median smoothing kills vibrato wobble without blurring note changes.
    sm = midi.copy()
    for i in range(len(midi)):
        lo, hi = max(0, i - 2), min(len(midi), i + 3)
        w = midi[lo:hi][np.isfinite(midi[lo:hi])]
        if w.size:
            sm[i] = float(np.median(w))
    return times, sm, np.isfinite(sm)


def transcribe_hum(samples: np.ndarray, sr: int, *, spb: Optional[float] = None,
                   bpb: int = 4, quantize: bool = True, min_note: float = 0.09,
                   key: Optional[str] = None, scale: str = "major",
                   tolerance: float = 0.7) -> List[Note]:
    """Turn a hummed/sung recording into monophonic MIDI notes.

    ``spb`` (seconds per beat) enables grid quantising; ``key``/``scale`` snap
    pitches into a key. ``tolerance`` is how far (semitones) the pitch may drift
    before it counts as a new note — larger values ride through portamento.
    """
    times, midi, voiced = track_pitch(samples, sr)
    if not voiced.any():
        return []

    # --- segment the contour into notes -------------------------------------
    # Quantise every frame to its nearest semitone, then take RUNS of the same
    # semitone. A tracking window can't be used here: during portamento it
    # drifts along with the slide and never registers the note change. Runs
    # handle glides for free — the slide passes through intermediate semitones
    # only briefly, so those runs fall under min_note and get dropped.
    steps = np.where(voiced, np.round(midi), np.nan)
    segs = []                                    # [start_i, end_i, [pitches]]
    i = 0
    while i < len(steps):
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(steps) and voiced[j + 1] and steps[j + 1] == steps[i]:
            j += 1
        segs.append((i, j + 1, list(midi[i:j + 1])))
        i = j + 1

    # --- segments -> notes ---------------------------------------------------
    notes: List[Note] = []
    spreads: List[float] = []                     # pitch movement within each note
    for s, e, vals in segs:
        start = float(times[s])
        end = float(times[min(e, len(times) - 1)])
        if end - start < min_note or not vals:
            continue                              # too short to be a note
        # Median of the middle of the segment: skips the slide into the note.
        core = vals[len(vals) // 4: max(len(vals) // 4 + 1, len(vals) * 3 // 4)]
        pitch = int(round(float(np.median(core))))
        if key is not None:
            pitch = _snap_to_key(pitch, key, scale)
        arr = np.asarray(vals, dtype=float)
        spreads.append(float(np.percentile(arr, 90) - np.percentile(arr, 10)))
        notes.append(Note(pitch=pitch, start=start, duration=end - start, velocity=96))

    # --- drop portamento steps ----------------------------------------------
    # A slide shows up as a brief segment whose pitch sits *between* its
    # neighbours AND is still moving. The movement test is what separates a
    # glide from a genuinely fast stepwise note (C-E-G), which is held steady
    # for its whole length — filtering on shape alone deletes real melody.
    kept: List[Note] = []
    for idx, n in enumerate(notes):
        prev_p = notes[idx - 1].pitch if idx > 0 else None
        next_p = notes[idx + 1].pitch if idx + 1 < len(notes) else None
        # Measured on hummed material: glide segments span 0.6-0.7 semitones
        # while held notes (even fast ones) stay within 0.1-0.3. 0.45 splits
        # them with margin on both sides.
        sweeping = spreads[idx] > 0.45
        if (prev_p is not None and next_p is not None and n.duration < 0.16
                and min(prev_p, next_p) < n.pitch < max(prev_p, next_p)
                and sweeping):
            continue
        kept.append(n)
    notes = kept

    # --- merge repeats split by a brief unvoiced gap (breath, consonant) -----
    merged: List[Note] = []
    for n in notes:
        if merged and n.pitch == merged[-1].pitch and n.start - (merged[-1].start + merged[-1].duration) < 0.12:
            prev = merged[-1]
            prev.duration = n.start + n.duration - prev.start
        else:
            merged.append(n)

    # --- quantise to the grid ------------------------------------------------
    if quantize and spb and spb > 0:
        grid = spb / 4.0                          # sixteenth-note grid
        out: List[Note] = []
        for n in merged:
            start = round(n.start / grid) * grid
            dur = max(round(n.duration / grid) * grid, grid)
            out.append(Note(n.pitch, round(start, 6), round(dur, 6), n.velocity))
        merged = out
    return merged
