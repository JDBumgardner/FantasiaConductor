"""Metronome clicks mixed into live playback (not bounce/export).

Clicks are short decaying sines. The first beat of each bar is louder (and a
bit higher) than the others. Beat phase is locked to the timeline origin so
the click lines up with the grid after scrub + play.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

CLICK_SECONDS = 0.024
ACCENT_HZ = 1320.0
BEAT_HZ = 880.0
ACCENT_PEAK = 0.45
BEAT_PEAK = 0.18


def make_click(
    sr: int,
    freq: float = BEAT_HZ,
    peak: float = BEAT_PEAK,
    duration: float = CLICK_SECONDS,
) -> np.ndarray:
    """One mono click: sine + exponential decay. Sample 0 is zeroed so mixing
    at a beat boundary does not inject a DC pop."""
    n = max(1, int(round(duration * sr)))
    t = np.arange(n, dtype=np.float64) / float(sr)
    env = np.exp(-t * 220.0)
    env[0] = 0.0
    return (np.sin(2.0 * np.pi * freq * t) * env * peak).astype(np.float32)


def make_click_bank(sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(accent, beat)`` templates at ``sr``."""
    return (
        make_click(sr, ACCENT_HZ, ACCENT_PEAK),
        make_click(sr, BEAT_HZ, BEAT_PEAK),
    )


def mix_metronome(
    out: np.ndarray,
    start_frame: int,
    sr: int,
    tempo: float,
    beats_per_bar: int = 4,
    clicks: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> None:
    """Add metronome clicks into a stereo ``(num_frames, 2)`` block in place."""
    num_frames = int(out.shape[0])
    if num_frames <= 0 or tempo <= 0 or beats_per_bar < 1 or sr <= 0:
        return
    accent, beat = clicks if clicks is not None else make_click_bank(sr)
    click_len = max(len(accent), len(beat))
    frames_per_beat = (60.0 / tempo) * sr
    if frames_per_beat < 1.0:
        return

    # Include beats whose click tail still overlaps this block.
    first = int(math.floor((start_frame - click_len + 1) / frames_per_beat))
    last = int(math.floor((start_frame + num_frames - 1) / frames_per_beat))
    for b in range(max(0, first), last + 1):
        onset = int(round(b * frames_per_beat))
        click = accent if (b % beats_per_bar) == 0 else beat
        src = 0
        dst = onset - start_frame
        if dst < 0:
            src = -dst
            dst = 0
        n = min(len(click) - src, num_frames - dst)
        if n > 0:
            chunk = click[src : src + n]
            out[dst : dst + n, 0] += chunk
            out[dst : dst + n, 1] += chunk


def render_metronome_block(
    start_frame: int,
    num_frames: int,
    sr: int,
    tempo: float,
    beats_per_bar: int = 4,
    clicks: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """Stereo metronome block as ``float32`` ``(num_frames, 2)``."""
    out = np.zeros((max(0, int(num_frames)), 2), dtype=np.float32)
    mix_metronome(out, start_frame, sr, tempo, beats_per_bar, clicks)
    return out
