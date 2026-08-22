"""Metronome click placement, downbeat accent, and tempo (no audio device)."""

from __future__ import annotations

import numpy as np

from fantasia_core.document import Project
from fantasia_core.engine import AudioPool, PlaybackEngine, render_metronome_block
from fantasia_core.engine.metronome import ACCENT_PEAK, BEAT_PEAK


SR = 44100


def _peak_near(buf: np.ndarray, t: float, window: float = 0.02) -> float:
    i = int(round(t * SR))
    w = int(window * SR)
    sl = buf[max(0, i) : i + w]
    return float(np.max(np.abs(sl))) if len(sl) else 0.0


def test_block_shape_and_dtype():
    block = render_metronome_block(0, 2048, SR, tempo=120.0, beats_per_bar=4)
    assert block.shape == (2048, 2)
    assert block.dtype == np.float32


def test_click_on_first_beat():
    block = render_metronome_block(0, 2048, SR, tempo=120.0)
    assert np.max(np.abs(block)) > 0.0


def test_silent_between_beats():
    # 120 BPM → beats every 0.5s. A block at 0.2s sits between beat 0 and 1,
    # after the click tail (~24ms) has died out.
    start = int(0.2 * SR)
    block = render_metronome_block(start, 1024, SR, tempo=120.0)
    assert np.max(np.abs(block)) == 0.0


def test_downbeat_louder_than_other_beats():
    # One bar at 120 BPM = 2.0s. Beat 0 is the downbeat; 1–3 are quieter.
    bar = render_metronome_block(0, int(2.0 * SR), SR, tempo=120.0, beats_per_bar=4)
    down = _peak_near(bar, 0.0)
    others = [_peak_near(bar, t) for t in (0.5, 1.0, 1.5)]
    assert down > 0.0
    assert all(p > 0.0 for p in others)
    assert down > max(others)
    assert max(others) / down < 0.6
    # Other beats are the same click (within float noise).
    assert max(others) - min(others) < 1e-6


def test_next_bar_downbeat_is_also_accented():
    two_bars = render_metronome_block(0, int(2.5 * SR), SR, tempo=120.0, beats_per_bar=4)
    first = _peak_near(two_bars, 0.0)
    next_down = _peak_near(two_bars, 2.0)
    mid = _peak_near(two_bars, 0.5)
    assert abs(first - next_down) < 1e-6
    assert next_down > mid


def test_tempo_changes_spacing():
    # 60 BPM → beats every 1.0s. Mid-block at 0.5s is silent; t=1.0 has a click.
    mid = render_metronome_block(int(0.4 * SR), 2048, SR, tempo=60.0)
    assert np.max(np.abs(mid)) == 0.0
    at_beat = render_metronome_block(int(1.0 * SR), 2048, SR, tempo=60.0)
    assert np.max(np.abs(at_beat)) > 0.0


def test_click_tail_spans_block_boundary():
    # Start a few samples after a beat so the remaining tail still mixes in.
    tail = render_metronome_block(8, 256, SR, tempo=120.0)
    assert np.max(np.abs(tail)) > 0.0


def test_playback_engine_defaults_metronome_off():
    engine = PlaybackEngine(Project(), AudioPool(SR), SR)
    assert engine.metronome_enabled is False


def test_accent_peak_constant_is_louder():
    assert ACCENT_PEAK > BEAT_PEAK
