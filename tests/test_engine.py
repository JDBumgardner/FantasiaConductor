"""Audio engine: buffer pool, peaks, and the mixer (M3, no audio device)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from fantasia_core.document import Project
from fantasia_core.engine import AudioPool, render_block

SAMPLES = pathlib.Path(__file__).resolve().parent.parent / "assets" / "samples"
DRUMS = str(SAMPLES / "drums.wav")
BASS = str(SAMPLES / "bass.wav")

pytestmark = pytest.mark.skipif(
    not (SAMPLES / "drums.wav").exists(),
    reason="demo audio not generated (run tools/make_demo_audio.py)",
)


def _project_with(path: str, start: float = 0.0, dur: float = 2.0) -> tuple[Project, AudioPool]:
    p = Project(sample_rate=44100)
    t = p.add_track("A")
    p.add_clip(t.id, start=start, duration=dur, name="c", source_path=path)
    pool = AudioPool(p.sample_rate)
    pool.preload(p)
    return p, pool


def test_load_and_duration():
    pool = AudioPool(44100)
    data = pool.load(DRUMS)
    assert data.dtype == np.float32 and data.ndim == 2
    assert abs(pool.duration(DRUMS) - 4.0) < 0.01


def test_render_block_has_audio():
    p, pool = _project_with(DRUMS)
    block = render_block(p, pool, start_frame=0, num_frames=2048, sr=44100)
    assert block.shape == (2048, 2)
    assert np.max(np.abs(block)) > 0.0  # first kick is at t=0


def test_block_before_clip_is_silent():
    p, pool = _project_with(DRUMS, start=5.0, dur=2.0)  # clip starts at 5s
    block = render_block(p, pool, start_frame=0, num_frames=2048, sr=44100)
    assert np.max(np.abs(block)) == 0.0


def test_mute_silences_track():
    p, pool = _project_with(DRUMS)
    p.tracks[0].mute = True
    block = render_block(p, pool, 0, 2048, 44100)
    assert np.max(np.abs(block)) == 0.0


def test_solo_isolates():
    p, pool = _project_with(DRUMS)
    t2 = p.add_track("B")
    p.add_clip(t2.id, 0.0, 2.0, "c2", source_path=BASS)
    pool.preload(p)
    full = render_block(p, pool, 0, 2048, 44100)
    p.tracks[0].solo = True  # solo drums only
    soloed = render_block(p, pool, 0, 2048, 44100)
    # Soloed mix should differ from the full mix (bass removed).
    assert not np.allclose(full, soloed)


def test_gain_scales_amplitude():
    p, pool = _project_with(DRUMS)
    loud = render_block(p, pool, 0, 2048, 44100)
    p.tracks[0].gain_db = -20.0
    quiet = render_block(p, pool, 0, 2048, 44100)
    assert np.max(np.abs(quiet)) < np.max(np.abs(loud))


def test_reverse_flips_audio():
    p, pool = _project_with(DRUMS, start=0.0, dur=4.0)
    fwd = render_block(p, pool, 0, 4096, 44100)
    p.tracks[0].clips[0].reversed = True
    rev = render_block(p, pool, 0, 4096, 44100)
    # Reversed playback differs from forward (kick is at the very start forward).
    assert not np.allclose(fwd, rev)


def test_bounce_matches_length():
    from fantasia_core.engine import bounce_to_array

    p, pool = _project_with(DRUMS, start=0.0, dur=2.0)
    t2 = p.add_track("B")
    p.add_clip(t2.id, 1.0, 3.0, "c2", source_path=BASS)
    pool.preload(p)
    mix = bounce_to_array(p, pool, 44100)
    assert mix.shape == (int(p.duration * 44100), 2)  # spans to the last clip end
    assert np.max(np.abs(mix)) > 0.0


def test_track_fx_changes_output():
    from fantasia_core.engine import FxHost

    p, pool = _project_with(DRUMS, start=0.0, dur=2.0)
    dry = render_block(p, pool, 0, 8192, 44100)
    p.tracks[0].fx = [{"type": "reverb", "params": {"wet": 0.7}}]
    wet = render_block(p, pool, 0, 8192, 44100, fx_host=FxHost())
    assert not np.allclose(dry, wet)


def test_pitch_shift_changes_output_same_length():
    p, pool = _project_with(BASS, start=0.0, dur=2.0)
    dry = render_block(p, pool, 0, 8192, 44100)
    p.tracks[0].clips[0].pitch_semitones = 5.0
    shifted = render_block(p, pool, 0, 8192, 44100)
    assert not np.allclose(dry, shifted)
    # Pitch shift preserves buffer length (so timeline timing is unchanged).
    assert len(pool.load_pitched(BASS, 5.0)) == len(pool.load(BASS))


def test_peaks_shape():
    pool = AudioPool(44100)
    mins, maxs = pool.peaks(DRUMS, 0.0, 2.0, buckets=200)
    assert len(mins) == len(maxs) == 200
    assert np.any(maxs > 0)


# ---- mixing FX (EQ / colour / dynamics) -----------------------------------
def _fx_run(spec, x, sr=44100):
    from fantasia_core.engine.fx import build_board
    board = build_board([spec])
    assert board is not None, f"{spec['type']} failed to build"
    return board(x.T.astype(np.float32), sr).T


def _band(x, lo, hi, sr=44100):
    S = np.abs(np.fft.rfft(x[:, 0]))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return float(np.sum(S[(f >= lo) & (f < hi)]))


def _two_tone(sr=44100):
    t = np.arange(sr) / sr
    sig = 0.3 * np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 5000 * t)
    return np.stack([sig] * 2, axis=1).astype(np.float32)


def test_eq_bands_target_the_right_frequencies():
    x = _two_tone()
    low0, high0 = _band(x, 50, 200), _band(x, 4000, 6000)
    y = _fx_run({"type": "eq_low_shelf", "params": {"freq": 200, "gain": -18, "q": 0.7}}, x)
    assert _band(y, 50, 200) / low0 < 0.4          # bass cut
    assert 0.9 < _band(y, 4000, 6000) / high0 < 1.1  # treble untouched
    y = _fx_run({"type": "eq_high_shelf", "params": {"freq": 3000, "gain": -18, "q": 0.7}}, x)
    assert _band(y, 4000, 6000) / high0 < 0.4
    assert 0.9 < _band(y, 50, 200) / low0 < 1.1
    y = _fx_run({"type": "eq_peak", "params": {"freq": 5000, "gain": -24, "q": 2.0}}, x)
    assert _band(y, 4000, 6000) / high0 < 0.4      # notched the 5k tone only
    assert 0.9 < _band(y, 50, 200) / low0 < 1.1


def test_compressor_narrows_dynamic_range():
    sr = 44100
    t = np.arange(sr) / sr
    env = np.concatenate([np.full(sr // 2, 0.9), np.full(sr // 2, 0.15)])
    x = np.stack([env * np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype(np.float32)
    y = _fx_run({"type": "compressor",
                 "params": {"threshold": -20, "ratio": 8, "attack": 5, "release": 80}}, x)
    before = np.max(np.abs(x[: sr // 2])) / np.max(np.abs(x[sr // 2:]))
    after = np.max(np.abs(y[: sr // 2])) / np.max(np.abs(y[sr // 2:]))
    assert after < before


def test_limiter_caps_without_boosting_quiet_signal():
    """pedalboard's own Limiter adds makeup gain (a maximizer) and would push a
    quiet track to full scale — ours must only cap."""
    sr = 44100
    t = np.arange(sr) / sr
    quiet = np.stack([0.1 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype(np.float32)
    loud = np.stack([0.9 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype(np.float32)
    spec = {"type": "limiter", "params": {"threshold": -6.0}}
    q = _fx_run(spec, quiet)[4410:]
    ld = _fx_run(spec, loud)[4410:]
    assert np.max(np.abs(q)) < 0.15          # left alone, NOT boosted
    assert np.max(np.abs(ld)) < 0.65         # capped near -6 dB (0.501)


def test_saturator_adds_harmonics_without_raising_peak():
    sr = 44100
    t = np.arange(sr) / sr
    pure = np.stack([0.5 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1).astype(np.float32)
    y = _fx_run({"type": "saturator", "params": {"drive": 15, "output": -9}}, pure)
    assert _band(y, 600, 1200) > 50 * max(_band(pure, 600, 1200), 1e-9)
    assert np.max(np.abs(y)) <= np.max(np.abs(pure)) + 1e-3
