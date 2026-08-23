"""Pitch-preserving time-stretch (tempo follow, clip resize, Time Stretch menu)."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from fantasia_core.engine.buffers import AudioPool
from fantasia_core.stretch import available, stretch


def _tone(sr: int, hz: float, seconds: float) -> np.ndarray:
    n = int(sr * seconds)
    t = np.linspace(0.0, seconds, n, endpoint=False)
    mono = np.sin(2.0 * np.pi * hz * t).astype(np.float32)
    return np.stack([mono, mono], axis=1)


def _peak_hz(x: np.ndarray, sr: int) -> float:
    mono = x[:, 0] if x.ndim == 2 else x
    window = np.hanning(len(mono))
    spec = np.abs(np.fft.rfft(mono * window))
    return float(np.argmax(spec) * sr / len(mono))


def test_available_without_rubberband_cli():
    assert available() is True


def test_stretch_length_and_pitch():
    sr = 22050
    x = _tone(sr, 440.0, 0.25)
    y = stretch(x, sr, 2.0)
    assert abs(len(y) / sr - 0.5) < 0.06
    assert abs(_peak_hz(y, sr) - 440.0) < 20.0
    z = stretch(x, sr, 0.5)
    assert abs(len(z) / sr - 0.125) < 0.04
    assert abs(_peak_hz(z, sr) - 440.0) < 20.0


def test_varispeed_would_shift_pitch_but_warp_does_not(tmp_path):
    sr = 22050
    x = _tone(sr, 440.0, 0.25)
    path = tmp_path / "tone.wav"
    sf.write(path, x, sr)
    pool = AudioPool(sr)
    # Half duration at original pitch (professional warp).
    warped = pool.load_warped(str(path), 0.0, 0.25, 0.125)
    assert warped is not None
    assert abs(len(warped) / sr - 0.125) < 0.02
    assert abs(_peak_hz(warped, sr) - 440.0) < 20.0
    # Linear resample of the same buffer *does* raise pitch — the old bug.
    from fantasia_core.engine.buffers import resample_to_length

    cheap = resample_to_length(x, len(warped))
    assert _peak_hz(cheap, sr) > 700.0


def test_load_warped_compute_false_is_cache_only(tmp_path):
    sr = 22050
    x = _tone(sr, 330.0, 0.2)
    path = tmp_path / "tone.wav"
    sf.write(path, x, sr)
    pool = AudioPool(sr)
    assert pool.load_warped(str(path), 0.0, 0.2, 0.1, compute=False) is None
    filled = pool.load_warped(str(path), 0.0, 0.2, 0.1, compute=True)
    assert filled is not None
    hit = pool.load_warped(str(path), 0.0, 0.2, 0.1, compute=False)
    assert hit is filled
