"""Generate a few short synthesized loops so there's real audio to import/test.

Writes 4 mono 44.1 kHz WAV loops (2 bars @ 120 BPM = 4.0 s) into
``assets/samples/``. Deliberately simple synthesis — enough to hear parts and
test mixing, not to win a Grammy.

Run: ``.venv/bin/python tools/make_demo_audio.py``
"""

from __future__ import annotations

import pathlib

import numpy as np
import soundfile as sf

SR = 44100
BPM = 120.0
BEAT = 60.0 / BPM          # 0.5 s
BAR = BEAT * 4             # 2.0 s
TOTAL = BAR * 2            # 4.0 s, 2-bar loop
N = int(TOTAL * SR)

OUT = pathlib.Path(__file__).resolve().parent.parent / "assets" / "samples"


def _t(n: int) -> np.ndarray:
    return np.arange(n) / SR


def _place(buf: np.ndarray, at_sec: float, sound: np.ndarray) -> None:
    i = int(at_sec * SR)
    j = min(len(buf), i + len(sound))
    if i < len(buf):
        buf[i:j] += sound[: j - i]


def _kick() -> np.ndarray:
    n = int(0.18 * SR)
    t = _t(n)
    freq = 120.0 * np.exp(-t * 30) + 45.0
    phase = 2 * np.pi * np.cumsum(freq) / SR
    env = np.exp(-t * 22)
    return (np.sin(phase) * env * 0.9).astype(np.float32)


def _snare() -> np.ndarray:
    n = int(0.2 * SR)
    t = _t(n)
    noise = np.random.uniform(-1, 1, n)
    tone = np.sin(2 * np.pi * 190 * t)
    env = np.exp(-t * 26)
    return ((0.7 * noise + 0.3 * tone) * env * 0.5).astype(np.float32)


def _hat() -> np.ndarray:
    n = int(0.04 * SR)
    t = _t(n)
    noise = np.random.uniform(-1, 1, n)
    env = np.exp(-t * 180)
    return (noise * env * 0.3).astype(np.float32)


def _pluck(freq: float, dur: float, gain: float = 0.5) -> np.ndarray:
    n = int(dur * SR)
    t = _t(n)
    wave = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    env = np.exp(-t * 6)
    return (wave * env * gain).astype(np.float32)


def _pad(freqs, dur: float, gain: float = 0.28) -> np.ndarray:
    n = int(dur * SR)
    t = _t(n)
    wave = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    attack = np.clip(t / 0.15, 0, 1)
    release = np.clip((dur - t) / 0.3, 0, 1)
    return (wave * attack * release * gain).astype(np.float32)


def make_drums() -> np.ndarray:
    buf = np.zeros(N, dtype=np.float32)
    for bar in (0.0, BAR):
        _place(buf, bar + 0 * BEAT, _kick())
        _place(buf, bar + 2 * BEAT, _kick())
        _place(buf, bar + 1 * BEAT, _snare())
        _place(buf, bar + 3 * BEAT, _snare())
    for k in range(int(TOTAL / (BEAT / 2))):
        _place(buf, k * (BEAT / 2), _hat())
    return buf


def make_bass() -> np.ndarray:
    buf = np.zeros(N, dtype=np.float32)
    pattern = [55.00, 55.00, 43.65, 49.00]  # A1, A1, F1, G1 (i - i - VI - VII)
    for bar in (0.0, BAR):
        for i, f in enumerate(pattern):
            _place(buf, bar + i * BEAT, _pluck(f, BEAT * 0.9, gain=0.6))
    return buf


def make_keys() -> np.ndarray:
    buf = np.zeros(N, dtype=np.float32)
    a_minor = [220.00, 261.63, 329.63]  # A3 C4 E4
    _place(buf, 0.0, _pad(a_minor, BAR - 0.1))
    _place(buf, BAR, _pad(a_minor, BAR - 0.1))
    return buf


def make_lead() -> np.ndarray:
    buf = np.zeros(N, dtype=np.float32)
    melody = [(0.0, 440.0, BEAT), (0.5, 523.25, BEAT), (1.0, 493.88, BEAT * 2),
              (2.0, 659.25, BEAT), (2.5, 587.33, BEAT), (3.0, 440.0, BEAT)]
    for at, f, dur in melody:
        _place(buf, at, _pluck(f, dur, gain=0.4))
    return buf


def main() -> None:
    np.random.seed(7)  # deterministic drums
    OUT.mkdir(parents=True, exist_ok=True)
    tracks = {
        "drums": make_drums(),
        "bass": make_bass(),
        "keys": make_keys(),
        "lead": make_lead(),
    }
    for name, buf in tracks.items():
        peak = float(np.max(np.abs(buf))) or 1.0
        buf = (buf / peak * 0.8).astype(np.float32)  # normalize to -2 dBFS-ish
        path = OUT / f"{name}.wav"
        sf.write(path, buf, SR, subtype="PCM_16")
        print(f"wrote {path}  ({len(buf) / SR:.2f}s)")


if __name__ == "__main__":
    main()
