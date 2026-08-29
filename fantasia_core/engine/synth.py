"""A small subtractive synthesizer (pure NumPy) for synth tracks.

Signal path per note: 3 oscillators (slightly detuned saw stack by default)
→ resonant low-pass filter (with a coarse envelope sweep) → ADSR amplitude
envelope → gain. A clip's notes are summed. Like the MIDI renderer, results
are cached and only rendered off the audio thread (:meth:`render`/:meth:`warm`);
the audio callback reads the cache (:meth:`cached`).

The patch is a plain dict so it serializes with the track and drives the synth
panel UI directly.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

WAVEFORMS = ["sine", "saw", "square", "triangle"]

# Default voice: three slightly detuned saws through a low-pass so a new
# MIDI track is audible without being a raw buzzy stack.
DEFAULT_PATCH: Dict[str, object] = {
    "osc1": "saw",
    "osc2": "saw",
    "osc3": "saw",
    "mix": 1.0,          # 0 = osc1 only, 1 = equal detuned trio
    "detune": 0.12,      # osc2/osc3 spread in semitones (±)
    "attack": 0.01,      # s
    "decay": 0.22,       # s
    "sustain": 0.68,     # 0..1
    "release": 0.20,     # s
    "cutoff": 1600.0,    # Hz (base) — audible, still filtered
    "resonance": 0.28,   # 0..1
    "env_amount": 800.0,  # Hz added at envelope peak
    "gain": 0.48,        # 0..1
}


def _osc(kind: str, freq: float, t: np.ndarray) -> np.ndarray:
    phase = freq * t
    if kind == "sine":
        return np.sin(2 * np.pi * phase)
    if kind == "square":
        return np.sign(np.sin(2 * np.pi * phase))
    if kind == "triangle":
        return 2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5))) - 1.0
    # saw (default)
    return 2.0 * (phase - np.floor(phase + 0.5))


def _adsr(patch, dur: float, n: int, sr: int) -> np.ndarray:
    a = max(1, int(float(patch["attack"]) * sr))
    d = max(1, int(float(patch["decay"]) * sr))
    s = float(patch["sustain"])
    r = max(1, int(float(patch["release"]) * sr))
    sus_end = int(dur * sr)
    env = np.zeros(n, dtype=np.float32)

    ae = min(a, n)
    if ae > 0:
        env[:ae] = np.linspace(0.0, 1.0, ae, endpoint=False)
    de = min(a + d, sus_end, n)
    if de > ae:
        env[ae:de] = np.linspace(1.0, s, de - ae, endpoint=False)
    se = min(sus_end, n)
    if se > de:
        env[de:se] = s
    re = min(se + r, n)
    if re > se:
        env[se:re] = np.linspace(env[se - 1] if se > 0 else s, 0.0, re - se, endpoint=False)
    return env


def _lowpass(fc: float, q: float, sr: int) -> Tuple[list, list]:
    """RBJ resonant low-pass biquad coefficients."""
    fc = max(30.0, min(fc, sr * 0.45))
    w0 = 2.0 * math.pi * fc / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * max(0.5, q))
    b0, b1, b2 = (1 - cw) / 2, 1 - cw, (1 - cw) / 2
    a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
    return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]


def _filter(sig: np.ndarray, patch, env: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import lfilter, lfilter_zi

    q = 0.6 + float(patch["resonance"]) * 8.0
    base = float(patch["cutoff"])
    amt = float(patch["env_amount"])
    n = len(sig)
    chunks = 16
    idx = np.linspace(0, n, chunks + 1).astype(int)
    out = np.empty_like(sig)
    zi = None
    for i in range(chunks):
        a0, a1 = idx[i], idx[i + 1]
        if a1 <= a0:
            continue
        env_v = float(env[a0]) if a0 < len(env) else 0.0
        b, a = _lowpass(base + amt * env_v, q, sr)
        if zi is None:
            zi = lfilter_zi(b, a) * sig[a0]
        seg, zi = lfilter(b, a, sig[a0:a1], zi=zi)
        out[a0:a1] = seg
    return out


def render_note(patch, pitch: int, dur: float, sr: int) -> np.ndarray:
    freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
    n = int((dur + float(patch["release"])) * sr)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n) / sr
    mix = float(patch["mix"])
    detune = float(patch["detune"])
    o1 = _osc(str(patch["osc1"]), freq, t)
    o2 = _osc(str(patch.get("osc2", "saw")), freq * (2.0 ** (detune / 12.0)), t)
    o3 = _osc(str(patch.get("osc3", "saw")), freq * (2.0 ** (-detune / 12.0)), t)
    trio = (o1 + o2 + o3) / 3.0
    sig = ((1.0 - mix) * o1 + mix * trio).astype(np.float32)
    env = _adsr(patch, dur, n, sr)
    sig = _filter(sig, patch, env, sr)
    sig *= env * float(patch["gain"])
    return sig.astype(np.float32)


def render_clip(patch, clip, sr: int) -> np.ndarray:
    patch = {**DEFAULT_PATCH, **(patch or {})}  # fill any unset params with defaults
    total = max(int(clip.duration * sr), 0)
    out = np.zeros((total, 2), dtype=np.float32)
    if total == 0:
        return out
    for note in clip.notes:
        s = int(note.start * sr)
        if s >= total:
            continue
        mono = render_note(patch, int(note.pitch), float(note.duration), sr) * (note.velocity / 127.0)
        e = min(s + len(mono), total)
        length = e - s
        if length <= 0:
            continue
        out[s:e, 0] += mono[:length]
        out[s:e, 1] += mono[:length]
    return out


class SynthRenderer:
    def __init__(self, sample_rate: int = 44100) -> None:
        self.sr = sample_rate
        self._cache: Dict[tuple, np.ndarray] = {}

    def _key(self, clip, patch) -> tuple:  # noqa: ANN001
        notes = tuple(
            (n.pitch, round(n.start, 4), round(n.duration, 4), n.velocity) for n in clip.notes
        )
        patch_sig = tuple(sorted((k, v) for k, v in patch.items()))
        return (patch_sig, round(clip.duration, 4), notes)

    def cached(self, clip, patch) -> Optional[np.ndarray]:  # noqa: ANN001
        return self._cache.get(self._key(clip, patch))

    def render(self, clip, patch) -> np.ndarray:  # noqa: ANN001
        key = self._key(clip, patch)
        buf = self._cache.get(key)
        if buf is None:
            buf = render_clip(patch or {}, clip, self.sr)
            self._cache[key] = buf
        return buf

    def warm(self, project) -> None:  # noqa: ANN001
        for track in project.tracks:
            if not getattr(track, "is_synth", False):
                continue
            # Use the same patch expression as the mixer's cached() lookup so the
            # cache key matches (render_clip fills defaults internally).
            patch = getattr(track, "synth", None) or {}
            for clip in track.clips:
                if clip.content_type == "midi":
                    self.render(clip, patch)
