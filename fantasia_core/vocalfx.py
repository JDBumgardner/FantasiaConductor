"""WORLD-vocoder vocal engine: formant-preserving pitch operations.

The WORLD vocoder decomposes a voice into three streams — fundamental pitch
(f0), spectral envelope (sp, the formants/timbre) and aperiodicity (ap, the
breathy/voiced balance). Editing f0 or sp and resynthesizing changes pitch or
character *without* the chipmunk formant-shift that plain pitch-shifting causes.

This module is the shared core for both singing synthesis (:mod:`sing`) and the
vocal-FX tools (autotune, harmony, formant shift, de-ess). Headless, CPU-only.
"""

from __future__ import annotations

import numpy as np

FRAME_PERIOD = 5.0  # ms per WORLD frame

_SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "pentatonic": [0, 2, 4, 7, 9],
    "chromatic": list(range(12)),
}
_NOTE_PC = {"c": 0, "c#": 1, "db": 1, "d": 2, "d#": 3, "eb": 3, "e": 4, "f": 5,
            "f#": 6, "gb": 6, "g": 7, "g#": 8, "ab": 8, "a": 9, "a#": 10, "bb": 10, "b": 11}


def available() -> bool:
    try:
        import pyworld  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


# ---- WORLD analysis / synthesis -----------------------------------------
def to_mono(x: np.ndarray) -> np.ndarray:
    """Coerce any (frames,) or (frames, channels) buffer to a 1-D mono float64
    array. AudioPool returns 2-D (frames, channels) even for mono, and WORLD /
    librosa need 1-D — so every entry point downmixes here."""
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float64)


def analyze(x: np.ndarray, sr: int):
    """Return (f0, sp, ap) for a signal (downmixed to mono)."""
    import pyworld as pw

    x = to_mono(x)
    f0, t = pw.harvest(x, sr, frame_period=FRAME_PERIOD)
    f0 = pw.stonemask(x, f0, t, sr)  # refine
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    return f0, sp, ap


def synth(f0: np.ndarray, sp: np.ndarray, ap: np.ndarray, sr: int) -> np.ndarray:
    import pyworld as pw

    y = pw.synthesize(np.ascontiguousarray(f0.astype(np.float64)),
                      np.ascontiguousarray(sp.astype(np.float64)),
                      np.ascontiguousarray(ap.astype(np.float64)), sr, FRAME_PERIOD)
    return y.astype(np.float32)


def _norm(x: np.ndarray, peak: float = 0.97) -> np.ndarray:
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return (x * (peak / p)).astype(np.float32) if p > peak else x.astype(np.float32)


def resample_frames(arr: np.ndarray, target: int) -> np.ndarray:
    """Linearly resample along the frame axis (axis 0) to ``target`` frames."""
    n = arr.shape[0]
    if n == target or n == 0:
        return arr
    src = np.linspace(0, n - 1, target)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = src - lo
    if arr.ndim == 1:
        return arr[lo] * (1 - frac) + arr[hi] * frac
    return arr[lo] * (1 - frac)[:, None] + arr[hi] * frac[:, None]


def vibrato_curve(n_frames: int, rate: float = 5.5, depth_semi: float = 0.28,
                  onset_s: float = 0.18) -> np.ndarray:
    """Multiplicative pitch curve for vibrato with a delayed onset."""
    tt = np.arange(n_frames) * (FRAME_PERIOD / 1000.0)
    onset = np.clip((tt - onset_s) / max(onset_s, 1e-3), 0.0, 1.0)
    return 2.0 ** (depth_semi / 12.0 * np.sin(2 * np.pi * rate * tt) * onset)


# ---- pitch helpers -------------------------------------------------------
def midi_to_hz(m):
    return 440.0 * (2.0 ** ((np.asarray(m) - 69) / 12.0))


def hz_to_midi(hz):
    hz = np.asarray(hz, dtype=np.float64)
    out = np.zeros_like(hz)
    pos = hz > 0
    out[pos] = 69 + 12 * np.log2(hz[pos] / 440.0)
    return out


def key_to_pc(key) -> int:
    if isinstance(key, str):
        return _NOTE_PC.get(key.strip().lower(), 0)
    return int(key) % 12


def _snap_midi(m: float, key_pc: int, allowed: list) -> float:
    base = int(round(m))
    for d in range(0, 8):
        for cand in (base - d, base + d):
            if (cand - key_pc) % 12 in allowed:
                return float(cand)
    return float(base)


# ---- vocal FX ------------------------------------------------------------
def autotune(x: np.ndarray, sr: int, key="c", scale: str = "major",
             strength: float = 1.0) -> np.ndarray:
    """Snap the voice's pitch to the nearest note in a scale (formant-preserving)."""
    f0, sp, ap = analyze(x, sr)
    voiced = f0 > 0
    allowed = _SCALES.get(scale, _SCALES["major"])
    kpc = key_to_pc(key)
    midi = hz_to_midi(f0)
    new_midi = midi.copy()
    for i in np.nonzero(voiced)[0]:
        snapped = _snap_midi(midi[i], kpc, allowed)
        new_midi[i] = midi[i] + float(strength) * (snapped - midi[i])
    new_f0 = np.where(voiced, midi_to_hz(new_midi), 0.0)
    return _norm(synth(new_f0, sp, ap, sr))


def shift_pitch(x: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Formant-preserving pitch shift (for harmony voices)."""
    f0, sp, ap = analyze(x, sr)
    new_f0 = np.where(f0 > 0, f0 * (2.0 ** (semitones / 12.0)), 0.0)
    return _norm(synth(new_f0, sp, ap, sr))


def formant_shift(x: np.ndarray, sr: int, ratio: float = 1.0) -> np.ndarray:
    """Warp the spectral envelope's frequency axis (>1 = brighter/smaller, <1 =
    darker/larger). Changes vocal character/gender without changing pitch."""
    f0, sp, ap = analyze(x, sr)
    bins = sp.shape[1]
    src = np.clip(np.arange(bins) / max(ratio, 1e-3), 0, bins - 1)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, bins - 1)
    frac = src - lo
    sp2 = sp[:, lo] * (1 - frac) + sp[:, hi] * frac
    return _norm(synth(f0, sp2, ap, sr))


def deess(x: np.ndarray, sr: int, amount: float = 0.6) -> np.ndarray:
    """Tame sibilance: attenuate the 5–9kHz band on frames where it spikes."""
    x = to_mono(x).astype(np.float32)
    n_fft = 1024
    hop = 256
    try:
        import librosa

        S = librosa.stft(x, n_fft=n_fft, hop_length=hop)
    except Exception:  # noqa: BLE001
        return x
    freqs = np.linspace(0, sr / 2, S.shape[0])
    band = (freqs >= 5000) & (freqs <= 9000)
    mag = np.abs(S)
    band_energy = mag[band].mean(axis=0) + 1e-9
    thresh = np.median(band_energy) * 1.6
    over = np.clip(band_energy / thresh, 1.0, 4.0)
    gain = np.ones_like(band_energy)
    gain = np.where(band_energy > thresh, 1.0 - amount * (1 - 1 / over), 1.0)
    S[band] *= gain[None, :]
    import librosa

    return librosa.istft(S, hop_length=hop, length=len(x)).astype(np.float32)


def double(x: np.ndarray, sr: int, detune_cents: float = 12.0,
           delay_ms: float = 22.0) -> np.ndarray:
    """Vocal doubling: blend a slightly detuned, slightly delayed copy for thickness."""
    x = to_mono(x).astype(np.float32)
    copy = shift_pitch(x, sr, detune_cents / 100.0)
    d = int(sr * delay_ms / 1000.0)
    if d > 0:
        copy = np.concatenate([np.zeros(d, dtype=np.float32), copy])[: len(x)]
    if len(copy) < len(x):
        copy = np.pad(copy, (0, len(x) - len(copy)))
    mix = x + 0.7 * copy[: len(x)]
    peak = float(np.max(np.abs(mix))) or 1.0
    return (mix / peak * 0.97).astype(np.float32)
