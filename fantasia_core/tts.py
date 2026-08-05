"""Text-to-speech via Kokoro (mlx-audio) — runs natively on the Apple GPU (MLX),
so it sidesteps the PyTorch-MPS hang that afflicts MusicGen. Headless (no Qt).

The model (~few hundred MB) downloads on first use and loads in ~10s; synthesis
is fast. Kokoro outputs 24kHz mono; we resample to the project rate on write.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

_MODEL = None
_NAME = os.environ.get("FANTASIA_KOKORO", "mlx-community/Kokoro-82M-bf16")
KOKORO_SR = 24000
DEFAULT_VOICE = "af_heart"

# (id, human label). Prefix: a=American / b=British ; f=female / m=male.
VOICES: List[Tuple[str, str]] = [
    ("af_heart", "American · Female · Heart"),
    ("af_bella", "American · Female · Bella"),
    ("af_nicole", "American · Female · Nicole"),
    ("am_michael", "American · Male · Michael"),
    ("am_adam", "American · Male · Adam"),
    ("bf_emma", "British · Female · Emma"),
    ("bf_isabella", "British · Female · Isabella"),
    ("bm_george", "British · Male · George"),
    ("bm_lewis", "British · Male · Lewis"),
]


def available() -> bool:
    try:
        import misaki  # noqa: F401
        import mlx.core  # noqa: F401
        import mlx_audio  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _load():
    global _MODEL
    if _MODEL is None:
        from mlx_audio.tts.utils import load_model

        _MODEL = load_model(_NAME)
    return _MODEL


def synthesize(text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0):
    """Return ``(mono float32 audio, sample_rate)`` for the spoken text."""
    model = _load()
    lang = "b" if voice.startswith("b") else "a"  # British vs American G2P
    segs, sr = [], KOKORO_SR
    for r in model.generate(text=text, voice=voice, speed=float(speed), lang_code=lang):
        segs.append(np.asarray(r.audio))
        sr = getattr(r, "sample_rate", KOKORO_SR)
    if not segs:
        return np.zeros((0,), dtype=np.float32), sr
    audio = np.concatenate(segs).astype(np.float32)
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio / peak * 0.95).astype(np.float32), sr


def synthesize_to_file(text: str, path: str, voice: str = DEFAULT_VOICE,
                       speed: float = 1.0, sr_out: int = 44100) -> float:
    """Synthesize and write a mono WAV at ``sr_out``; returns duration in seconds."""
    import soundfile as sf

    audio, sr = synthesize(text, voice, speed)
    if sr != sr_out and len(audio):
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_out).astype(np.float32)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    sf.write(path, audio, sr_out, subtype="PCM_16")
    return len(audio) / sr_out
