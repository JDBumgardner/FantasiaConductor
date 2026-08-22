"""Text-to-speech, two backends, both on the Apple GPU via MLX — which is why
neither hits the PyTorch-MPS hang that afflicts MusicGen. Headless (no Qt).

* **kokoro** (default) — ~300MB, loads in ~10s, synthesis is near-instant, and
  picks a timbre from a fixed list of nine. 24kHz mono.
* **clone** — Chatterbox Turbo (Resemble AI), ~0.7GB 8-bit: reproduces a voice
  zero-shot from a few seconds of reference audio, no training. Runs about 3-4x
  slower than real time, so repeated short calls go through the syllable cache
  below. 24kHz.

The clone backend has no built-in voices — the timbre comes entirely from the
reference clip — so it needs a ``ref_voice`` from :mod:`fantasia_core.voices`.

Fish Speech S2 Pro was the first choice here and is a stronger model, but every
build of it drives this 8GB machine into swap: the 11GB bf16 weights were killed
before producing anything, and the 4.1GB 4-bit build took 1438s to synthesize
3.95s of audio (364x slower than real time). Chatterbox is the largest cloning
model that actually fits alongside the DAW; on a 32GB+ box, revisiting Fish is
worthwhile.
"""

from __future__ import annotations

import hashlib
import os
from typing import List, Optional, Tuple

import numpy as np

_MODEL = None
_CLONE = None
_NAME = os.environ.get("FANTASIA_KOKORO", "mlx-community/Kokoro-82M-bf16")
CLONE_NAME = os.environ.get("FANTASIA_CLONE", "mlx-community/chatterbox-turbo-8bit")
KOKORO_SR = 24000
CLONE_SR = 24000
DEFAULT_VOICE = "af_heart"
BACKENDS = ("kokoro", "clone")
DEFAULT_BACKEND = os.environ.get("FANTASIA_TTS_BACKEND", "kokoro")

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


def clone_downloaded() -> bool:
    """True if the cloning weights are already in the HF cache.

    Checked before offering the backend in the UI so a click can't silently kick
    off a download.
    """
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(CLONE_NAME, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def backends() -> List[Tuple[str, str]]:
    """``(id, label)`` for the backends usable right now."""
    out = [("kokoro", "Kokoro — fast, nine fixed voices")]
    if available():
        out.append(("clone", "Chatterbox — clones a reference voice"
                             + ("" if clone_downloaded() else "  (downloads ~0.7GB)")))
    return out


def _load():
    global _MODEL
    if _MODEL is None:
        from mlx_audio.tts.utils import load_model

        _MODEL = load_model(_NAME)
    return _MODEL


def _load_clone():
    global _CLONE
    if _CLONE is None:
        from mlx_audio.tts.utils import load_model

        _CLONE = load_model(CLONE_NAME)
    return _CLONE


def unload_clone() -> None:
    """Drop the cloning weights. Worth calling before a heavy render on a small
    machine — fitting in memory is this backend's whole reason for existing."""
    global _CLONE
    _CLONE = None


def _synth_kokoro(text: str, voice: str, speed: float):
    model = _load()
    lang = "b" if voice.startswith("b") else "a"  # British vs American G2P
    segs, sr = [], KOKORO_SR
    for r in model.generate(text=text, voice=voice, speed=float(speed), lang_code=lang):
        segs.append(np.asarray(r.audio))
        sr = getattr(r, "sample_rate", KOKORO_SR)
    if not segs:
        return np.zeros((0,), dtype=np.float32), sr
    return np.concatenate(segs).astype(np.float32), sr


def _synth_clone(text: str, ref_voice, exaggeration: float, temperature: float):
    from fantasia_core import voices as voice_cat

    ref = voice_cat.get(ref_voice) if isinstance(ref_voice, str) else ref_voice
    if ref is None:
        raise ValueError(
            "the clone backend has no built-in voices — pick a reference voice "
            "from the catalog (fantasia_core.voices) or seed it with build_builtin()")

    model = _load_clone()
    segs, sr = [], CLONE_SR
    # ref_audio is raw samples; sample_rate tells Chatterbox what rate they are at.
    for r in model.generate(text=text, ref_audio=voice_cat.load_ref(ref),
                            sample_rate=voice_cat.REF_SR,
                            exaggeration=float(exaggeration),
                            temperature=float(temperature), verbose=False):
        segs.append(np.asarray(r.audio))
        sr = getattr(r, "sample_rate", CLONE_SR)
    if not segs:
        return np.zeros((0,), dtype=np.float32), sr
    return np.concatenate(segs).astype(np.float32), sr


def synthesize(text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0,
               backend: Optional[str] = None, ref_voice=None,
               exaggeration: float = 0.0, temperature: float = 0.8,
               cache: bool = False):
    """Return ``(mono float32 audio, sample_rate)`` for the spoken text.

    ``ref_voice`` (a slug or a :class:`~fantasia_core.voices.RefVoice`) selects
    the clone backend implicitly — a reference clip is only meaningful there.
    ``cache`` memoizes short utterances — singing calls this once per syllable
    and lyrics repeat heavily, which matters a lot at cloning speeds.
    """
    backend = backend or ("clone" if ref_voice is not None else DEFAULT_BACKEND)
    if backend not in BACKENDS:
        raise ValueError(f"unknown TTS backend {backend!r}; expected one of {BACKENDS}")

    key = None
    if cache:
        slug = getattr(ref_voice, "slug", ref_voice)
        key = _cache_key(text, backend, voice, slug, speed, exaggeration)
        hit = _cache_get(key)
        if hit is not None:
            return hit

    if backend == "clone":
        audio, sr = _synth_clone(text, ref_voice, exaggeration, temperature)
    else:
        audio, sr = _synth_kokoro(text, voice, speed)

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio = (audio / peak * 0.95).astype(np.float32)
    if key is not None:
        _cache_put(key, audio, sr)
    return audio, sr


# --- syllable cache ---------------------------------------------------------
# Cloning runs ~4x slower than real time; a sung line asks for one utterance per
# syllable and syllables repeat, so caching turns most of a verse into lookups.
_MEM: dict = {}
_MEM_MAX = 512


def _cache_key(*parts) -> str:
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _cache_dir() -> str:
    d = os.path.join(os.getcwd(), ".fantasia_cache", "tts")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_get(key: str):
    if key in _MEM:
        return _MEM[key]
    path = os.path.join(_cache_dir(), key + ".npy")
    if os.path.exists(path):
        try:
            d = np.load(path, allow_pickle=False)
            val = (d.astype(np.float32), int(_SR_OF.get(key) or _read_sr(key)))
            _MEM[key] = val
            return val
        except Exception:  # noqa: BLE001 — a corrupt cache entry just misses
            return None
    return None


_SR_OF: dict = {}


def _read_sr(key: str) -> int:
    path = os.path.join(_cache_dir(), key + ".sr")
    try:
        return int(open(path).read().strip())
    except Exception:  # noqa: BLE001
        return CLONE_SR


def _cache_put(key: str, audio: np.ndarray, sr: int) -> None:
    if len(_MEM) >= _MEM_MAX:
        _MEM.clear()
    _MEM[key] = (audio, sr)
    _SR_OF[key] = sr
    try:
        np.save(os.path.join(_cache_dir(), key + ".npy"), audio)
        with open(os.path.join(_cache_dir(), key + ".sr"), "w") as fh:
            fh.write(str(sr))
    except Exception:  # noqa: BLE001 — cache is an optimization, never fatal
        pass


def synthesize_to_file(text: str, path: str, voice: str = DEFAULT_VOICE,
                       speed: float = 1.0, sr_out: int = 44100,
                       backend: Optional[str] = None, ref_voice=None,
                       exaggeration: float = 0.0) -> float:
    """Synthesize and write a mono WAV at ``sr_out``; returns duration in seconds."""
    import soundfile as sf

    audio, sr = synthesize(text, voice, speed, backend=backend,
                           ref_voice=ref_voice, exaggeration=exaggeration)
    if sr != sr_out and len(audio):
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_out).astype(np.float32)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    sf.write(path, audio, sr_out, subtype="PCM_16")
    return len(audio) / sr_out
