"""Voice conversion — recast an existing vocal in someone else's timbre.

This is the fix for the vocoder character that :mod:`fantasia_core.sing` leaves
behind. WORLD gets the notes right but sounds synthetic; running the result
through a neural voice-conversion model re-renders it with a real voice's
texture while keeping the performance.

Uses Seed-VC, which is *zero-shot*: it clones from a reference clip with no
training step, so it shares the catalog in :mod:`fantasia_core.voices` with the
TTS cloning backend. (Classic RVC would need a trained model per voice, hours of
it, and its Python package pins fairseq and numpy<=1.23 — both unusable here.)

Three things about the upstream library are load-bearing, all verified by
measurement rather than docs:

* **CPU only.** seed_vc picks its device at import, preferring MPS, and MPS then
  dies on a float64 op inside the pitch extractor. MPS has to be hidden *before*
  the import.
* **The streaming path is the only working one.** ``api.inference`` without
  ``streaming`` always calls ``load_models_realtime``, which sets ``f0_fn=None``
  — so pitch conditioning, i.e. singing, raises TypeError. ``streaming=True``
  plus ``realtime=False`` routes through the loader that does build the
  extractor.
* **auto_f0_adjust must stay off.** It shifts the performance into the target
  speaker's comfortable range: a C4-D4-E4-G4 melody came back at MIDI 43-53,
  transposed down more than an octave. Fine for speech, ruinous for a scored
  vocal.

Roughly 10x slower than real time on CPU once warm, plus a ~27s one-off model
load, so this is an offline render — never call it on the audio thread.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

# One loaded model + precomputed target features, keyed by reference-voice slug.
# Reloading per call costs ~85s, which dominated everything before caching.
_STATE: dict = {}
_MODEL_SR = 44100
# Diffusion steps past ~10 cost time without measurably changing the output:
# 4 steps and 10 steps both landed at ~90s for the same clip.
DEFAULT_STEPS = 10
# Long inputs are converted in windows and crossfaded; the model's own context
# window is 30s, and staying well inside it keeps memory flat on a small machine.
CHUNK_SECONDS = 10.0
CROSSFADE_SECONDS = 0.12


def available() -> bool:
    try:
        import seed_vc  # noqa: F401
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def checkpoint_dir() -> str:
    """Where the converter's weights live (~3.5GB)."""
    d = os.path.join(os.getcwd(), ".fantasia_cache", "seedvc")
    os.makedirs(d, exist_ok=True)
    return d


def _api():
    """Import seed_vc's API with the compatibility fixes applied first."""
    import torch

    # Must precede the seed_vc import: api.py resolves its device at module load.
    torch.backends.mps.is_available = lambda: False

    # seed_vc downloads into a literal "./checkpoints" next to the working
    # directory, i.e. into the repo. Redirect before any seed_vc module binds
    # the name via `from .hf_utils import load_custom_model_from_hf`.
    from seed_vc import hf_utils as _hf

    if not getattr(_hf, "_fantasia_patched", False):
        from huggingface_hub import hf_hub_download

        def _load(repo_id, model_filename="pytorch_model.bin", config_filename=None):
            cache = checkpoint_dir()
            model = hf_hub_download(repo_id=repo_id, filename=model_filename,
                                    cache_dir=cache)
            if config_filename is None:
                return model
            return model, hf_hub_download(repo_id=repo_id, filename=config_filename,
                                          cache_dir=cache)

        _hf.load_custom_model_from_hf = _load
        _hf._fantasia_patched = True

    from seed_vc.modules.bigvgan import bigvgan as _bv

    if not getattr(_bv.BigVGAN, "_fantasia_patched", False):
        # BigVGAN's PyTorchModelHubMixin override still declares proxies and
        # resume_download as required; huggingface_hub stopped passing them.
        # Defaulting them here beats pinning an old hub, which mlx-audio needs
        # current.
        orig = _bv.BigVGAN._from_pretrained.__func__
        _bv.BigVGAN._from_pretrained = classmethod(
            lambda cls, **kw: orig(cls, **{"proxies": None, "resume_download": None, **kw}))
        _bv.BigVGAN._fantasia_patched = True

    from seed_vc import api

    return api


def _audio_data(api, mono: np.ndarray, sr: int):
    """Wrap mono float32 as seed_vc's AudioData, whose samples are int16."""
    mono = np.clip(np.asarray(mono, dtype=np.float32), -1.0, 1.0)
    return api.AudioData(mono * 32767.0, None, len(mono) / sr, len(mono), sr, None)


def _state_for(api, ref, slug: str):
    st = _STATE.get(slug)
    if st is None:
        from fantasia_core import voices as voice_cat

        target = _audio_data(api, voice_cat.load_ref(ref), voice_cat.REF_SR)
        st = (api.create_v1_stream_state(target=target, f0_condition=True,
                                         fp16=False, realtime=False), target)
        _STATE[slug] = st
    return st


def unload() -> None:
    """Release the converter's models (a few hundred MB)."""
    _STATE.clear()


def convert(audio: np.ndarray, sr: int, ref_voice, *, steps: int = DEFAULT_STEPS,
            semitones: int = 0, fit_range: bool = False,
            progress=None) -> Tuple[np.ndarray, int]:
    """Recast ``audio`` in the timbre of ``ref_voice``; returns ``(mono, sr)``.

    ``fit_range`` transposes the performance into the target's natural range.
    Off by default because it moves written notes — see the module docstring.
    """
    from fantasia_core import voices as voice_cat

    ref = voice_cat.get(ref_voice) if isinstance(ref_voice, str) else ref_voice
    if ref is None:
        raise ValueError(f"no reference voice named {ref_voice!r} in the catalog")

    api = _api()
    state, target = _state_for(api, ref, ref.slug)

    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=1 if mono.shape[0] > mono.shape[1] else 0)
    if not len(mono):
        return np.zeros((0,), dtype=np.float32), sr

    step = int(CHUNK_SECONDS * sr)
    fade = int(CROSSFADE_SECONDS * sr)
    pieces, out_sr = [], sr
    starts = list(range(0, len(mono), step)) or [0]
    for i, start in enumerate(starts):
        if progress:
            progress(i, len(starts))
        # Overlap each window by the crossfade so the seams can be blended.
        lo = max(0, start - (fade if i else 0))
        chunk = mono[lo:start + step]
        res = api.inference(source=_audio_data(api, chunk, sr), target=target,
                            f0_condition=True, auto_f0_adjust=bool(fit_range),
                            semi_tone_shift=int(semitones), diffusion_steps=int(steps),
                            fp16=False, streaming=True, realtime=False,
                            end_of_stream=True, stream_state=state)
        w = api.get_audio_numpy(res).astype(np.float32)
        out_sr = int(res.sample_rate)
        pieces.append(w)

    out = pieces[0]
    for nxt in pieces[1:]:
        n = min(fade, len(out), len(nxt))
        if n > 0:
            ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
            out = np.concatenate([out[:-n], out[-n:] * (1 - ramp) + nxt[:n] * ramp, nxt[n:]])
        else:
            out = np.concatenate([out, nxt])

    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0:
        out = (out / peak * 0.95).astype(np.float32)
    if progress:
        progress(len(starts), len(starts))
    return out.astype(np.float32), out_sr


def convert_file(in_path: str, out_path: str, ref_voice, *, sr_out: int = 44100,
                 steps: int = DEFAULT_STEPS, semitones: int = 0,
                 fit_range: bool = False, progress=None) -> float:
    """Convert a WAV on disk; returns the output duration in seconds."""
    import soundfile as sf

    y, sr = sf.read(in_path, dtype="float32", always_2d=True)
    out, out_sr = convert(y.mean(axis=1), sr, ref_voice, steps=steps,
                          semitones=semitones, fit_range=fit_range, progress=progress)
    if out_sr != sr_out and len(out):
        import librosa

        out = librosa.resample(out, orig_sr=out_sr, target_sr=sr_out).astype(np.float32)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    sf.write(out_path, out, sr_out, subtype="PCM_16")
    return len(out) / sr_out
