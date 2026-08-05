"""Local text→audio generation via Meta's MusicGen (Hugging Face transformers).

Headless (no Qt). ``torch`` / ``transformers`` and the model are imported/loaded
lazily so importing this module — or the app — stays cheap; the ~2GB model
downloads on first use and is cached by Hugging Face. Generation is slow on
CPU/MPS, so callers must run :func:`generate` off the UI thread.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

_MODEL = None
_PROCESSOR = None
_MODEL_NAME = os.environ.get("FANTASIA_MUSICGEN", "facebook/musicgen-small")

TOKENS_PER_SEC = 50  # MusicGen's audio-token rate


def available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _device():
    """Device for MusicGen. Defaults to CPU: MusicGen generation *hangs* on MPS
    with the current torch stack (indefinite GPU wait), whereas CPU is slow
    (~2 min per 4 s clip) but reliable. Opt back into MPS with
    ``FANTASIA_MUSICGEN_DEVICE=mps`` once the torch/MPS bug is resolved."""
    override = os.environ.get("FANTASIA_MUSICGEN_DEVICE")
    if override:
        return override
    return "cpu"


def _limit_threads() -> None:
    """Leave CPU headroom for the real-time audio callback + UI while generating.
    Without this, CPU MusicGen pegs every core and playback stutters/drops out."""
    import torch

    try:
        cap = max(1, (os.cpu_count() or 4) - 4)
        torch.set_num_threads(cap)
    except Exception:  # noqa: BLE001
        pass


def _load():
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        _limit_threads()
        _PROCESSOR = AutoProcessor.from_pretrained(_MODEL_NAME)
        _MODEL = MusicgenForConditionalGeneration.from_pretrained(_MODEL_NAME)
        _MODEL.to(_device())
    return _MODEL, _PROCESSOR


def generate(prompt: str, duration: float = 4.0, sr_out: int = 44100,
             guidance: float = 3.0) -> np.ndarray:
    """Generate mono ``float32`` audio (1-D) at ``sr_out`` from a text prompt.

    ``guidance`` is the classifier-free guidance scale: ~3 = best prompt adherence,
    1 = 'draft' (no CFG → a single forward pass per step, roughly 2× faster)."""
    import torch

    model, processor = _load()
    device = next(model.parameters()).device

    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
    max_tokens = int(max(64, min(1500, round(duration * TOKENS_PER_SEC))))
    with torch.no_grad():
        wav = model.generate(**inputs, do_sample=True, guidance_scale=float(guidance),
                             max_new_tokens=max_tokens)

    audio = wav[0, 0].detach().cpu().numpy().astype(np.float32)  # mono
    sr = model.config.audio_encoder.sampling_rate  # typically 32000
    if sr != sr_out:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_out).astype(np.float32)

    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio / peak * 0.9).astype(np.float32)


def generate_to_file(prompt: str, duration: float, sr_out: int, path: str,
                     guidance: float = 3.0) -> float:
    """Generate and write a mono WAV to ``path``; returns its duration in seconds."""
    import soundfile as sf

    audio = generate(prompt, duration, sr_out, guidance=guidance)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    sf.write(path, audio, sr_out, subtype="PCM_16")
    return len(audio) / sr_out
