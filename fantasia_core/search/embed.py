"""CLAP embeddings — text and audio in one joint space (via `transformers`).

``get_text_features`` and ``get_audio_features`` land in the same 512-d space, so
a text query and an audio clip are directly comparable. Vectors are L2-normalized
so a dot product is cosine similarity. The model (~2GB) downloads on first use.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np

_MODEL = None
_PROCESSOR = None
_NAME = os.environ.get("FANTASIA_CLAP", "laion/clap-htsat-unfused")
DIM = 512
CLAP_SR = 48000


def available() -> bool:
    try:
        import torch  # noqa: F401
        from transformers import ClapModel  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _device():
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load():
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        from transformers import ClapModel, ClapProcessor

        _PROCESSOR = ClapProcessor.from_pretrained(_NAME)
        _MODEL = ClapModel.from_pretrained(_NAME).to(_device())
        _MODEL.eval()
    return _MODEL, _PROCESSOR


def _to_numpy(out) -> np.ndarray:
    """CLAP feature getters return a bare tensor on older transformers and a
    ``BaseModelOutputWithPooling`` (projection in ``pooler_output``) on 5.x."""
    tensor = getattr(out, "pooler_output", None)
    if tensor is None:
        tensor = out  # already a tensor
    return tensor.cpu().numpy()


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    return (v / norm).astype(np.float32)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return ``(n, 512)`` normalized text embeddings."""
    import torch

    model, proc = _load()
    inputs = proc(text=list(texts), return_tensors="pt", padding=True).to(_device())
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    return _normalize(_to_numpy(feats))


def embed_audio(samples: np.ndarray, sr: int) -> np.ndarray:
    """Return a single ``(512,)`` normalized audio embedding."""
    import torch

    mono = samples if samples.ndim == 1 else samples.mean(axis=1)
    if sr != CLAP_SR:
        import librosa

        mono = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=CLAP_SR)
    model, proc = _load()
    clip = [mono.astype(np.float32)]
    try:  # transformers 5.x renamed the processor kwarg audios -> audio
        inputs = proc(audio=clip, sampling_rate=CLAP_SR, return_tensors="pt")
    except (TypeError, ValueError):
        inputs = proc(audios=clip, sampling_rate=CLAP_SR, return_tensors="pt")
    inputs = inputs.to(_device())
    with torch.no_grad():
        feats = model.get_audio_features(**inputs)
    return _normalize(_to_numpy(feats))[0]


def embed_audio_file(path: str) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    return embed_audio(data, sr)
