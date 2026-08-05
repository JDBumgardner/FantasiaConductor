"""Instrument/stem separation via Demucs (htdemucs).

Splits any audio file into 4 stems — drums, bass, vocals, other — so a user (or
the agent) can isolate and reuse individual instruments from imported or
generated audio. Headless (no Qt). ``torch``/``demucs`` and the model load
lazily; the ~300MB model downloads on first use. Separation is slow, so callers
must run it off the UI thread.
"""

from __future__ import annotations

import os
import pathlib
import uuid
from typing import List, Tuple

import numpy as np

_MODEL = None
_MODEL_NAME = os.environ.get("FANTASIA_DEMUCS", "htdemucs")


def available() -> bool:
    try:
        import demucs  # noqa: F401
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _device() -> str:
    """CPU by default — same MPS caution as MusicGen. Override with
    ``FANTASIA_DEMUCS_DEVICE=mps``."""
    return os.environ.get("FANTASIA_DEMUCS_DEVICE", "cpu")


def _model():
    global _MODEL
    if _MODEL is None:
        import torch
        from demucs.pretrained import get_model

        try:  # leave CPU headroom for the audio callback + UI during separation
            torch.set_num_threads(max(1, (os.cpu_count() or 4) - 4))
        except Exception:  # noqa: BLE001
            pass
        _MODEL = get_model(_MODEL_NAME)
        _MODEL.eval()
    return _MODEL


def stem_names() -> List[str]:
    return list(_model().sources)


def separate_to_files(path: str, out_dir: str, sr_out: int = 44100) -> List[Tuple[str, str, float]]:
    """Separate ``path`` into stems written as WAVs under ``out_dir``.

    Returns ``[(stem_name, wav_path, duration_seconds), ...]``."""
    import soundfile as sf
    import torch
    from demucs.apply import apply_model

    model = _model()
    model_sr = model.samplerate

    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (n, channels)
    wav = data.T  # (channels, n)
    if wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)  # demucs wants stereo
    elif wav.shape[0] > 2:
        wav = wav[:2]
    if sr != model_sr:
        import librosa

        wav = np.stack([librosa.resample(wav[c], orig_sr=sr, target_sr=model_sr)
                        for c in range(wav.shape[0])])

    tensor = torch.from_numpy(np.ascontiguousarray(wav))
    ref = tensor.mean(0)
    tensor = (tensor - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        sources = apply_model(model, tensor[None], device=_device(), split=True,
                              overlap=0.25, progress=False)[0]
    sources = sources * ref.std() + ref.mean()

    out_dir_p = pathlib.Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    stem_base = pathlib.Path(path).stem
    tag = uuid.uuid4().hex[:6]
    results: List[Tuple[str, str, float]] = []
    for name, src in zip(model.sources, sources):
        arr = src.detach().cpu().numpy()  # (2, n) at model_sr
        if model_sr != sr_out:
            import librosa

            arr = np.stack([librosa.resample(arr[c], orig_sr=model_sr, target_sr=sr_out)
                            for c in range(arr.shape[0])])
        stereo = arr.T.astype(np.float32)  # (n, 2)
        wav_path = str(out_dir_p / f"{stem_base}_{name}_{tag}.wav")
        sf.write(wav_path, stereo, sr_out, subtype="PCM_16")
        results.append((name, wav_path, len(stereo) / sr_out))
    return results
