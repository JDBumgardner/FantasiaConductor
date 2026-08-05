"""Audio → MIDI transcription via Spotify's ``basic-pitch`` (polyphonic).

Kept headless (no Qt). ``basic-pitch`` (and TensorFlow) is imported lazily inside
the functions so importing this module — or the app — stays cheap; the heavy
model only loads when transcription actually runs (off the UI thread).

Notes on the pinned model/back-end:
* We use the bundled **TFLite** model (``nmp.tflite``) — it loads cleanly and is
  fast, avoiding TF-SavedModel loading issues under TensorFlow 2.16 / Keras 3.
* A small shim restores ``scipy.signal.gaussian`` (moved to ``scipy.signal.windows``
  in newer SciPy) that basic-pitch 0.3.0 still expects.
"""

from __future__ import annotations

import os
import tempfile
from typing import List

import numpy as np

from fantasia_core.document.model import Note


def available() -> bool:
    try:
        import basic_pitch  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _tflite_model_path() -> str:
    import pathlib

    from basic_pitch import ICASSP_2022_MODEL_PATH

    return str(pathlib.Path(ICASSP_2022_MODEL_PATH).parent / "nmp.tflite")


def transcribe_file(path: str) -> List[Note]:
    """Transcribe an audio file to a list of :class:`Note` (times from t=0)."""
    import scipy.signal

    if not hasattr(scipy.signal, "gaussian"):  # basic-pitch 0.3.0 compat
        scipy.signal.gaussian = scipy.signal.windows.gaussian

    from basic_pitch.inference import predict

    _, _, events = predict(path, _tflite_model_path())
    notes: List[Note] = []
    for ev in events:
        start, end, pitch, amp = float(ev[0]), float(ev[1]), int(ev[2]), float(ev[3])
        notes.append(
            Note(
                pitch=pitch,
                start=max(0.0, start),
                duration=max(0.05, end - start),
                velocity=int(max(1, min(127, round(amp * 127)))),
            )
        )
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def transcribe_audio(samples: np.ndarray, sr: int) -> List[Note]:
    """Transcribe an in-memory buffer ``(frames,)`` or ``(frames, ch)``."""
    import soundfile as sf

    mono = samples if samples.ndim == 1 else samples.mean(axis=1)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        sf.write(tmp.name, mono.astype(np.float32), sr, subtype="PCM_16")
        return transcribe_file(tmp.name)
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass
