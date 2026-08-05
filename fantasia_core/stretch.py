"""Time-stretching — change a clip's length without changing its pitch, via the
Rubber Band library (`pyrubberband`, which shells out to the `rubberband` CLI).

``factor`` is a **duration multiplier**: 2.0 = twice as long (half speed), 0.5 =
half as long (double speed), pitch preserved either way. High quality; CPU; fast.
Headless (no Qt).
"""

from __future__ import annotations

import shutil

import numpy as np


def available() -> bool:
    try:
        import pyrubberband  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return shutil.which("rubberband") is not None  # the CLI must be installed


def stretch(x: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Stretch ``x`` by ``factor`` (duration multiplier), preserving pitch."""
    import pyrubberband as pyrb

    factor = max(0.1, min(10.0, float(factor)))
    if abs(factor - 1.0) < 1e-3:
        return np.asarray(x, dtype=np.float32)
    data = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    y = pyrb.time_stretch(data, sr, 1.0 / factor)  # pyrb rate = speed = 1/duration-factor
    return np.asarray(y, dtype=np.float32)


def stretch_to_file(x: np.ndarray, sr: int, factor: float, path: str) -> float:
    """Stretch and write a WAV; returns the new duration in seconds."""
    import os

    import soundfile as sf

    y = stretch(x, sr, factor)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    sf.write(path, y, sr, subtype="PCM_16")
    return len(y) / sr
