"""Time-stretching — change a clip's length without changing its pitch.

``factor`` is a **duration multiplier**: 2.0 = twice as long (half speed), 0.5 =
half as long (double speed). Backends, in order:

1. ``pedalboard.time_stretch`` (core dep; bundles Rubber Band)
2. ``pyrubberband`` + the ``rubberband`` CLI, if installed
3. ``librosa.effects.time_stretch`` (phase vocoder)

Never falls back to varispeed resampling, which would change pitch.
"""

from __future__ import annotations

import shutil

import numpy as np


def available() -> bool:
    """True when a pitch-preserving stretcher can run (always, via pedalboard/librosa)."""
    return True


def _as_framed(x: np.ndarray) -> tuple[np.ndarray, bool]:
    data = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if data.ndim == 1:
        return data.reshape(-1, 1), True
    return data, False


def _stretch_pedalboard(x: np.ndarray, sr: int, factor: float, quality: bool) -> np.ndarray:
    import pedalboard as pb

    data, squeeze = _as_framed(x)
    y = pb.time_stretch(
        data, float(sr), stretch_factor=1.0 / factor, high_quality=quality,
    )
    y = np.asarray(y, dtype=np.float32)
    return y[:, 0] if squeeze and y.ndim == 2 else y


def _stretch_rubberband_cli(x: np.ndarray, sr: int, factor: float) -> np.ndarray:
    import pyrubberband as pyrb

    data, squeeze = _as_framed(x)
    y = np.asarray(pyrb.time_stretch(data, sr, 1.0 / factor), dtype=np.float32)
    return y[:, 0] if squeeze and y.ndim == 2 else y


def _stretch_librosa(x: np.ndarray, sr: int, factor: float) -> np.ndarray:
    import librosa

    data, squeeze = _as_framed(x)
    rate = 1.0 / factor
    chans = [librosa.effects.time_stretch(data[:, c], rate=rate) for c in range(data.shape[1])]
    y = np.stack(chans, axis=1).astype(np.float32)
    return y[:, 0] if squeeze else y


def stretch(x: np.ndarray, sr: int, factor: float, quality: bool = True) -> np.ndarray:
    """Stretch ``x`` by ``factor`` (duration multiplier), preserving pitch."""
    factor = max(0.05, min(20.0, float(factor)))
    if abs(factor - 1.0) < 1e-3:
        return np.asarray(x, dtype=np.float32)

    errors: list[str] = []
    try:
        return _stretch_pedalboard(x, sr, factor, quality)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pedalboard: {exc}")
    if shutil.which("rubberband") is not None:
        try:
            return _stretch_rubberband_cli(x, sr, factor)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"rubberband: {exc}")
    try:
        return _stretch_librosa(x, sr, factor)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"librosa: {exc}")
    raise RuntimeError("pitch-preserving time-stretch failed: " + "; ".join(errors))


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
