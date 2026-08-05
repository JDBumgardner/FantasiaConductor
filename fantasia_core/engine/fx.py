"""Per-track effects via `pedalboard`.

An FX chain is stored on a track as a list of ``{"type", "params"}`` dicts
(serializable). :class:`FxHost` turns that into a ``pedalboard.Pedalboard`` and
processes audio, keeping one board per track so effect state (e.g. reverb tails)
carries across playback blocks. Rebuilds only when the spec changes.

pedalboard uses ``(num_channels, num_samples)`` arrays, so we transpose in/out.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pedalboard as pb
except Exception:  # noqa: BLE001
    pb = None


def _make(spec: dict):
    """Build one pedalboard plugin from a spec dict, or None."""
    if pb is None:
        return None
    kind = spec.get("type")
    p = spec.get("params", {})
    try:
        if kind == "reverb":
            return pb.Reverb(
                room_size=p.get("room_size", 0.6),
                wet_level=p.get("wet", 0.35),
                dry_level=p.get("dry", 0.7),
            )
        if kind == "delay":
            return pb.Delay(
                delay_seconds=p.get("time", 0.25),
                feedback=p.get("feedback", 0.3),
                mix=p.get("mix", 0.3),
            )
        if kind == "lowpass":
            return pb.LowpassFilter(cutoff_frequency_hz=p.get("cutoff", 1200.0))
        if kind == "highpass":
            return pb.HighpassFilter(cutoff_frequency_hz=p.get("cutoff", 250.0))
        if kind == "chorus":
            return pb.Chorus()
        if kind == "distortion":
            return pb.Distortion(drive_db=p.get("drive", 12.0))
        if kind == "compressor":
            return pb.Compressor(
                threshold_db=p.get("threshold", -16.0), ratio=p.get("ratio", 4.0)
            )
    except Exception:  # noqa: BLE001 — bad params shouldn't crash audio
        return None
    return None


def build_board(specs: List[dict]):
    if pb is None:
        return None
    plugins = [pl for pl in (_make(s) for s in specs) if pl is not None]
    return pb.Pedalboard(plugins) if plugins else None


class FxHost:
    """Caches a Pedalboard per track and processes stereo blocks statefully."""

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[str, object]] = {}  # track_id -> (sig, board)

    def process(self, track, audio: np.ndarray, sr: int) -> np.ndarray:  # noqa: ANN001
        specs = getattr(track, "fx", None) or []
        if not specs or pb is None:
            return audio
        sig = repr(specs)
        entry = self._cache.get(track.id)
        if entry is None or entry[0] != sig:
            board = build_board(specs)
            self._cache[track.id] = (sig, board)
        else:
            board = entry[1]
        if board is None:
            return audio

        try:
            out = board(audio.T.astype(np.float32), sr, reset=False).T
        except Exception:  # noqa: BLE001
            return audio

        # Streaming plugins return equal length, but clamp defensively so the
        # caller's fixed-size mix buffer is never mismatched.
        n = len(audio)
        if len(out) < n:
            pad = np.zeros((n - len(out), out.shape[1]), dtype=np.float32)
            out = np.vstack([out, pad])
        elif len(out) > n:
            out = out[:n]
        return out
