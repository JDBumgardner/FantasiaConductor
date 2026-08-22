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

        # ---- EQ bands -----------------------------------------------------
        # A "bell": boost/cut a band centred on freq. Q sets its width
        # (0.7 broad and musical, 4+ surgical).
        if kind == "eq_peak":
            return pb.PeakFilter(cutoff_frequency_hz=p.get("freq", 1000.0),
                                 gain_db=p.get("gain", 0.0), q=p.get("q", 1.0))
        # Shelves tilt everything above/below the corner instead of a band.
        if kind == "eq_low_shelf":
            return pb.LowShelfFilter(cutoff_frequency_hz=p.get("freq", 200.0),
                                     gain_db=p.get("gain", 0.0), q=p.get("q", 0.7))
        if kind == "eq_high_shelf":
            return pb.HighShelfFilter(cutoff_frequency_hz=p.get("freq", 6000.0),
                                      gain_db=p.get("gain", 0.0), q=p.get("q", 0.7))

        # ---- dynamics -----------------------------------------------------
        if kind == "compressor":
            return pb.Compressor(
                threshold_db=p.get("threshold", -16.0), ratio=p.get("ratio", 4.0),
                attack_ms=p.get("attack", 10.0), release_ms=p.get("release", 100.0),
            )
        if kind == "limiter":
            # NOT pedalboard's Limiter: that one applies makeup gain up to the
            # threshold (a maximizer), so it can push a quiet track to full
            # scale — the opposite of what you want on an insert. A compressor
            # with a near-infinite ratio and fast attack caps predictably.
            return pb.Compressor(
                threshold_db=p.get("threshold", -1.0), ratio=p.get("ratio", 20.0),
                attack_ms=p.get("attack", 1.0), release_ms=p.get("release", 100.0),
            )
        if kind == "gate":
            return pb.NoiseGate(threshold_db=p.get("threshold", -50.0),
                                ratio=p.get("ratio", 4.0),
                                attack_ms=p.get("attack", 1.0),
                                release_ms=p.get("release", 100.0))

        # ---- colour -------------------------------------------------------
        # Saturation = drive into a soft curve for harmonics, then pull the
        # level back so it warms rather than just gets louder.
        if kind == "saturator":
            drive = float(p.get("drive", 5.0))
            return pb.Pedalboard([
                pb.Distortion(drive_db=drive),
                pb.Gain(gain_db=p.get("output", -drive * 0.6)),
            ])
        if kind == "gain":
            return pb.Gain(gain_db=p.get("gain", 0.0))
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
