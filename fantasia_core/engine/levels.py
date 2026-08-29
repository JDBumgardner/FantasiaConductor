"""Per-track peak tap for header meters.

The audio callback only updates a peak hold. The UI timer consumes and paints.
A torn read is preferable to locking or allocating on the callback.
"""

from __future__ import annotations

import math

import numpy as np

CAP = 64


def amp_to_db(amp: float, floor_db: float = -60.0) -> float:
    """Peak amplitude → dBFS. Silence maps to ``floor_db``."""
    if amp <= 1e-8:
        return float(floor_db)
    return max(float(floor_db), min(12.0, 20.0 * math.log10(float(amp))))


class LevelTap:
    """Peak-hold slots filled by the mixer, drained by the UI."""

    def __init__(self) -> None:
        self._ids: list[str] = [""] * CAP
        self._peaks = np.zeros(CAP, dtype=np.float32)
        self._n = 0
        self._index: dict[str, int] = {}

    def reset(self) -> None:
        self._ids = [""] * CAP
        self._peaks.fill(0.0)
        self._n = 0
        self._index.clear()

    def write_peak(self, track_id: str, peak: float) -> None:
        """Audio-thread entry. Must not allocate a large buffer or lock."""
        try:
            if not track_id:
                return
            i = self._index.get(track_id)
            if i is None:
                if self._n >= CAP:
                    return
                i = self._n
                self._n += 1
                self._ids[i] = track_id
                self._index[track_id] = i
            p = float(peak)
            if p > self._peaks[i]:
                self._peaks[i] = p
        except Exception:  # noqa: BLE001 — visualization must never break audio
            return

    def write(self, track_id: str, block: np.ndarray) -> None:
        try:
            if block is None or getattr(block, "size", 0) == 0:
                return
            peak = max(abs(float(np.max(block))), abs(float(np.min(block))))
            self.write_peak(track_id, peak)
        except Exception:  # noqa: BLE001
            return

    def consume(self) -> dict[str, float]:
        """UI thread: current peaks in linear amplitude, then clear the hold."""
        out = {
            self._ids[i]: float(self._peaks[i])
            for i in range(self._n)
            if self._ids[i]
        }
        self._peaks[: self._n] = 0.0
        return out
