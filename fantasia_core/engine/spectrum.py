"""Off-thread spectral tap for the stock EQ analyzer.

The audio callback is only allowed to memcpy into a pre-allocated ring. FFT,
windowing, and painting happen on the UI timer. If the ring wraps mid-read the
analyzer may tear by a few samples — that is preferred to locking the callback.

Dropping a visual block is always legal; blocking or allocating in ``write``
is not.
"""

from __future__ import annotations

import numpy as np


class SpectrumTap:
    """Latest-N-samples ring filled by the mixer, read by the EQ view."""

    def __init__(self, n_samples: int = 4096) -> None:
        self._n = int(max(256, n_samples))
        self._buf = np.zeros((self._n, 2), dtype=np.float32)
        self._pos = 0
        self._gen = 0

    @property
    def n_samples(self) -> int:
        return self._n

    @property
    def generation(self) -> int:
        """Monotonic write counter; the UI uses this to skip redundant FFTs."""
        return self._gen

    def write(self, block: np.ndarray) -> None:
        """Audio-thread entry. Must not allocate, lock, or raise."""
        try:
            if block is None or block.size == 0:
                return
            # Mixer blocks are (frames, 2). Accept a view; never copy the source.
            if block.ndim == 1:
                n = min(int(block.shape[0]), self._n)
                src_l = src_r = block[:n]
            else:
                n = min(int(block.shape[0]), self._n)
                src_l = block[:n, 0]
                src_r = block[:n, 1] if block.shape[1] > 1 else src_l
            pos = self._pos
            room = self._n - pos
            if n <= room:
                self._buf[pos:pos + n, 0] = src_l
                self._buf[pos:pos + n, 1] = src_r
            else:
                self._buf[pos:, 0] = src_l[:room]
                self._buf[pos:, 1] = src_r[:room]
                extra = n - room
                self._buf[:extra, 0] = src_l[room:]
                self._buf[:extra, 1] = src_r[room:]
            self._pos = (pos + n) % self._n
            self._gen += 1
        except Exception:  # noqa: BLE001 — visualization must never break audio
            return

    def snapshot(self) -> np.ndarray:
        """Chronological copy for the UI thread. Allocation is allowed here."""
        pos = self._pos
        out = np.empty((self._n, 2), dtype=np.float32)
        out[: self._n - pos] = self._buf[pos:]
        if pos:
            out[self._n - pos:] = self._buf[:pos]
        return out


def spectrum_db(stereo: np.ndarray, sr: float, n_fft: int = 2048
                ) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed rFFT of a stereo block, mixed to mono. UI thread only.

    Returns ``(freqs_hz, magnitude_dbfs)``. 0 dBFS is a full-scale sinusoid.
    """
    if stereo is None or len(stereo) == 0:
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        return freqs, np.full(len(freqs), -120.0)
    n_fft = int(n_fft)
    mono = stereo[:, 0] if stereo.ndim == 2 else stereo
    if stereo.ndim == 2 and stereo.shape[1] > 1:
        mono = 0.5 * (stereo[:, 0] + stereo[:, 1])
    if len(mono) < n_fft:
        pad = np.zeros(n_fft, dtype=np.float32)
        pad[-len(mono):] = mono
        mono = pad
    else:
        mono = np.asarray(mono[-n_fft:], dtype=np.float32)
    window = np.hanning(n_fft).astype(np.float32)
    spec = np.fft.rfft(mono * window)
    # Window coherent-gain compensation so a 0 dBFS sine reads near 0 dBFS.
    scale = (len(window) / 2.0) * (float(window.mean()) * 2.0 or 1.0)
    mag = np.abs(spec) / max(scale, 1e-12)
    db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sr))
    return freqs, db
