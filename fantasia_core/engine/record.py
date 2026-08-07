"""Microphone capture via a sounddevice input stream.

Mirror image of :mod:`playback`: an ``InputStream`` whose callback appends
incoming frames to a list (cheap, allocation-light); :meth:`stop` concatenates
them into one mono float32 buffer. All heavy work (writing a WAV, adding a clip)
happens on the caller's thread after :meth:`stop`.

First use triggers the macOS microphone-permission prompt; if denied,
:meth:`start` returns ``False``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:  # optional at import time
    import sounddevice as sd
except Exception:  # noqa: BLE001
    sd = None


def list_input_devices(refresh: bool = False):
    """Return ``[(index, name)]`` for every device with an input channel."""
    if sd is None:
        return []
    if refresh:
        try:
            sd._terminate()
            sd._initialize()
        except Exception:  # noqa: BLE001
            pass
    try:
        devices = sd.query_devices()
    except Exception:  # noqa: BLE001
        return []
    return [(i, d.get("name", f"Device {i}"))
            for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]


class Recorder:
    def __init__(self, sample_rate: int = 44100, channels: int = 1, device=None) -> None:
        self.sr = sample_rate
        self.channels = channels
        self.input_device = device
        self._chunks: list = []
        self._stream = None
        self._recording = False
        self.overflows = 0        # PortAudio dropped input blocks (choppy audio!)
        self.dropped_frames = 0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def elapsed_seconds(self) -> float:
        frames = sum(len(c) for c in self._chunks)
        return frames / self.sr

    @property
    def had_dropouts(self) -> bool:
        return self.overflows > 0

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        # An input overflow means the OS dropped audio before it reached us —
        # the take ends up choppy with no silent gaps to detect afterwards, so
        # it has to be caught here.
        if status and getattr(status, "input_overflow", False):
            self.overflows += 1
            self.dropped_frames += frames
        if self._recording:
            self._chunks.append(indata.copy())

    def start(self) -> bool:
        """Open the mic stream. Returns False if no input device / permission denied."""
        if sd is None:
            return False
        self._chunks = []
        self.overflows = 0
        self.dropped_frames = 0
        try:
            self._stream = sd.InputStream(
                samplerate=self.sr, channels=self.channels, dtype="float32",
                device=self.input_device, callback=self._callback,
                # A generous buffer: the UI thread does heavy work (rendering,
                # generation), and a small blocksize overflows under that load.
                blocksize=4096, latency="high",
            )
            self._stream.start()
        except Exception:  # noqa: BLE001 — no device / denied permission
            self._stream = None
            return False
        self._recording = True
        return True

    def stop(self) -> np.ndarray:
        """Stop and return the recorded mono float32 buffer (may be empty)."""
        self._recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
        if not self._chunks:
            return np.zeros((0,), dtype=np.float32)
        data = np.concatenate(self._chunks, axis=0)
        if data.ndim > 1:  # downmix to mono
            data = data.mean(axis=1)
        return data.astype(np.float32)

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
        self._recording = False
