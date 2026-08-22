"""Real-time playback via a sounddevice output stream.

The audio callback pulls one block from :func:`render_block` at the current
frame cursor and advances. All heavy work (decoding) happened at load time, so
the callback only slices + sums NumPy buffers.

The stream is created lazily on first ``play()`` so the app runs fine on a
machine with no audio output (and so headless tests never open a device). The
UI polls :attr:`playhead` on a timer to move the on-screen playhead.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from fantasia_core.engine.fx import FxHost
from fantasia_core.engine.metronome import make_click_bank, mix_metronome
from fantasia_core.engine.mixer import render_block

try:  # optional at import time
    import sounddevice as sd
except Exception:  # noqa: BLE001
    sd = None


def refresh_devices() -> None:
    """Re-read the OS device list. PortAudio snapshots devices at init, so a
    long-lived process misses hotplugs (e.g. Bluetooth headphones dropping during
    a long render) until it is re-initialized. Caller must ensure no stream is open."""
    if sd is None:
        return
    try:
        sd._terminate()
        sd._initialize()
    except Exception:  # noqa: BLE001
        pass


def list_output_devices(refresh: bool = False):
    """Return ``[(index, name)]`` for every device with an output channel.

    Empty when there is no audio backend (headless/CI). ``refresh`` re-reads the
    OS device list first (only safe when no stream is open)."""
    if sd is None:
        return []
    if refresh:
        refresh_devices()
    try:
        devices = sd.query_devices()
    except Exception:  # noqa: BLE001
        refresh_devices()
        try:
            devices = sd.query_devices()
        except Exception:  # noqa: BLE001
            return []
    out = []
    for i, d in enumerate(devices):
        if d.get("max_output_channels", 0) > 0:
            out.append((i, d.get("name", f"Device {i}")))
    return out


def default_output_device():
    """Index of the system default output device, or ``None``."""
    if sd is None:
        return None
    try:
        dev = sd.default.device[1]
        return dev if dev is not None and dev >= 0 else None
    except Exception:  # noqa: BLE001
        return None


class PlaybackEngine:
    def __init__(self, project, pool, sample_rate: int = 44100, block: int = 1024) -> None:  # noqa: ANN001
        self.project = project
        self.pool = pool
        self.sr = sample_rate
        self.block = block
        self.loop = False
        self.metronome_enabled = False
        self._metro_clicks = make_click_bank(self.sr)
        self._cursor = 0  # frames
        self._playing = False
        self._stream = None
        self.output_device = None  # None = system default; else sounddevice index
        self._fx_host = FxHost()  # persistent per-track FX state for smooth tails
        self.midi_renderer = None  # set by the app; audio callback only reads its cache
        self.synth_renderer = None

    # ---- state -----------------------------------------------------------
    def set_project(self, project) -> None:  # noqa: ANN001
        self.project = project

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def playhead(self) -> float:
        return self._cursor / self.sr

    def set_playhead_seconds(self, seconds: float) -> None:
        self._cursor = max(0, int(seconds * self.sr))

    def _end_frame(self) -> int:
        return int(self.project.duration * self.sr)

    # ---- audio callback --------------------------------------------------
    def _callback(self, outdata, frames, time_info, status) -> None:  # noqa: ANN001
        if not self._playing:
            outdata.fill(0.0)
            return
        block = render_block(
            self.project, self.pool, self._cursor, frames, self.sr,
            fx_host=self._fx_host, midi_renderer=self.midi_renderer,
            synth_renderer=self.synth_renderer,
            warp_compute=False,
        )
        if self.metronome_enabled:
            mix_metronome(
                block, self._cursor, self.sr,
                self.project.tempo, self.project.beats_per_bar,
                self._metro_clicks,
            )
        np.clip(block, -1.0, 1.0, out=block)
        outdata[:] = block
        self._cursor += frames

        end = self._end_frame()
        if self._cursor >= end:
            if self.loop and end > 0:
                self._cursor = 0
            else:
                self._playing = False

    # ---- transport -------------------------------------------------------
    def _try_open(self, device) -> bool:
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sr, channels=2, blocksize=self.block,
                dtype="float32", device=device, callback=self._callback,
            )
            self._stream.start()
            self.output_device = device  # remember what actually opened
            return True
        except Exception:  # noqa: BLE001
            self._stream = None
            return False

    def play(self) -> bool:
        """Start playback. Falls back to the system default (and a PortAudio
        refresh) if the chosen device has gone away. Returns False only if no
        device can be opened at all."""
        if sd is None:
            return False
        if self._stream is None:
            # 1) the chosen device, 2) the system default, 3) default after a
            #    device-list refresh (handles a stale/hot-unplugged device).
            if not self._try_open(self.output_device):
                if self.output_device is not None and self._try_open(None):
                    pass
                else:
                    refresh_devices()
                    if not self._try_open(None):
                        return False
        self._playing = True
        return True

    def refresh_devices(self) -> bool:
        """Re-read the OS device list (only when stopped). Returns True if done."""
        if self._playing or self._stream is not None:
            return False
        refresh_devices()
        return True

    def set_output_device(self, device) -> bool:
        """Route playback to a device index (``None`` = system default).

        Recreates the stream on the new device, resuming if it was playing."""
        if device == self.output_device and self._stream is not None:
            return True
        self.output_device = device
        was_playing = self._playing
        self._playing = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
        return self.play() if was_playing else True

    def stop(self) -> None:
        self._playing = False

    def close(self) -> None:
        self._playing = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
