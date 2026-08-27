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


def loop_render_plan(
    cursor: int,
    frames: int,
    loop: bool,
    loop_start: int,
    loop_end: int,
) -> tuple[list[tuple[int, int]], int]:
    """Split one callback into ``(src_frame, n)`` pieces so a loop wrap is seamless.

    Returns ``(pieces, new_cursor)``. When looping is off, a single piece.
    """
    if frames <= 0:
        return [], cursor
    if not loop or loop_end <= loop_start + 1:
        return [(cursor, frames)], cursor + frames
    lo, hi = int(loop_start), int(loop_end)
    pieces: list[tuple[int, int]] = []
    remaining = int(frames)
    cur = int(cursor)
    if cur >= hi:
        length = hi - lo
        cur = lo + ((cur - lo) % length) if length > 0 else lo
    guard = 0
    while remaining > 0 and guard < 16:
        guard += 1
        if cur >= hi:
            cur = lo
        room = hi - cur
        if room <= 0:
            cur = lo
            room = hi - lo
            if room <= 0:
                break
        n = min(remaining, room)
        pieces.append((cur, n))
        cur += n
        remaining -= n
        if cur >= hi:
            cur = lo
    return pieces, cur


class PlaybackEngine:
    # How long the callback has to produce a block: 8192 frames is ~186ms at
    # 44.1k. The size is set by what has to be survived rather than by how
    # little latency is achievable.
    #
    # Shortening the GIL switch interval keeps ordinary Python work from
    # starving the callback, but a call into C holds the lock for its whole
    # duration whatever that interval is — and Qt painting is exactly that. The
    # first paint of the synth panel measured ~97ms, which overran a 4096-frame
    # deadline and was heard as a skip. 8192 absorbs it.
    #
    # The cost is start-up latency on playback, not sync: this engine renders a
    # timeline rather than monitoring live input, so ~186ms before the first
    # sound is not something you play against. Drop it if that ever changes.
    def __init__(self, project, pool, sample_rate: int = 44100, block: int = 8192) -> None:  # noqa: ANN001
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
        self._underruns = 0
        self._fx_host = FxHost()  # persistent per-track FX state for smooth tails
        self.midi_renderer = None  # set by the app; audio callback only reads its cache
        self.synth_renderer = None
        self.plugin_renderer = None

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
    @property
    def underruns(self) -> int:
        """Blocks PortAudio could not get in time since playback started.

        Each one is a gap filled with silence — the popping you hear when the
        callback misses its deadline. Counted rather than ignored so a busy
        project is diagnosable instead of just sounding broken.
        """
        return self._underruns

    def _callback(self, outdata, frames, time_info, status) -> None:  # noqa: ANN001
        if status and getattr(status, "output_underflow", False):
            self._underruns += 1
        if not self._playing:
            outdata.fill(0.0)
            return
        loop_on = bool(self.loop or getattr(self.project, "loop_enabled", False))
        lo_f = hi_f = 0
        if loop_on:
            start_s, end_s = self.project.loop_bounds()
            lo_f, hi_f = int(start_s * self.sr), int(end_s * self.sr)
        pieces, new_cursor = loop_render_plan(
            self._cursor, frames, loop_on, lo_f, hi_f,
        )
        written = 0
        for src, n in pieces:
            block = render_block(
                self.project, self.pool, src, n, self.sr,
                fx_host=self._fx_host, midi_renderer=self.midi_renderer,
                synth_renderer=self.synth_renderer,
                plugin_renderer=self.plugin_renderer,
                warp_compute=False,
            )
            if self.metronome_enabled:
                mix_metronome(
                    block, src, self.sr,
                    self.project.tempo, self.project.beats_per_bar,
                    self._metro_clicks,
                )
            np.clip(block, -1.0, 1.0, out=block)
            outdata[written:written + n] = block
            written += n
        if written < frames:
            outdata[written:].fill(0.0)
        self._cursor = new_cursor

        if not loop_on:
            end = self._end_frame()
            if end > 0 and self._cursor >= end:
                self._playing = False

    # ---- transport -------------------------------------------------------
    @property
    def has_stream(self) -> bool:
        """True while a PortAudio stream is open. A device-list refresh
        (terminate/initialize) is only safe when this is False."""
        return self._stream is not None

    def _close_stream(self) -> None:
        """Tear the stream down. Uses abort() (discard buffered audio) rather
        than stop() (drain it) — draining blocks the caller, badly so on
        Bluetooth devices."""
        if self._stream is not None:
            try:
                self._stream.abort()
            except Exception:  # noqa: BLE001
                try:
                    self._stream.stop()
                except Exception:  # noqa: BLE001
                    pass
            try:
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None

    def _try_open(self, device) -> bool:
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sr, channels=2, blocksize=self.block,
                dtype="float32", device=device, callback=self._callback,
                latency="high",   # buy the callback time; see block= above
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

        The device is opened *immediately* to verify it works, so a bad choice
        is reported now instead of silently failing later at Play. Returns False
        if the requested device could not be opened (playback falls back to the
        system default so the app still makes sound)."""
        if sd is None:
            return False
        was_playing = self._playing
        self._playing = False
        self._close_stream()
        self.output_device = device

        ok = self._try_open(device)          # prove it can actually open
        if not ok:
            self._try_open(None)             # keep audio alive on the default

        if was_playing:
            self._playing = True             # resume on the new device
        else:
            # Idle: don't hold the device open — that would block the device-list
            # refresh (and other apps). play() reopens it.
            self._close_stream()
            self.output_device = device if ok else None
        return ok

    def stop(self) -> None:
        """Stop playback instantly.

        Deliberately does NOT tear down the stream: closing a device blocks the
        calling (UI) thread while the driver shuts down — on Bluetooth that
        reads as the app hanging. The callback outputs silence while stopped,
        and the open device makes the next Play instant. Use release_device()
        when the device itself genuinely has to be freed."""
        self._playing = False

    def release_device(self) -> None:
        """Close the stream so the OS device list can be re-read (or the device
        handed to another app). Only safe/needed when not playing."""
        if not self._playing:
            self._close_stream()

    def close(self) -> None:
        self._playing = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
