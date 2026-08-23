"""Audio buffer pool: decode files to float32, cache them, and derive waveform
peaks for display.

Everything is kept in memory as ``float32`` at the project sample rate so the
mixer (audio thread) can slice buffers without decoding or allocating source
data mid-callback. Waveform peaks are computed on demand for the UI.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import soundfile as sf


def resample_to_length(data: np.ndarray, dest_n: int) -> np.ndarray:
    """Linear-interpolate a ``(frames, ch)`` buffer to ``dest_n`` frames."""
    dest_n = max(1, int(dest_n))
    src_n = len(data)
    ch = data.shape[1] if data.ndim == 2 else 1
    if src_n == 0:
        return np.zeros((dest_n, ch), dtype=np.float32)
    if src_n == dest_n:
        return np.ascontiguousarray(data, dtype=np.float32)
    t_src = np.linspace(0.0, 1.0, src_n, endpoint=False)
    t_dst = np.linspace(0.0, 1.0, dest_n, endpoint=False)
    if data.ndim == 1:
        return np.interp(t_dst, t_src, data).astype(np.float32)
    chans = [np.interp(t_dst, t_src, data[:, c]) for c in range(data.shape[1])]
    return np.stack(chans, axis=1).astype(np.float32)


class AudioPool:
    def __init__(self, sample_rate: int = 44100) -> None:
        self.sr = sample_rate
        self._cache: Dict[str, np.ndarray] = {}     # path -> (frames, channels) float32
        self._reversed: Dict[str, np.ndarray] = {}  # path -> time-reversed cache
        self._pitched: Dict[str, np.ndarray] = {}   # "path@semis" -> pitch-shifted cache
        self._warped: Dict[tuple, np.ndarray] = {}  # tempo-follow fit cache
        self._mono: Dict[str, np.ndarray] = {}      # path -> (frames,) float32

    # ---- loading ---------------------------------------------------------
    def load(self, path: str) -> np.ndarray:
        """Return decoded audio as a ``(frames, channels)`` float32 array."""
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != self.sr:
            data = self._resample(data, sr, self.sr)
        self._cache[path] = data
        return data

    def _resample(self, data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        import librosa

        chans = [
            librosa.resample(data[:, c], orig_sr=sr_in, target_sr=sr_out)
            for c in range(data.shape[1])
        ]
        return np.stack(chans, axis=1).astype(np.float32)

    def load_pitched(self, path: str, semitones: float) -> np.ndarray:
        """Return a pitch-shifted (length-preserving) copy of the buffer, cached.

        Like :meth:`load_reversed`, this lets the mixer's slicing path stay the
        same — a pitched clip just reads from a different, same-length buffer.
        """
        key = f"{path}@{round(semitones, 3)}"
        cached = self._pitched.get(key)
        if cached is not None:
            return cached
        data = self.load(path)
        try:
            import pedalboard as pb

            board = pb.Pedalboard([pb.PitchShift(semitones=semitones)])
            out = board(data.T.astype(np.float32), self.sr).T
            # PitchShift preserves length; clamp defensively.
            if len(out) != len(data):
                out = out[: len(data)]
                if len(out) < len(data):
                    pad = np.zeros((len(data) - len(out), out.shape[1]), dtype=np.float32)
                    out = np.vstack([out, pad])
            out = np.ascontiguousarray(out, dtype=np.float32)
        except Exception:  # noqa: BLE001 — no pedalboard / failure: fall back to dry
            out = data
        self._pitched[key] = out
        return out

    def load_reversed(self, path: str) -> np.ndarray:
        """Return a time-reversed copy of the buffer (cached).

        A reversed clip is rendered as forward playback of this buffer, so the
        mixer's slicing path stays identical for both directions.
        """
        cached = self._reversed.get(path)
        if cached is None:
            cached = self.load(path)[::-1].copy()
            self._reversed[path] = cached
        return cached

    def duration(self, path: str) -> float:
        return len(self.load(path)) / self.sr

    def _warped_key(
        self,
        path: str,
        offset: float,
        src_span: float,
        dest_dur: float,
        pitch_semitones: float,
        reversed: bool,
    ) -> tuple:
        return (
            path,
            round(float(offset), 5),
            round(float(src_span), 5),
            round(float(dest_dur), 5),
            round(float(pitch_semitones), 3),
            bool(reversed),
        )

    def _source_region(
        self,
        path: str,
        offset: float,
        src_span: float,
        pitch_semitones: float,
        reversed: bool,
    ) -> np.ndarray:
        if pitch_semitones:
            base = self.load_pitched(path, pitch_semitones)
        else:
            base = self.load(path)
        src_n = max(1, int(round(src_span * self.sr)))
        if reversed:
            src0 = len(base) - int(round(offset * self.sr)) - src_n
            return base[max(0, src0) : max(0, src0) + src_n]
        src0 = int(round(offset * self.sr))
        return base[src0 : src0 + src_n]

    def load_warped(
        self,
        path: str,
        offset: float,
        src_span: float,
        dest_dur: float,
        pitch_semitones: float = 0.0,
        reversed: bool = False,
        quality: bool = True,
        compute: bool = True,
    ) -> np.ndarray | None:
        """Source region ``[offset, offset+src_span)`` fitted to ``dest_dur``.

        Always pitch-preserving (never varispeed). ``compute=False`` returns a
        cached buffer or ``None`` so the audio callback never stretches.
        ``quality`` is kept for callers; it selects a faster Rubber Band mode.
        """
        key = self._warped_key(path, offset, src_span, dest_dur, pitch_semitones, reversed)
        cached = self._warped.get(key)
        if cached is not None:
            return cached
        if not compute:
            return None
        dest_n = max(1, int(round(dest_dur * self.sr)))
        seg = self._source_region(path, offset, src_span, pitch_semitones, reversed)
        if len(seg) == 0:
            out = np.zeros((dest_n, 2), dtype=np.float32)
        elif abs(len(seg) - dest_n) <= 1:
            out = np.ascontiguousarray(seg, dtype=np.float32)
            if len(out) < dest_n:
                pad = np.zeros((dest_n - len(out), out.shape[1]), dtype=np.float32)
                out = np.vstack([out, pad])
            else:
                out = out[:dest_n]
        else:
            from fantasia_core.stretch import stretch

            factor = dest_dur / max(src_span, 1e-9)
            out = stretch(seg, self.sr, factor, quality=quality)
            if len(out) != dest_n:
                out = resample_to_length(out, dest_n)
        self._warped[key] = out
        return out

    def preload(self, project) -> None:  # noqa: ANN001
        from fantasia_core.document.tempo import source_span

        for track in project.tracks:
            for clip in track.clips:
                if clip.source_path:
                    try:
                        self.load(clip.source_path)
                        span = source_span(clip)
                        if abs(span - clip.duration) > 0.003:
                            self.load_warped(
                                clip.source_path, clip.source_offset, span, clip.duration,
                                clip.pitch_semitones, clip.reversed, quality=True,
                                compute=True,
                            )
                    except Exception:  # noqa: BLE001 — missing/broken file shouldn't crash the UI
                        pass

    # ---- waveform peaks --------------------------------------------------
    def _mono_of(self, path: str) -> np.ndarray:
        cached = self._mono.get(path)
        if cached is None:
            cached = self.load(path).mean(axis=1).astype(np.float32)
            self._mono[path] = cached
        return cached

    def peaks(
        self, path: str, start_sec: float, dur_sec: float, buckets: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-bucket ``(mins, maxs)`` for the segment ``[start, start+dur)``.

        Buckets map left→right across the clip's on-screen width.
        """
        try:
            mono = self._mono_of(path)
        except Exception:  # noqa: BLE001
            return np.zeros(0), np.zeros(0)

        s = max(0, int(start_sec * self.sr))
        e = min(len(mono), int((start_sec + dur_sec) * self.sr))
        seg = mono[s:e]
        if len(seg) == 0 or buckets <= 0:
            return np.zeros(0), np.zeros(0)

        buckets = min(buckets, len(seg))
        # Pad to an even multiple so we can reshape and reduce vectorised.
        per = len(seg) // buckets
        usable = per * buckets
        block = seg[:usable].reshape(buckets, per)
        mins = block.min(axis=1)
        maxs = block.max(axis=1)
        return mins, maxs
