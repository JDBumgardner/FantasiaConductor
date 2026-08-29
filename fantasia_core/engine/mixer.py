"""Offline/real-time block mixer.

``render_block`` produces one stereo block ``(num_frames, 2)`` for the timeline
window ``[start_frame, start_frame + num_frames)``. It is a pure function of the
project + audio pool (+ an optional stateful FX host), so it's used identically
by the playback callback, the offline bounce, and unit tests.

Signal flow per track: clips (placement, source_offset, reverse, pitch, clip
gain, fades) → summed into a track buffer → per-track FX chain → track gain +
constant-power pan → summed into the master → master FX → master gain + pan.

An optional :class:`~fantasia_core.engine.spectrum.SpectrumTap` may snapshot a
track's pre-FX audio (or the pre-master mix) for the EQ analyzer. The callback
only memcpy's; FFT is the UI's job.
"""

from __future__ import annotations

import math

import numpy as np


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _pan_gains(pan: float) -> tuple[float, float]:
    """Constant-power pan; ``pan`` in [-1, 1] (0 = center)."""
    p = max(-1.0, min(1.0, pan))
    angle = (p + 1.0) * (math.pi / 4.0)
    return math.cos(angle), math.sin(angle)


def _fade_env(clip, a: int, b: int, clip_start: int, clip_end: int, sr: int) -> np.ndarray:
    n = b - a
    env = np.ones(n, dtype=np.float32)
    fin = int(clip.fade_in * sr)
    fout = int(clip.fade_out * sr)
    if fin <= 0 and fout <= 0:
        return env
    frames = np.arange(a, b)
    if fin > 0:
        env *= np.clip((frames - clip_start) / fin, 0.0, 1.0).astype(np.float32)
    if fout > 0:
        env *= np.clip((clip_end - frames) / fout, 0.0, 1.0).astype(np.float32)
    return env


def _clip_buffer(pool, clip, sr: int, clip_len_frames: int, warp_compute: bool = True):  # noqa: ANN001
    """Pick the source buffer + start index, honouring pitch, reverse, and tempo warp.

    If ``source_duration`` differs from ``duration`` (tempo follow / stretch),
    the source region is fitted to the clip length with pitch preserved.
    """
    from fantasia_core.document.tempo import source_span

    span = source_span(clip)
    dest = float(clip.duration)
    if dest > 0 and abs(span - dest) > 0.003 and hasattr(pool, "load_warped"):
        data = pool.load_warped(
            clip.source_path, clip.source_offset, span, dest,
            getattr(clip, "pitch_semitones", 0.0), clip.reversed,
            quality=True, compute=warp_compute,
        )
        if data is not None:
            return data, 0
        # Cache miss on the audio thread: play the source at original pitch
        # (trimmed/padded) until preload finishes the real stretch.
        dest_n = max(1, int(round(dest * sr)))
        try:
            seg = pool._source_region(
                clip.source_path, clip.source_offset, span,
                getattr(clip, "pitch_semitones", 0.0), clip.reversed,
            )
        except Exception:  # noqa: BLE001
            seg = np.zeros((0, 2), dtype=np.float32)
        if len(seg) >= dest_n:
            return np.ascontiguousarray(seg[:dest_n], dtype=np.float32), 0
        if len(seg) == 0:
            return np.zeros((dest_n, 2), dtype=np.float32), 0
        pad = np.zeros((dest_n - len(seg), seg.shape[1]), dtype=np.float32)
        return np.vstack([seg, pad]), 0
    if clip.pitch_semitones:
        base = pool.load_pitched(clip.source_path, clip.pitch_semitones)
    else:
        base = pool.load(clip.source_path)
    if clip.reversed:
        data = base[::-1]
        src0 = len(data) - int(clip.source_offset * sr) - clip_len_frames
    else:
        data = base
        src0 = int(clip.source_offset * sr)
    return data, src0


def render_track_block(
    track, pool, start_frame: int, num_frames: int, sr: int, midi_renderer=None, synth_renderer=None,
    plugin_renderer=None,
    warp_compute: bool = True,
    stats=None,
) -> np.ndarray:  # noqa: ANN001
    """Render one track's clips (pre-FX, pre-track-gain/pan) as ``(num_frames, 2)``.

    MIDI clips read their pre-rendered buffer from ``midi_renderer`` (callback-safe
    cache); audio clips read from the pool. From there the mixing path is shared.
    """
    out = np.zeros((num_frames, 2), dtype=np.float32)
    block_start = start_frame
    block_end = start_frame + num_frames

    for clip in track.clips:
        clip_start = int(clip.start * sr)
        clip_end = int((clip.start + clip.duration) * sr)
        a = max(block_start, clip_start)
        b = min(block_end, clip_end)
        if b <= a:
            continue

        if clip.content_type == "midi":
            plug = getattr(track, "plugin", "")
            if plug and plugin_renderer is not None:
                data = plugin_renderer.cached(clip, plug,
                                              getattr(track, "plugin_state", ""), track.id)
            elif getattr(track, "is_synth", False) and synth_renderer is not None:
                data = synth_renderer.cached(clip, getattr(track, "synth", None) or {})
            elif midi_renderer is not None:
                data = midi_renderer.cached(clip, track.instrument, getattr(track, "is_drum", False))
            else:
                continue
            if data is None:  # not yet rendered (warming pending) → silent
                # A silent clip is heard as a gap even though the callback met
                # its deadline, so it is counted separately from a dropout.
                if stats is not None:
                    stats.misses += 1
                continue
            src0 = 0
        else:
            if not clip.source_path:
                continue
            try:
                data, src0 = _clip_buffer(
                    pool, clip, sr, clip_end - clip_start, warp_compute=warp_compute,
                )
            except Exception:  # noqa: BLE001
                continue

        src_a = src0 + (a - clip_start)
        if src_a < 0 or src_a >= len(data):
            continue
        length = min(b - a, len(data) - src_a)
        if length <= 0:
            continue
        b = a + length

        seg = data[src_a : src_a + length]
        if seg.shape[1] == 1:
            seg_l = seg_r = seg[:, 0]
        else:
            seg_l, seg_r = seg[:, 0], seg[:, 1]

        env = _fade_env(clip, a, b, clip_start, clip_end, sr) * db_to_lin(clip.gain_db)
        pos = a - block_start
        out[pos : pos + length, 0] += seg_l * env
        out[pos : pos + length, 1] += seg_r * env

    return out


def render_block(
    project, pool, start_frame: int, num_frames: int, sr: int,
    fx_host=None, midi_renderer=None, synth_renderer=None, plugin_renderer=None,
    warp_compute: bool = True,
    spectrum_tap=None, spectrum_track_id=None,
    level_tap=None,
    apply_master: bool = True,
    stats=None,
) -> np.ndarray:  # noqa: ANN001
    """Render one stereo block of the full mix as ``float32`` ``(num_frames, 2)``."""
    out = np.zeros((num_frames, 2), dtype=np.float32)
    any_solo = any(t.solo for t in project.tracks)

    for track in project.tracks:
        if track.mute or (any_solo and not track.solo):
            continue
        tb = render_track_block(
            track, pool, start_frame, num_frames, sr, midi_renderer, synth_renderer,
            plugin_renderer,
            warp_compute=warp_compute,
            stats=stats,
        )
        if spectrum_tap is not None and spectrum_track_id == track.id:
            spectrum_tap.write(tb)
        if getattr(track, "fx", None) and fx_host is not None:
            tb = fx_host.process(track, tb, sr)
        tgain = db_to_lin(track.gain_db)
        lpan, rpan = _pan_gains(track.pan)
        out[:, 0] += tb[:, 0] * tgain * lpan
        out[:, 1] += tb[:, 1] * tgain * rpan
        if level_tap is not None and tb.size:
            peak = max(abs(float(np.max(tb))), abs(float(np.min(tb)))) * abs(tgain)
            level_tap.write_peak(track.id, peak)

    if apply_master:
        master = getattr(project, "master", None)
        if master is not None:
            if spectrum_tap is not None and spectrum_track_id == getattr(master, "id", None):
                spectrum_tap.write(out)
            if getattr(master, "mute", False):
                out[:] = 0.0
                return out
            if getattr(master, "fx", None) and fx_host is not None:
                out = fx_host.process(master, out, sr)
            mgain = db_to_lin(getattr(master, "gain_db", 0.0))
            if mgain != 1.0 or getattr(master, "pan", 0.0):
                lpan, rpan = _pan_gains(getattr(master, "pan", 0.0))
                out[:, 0] *= mgain * lpan
                out[:, 1] *= mgain * rpan
            if level_tap is not None and out.size:
                peak = max(abs(float(np.max(out))), abs(float(np.min(out))))
                level_tap.write_peak(getattr(master, "id", "master"), peak)

    return out
