"""Offline bounce: render the whole project to a stereo array / WAV file.

Uses the same :func:`render_block` as playback, so an exported mix matches what
you hear.
"""

from __future__ import annotations

import numpy as np

from fantasia_core.engine.fx import FxHost
from fantasia_core.engine.mixer import render_block


def bounce_to_array(project, pool, sr: int, block: int = 8192, midi_renderer=None, synth_renderer=None) -> np.ndarray:  # noqa: ANN001
    total = int(project.duration * sr)
    out = np.zeros((max(total, 0), 2), dtype=np.float32)
    if hasattr(pool, "preload"):
        pool.preload(project)
    if midi_renderer is not None:
        midi_renderer.warm(project)  # ensure MIDI buffers are cached before mixing
    if synth_renderer is not None:
        synth_renderer.warm(project)
    fx_host = FxHost()  # fresh FX state, carried across blocks for smooth tails
    pos = 0
    while pos < total:
        n = min(block, total - pos)
        out[pos : pos + n] = render_block(
            project, pool, pos, n, sr,
            fx_host=fx_host, midi_renderer=midi_renderer, synth_renderer=synth_renderer,
        )
        pos += n
    np.clip(out, -1.0, 1.0, out=out)
    return out


def _apply_loudness(mix: np.ndarray, sr: int, mode) -> np.ndarray:
    """Post-process a mix: 'normalize' (peak → −1 dBFS) or 'limiter' (gain into a
    brickwall limiter for loudness, then normalize). None/'none' returns as-is."""
    if not mode or mode == "none" or mix.size == 0:
        return mix
    ceiling = 10.0 ** (-1.0 / 20.0)  # −1 dBFS
    if mode == "normalize":
        peak = float(np.max(np.abs(mix))) or 1.0
        return (mix * (ceiling / peak)).astype(np.float32)
    if mode == "limiter":
        out = mix
        try:
            from pedalboard import Gain, Limiter, Pedalboard

            board = Pedalboard([Gain(gain_db=4.0), Limiter(threshold_db=-1.0, release_ms=120.0)])
            out = board(mix.T.astype(np.float32), float(sr)).T  # pedalboard: (ch, n)
        except Exception:  # noqa: BLE001
            out = mix
        peak = float(np.max(np.abs(out))) or 1.0
        if peak > ceiling:
            out = out * (ceiling / peak)
        return out.astype(np.float32)
    return mix


def bounce_to_file(project, pool, sr: int, path: str, midi_renderer=None,  # noqa: ANN001
                   synth_renderer=None, subtype=None, loudness=None) -> float:
    """Render the whole project to ``path``. ``subtype`` (e.g. PCM_24, FLOAT,
    MPEG_LAYER_III) picks the encoding; None uses the format's default. Returns
    duration in seconds."""
    import soundfile as sf

    mix = bounce_to_array(project, pool, sr, midi_renderer=midi_renderer, synth_renderer=synth_renderer)
    mix = _apply_loudness(mix, sr, loudness)
    if subtype:
        sf.write(path, mix, sr, subtype=subtype)
    else:
        sf.write(path, mix, sr)
    return len(mix) / sr


def bounce_track_to_file(project, pool, sr: int, path: str, track_id: str,  # noqa: ANN001
                         midi_renderer=None, synth_renderer=None, subtype=None,
                         loudness=None) -> float:
    """Render a single track in isolation (its own gain/pan/FX) to ``path`` —
    for stem export. Temporarily solos the track, then restores mute/solo."""
    import soundfile as sf

    saved = [(t.mute, t.solo) for t in project.tracks]
    try:
        for t in project.tracks:
            t.solo = False
            t.mute = t.id != track_id
        mix = bounce_to_array(project, pool, sr, midi_renderer=midi_renderer,
                              synth_renderer=synth_renderer)
    finally:
        for t, (m, s) in zip(project.tracks, saved):
            t.mute, t.solo = m, s
    mix = _apply_loudness(mix, sr, loudness)
    if subtype:
        sf.write(path, mix, sr, subtype=subtype)
    else:
        sf.write(path, mix, sr)
    return len(mix) / sr
