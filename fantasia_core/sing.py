"""Singing synthesis — map a spoken voice onto a drawn melody, WORLD-vocoder edition.

For each note we synthesize the lyric syllable with Kokoro TTS, then use the
WORLD vocoder (:mod:`vocalfx`) to resynthesize it at the note's pitch while
*preserving the formants* — so it sounds like a voice singing that pitch, not
pitch-shifted speech. Time-scaling is done on the WORLD frames (formant-safe),
and we add vibrato and a portamento glide between notes for a legato feel.

Input: a melody (list of Note) + lyrics (one token per note). Headless (no Qt).
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np


def available() -> bool:
    try:
        from fantasia_core import tts, vocalfx
    except Exception:  # noqa: BLE001
        return False
    return tts.available() and vocalfx.available()


def _midi_hz(pitch: int) -> float:
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


def split_lyrics(text: str, n: int) -> List[str]:
    """Split lyric text into ``n`` tokens (one per note). Hyphens split a word
    into syllables (``won-der-ful``); otherwise split on whitespace. Pads with a
    hummed vowel if there are fewer tokens than notes."""
    raw = text.replace("-", " ").split()
    if not raw:
        raw = ["la"]
    if len(raw) < n:
        raw = raw + ["la"] * (n - len(raw))
    elif len(raw) > n:
        raw = raw[: n - 1] + [" ".join(raw[n - 1:])]
    return raw


def _render_note(y: np.ndarray, sr: int, target_hz: float, target_len: int,
                 prev_hz: float = 0.0) -> np.ndarray:
    """WORLD-resynthesize one syllable at ``target_hz``, stretched to ``target_len``
    samples, with vibrato + a portamento glide in from ``prev_hz``."""
    from fantasia_core import vocalfx as vf

    f0, sp, ap = vf.analyze(y, sr)
    if len(f0) == 0:
        return np.zeros(target_len, dtype=np.float32)
    tframes = max(2, int(round(target_len / sr * 1000.0 / vf.FRAME_PERIOD)))
    vuv = vf.resample_frames((f0 > 0).astype(np.float64), tframes) > 0.5
    sp_r = vf.resample_frames(sp, tframes)
    ap_r = vf.resample_frames(ap, tframes)

    pitch = np.full(tframes, float(target_hz), dtype=np.float64)
    if prev_hz and prev_hz > 0:  # portamento into the note (~60ms)
        gl = min(int(0.06 * 1000.0 / vf.FRAME_PERIOD), tframes)
        if gl > 1:
            ramp = np.linspace(0.0, 1.0, gl)
            pitch[:gl] = prev_hz * (1 - ramp) + target_hz * ramp
    pitch *= vf.vibrato_curve(tframes)
    new_f0 = np.where(vuv, pitch, 0.0)

    out = vf.synth(new_f0, sp_r, ap_r, sr)
    if len(out) >= target_len:
        return out[:target_len]
    return np.pad(out, (0, target_len - len(out)))


def sing_notes(notes: Sequence, lyrics: str, voice: str = "af_heart",
               sr: int = 44100) -> np.ndarray:
    """Render a melody + lyrics to a mono float32 buffer at ``sr``."""
    import librosa

    from fantasia_core import tts

    ordered = sorted(notes, key=lambda n: n.start)
    if not ordered:
        return np.zeros((0,), dtype=np.float32)
    tokens = split_lyrics(lyrics, len(ordered))
    total = int(math.ceil(max(n.start + n.duration for n in ordered) * sr)) + sr // 5
    out = np.zeros(total, dtype=np.float32)
    xfade = int(0.02 * sr)
    prev_hz = 0.0

    for note, token in zip(ordered, tokens):
        try:
            syl, ssr = tts.synthesize(token, voice=voice)
        except Exception:  # noqa: BLE001
            continue
        if len(syl) == 0:
            continue
        if ssr != sr:
            syl = librosa.resample(syl, orig_sr=ssr, target_sr=sr).astype(np.float32)
        target_hz = _midi_hz(note.pitch)
        target_len = max(int(note.duration * sr), 1)
        seg = _render_note(syl, sr, target_hz, target_len, prev_hz)
        f = min(len(seg) // 8, xfade)
        if f > 0:
            seg[:f] *= np.linspace(0.0, 1.0, f)
            seg[-f:] *= np.linspace(1.0, 0.0, f)
        seg *= note.velocity / 127.0
        pos = int(note.start * sr)
        end = min(pos + len(seg), total)
        out[pos:end] += seg[: end - pos]
        prev_hz = target_hz

    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.95).astype(np.float32)


def sing_to_file(notes: Sequence, lyrics: str, path: str,
                 voice: str = "af_heart", sr: int = 44100) -> float:
    import os

    import soundfile as sf

    audio = sing_notes(notes, lyrics, voice=voice, sr=sr)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")
    return len(audio) / sr
