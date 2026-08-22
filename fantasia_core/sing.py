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
from typing import List, Sequence, Tuple

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
    return [t for t, _ in split_lyrics_joined(text, n)]


def split_lyrics_joined(text: str, n: int) -> List[Tuple[str, bool]]:
    """As :func:`split_lyrics`, but each token carries whether it continues the
    previous word.

    Phrase synthesis needs this: rejoining ``won-der-ful`` as "won der ful" makes
    the TTS pause between syllables — reintroducing exactly the gaps that made
    singing sound chopped up — while "wonderful" is spoken as one smooth word.
    """
    out: List[Tuple[str, bool]] = []
    for word in text.split():
        for i, syl in enumerate(word.split("-")):
            if syl:
                out.append((syl, i > 0))
    if not out:
        out = [("la", False)]
    if len(out) < n:
        out += [("la", False)] * (n - len(out))
    elif len(out) > n:
        tail = " ".join(t for t, _ in out[n - 1:])
        out = out[: n - 1] + [(tail, False)]
    return out


def phrase_text(tokens: List[Tuple[str, bool]]) -> str:
    """Rebuild speakable text from tokens, keeping words whole."""
    parts = []
    for i, (tok, cont) in enumerate(tokens):
        parts.append(tok if (cont and i) else (" " + tok if i else tok))
    return "".join(parts).strip()


def trim_silence(y: np.ndarray, sr: int, thresh: float = 0.02,
                 keep_ms: float = 8.0) -> np.ndarray:
    """Strip the silence a TTS engine pads around a short utterance.

    Kokoro returns ~400ms of lead-in and ~500ms of tail for a single syllable,
    against only ~350ms of actual voice. Stretching that whole buffer onto a note
    leaves the note roughly 60% silent — which is what made sung lines sound
    chopped up. Trim first, then stretch, so the note is voiced end to end.
    """
    if len(y) == 0:
        return y
    env = np.abs(y)
    peak = float(env.max())
    if peak <= 0:
        return y
    voiced = np.nonzero(env > thresh * peak)[0]
    if len(voiced) == 0:
        return y
    pad = int(keep_ms * sr / 1000.0)      # a hair of room so onsets aren't clipped
    lo = max(0, voiced[0] - pad)
    hi = min(len(y), voiced[-1] + pad + 1)
    return y[lo:hi]


# A sung phrase is synthesized as ONE utterance and warped onto its notes, so
# consonants and vowels blend the way they do in speech. Past ~8 syllables the
# even-split segmentation below gets unreliable, and singers breathe anyway, so
# long lines are cut into phrases of this many notes.
PHRASE_MAX = 8


def syllable_bounds(y: np.ndarray, sr: int, n: int, frame: int = 512,
                    hop: int = 128) -> List[int]:
    """Sample offsets splitting a spoken phrase into ``n`` segments, one per note.

    Taking the n-1 globally quietest points does not work: the quietest places in
    an utterance are its own head and tail, so the outer syllables collapse to a
    few milliseconds and their notes come out unvoiced. Each boundary is instead
    searched for only near where an even split would put it, which bounds how
    lopsided the segments can get.
    """
    import librosa

    if n <= 1 or len(y) == 0:
        return [0, len(y)]
    env = np.abs(librosa.util.frame(y, frame_length=frame, hop_length=hop)).max(axis=0)
    env = env / (float(env.max()) or 1.0)
    nf = len(env)
    seg = nf / n
    bounds = [0]
    for k in range(1, n):
        centre = k * seg
        half = seg * 0.4                      # stay within 40% of the even split
        lo, hi = int(max(1, centre - half)), int(min(nf - 1, centre + half))
        idx = int(centre) if hi <= lo else lo + int(np.argmin(env[lo:hi]))
        bounds.append(int(idx * hop + frame / 2))
    bounds.append(len(y))
    for i in range(1, len(bounds)):           # keep strictly increasing
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = min(len(y), bounds[i - 1] + 1)
    return bounds


def _render_phrase(y: np.ndarray, sr: int, notes: Sequence,
                   prev_hz: float = 0.0) -> List[Tuple[float, np.ndarray]]:
    """Warp one spoken phrase across ``notes``; returns ``(start, audio)`` per note.

    The whole phrase goes through a single WORLD resynthesis, so the vocal tract
    moves continuously instead of restarting at every note. The result is only
    cut apart afterwards to honour rests — consecutive notes land sample-adjacent,
    so nothing is lost where the line is legato.
    """
    from fantasia_core import vocalfx as vf

    f0, sp, ap = vf.analyze(y, sr)
    if len(f0) == 0:
        return []
    fps = 1000.0 / vf.FRAME_PERIOD
    bounds = syllable_bounds(y, sr, len(notes))
    fb = [min(len(f0), max(0, int(b / sr * fps))) for b in bounds]
    fb[-1] = len(f0)

    sp_parts, ap_parts, vuv_parts, pitch_parts, counts = [], [], [], [], []
    for i, note in enumerate(notes):
        lo = fb[i]
        hi = max(lo + 2, fb[i + 1])
        tframes = max(2, int(round(note.duration * fps)))
        sp_parts.append(vf.resample_frames(sp[lo:hi], tframes))
        ap_parts.append(vf.resample_frames(ap[lo:hi], tframes))
        vuv_parts.append(vf.resample_frames((f0[lo:hi] > 0).astype(np.float64),
                                            tframes) > 0.5)
        hz = _midi_hz(note.pitch)
        pitch = np.full(tframes, hz, dtype=np.float64)
        if prev_hz > 0:                       # portamento into the note (~60ms)
            gl = min(int(0.06 * fps), tframes)
            if gl > 1:
                r = np.linspace(0.0, 1.0, gl)
                pitch[:gl] = prev_hz * (1 - r) + hz * r
        pitch *= vf.vibrato_curve(tframes)
        pitch_parts.append(pitch)
        counts.append(tframes)
        prev_hz = hz

    out = vf.synth(np.where(np.concatenate(vuv_parts), np.concatenate(pitch_parts), 0.0),
                   np.concatenate(sp_parts, axis=0),
                   np.concatenate(ap_parts, axis=0), sr)
    if len(out) == 0:
        return []

    # Slice back per note in proportion to the frames each contributed.
    total = float(sum(counts))
    pieces, acc = [], 0
    for note, c in zip(notes, counts):
        lo = int(len(out) * acc / total)
        acc += c
        hi = int(len(out) * acc / total)
        pieces.append((note.start, out[lo:hi].astype(np.float32) * (note.velocity / 127.0)))
    return pieces


def _phrase_chunks(notes: Sequence, tokens: List[str]):
    """Group (note, token) pairs into singable phrases of at most PHRASE_MAX."""
    for i in range(0, len(notes), PHRASE_MAX):
        yield list(notes[i:i + PHRASE_MAX]), tokens[i:i + PHRASE_MAX]


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
               sr: int = 44100, ref_voice=None, backend=None,
               per_syllable: bool = False) -> np.ndarray:
    """Render a melody + lyrics to a mono float32 buffer at ``sr``.

    Each phrase (up to :data:`PHRASE_MAX` notes) is spoken as a single utterance
    and warped onto its notes in one WORLD pass. Synthesizing syllables in
    isolation instead — the old behaviour, still available as ``per_syllable``
    and used as a fallback — gives no coarticulation and restarts the vocoder at
    every note, which is what made sung lines sound chopped up.

    ``ref_voice`` picks a cloned timbre from :mod:`fantasia_core.voices`. Phrase
    synthesis also makes that far cheaper: one cloning call per phrase rather
    than one per syllable.
    """
    import librosa

    from fantasia_core import tts

    ordered = sorted(notes, key=lambda n: n.start)
    if not ordered:
        return np.zeros((0,), dtype=np.float32)
    tokens = split_lyrics_joined(lyrics, len(ordered))
    total = int(math.ceil(max(n.start + n.duration for n in ordered) * sr)) + sr // 5
    out = np.zeros(total, dtype=np.float32)
    xfade = int(0.02 * sr)

    def _speak(text):
        try:
            y, ssr = tts.synthesize(text, voice=voice, backend=backend,
                                    ref_voice=ref_voice, cache=True)
        except Exception:  # noqa: BLE001
            return None
        if len(y) == 0:
            return None
        if ssr != sr:
            y = librosa.resample(y, orig_sr=ssr, target_sr=sr).astype(np.float32)
        y = trim_silence(y, sr)      # before any stretching — see trim_silence
        return y if len(y) else None

    def _place(start_s: float, seg: np.ndarray) -> None:
        if len(seg) == 0:
            return
        f = min(len(seg) // 8, xfade)
        if f > 0:
            seg = seg.copy()
            seg[:f] *= np.linspace(0.0, 1.0, f)
            seg[-f:] *= np.linspace(1.0, 0.0, f)
        pos = max(0, int(start_s * sr))
        end = min(pos + len(seg), total)
        if end > pos:
            out[pos:end] += seg[: end - pos]

    prev_hz = 0.0
    for chunk_notes, chunk_tokens in _phrase_chunks(ordered, tokens):
        pieces = []
        if not per_syllable:
            spoken = _speak(phrase_text(chunk_tokens))
            if spoken is not None:
                try:
                    pieces = _render_phrase(spoken, sr, chunk_notes, prev_hz)
                except Exception:  # noqa: BLE001 — fall back to per-syllable below
                    pieces = []
        if not pieces:
            # Fallback: the old one-utterance-per-syllable path.
            for note, (token, _cont) in zip(chunk_notes, chunk_tokens):
                syl = _speak(token)
                if syl is None:
                    continue
                seg = _render_note(syl, sr, _midi_hz(note.pitch),
                                   max(int(note.duration * sr), 1), prev_hz)
                seg = seg * (note.velocity / 127.0)
                pieces.append((note.start, seg))
                prev_hz = _midi_hz(note.pitch)
        else:
            prev_hz = _midi_hz(chunk_notes[-1].pitch)
        for start_s, seg in pieces:
            _place(start_s, seg)

    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.95).astype(np.float32)


def sing_to_file(notes: Sequence, lyrics: str, path: str,
                 voice: str = "af_heart", sr: int = 44100,
                 ref_voice=None, backend=None, per_syllable: bool = False) -> float:
    import os

    import soundfile as sf

    audio = sing_notes(notes, lyrics, voice=voice, sr=sr, ref_voice=ref_voice,
                       backend=backend, per_syllable=per_syllable)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")
    return len(audio) / sr
