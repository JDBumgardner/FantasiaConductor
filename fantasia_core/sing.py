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
# Speaking rates tried for a phrase, fastest first. Dense lines need the slow
# ones: at normal rate Kokoro runs "fantasia conductor sings" together in 1.6s,
# 86% voiced across two unbroken runs, and no segmenter finds the syllables in
# that — the last note came out holding a second of "sss". At 0.5x the same line
# is 3.4s and breaks into 7 voiced runs for 8 syllables.
#
# Slowing down is not free: it opens real gaps between syllables and roughly
# doubles the envelope discontinuity on easy lines. So the rate is chosen per
# phrase rather than fixed — the fastest one whose segmentation gives every note
# a vowel wins, leaving simple phrases smooth and only paying for dense ones.
PHRASE_SPEEDS = (1.0, 0.7, 0.5)


def word_groups(tokens: List[Tuple[str, bool]]) -> List[int]:
    """Syllables per word, from the continuation flags: fan-ta-si-a con-duc-tor
    sings -> [4, 3, 1]."""
    groups: List[int] = []
    for _tok, cont in tokens:
        if cont and groups:
            groups[-1] += 1
        else:
            groups.append(1)
    return groups


def _split_evenly(env: np.ndarray, lo: int, hi: int, n: int) -> List[int]:
    """n-1 boundaries inside [lo, hi), each snapped to the quietest frame near
    where an even split would put it."""
    if n <= 1 or hi - lo < 2:
        return []
    seg = (hi - lo) / n
    out = []
    for k in range(1, n):
        centre = lo + k * seg
        half = seg * 0.4                      # bound how lopsided a split can get
        a, b = int(max(lo + 1, centre - half)), int(min(hi - 1, centre + half))
        out.append(int(centre) if b <= a else a + int(np.argmin(env[a:b])))
    return out


def syllable_bounds(y: np.ndarray, sr: int, groups, frame: int = 512,
                    hop: int = 128) -> List[int]:
    """Sample offsets splitting a spoken phrase into one segment per note.

    ``groups`` is syllables-per-word (see :func:`word_groups`); an int means a
    single word of that many syllables.

    Splitting the whole phrase evenly does not survive real lyrics: "fantasia
    conductor sings" is 4+3+1 syllables over words of very different lengths, so
    evenly spaced boundaries drift and the error compounds — the last note ended
    up holding only the "s" of "sings", a full second of unvoiced hiss. Words are
    therefore located first (the TTS puts a genuine pause at every space, which
    is far easier to find than a boundary inside a word) and only then is each
    word split by its own syllable count, where even spacing does hold up.

    Taking the globally quietest points instead fails differently: the quietest
    places in an utterance are its head and tail, which collapses the outer
    syllables to a few milliseconds.
    """
    import librosa

    if isinstance(groups, int):
        groups = [groups]
    groups = [g for g in groups if g > 0] or [1]
    n = sum(groups)
    if n <= 1 or len(y) == 0:
        return [0, len(y)]

    env = np.abs(librosa.util.frame(y, frame_length=frame, hop_length=hop)).max(axis=0)
    env = env / (float(env.max()) or 1.0)
    nf = len(env)

    # 1. word boundaries — the real pauses. Each search is fenced so that the
    #    words on either side keep room for their own syllables; without that
    #    fence the last window reaches the utterance's silent tail, picks it as
    #    the "gap", and the final word collapses to a few milliseconds.
    min_syl = max(1.0, nf / n * 0.4)
    word_edges = [0]
    if len(groups) > 1:
        acc = 0
        for wi, g in enumerate(groups[:-1]):
            acc += g
            centre = acc / n * nf
            span = nf / len(groups) * 0.45
            floor = word_edges[-1] + groups[wi] * min_syl
            ceil = nf - sum(groups[wi + 1:]) * min_syl
            a = int(max(1, centre - span, floor))
            b = int(min(nf - 1, centre + span, ceil))
            word_edges.append(int(min(max(centre, a), max(a, b)))
                              if b <= a else a + int(np.argmin(env[a:b])))
        for i in range(1, len(word_edges)):   # keep increasing
            if word_edges[i] <= word_edges[i - 1]:
                word_edges[i] = word_edges[i - 1] + 1
    word_edges.append(nf)

    # 2. syllables inside each word, where even spacing is a fair assumption
    frames: List[int] = [0]
    for wi, g in enumerate(groups):
        lo, hi = word_edges[wi], word_edges[wi + 1]
        frames.extend(_split_evenly(env, lo, hi, g))
        frames.append(hi)
    frames = frames[:n] + [nf]

    bounds = [0] + [int(f * hop + frame / 2) for f in frames[1:-1]] + [len(y)]
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = min(len(y), bounds[i - 1] + 1)
    return bounds


def _sustain_map(voiced: np.ndarray, tframes: int) -> np.ndarray:
    """Frame indices for stretching a syllable to ``tframes``, holding the vowel.

    Uniform stretching is what a vocoder does, not what a singer does. Held
    "sings" spoken in 150ms became a full second of sibilant hiss, because the
    unvoiced "s" was stretched by the same factor as everything else. A singer
    keeps the consonants roughly their natural length and sustains the vowel.

    Consonant frames are therefore allowed to grow only a little, and whatever
    time is left over goes to the voiced run.
    """
    n = len(voiced)
    if n == 0:
        return np.zeros(tframes, dtype=np.float64)
    if not voiced.any() or tframes <= n:
        return np.linspace(0, n - 1, tframes)          # nothing to sustain

    idx = np.nonzero(voiced)[0]
    v0, v1 = int(idx[0]), int(idx[-1]) + 1             # the vowel run
    head, tail = v0, n - v1
    # Consonants may stretch up to 1.5x; the vowel absorbs the rest.
    grow = min(1.5, tframes / n)
    h_out = int(round(head * grow))
    t_out = int(round(tail * grow))
    v_out = max(1, tframes - h_out - t_out)
    parts = []
    if h_out:
        parts.append(np.linspace(0, v0, h_out, endpoint=False))
    parts.append(np.linspace(v0, v1 - 1, v_out))
    if t_out:
        parts.append(np.linspace(v1, n - 1, t_out))
    out = np.concatenate(parts)
    if len(out) != tframes:                            # rounding guard
        out = np.interp(np.linspace(0, len(out) - 1, tframes),
                        np.arange(len(out)), out)
    return out


def _tighten(lo: int, hi: int, voiced: np.ndarray, energy: np.ndarray,
             floor: float = 0.06) -> Tuple[int, int]:
    """Shrink a syllable's frame span to its voiced core plus its consonants.

    Speaking the phrase slowly is what makes syllables findable at all, but it
    also leaves real gaps between them. Carrying that dead air into the note
    puts the choppiness back, so each segment is trimmed to the part that
    actually sounds before it is stretched.
    """
    if hi - lo < 3:
        return lo, hi
    span = np.arange(lo, hi)
    keep = voiced[lo:hi] | (energy[lo:hi] > floor)
    if not keep.any():
        return lo, hi
    idx = span[keep]
    a, b = int(idx[0]), int(idx[-1]) + 1
    return (a, b) if b - a >= 2 else (lo, hi)


def _compress_map(voiced: np.ndarray, tframes: int) -> np.ndarray:
    """Frame indices for fitting a syllable into a note SHORTER than it.

    Resampling the whole syllable faster is what makes short notes mushy: the
    consonants are sped up along with the vowel, so the word stops being
    articulated. A singer on a short note does not talk faster — they clip the
    vowel and keep the consonants intact. On a 0.25-beat note (125ms against a
    ~400ms spoken syllable) that is the difference between a word and a blur.
    """
    n = len(voiced)
    if n == 0:
        return np.zeros(tframes, dtype=np.float64)
    if tframes >= n or not voiced.any():
        return np.linspace(0, max(n - 1, 0), tframes)

    idx = np.nonzero(voiced)[0]
    v0, v1 = int(idx[0]), int(idx[-1]) + 1
    head, tail = v0, n - v1
    # Only worth doing if the consonants actually fit with a vowel left over.
    if head + tail + 2 > tframes:
        return np.linspace(0, n - 1, tframes)

    vowel_out = tframes - head - tail
    parts = []
    if head:
        parts.append(np.arange(0, v0, dtype=np.float64))
    parts.append(np.linspace(v0, v1 - 1, vowel_out))
    if tail:
        parts.append(np.arange(v1, n, dtype=np.float64))
    out = np.concatenate(parts)
    if len(out) != tframes:
        out = np.interp(np.linspace(0, len(out) - 1, tframes),
                        np.arange(len(out)), out)
    return out


def _take(arr: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """Sample ``arr`` (frames on axis 0) at fractional frame positions."""
    i = np.clip(np.rint(mapping).astype(int), 0, len(arr) - 1)
    return arr[i]


def syllable_bounds_from_words(y: np.ndarray, sr: int, groups, word_edges,
                               frame: int = 512, hop: int = 128) -> List[int]:
    """Syllable offsets when the word offsets are already known exactly.

    Guessing where one word ends and the next begins was the dominant error:
    on "let me sing a song for you to-day now" the guessed split handed "let"
    87ms and "sing" 258ms of source, so identical half-note values came out
    stretched 5.7x and 1.9x — audibly, one dragging and the next clipped.
    Speaking each word separately removes that guess entirely; only the split
    *inside* a multi-syllable word is still inferred, where the syllables really
    are about even and both sides are the same word anyway.
    """
    import librosa

    env = np.abs(librosa.util.frame(y, frame_length=frame, hop_length=hop)).max(axis=0)
    env = env / (float(env.max()) or 1.0)
    to_f = lambda smp: min(len(env) - 1, max(0, int(smp / hop)))  # noqa: E731

    bounds = [0]
    for wi, g in enumerate(groups):
        lo_s, hi_s = word_edges[wi], word_edges[wi + 1]
        if g > 1:
            for f in _split_evenly(env, to_f(lo_s), to_f(hi_s), g):
                bounds.append(int(f * hop + frame / 2))
        bounds.append(hi_s)
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = min(len(y), bounds[i - 1] + 1)
    return bounds


def _render_phrase(y: np.ndarray, sr: int, notes: Sequence, prev_hz: float = 0.0,
                   groups=None, word_edges=None) -> List[Tuple[float, np.ndarray]]:
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
    # Per-frame energy, used to trim dead air at each syllable's edges.
    step = max(1, int(round(sr * vf.FRAME_PERIOD / 1000.0)))
    pad = np.pad(y, (0, step))
    energy = np.array([float(np.abs(pad[i * step:(i + 1) * step]).max())
                       for i in range(len(f0))])
    energy /= (float(energy.max()) or 1.0)
    voiced_all = f0 > 0
    bounds = (syllable_bounds_from_words(y, sr, groups, word_edges)
              if word_edges is not None
              else syllable_bounds(y, sr, groups or len(notes)))
    fb = [min(len(f0), max(0, int(b / sr * fps))) for b in bounds]
    fb[-1] = len(f0)

    sp_parts, ap_parts, vuv_parts, pitch_parts, counts = [], [], [], [], []
    for i, note in enumerate(notes):
        lo = fb[i]
        hi = max(lo + 2, fb[i + 1])
        lo, hi = _tighten(lo, min(hi, len(f0)), voiced_all, energy)
        hi = max(lo + 2, hi)
        tframes = max(2, int(round(note.duration * fps)))
        seg_voiced = f0[lo:hi] > 0
        m = (_sustain_map(seg_voiced, tframes) if tframes >= (hi - lo)
             else _compress_map(seg_voiced, tframes))
        sp_parts.append(_take(sp[lo:hi], m))
        ap_parts.append(_take(ap[lo:hi], m))
        vuv_parts.append(_take(seg_voiced.astype(np.float64), m) > 0.5)
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

    # A segment with no voiced frame means the segmentation missed that
    # syllable's vowel; the caller retries more slowly rather than render hiss.
    if not all(v.any() for v in vuv_parts):
        return []

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


def _chunk_words(tokens: List[Tuple[str, bool]]) -> List[str]:
    """Rebuild the whole words in a chunk: [(fan,F),(ta,T)] -> ["fanta"]."""
    words: List[str] = []
    for tok, cont in tokens:
        if cont and words:
            words[-1] += tok
        else:
            words.append(tok)
    return words


def _speak_words(speak, words: List[str], speed: float):
    """Speak each word on its own; returns the joined buffer and the exact
    sample offset of every word boundary in it."""
    parts, edges = [], [0]
    for w in words:
        y = speak(w, speed)
        if y is None or len(y) == 0:
            return None, None
        parts.append(y)
        edges.append(edges[-1] + len(y))
    if not parts:
        return None, None
    return np.concatenate(parts), edges


def _edges_from_durations(y: np.ndarray, sr: int, word_lens: List[int],
                          frame: int = 512, hop: int = 128) -> List[int]:
    """Word boundaries inside a connected phrase, guided by how long each word
    takes on its own.

    Speaking every word separately gets the boundaries exactly right but ruins
    the delivery: an isolated "let" is produced with a fully released /t/ that
    connected speech reduces, and nothing flows into the next word. Speaking the
    phrase whole keeps the flow but leaves the boundaries to guesswork, and a
    flat guess gave "let" 87ms against "sing" 258ms.

    So the words are still spoken separately, but only to measure them: their
    relative lengths place the joins in the connected take, and each join is
    then nudged to the quietest frame nearby.
    """
    import librosa

    total = float(sum(word_lens)) or 1.0
    env = np.abs(librosa.util.frame(y, frame_length=frame, hop_length=hop)).max(axis=0)
    env = env / (float(env.max()) or 1.0)
    nf = len(env)

    edges, acc = [0], 0
    for wl in word_lens[:-1]:
        acc += wl
        centre = acc / total * nf
        # Only a small nudge: the proportion is the good estimate here, the
        # energy dip just snaps it onto the actual consonant boundary.
        half = max(2.0, nf * 0.03)
        a = int(max(1, centre - half))
        b = int(min(nf - 1, centre + half))
        f = int(centre) if b <= a else a + int(np.argmin(env[a:b]))
        edges.append(int(f * hop + frame / 2))
    edges.append(len(y))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = min(len(y), edges[i - 1] + 1)
    return edges


def _phrase_chunks(notes: Sequence, tokens: List[Tuple[str, bool]]):
    """Group (note, token) pairs into singable phrases of at most PHRASE_MAX.

    Breaks land only where a word starts. Cutting every PHRASE_MAX notes
    regardless split "to-day" down the middle, and the halves were then spoken
    as two separate words — losing both the pronunciation and the coarticulation
    that keeping the word whole is the entire point of.
    """
    start, n = 0, len(notes)
    while start < n:
        end = min(start + PHRASE_MAX, n)
        if end < n:
            back = end
            while back > start + 1 and tokens[back][1]:   # [1] = continues a word
                back -= 1
            if back > start:
                end = back
        yield list(notes[start:end]), tokens[start:end]
        start = end


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
    # Just enough to stop a click at the join. len//8 of a 125ms note was 25%
    # of it faded, which is most of why short syllables came out weak.
    xfade = int(0.006 * sr)

    def _speak(text, speed: float = 1.0):
        try:
            y, ssr = tts.synthesize(text, voice=voice, backend=backend,
                                    ref_voice=ref_voice, speed=speed, cache=True)
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
        f = min(len(seg) // 10, xfade)
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
            groups = word_groups(chunk_tokens)
            words = _chunk_words(chunk_tokens)
            text = phrase_text(chunk_tokens)
            for speed in PHRASE_SPEEDS:
                # Separate takes measure the words; the connected take is what
                # actually gets sung, so consonants and word joins stay natural.
                _joined, per_word = _speak_words(_speak, words, speed)
                spoken = _speak(text, speed)
                if spoken is None or per_word is None:
                    continue
                lens = [per_word[i + 1] - per_word[i] for i in range(len(words))]
                edges = _edges_from_durations(spoken, sr, lens)
                try:
                    pieces = _render_phrase(spoken, sr, chunk_notes, prev_hz,
                                            groups=groups, word_edges=edges)
                except Exception:  # noqa: BLE001 — fall back to per-syllable below
                    pieces = []
                if pieces:
                    break
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
