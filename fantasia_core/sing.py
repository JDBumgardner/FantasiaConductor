"""Singing: lyric handling, and the bridge to the singing-voice synthesizer.

The rendering lives in :mod:`fantasia_core.svs`, which drives a DiffSinger
voicebank. This module keeps the lyric utilities that turn a line of text into
one token per note, and the entry points the app and agent already call.

It used to synthesize singing itself, by speaking each syllable with a TTS
engine and pushing it to the note's pitch with the WORLD vocoder. That could
not sound right on a held note — a spoken vowel stretched several times its
length has none of the movement a sustained note has — and six rounds of better
alignment, segmentation and sustain all measured better while still sounding
synthetic. A voicebank sings at the written pitch instead, so nothing is
stretched. The vocoder path is gone; :mod:`fantasia_core.vocalfx` still provides
WORLD for its actual strength, editing a real recorded vocal.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def available() -> bool:
    """True when singing can be rendered — i.e. a voicebank is installed."""
    try:
        from fantasia_core import svs

        return svs.available() and any(b.ready for b in svs.list_voicebanks())
    except Exception:  # noqa: BLE001
        return False



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


def sing_notes(notes: Sequence, lyrics: str, voice: str = "", sr: int = 44100,
               ref_voice=None, backend=None, per_syllable: bool = False,
               engine: Optional[str] = None, voicebank: Optional[str] = None,
               speaker: Optional[str] = None) -> np.ndarray:
    """Render a melody + lyrics as sung audio (mono float32 at ``sr``).

    ``voice``, ``ref_voice``, ``backend`` and ``per_syllable`` are leftovers from
    the speech-based engine and are ignored; ``voicebank`` and ``speaker`` are
    what select a voice now. See :func:`fantasia_core.svs.sing_notes`.
    """
    from fantasia_core import svs

    return svs.sing_notes(notes, lyrics, voicebank=voicebank, speaker=speaker, sr=sr)


def sing_to_file(notes: Sequence, lyrics: str, path: str, voice: str = "",
                 sr: int = 44100, ref_voice=None, backend=None,
                 per_syllable: bool = False, engine: Optional[str] = None,
                 voicebank: Optional[str] = None,
                 speaker: Optional[str] = None) -> float:
    """Render to a mono WAV; returns duration in seconds."""
    from fantasia_core import svs

    return svs.sing_to_file(notes, lyrics, path, voicebank=voicebank,
                            speaker=speaker, sr=sr)
