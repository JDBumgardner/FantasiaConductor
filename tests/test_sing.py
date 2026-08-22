"""Singing synthesis: the phrase pipeline and the three bugs that made it choppy.

All three were real and none were caught by ear-free "did it change" checks, so
each gets a test pinning the specific behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from fantasia_core import sing

SR = 44100


def _speech(seconds=1.0, lead=0.4, trail=0.5, sr=SR):
    """A blip of 'voice' padded with silence, the way a TTS engine returns a
    short isolated utterance."""
    n = int(seconds * sr)
    y = np.zeros(n, dtype=np.float32)
    a, b = int(lead * sr), n - int(trail * sr)
    t = np.arange(b - a) / sr
    y[a:b] = (0.5 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    return y


# ---- trim_silence ------------------------------------------------------
def test_trim_silence_removes_tts_padding():
    """Kokoro pads ~400ms before and ~500ms after a syllable. Stretching that
    onto a note left the note ~60% silent — the original choppiness."""
    y = _speech(1.3, lead=0.4, trail=0.5)
    t = sing.trim_silence(y, SR)
    assert len(t) < len(y) * 0.5
    assert np.max(np.abs(t)) == pytest.approx(np.max(np.abs(y)), rel=1e-3)


def test_trim_silence_keeps_a_little_head_room():
    """Trimming flush to the first loud sample clips the onset of a consonant."""
    y = _speech(1.0, lead=0.3, trail=0.3)
    t = sing.trim_silence(y, SR, keep_ms=8.0)
    voiced = int(0.4 * SR)                      # 1.0 - 0.3 - 0.3
    assert len(t) >= voiced
    assert len(t) <= voiced + int(0.02 * SR) + 2


def test_trim_silence_passes_through_silence_and_empty():
    assert len(sing.trim_silence(np.zeros(1000, dtype=np.float32), SR)) == 1000
    assert len(sing.trim_silence(np.zeros(0, dtype=np.float32), SR)) == 0


# ---- word-aware lyric splitting ----------------------------------------
def test_hyphenated_syllables_rejoin_as_one_word():
    """Rejoining as "fan ta si a" makes the TTS pause between syllables, which
    puts the gaps straight back. It has to speak "fantasia"."""
    toks = sing.split_lyrics_joined("fan-ta-si-a", 4)
    assert [t for t, _ in toks] == ["fan", "ta", "si", "a"]
    assert sing.phrase_text(toks) == "fantasia"


def test_separate_words_keep_their_spaces():
    toks = sing.split_lyrics_joined("let me sing", 3)
    assert sing.phrase_text(toks) == "let me sing"


def test_mixed_words_and_syllables():
    toks = sing.split_lyrics_joined("sing to-day", 3)
    assert sing.phrase_text(toks) == "sing today"


def test_split_lyrics_still_pads_and_truncates():
    assert sing.split_lyrics("la", 3) == ["la", "la", "la"]
    assert len(sing.split_lyrics("a b c d e", 3)) == 3


# ---- syllable segmentation ---------------------------------------------
def test_segmentation_does_not_collapse_the_outer_syllables():
    """The quietest points of an utterance are its own head and tail. Picking the
    globally quietest valleys gave 17ms and 9ms end segments, so those notes came
    out unvoiced."""
    y = _speech(1.2, lead=0.02, trail=0.02)
    b = sing.syllable_bounds(y, SR, 4)
    assert len(b) == 5
    durs = [(b[i + 1] - b[i]) / SR for i in range(4)]
    even = (len(y) / SR) / 4
    assert min(durs) > even * 0.4, f"a segment collapsed: {durs}"


def test_segmentation_is_strictly_increasing():
    b = sing.syllable_bounds(_speech(0.3), SR, 6)   # more notes than room
    assert all(b[i] < b[i + 1] for i in range(len(b) - 1))


def test_segmentation_of_a_single_note_is_the_whole_clip():
    y = _speech(0.5)
    assert sing.syllable_bounds(y, SR, 1) == [0, len(y)]


# ---- phrase chunking ---------------------------------------------------
def test_long_lines_are_cut_into_singable_phrases():
    """One 20-syllable utterance segments badly, and singers breathe."""
    notes = [type("N", (), {"start": i * 0.5, "duration": 0.5,
                            "pitch": 60, "velocity": 100})() for i in range(20)]
    toks = sing.split_lyrics_joined(" ".join(["la"] * 20), 20)
    chunks = list(sing._phrase_chunks(notes, toks))
    assert all(len(cn) <= sing.PHRASE_MAX for cn, _ in chunks)
    assert sum(len(cn) for cn, _ in chunks) == 20      # nothing dropped
