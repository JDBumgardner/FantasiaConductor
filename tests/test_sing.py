"""Lyric handling: turning a line of text into one token per note.

The synthesis itself moved to fantasia_core.svs and is covered in test_svs.py.
What survives here is the lyric layer, which svs still depends on — a word must
stay whole across the notes it spans, and hyphen-split syllables must rejoin
into the word they came from.
"""

from __future__ import annotations

import numpy as np
import pytest

from fantasia_core import sing

SR = 44100

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

def test_word_groups_counts_syllables_per_word():
    toks = sing.split_lyrics_joined("fan-ta-si-a con-duc-tor sings", 8)
    assert sing.word_groups(toks) == [4, 3, 1]
