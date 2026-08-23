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


# ---- why singing is unavailable -----------------------------------------
def test_no_voicebanks_points_at_the_importer(tmp_path, monkeypatch):
    """A fresh install has the packages but no voices. Telling someone to pip
    install then sends them the wrong way entirely."""
    monkeypatch.setenv("FANTASIA_VOICEBANKS", str(tmp_path / "none"))
    msg = sing.why_unavailable()
    assert msg and "voicebank" in msg.lower()
    assert "pip install" not in msg
    assert not sing.available()


def test_missing_packages_says_pip(monkeypatch):
    from fantasia_core import svs

    monkeypatch.setattr(svs, "available", lambda: False)
    msg = sing.why_unavailable()
    assert "pip install" in msg


def test_a_bank_without_a_vocoder_is_called_out(tmp_path, monkeypatch):
    """It loads fine and yields a spectrogram with no sound, so the reason has
    to be stated rather than left as silence."""
    from fantasia_core import svs

    monkeypatch.setattr(svs, "available", lambda: True)
    monkeypatch.setattr(svs, "list_voicebanks", lambda: [
        svs.VoicebankInfo("x", "Silent Bank", "/tmp/x", ["a"], False, "")])
    msg = sing.why_unavailable()
    assert "vocoder" in msg.lower()
    assert "Silent Bank" in msg


def test_available_when_a_ready_bank_exists(monkeypatch):
    from fantasia_core import svs

    monkeypatch.setattr(svs, "available", lambda: True)
    monkeypatch.setattr(svs, "list_voicebanks", lambda: [
        svs.VoicebankInfo("x", "Good", "/tmp/x", ["a"], True, "")])
    assert sing.why_unavailable() is None
    assert sing.available()
