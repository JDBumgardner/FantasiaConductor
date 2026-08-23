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


# ---- word-structure segmentation ---------------------------------------
def test_word_groups_counts_syllables_per_word():
    toks = sing.split_lyrics_joined("fan-ta-si-a con-duc-tor sings", 8)
    assert sing.word_groups(toks) == [4, 3, 1]


def test_bounds_respect_word_structure():
    """Splitting the phrase evenly ignores that words differ in length, and the
    error compounds — the last note ended up holding only the 's' of 'sings'."""
    y = _speech(2.4, lead=0.02, trail=0.02)
    b = sing.syllable_bounds(y, SR, [4, 3, 1])
    assert len(b) == 9                          # 8 syllables -> 9 edges
    # The single-syllable word must not be squeezed to nothing.
    assert (b[8] - b[7]) / SR > 0.05


def test_bounds_accept_a_plain_count():
    y = _speech(1.0)
    assert len(sing.syllable_bounds(y, SR, 4)) == 5


# ---- vowel sustain ------------------------------------------------------
def test_sustain_map_holds_the_vowel_not_the_consonants():
    """Stretching a syllable uniformly turned a 150ms 'sings' into a second of
    sibilant hiss. The vowel should absorb the added time."""
    voiced = np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=bool)
    m = sing._sustain_map(voiced, 40)
    assert len(m) == 40
    stretched = voiced[np.rint(m).astype(int)]
    assert stretched.mean() > 0.75              # natural is 0.50
    assert stretched[0] == False and stretched[-1] == False   # consonants survive


def test_sustain_map_is_uniform_when_there_is_no_vowel():
    voiced = np.zeros(8, dtype=bool)
    m = sing._sustain_map(voiced, 20)
    assert len(m) == 20
    assert np.all(np.diff(m) >= 0)


def test_sustain_map_compressing_does_not_expand():
    m = sing._sustain_map(np.array([0, 1, 1, 0], dtype=bool), 2)
    assert len(m) == 2


# ---- adaptive speaking rate --------------------------------------------
def test_phrase_speeds_go_fastest_first():
    """Slowing down fixes dense lines but roughly doubles envelope
    discontinuity, so it must only be reached for phrases that need it."""
    assert sing.PHRASE_SPEEDS[0] == 1.0
    assert list(sing.PHRASE_SPEEDS) == sorted(sing.PHRASE_SPEEDS, reverse=True)


def test_render_phrase_rejects_a_segment_with_no_vowel():
    """This is the signal the retry loop keys on: a voiceless segment means the
    segmentation missed that syllable, and rendering it produces hiss."""
    sr = SR
    y = np.zeros(int(0.6 * sr), dtype=np.float32)      # silence: no vowels at all
    notes = [type("N", (), {"start": 0.0, "duration": 0.5, "pitch": 60,
                            "velocity": 100})()]
    assert sing._render_phrase(y, sr, notes, 0.0, groups=[1]) == []


# ---- per-word synthesis (exact boundaries) ------------------------------
def test_chunk_words_rebuilds_whole_words():
    toks = sing.split_lyrics_joined("fan-ta-si-a con-duc-tor sings", 8)
    assert sing._chunk_words(toks) == ["fantasia", "conductor", "sings"]


def test_speak_words_reports_exact_joins():
    """Guessing word boundaries handed 'let' 87ms and 'sing' 258ms of source, so
    two identical note values stretched 5.7x and 1.9x — one dragged, the next
    clipped. Speaking each word separately makes the joins exact."""
    lens = {"let": 4410, "me": 8820, "sing": 2205}
    buf, edges = sing._speak_words(
        lambda w, spd: np.ones(lens[w], dtype=np.float32), list(lens), 1.0)
    assert edges == [0, 4410, 13230, 15435]
    assert len(buf) == 15435


def test_speak_words_gives_up_if_a_word_fails():
    buf, edges = sing._speak_words(lambda w, spd: None, ["a"], 1.0)
    assert buf is None and edges is None


def test_bounds_from_words_uses_the_known_edges():
    """Single-syllable words need no guessing at all; only inside a
    multi-syllable word is a split still inferred."""
    y = _speech(1.5, lead=0.0, trail=0.0)
    edges = [0, 22050, 44100, 66150]                 # three words, 0.5s each
    b = sing.syllable_bounds_from_words(y, SR, [1, 2, 1], edges)
    assert b[0] == 0
    assert 22050 in b and 66150 in b                 # word joins preserved exactly
    assert len(b) == 5                               # 4 syllables -> 5 edges
    assert b[1] == 22050                             # first word is one syllable


def test_chunking_never_splits_a_word():
    """A break every PHRASE_MAX notes regardless cut "to-day" in half, and the
    two syllables were then spoken as separate words."""
    toks = sing.split_lyrics_joined("let me sing a song for you to-day now", 10)
    chunks = list(sing._phrase_chunks(list(range(10)), toks))
    assert all(not ct[0][1] for _cn, ct in chunks), "a chunk starts mid-word"
    assert sing._chunk_words(chunks[-1][1])[0] == "today"
    assert sum(len(cn) for cn, _ in chunks) == 10       # nothing dropped


def test_chunking_still_bounds_phrase_length():
    toks = sing.split_lyrics_joined(" ".join(["la"] * 20), 20)
    chunks = list(sing._phrase_chunks(list(range(20)), toks))
    assert all(len(cn) <= sing.PHRASE_MAX for cn, _ in chunks)


def test_a_word_longer_than_the_chunk_limit_still_progresses():
    """A 10-syllable word cannot fit in an 8-note chunk; it must not loop."""
    toks = [("syl", i > 0) for i in range(10)]
    chunks = list(sing._phrase_chunks(list(range(10)), toks))
    assert sum(len(cn) for cn, _ in chunks) == 10


# ---- duration-guided boundaries in a connected phrase -------------------
def test_edges_from_durations_follow_relative_word_length():
    """Per-word takes align perfectly but ruin delivery — an isolated "let" has
    a hard released /t/ that connected speech reduces, and nothing flows into
    the next word. The words are measured separately, but the connected take is
    what gets sung."""
    sr = SR
    y = _speech(3.0, lead=0.0, trail=0.0)
    edges = sing._edges_from_durations(y, sr, [1000, 3000, 1000])   # 1:3:1
    assert edges[0] == 0 and edges[-1] == len(y)
    spans = [edges[i + 1] - edges[i] for i in range(3)]
    assert spans[1] > spans[0] * 2          # the long word gets the long span
    assert abs(spans[0] - spans[2]) < len(y) * 0.12


def test_edges_from_durations_are_strictly_increasing():
    y = _speech(0.4, lead=0.0, trail=0.0)
    edges = sing._edges_from_durations(y, SR, [10, 10, 10, 10, 10, 10])
    assert all(edges[i] < edges[i + 1] for i in range(len(edges) - 1))


def test_edges_from_durations_handles_a_single_word():
    y = _speech(0.5)
    assert sing._edges_from_durations(y, SR, [100]) == [0, len(y)]


# ---- short notes --------------------------------------------------------
def test_short_notes_keep_natural_rate_rather_than_speeding_up():
    """"sing" spoken over 69 frames squeezed into a 25-frame note plays 2.8x too
    fast and turns to mush. The onset should be articulated normally and the
    rest simply not heard."""
    m = sing._compress_map(np.ones(69, dtype=bool), 25)
    assert len(m) == 25
    rate = float(np.median(np.diff(m)))
    assert rate <= 1.4, f"playing back at {rate:.1f}x is a speed-up, not singing"
    assert m[0] == 0.0                       # the attack is kept


def test_compress_absorbs_a_mild_overrun_instead_of_truncating():
    m = sing._compress_map(np.ones(30, dtype=bool), 25)
    assert m[-1] >= 28                       # nearly all of it still fits


def test_compress_never_indexes_past_the_source():
    m = sing._compress_map(np.ones(10, dtype=bool), 4)
    assert m.max() <= 9


def test_compress_map_is_a_noop_when_the_note_is_long_enough():
    voiced = np.zeros(20, dtype=bool)
    voiced[5:15] = True
    m = sing._compress_map(voiced, 40)
    assert len(m) == 40
