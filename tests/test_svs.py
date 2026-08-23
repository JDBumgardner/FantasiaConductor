"""DiffSinger voicebank singing: planning a melody into what the models expect.

The ONNX chain itself needs a ~410MB voicebank, so these cover the layer that
translates notes and lyrics into tokens, words, rests and durations — which is
where the mistakes live.
"""

from __future__ import annotations

import numpy as np
import pytest

from fantasia_core import svs


class N:
    def __init__(self, pitch, start, duration, velocity=100):
        self.pitch, self.start, self.duration, self.velocity = pitch, start, duration, velocity


def test_g2p_maps_english_words_to_arpabet():
    """Voicebanks ship phoneme maps, not word dictionaries — OpenUtau does this
    conversion itself. CMUdict fills the gap and its inventory lines up."""
    assert svs.g2p("sing") == ["s", "ih", "ng"]
    assert svs.g2p("Today") == ["t", "ah", "d", "ey"]
    assert svs.g2p("qqzzx") is None


def test_g2p_strips_stress_and_punctuation():
    assert all(not p[-1].isdigit() for p in svs.g2p("wonderful"))
    assert svs.g2p("sing,") == svs.g2p("sing")


def test_a_word_may_span_several_notes():
    """"to-day" is one word over two notes: word_dur covers the whole word while
    note_dur stays per note, which is exactly how the models take it."""
    notes = [N(60, 0.0, 0.5), N(62, 0.5, 0.5)]
    p = svs.plan(notes, "to-day")
    assert p.word_div == [4]                     # t ah d ey, one word
    assert len(p.note_midi) == 2                 # still two notes
    assert p.word_dur[0] == sum(p.note_dur)


def test_separate_words_stay_separate():
    p = svs.plan([N(60, 0.0, 0.5), N(62, 0.5, 0.5)], "let me")
    assert len(p.word_div) == 2
    assert p.phones == ["l", "eh", "t", "m", "iy"]


def test_gaps_become_explicit_rests():
    """A gap that is simply omitted shifts everything after it."""
    notes = [N(60, 0.0, 0.5), N(62, 1.5, 0.5)]   # one second of silence between
    p = svs.plan(notes, "let me")
    assert svs.REST in p.phones
    rest_at = p.phones.index(svs.REST)
    assert p.word_div[[i for i, _ in enumerate(p.word_div)][0]] > 0
    assert sum(p.note_dur) > svs._frames(1.0, svs.HOP, svs.SR)


def test_leading_silence_is_preserved():
    p = svs.plan([N(60, 2.0, 0.5)], "la")
    assert p.phones[0] == svs.REST
    assert p.note_dur[0] >= svs._frames(1.9, svs.HOP, svs.SR)


def test_no_rest_when_notes_are_contiguous():
    p = svs.plan([N(60, 0.0, 0.5), N(62, 0.5, 0.5)], "let me")
    assert svs.REST not in p.phones


def test_unknown_words_still_produce_a_phoneme_per_note():
    p = svs.plan([N(60, 0.0, 0.5), N(62, 0.5, 0.5)], "zzq-xxv")
    assert len(p.phones) >= 1
    assert p.word_div and sum(p.word_div) == len(p.phones)


# ---- duration fitting ---------------------------------------------------
def test_phoneme_durations_are_scaled_to_fill_their_notes_exactly():
    """The duration model returns a shape, not a schedule. Unscaled, phonemes
    drift out of time with the notes within a couple of bars."""
    raw = np.array([2.0, 6.0, 2.0, 5.0, 5.0])
    out = svs._fit(raw, [3, 2], [40, 20])
    assert out.shape == (1, 5)
    assert out[0][:3].sum() == 40
    assert out[0][3:].sum() == 20
    assert (out > 0).all()


def test_fit_never_produces_a_zero_length_phoneme():
    out = svs._fit(np.array([100.0, 0.001, 0.001]), [3], [5])
    assert out[0].sum() == 5
    assert (out[0] >= 1).all()


def test_fit_handles_a_word_wanting_fewer_frames_than_phonemes():
    out = svs._fit(np.array([1.0, 1.0, 1.0, 1.0]), [4], [2])
    assert (out[0] >= 1).all()


# ---- voicebank discovery ------------------------------------------------
def test_missing_voicebank_is_named_in_the_error(tmp_path, monkeypatch):
    monkeypatch.setenv("FANTASIA_VOICEBANKS", str(tmp_path / "banks"))
    svs.unload()
    with pytest.raises(RuntimeError, match="no voicebanks installed"):
        svs.load()


def test_unknown_slug_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("FANTASIA_VOICEBANKS", str(tmp_path / "banks"))
    svs.unload()
    (tmp_path / "banks").mkdir(parents=True, exist_ok=True)
    with pytest.raises((RuntimeError, ValueError)):
        svs.load("nope")


def test_listing_an_empty_directory_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("FANTASIA_VOICEBANKS", str(tmp_path / "banks"))
    assert svs.list_voicebanks() == []


# ---- format differences between banks -----------------------------------
class _FakeInput:
    def __init__(self, name, type_): self.name, self.type = name, type_


class _FakeSession:
    def __init__(self, spec): self._spec = [_FakeInput(n, t) for n, t in spec]
    def get_inputs(self): return self._spec


def test_feed_drops_inputs_a_model_does_not_declare():
    """Peiton's duration model takes no speaker embedding; passing one is a
    hard ONNX error rather than something ignored."""
    sess = _FakeSession([("tokens", "tensor(int64)")])
    out = svs._feed(sess, {"tokens": np.array([[1, 2]]), "spk_embed": np.zeros((1, 2, 8))})
    assert set(out) == {"tokens"}


def test_feed_casts_to_the_type_each_model_declares():
    """TIGER declares depth as an int where the other banks use a float."""
    sess = _FakeSession([("depth", "tensor(int64)"), ("f0", "tensor(float)")])
    out = svs._feed(sess, {"depth": np.array(0.6), "f0": np.array([[440.0]], dtype=np.float64)})
    assert out["depth"].dtype == np.int64
    assert out["f0"].dtype == np.float32


def test_feed_passes_bool_inputs_through_as_bool():
    sess = _FakeSession([("note_rest", "tensor(bool)")])
    out = svs._feed(sess, {"note_rest": np.array([[0, 1]])})
    assert out["note_rest"].dtype == np.bool_


def test_phoneme_table_reads_a_plain_text_list(tmp_path):
    """TIGER ships phonemes.txt where the line number is the id, not JSON."""
    d = tmp_path / "bank"
    (d / "dsacoustic").mkdir(parents=True)
    (d / "dsacoustic" / "phonemes.txt").write_text("<PAD>\nAP\nSP\naa\nae\n")
    (d / "dsconfig.yaml").write_text("phonemes: dsacoustic/phonemes.txt\n")
    mod = svs._Module(d / "dsconfig.yaml")
    assert mod.phonemes["SP"] == 2
    assert mod.phonemes["ae"] == 4


def test_root_config_found_in_either_layout(tmp_path):
    """LUNAI banks keep dsconfig in configs/; Peiton puts it at the top level."""
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "dsconfig.yaml").write_text("acoustic: a.onnx\n")
    assert svs._root_config(flat) == flat / "dsconfig.yaml"

    nested = tmp_path / "nested"
    (nested / "configs").mkdir(parents=True)
    (nested / "configs" / "dsconfig.yaml").write_text("acoustic: a.onnx\n")
    assert svs._root_config(nested) == nested / "configs" / "dsconfig.yaml"


# ---- keeping the written melody -----------------------------------------
class _P:
    note_dur = [40, 40]
    note_midi = [62.0, 67.0]


def test_pitch_guard_bounds_drift_from_the_written_note():
    """Peiton's pitch model wanders about four semitones below the note across a
    half-note, which reads as out of tune rather than expressive.

    The bound is measured against a target smoothed across note boundaries, so
    only the steady middle of a note is checked here — near a transition the
    band deliberately sits between the two notes so leaps can glide.
    """
    base = np.concatenate([np.full(40, 62.0), np.full(40, 67.0)])[None].astype(np.float32)
    out = svs._hold_the_tune(base - 4.0, base, _P(), leeway=2.5)
    assert out.shape == base.shape
    steady = np.concatenate([out[0][15:35], out[0][55:75]])
    want = np.concatenate([base[0][15:35], base[0][55:75]])
    assert np.all(steady >= want - 2.6), "drift is not bounded inside a note"
    assert np.all(steady < want), "a flat prediction should still read as flat"


def test_pitch_guard_leaves_expression_alone():
    """Vibrato inside the bound must pass through; only the note transition is
    touched, and only slightly."""
    base = np.concatenate([np.full(40, 62.0), np.full(40, 67.0)])[None].astype(np.float32)
    expressive = base + np.sin(np.linspace(0, 8 * np.pi, 80))[None] * 0.6
    out = svs._hold_the_tune(expressive, base, _P(), leeway=2.5)
    steady = np.r_[0:35, 45:80]
    assert np.allclose(out[0][steady], expressive[0][steady], atol=1e-4)


def test_pitch_guard_can_be_disabled():
    base = np.full((1, 80), 62.0, dtype=np.float32)
    wild = base - 10.0
    assert np.allclose(svs._hold_the_tune(wild, base, _P(), leeway=0), wild)


# ---- hyphenation that is not a dictionary word --------------------------
def test_melisma_gives_each_syllable_its_own_note():
    """"a-lo-one" is one word stretched over three notes. It does not rejoin
    into a dictionary word, so each piece is phonemized on its own — which is
    what the singer is doing anyway."""
    notes = [N(65, 0.0, 0.5), N(64, 0.5, 0.5), N(64, 1.0, 2.0)]
    p = svs.plan(notes, "a-lo-one")
    assert len(p.note_midi) == 3
    assert p.phones == ["ah", "l", "ow", "w", "ah", "n"]
    assert p.word_div == [1, 2, 3]          # one entry per sung syllable


def test_invented_syllable_splits_still_phonemize():
    """"be-ter" is how the line scans, but "beter" is not a word."""
    p = svs.plan([N(67, 0.0, 0.5), N(65, 0.5, 0.5)], "be-ter")
    assert p.phones == ["b", "iy", "t", "er"]


def test_a_real_word_still_spans_its_notes_as_one_unit():
    """The fallback must not fire for words the dictionary knows."""
    p = svs.plan([N(60, 0.0, 0.5), N(62, 0.5, 0.5)], "to-day")
    assert p.word_div == [4]
    assert p.word_dur[0] == sum(p.note_dur)


def test_unknown_syllable_does_not_crash_the_plan():
    p = svs.plan([N(60, 0.0, 0.5), N(62, 0.5, 0.5)], "zqx-vbn")
    assert len(p.note_midi) == 2
    assert sum(p.word_div) == len(p.phones)
