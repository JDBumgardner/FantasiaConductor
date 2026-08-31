"""Choosing a scale should keep you in it, not merely tint the rows."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui import midi_ops  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def panel(qapp):
    from ui.piano_roll import PianoRollPanel

    p = PianoRollPanel()
    yield p
    p.close()
    p.deleteLater()
    qapp.processEvents()


def _choose(panel, root: str, scale: str) -> None:
    ri = next(i for i in range(panel.scale_root.count())
              if panel.scale_root.itemText(i) == root)
    panel.scale_root.setCurrentIndex(ri)
    si = [panel.scale_value.itemText(i) for i in range(panel.scale_value.count())].index(scale)
    panel.scale_value.setCurrentIndex(si)


# ---- the scales on offer ------------------------------------------------
def test_the_klezmer_scales_are_offered_by_that_name():
    """Freygish is the klezmer scale; it was present only as 'Phrygian
    Dominant', which is not what anyone looks for."""
    assert midi_ops.SCALES["Klezmer (Freygish)"] == [0, 1, 4, 5, 7, 8, 10]
    assert midi_ops.SCALES["Klezmer (Misheberakh)"] == [0, 2, 3, 6, 7, 9, 10]


def test_the_usual_scales_are_there_too():
    for name in ("Major", "Natural Minor", "Major Pentatonic", "Minor Pentatonic",
                 "Dorian", "Blues", "Harmonic Minor"):
        assert name in midi_ops.SCALES


def test_choosing_a_scale_sets_it_without_folding(panel):
    """Selecting a scale and hiding rows are separate decisions."""
    before = panel.view.fold
    _choose(panel, "D", "Dorian")
    assert panel.view.scale_name == "Dorian"
    assert panel.view.scale_root == 2
    assert panel.view.fold is before


# ---- nearest_in_scale ---------------------------------------------------
def test_a_pitch_already_in_the_scale_does_not_move():
    pcs = midi_ops.scale_pitch_classes("Major", 0)
    for p in (60, 62, 64, 65, 67, 69, 71):
        assert midi_ops.nearest_in_scale(p, pcs) == p


def test_an_out_of_scale_pitch_goes_to_the_closest_degree():
    # C Eb F G Bb — cases with an unambiguous winner, so a tie rule is not
    # what is being tested here.
    pcs = midi_ops.scale_pitch_classes("Minor Pentatonic", 0)
    assert midi_ops.nearest_in_scale(68, pcs) == 67     # G# -> G down 1, Bb is 2 up
    assert midi_ops.nearest_in_scale(69, pcs) == 70     # A  -> Bb up 1, G is 2 down
    assert midi_ops.nearest_in_scale(61, pcs) == 60     # C# -> C down 1, Eb is 2 up


def test_a_tie_resolves_upward():
    """In C major, F# sits a semitone from both F and G. Picking consistently
    matters more than which way: an unstable choice makes drawing feel random."""
    pcs = midi_ops.scale_pitch_classes("Major", 0)
    assert midi_ops.nearest_in_scale(66, pcs) == 67
    assert midi_ops.nearest_in_scale(61, pcs) == 62
    assert midi_ops.nearest_in_scale(66, pcs) == 67     # and it does not wander


def test_snapping_respects_the_pitch_range():
    pcs = midi_ops.scale_pitch_classes("Major", 0)
    assert midi_ops.nearest_in_scale(61, pcs, lo=62, hi=127) == 62
    assert midi_ops.nearest_in_scale(200, pcs, lo=0, hi=127) <= 127


def test_chromatic_snaps_nothing():
    assert midi_ops.nearest_in_scale(61, None) == 61


# ---- what the roll does with it -----------------------------------------
def test_a_drawn_pitch_is_pulled_into_the_scale(panel):
    """This is the bit that was missing: the rows were tinted, but a note drawn
    between them still landed out of key."""
    _choose(panel, "D", "Klezmer (Freygish)")            # D Eb F# G A Bb C
    assert panel.view.snap_pitch(64) == 63               # E  -> Eb
    assert panel.view.snap_pitch(65) == 66               # F  -> F#
    for in_key in (62, 63, 66, 67, 69, 70, 72):
        assert panel.view.snap_pitch(in_key) == in_key


def test_chromatic_leaves_drawn_pitches_alone(panel):
    _choose(panel, "C", "Chromatic")
    for p in range(60, 72):
        assert panel.view.snap_pitch(p) == p
