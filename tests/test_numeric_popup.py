"""Pan syntax, typed commit helpers, and exclusive solo-set math."""

from __future__ import annotations

import pytest

from ui.numeric_popup import format_pan, parse_number, parse_pan, solo_selection_states
from ui.transport_bar import _format_bar_beat


def test_format_pan():
    assert format_pan(0.0) == "C"
    assert format_pan(-0.02) == "2L"
    assert format_pan(0.03) == "3R"
    assert format_pan(-1.0) == "100L"


def test_parse_pan():
    assert parse_pan("C") == 0.0
    assert parse_pan("2L") == pytest.approx(-0.02)
    assert parse_pan("3R") == pytest.approx(0.03)
    assert parse_pan("L25") == pytest.approx(-0.25)
    assert parse_pan("-50") == pytest.approx(-0.50)
    assert parse_pan("-0.25") == pytest.approx(-0.25)
    assert parse_number("−12") == -12.0


def test_solo_selection_replaces_prior_solo_set():
    # C was soloed; A+B are now selected → S solos exactly A+B.
    got = solo_selection_states(["A", "B", "C"], ["A", "B"], {"C"})
    assert got == {"A": True, "B": True, "C": False}


def test_solo_selection_toggles_off_when_already_the_set():
    got = solo_selection_states(["A", "B", "C"], ["A", "B"], {"A", "B"})
    assert got == {"A": False, "B": False}


def test_bar_beat_clock_format():
    assert _format_bar_beat(0.0, 120.0, 4) == "001.1.00"
    assert _format_bar_beat(0.5, 120.0, 4) == "001.2.00"
    assert _format_bar_beat(2.0, 120.0, 4) == "002.1.00"
