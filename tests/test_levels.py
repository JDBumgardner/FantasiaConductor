"""Header peak tap — dB mapping and consume/reset."""

from __future__ import annotations

import numpy as np
import pytest

from fantasia_core.engine.levels import LevelTap, amp_to_db


def test_amp_to_db_floor_and_unity():
    assert amp_to_db(0.0) == -60.0
    assert amp_to_db(1.0) == 0.0
    assert amp_to_db(0.5) == pytest.approx(-6.02, abs=0.05)


def test_level_tap_holds_then_clears():
    tap = LevelTap()
    block = np.array([[0.25, -0.5], [0.1, 0.0]], dtype=np.float32)
    tap.write("t1", block)
    tap.write_peak("t1", 0.4)
    got = tap.consume()
    assert got["t1"] == pytest.approx(0.5)
    assert tap.consume()["t1"] == 0.0


def test_bounce_length_includes_four_bar_tail():
    from fantasia_core.document import Project
    from fantasia_core.engine import AudioPool, bounce_to_array

    p = Project(tempo=120.0, beats_per_bar=4, sample_rate=8000)
    t = p.add_track("A")
    p.add_clip(t.id, 0.0, 1.0, "c")
    mix = bounce_to_array(p, AudioPool(), 8000)
    assert mix.shape == (int(p.playback_end() * 8000), 2)
    assert p.playback_end() == 9.0
