"""Flagging bad audio blocks, so an intermittent stutter can be investigated."""

import numpy as np
import pytest

from fantasia_core.engine.dropouts import (
    SLOW,
    STARVED,
    UNDERFLOW,
    BlockStats,
    DropoutLog,
)


def test_it_keeps_only_the_most_recent_blocks():
    """A bounded ring: the callback must never grow this."""
    log = DropoutLog(capacity=16)
    log.mark_start(0.0)
    for i in range(50):
        log.record(float(i), i * 0.5, 10.0, 186.0, SLOW)
    rows = log.snapshot(limit=999)
    assert log.total == 50
    assert len(rows) == 16
    assert rows[-1]["since_start"] == 49.0        # newest last
    assert rows[0]["since_start"] == 34.0


def test_recording_does_not_allocate():
    """The audio callback may not allocate; the ring is preallocated."""
    log = DropoutLog(capacity=32)
    log.mark_start(0.0)
    before = log._rows.nbytes
    for i in range(200):
        log.record(float(i), 1.0, 20.0, 186.0, UNDERFLOW)
    assert log._rows.nbytes == before
    assert log._rows.shape[0] == 32


def test_times_are_relative_to_the_last_start():
    """'Does it cluster when the track starts' is the question being asked."""
    log = DropoutLog()
    log.mark_start(1000.0)
    log.record(1000.4, 0.4, 120.0, 186.0, SLOW)
    log.record(1009.0, 9.0, 120.0, 186.0, SLOW)
    assert [r["since_start"] for r in log.snapshot()] == [0.4, 9.0]
    s = log.summary()
    assert s["within_2s_of_start"] == 1
    assert s["share_at_start_pct"] == 50.0


def test_headroom_is_reported_against_the_deadline():
    log = DropoutLog()
    log.mark_start(0.0)
    log.record(0.0, 0.0, 93.0, 186.0, SLOW)
    assert log.snapshot()[0]["headroom_pct"] == 50.0


def test_an_unrendered_clip_is_distinguished_from_a_missed_deadline():
    """Silence because audio was not ready sounds like a gap but is a
    different fault from the callback running out of time."""
    log = DropoutLog()
    log.mark_start(0.0)
    log.record(0.0, 0.0, 5.0, 186.0, STARVED, misses=3)
    row = log.snapshot()[0]
    assert row["kind"] == "starved" and row["unrendered_clips"] == 3


def test_reset_clears_the_history():
    log = DropoutLog()
    log.mark_start(0.0)
    log.record(0.0, 0.0, 5.0, 186.0, SLOW)
    log.reset()
    assert log.total == 0 and log.snapshot() == []
    assert log.summary()["total"] == 0


def test_summary_of_nothing_says_so_rather_than_failing():
    assert DropoutLog().summary()["total"] == 0


def test_block_stats_counts_unrendered_clips():
    from types import SimpleNamespace as NS

    from fantasia_core.engine.mixer import render_block

    stats = BlockStats()
    clip = NS(id="c1", content_type="midi", start=0.0, duration=4.0, notes=[1],
              source_path=None, gain_db=0.0, fade_in=0.0, fade_out=0.0,
              reversed=False, is_midi=True)
    track = NS(id="t1", clips=[clip], mute=False, solo=False, gain_db=0.0, pan=0.0,
               fx=[], fx_wires=[], instrument=0, is_drum=False, is_synth=False,
               plugin="Vital", plugin_state="")
    project = NS(tracks=[track], sample_rate=44100, tempo=120.0, master=None)

    class _NeverReady:
        def cached(self, *a, **k):
            return None

    render_block(project, None, 0, 512, 44100, plugin_renderer=_NeverReady(),
                 warp_compute=False, apply_master=False, stats=stats)
    assert stats.misses == 1
