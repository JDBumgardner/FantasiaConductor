"""Arrangement grid interval + snap (no Qt)."""

from __future__ import annotations

from ui.grid import (
    GridSpec,
    adaptive_interval_beats,
    fixed_interval_beats,
    grid_interval_seconds,
    snap_time,
)


def test_fixed_quarter_is_one_beat():
    spec = GridSpec(kind="fixed", fixed_key="1/4")
    assert grid_interval_seconds(spec, tempo=120.0, beats_per_bar=4, pps=80) == 0.5


def test_fixed_sixteenth():
    spec = GridSpec(kind="fixed", fixed_key="1/16")
    assert grid_interval_seconds(spec, tempo=120.0, beats_per_bar=4, pps=80) == 0.125


def test_fixed_one_bar_follows_time_signature():
    spec = GridSpec(kind="fixed", fixed_key="1bar")
    assert grid_interval_seconds(spec, tempo=120.0, beats_per_bar=4, pps=80) == 2.0
    assert grid_interval_seconds(spec, tempo=120.0, beats_per_bar=3, pps=80) == 1.5


def test_off_has_no_interval():
    spec = GridSpec(kind="off")
    assert grid_interval_seconds(spec, tempo=120.0, beats_per_bar=4, pps=80) is None
    assert snap_time(1.37, None) == 1.37


def test_snap_to_sixteenth():
    spec = GridSpec(kind="fixed", fixed_key="1/16")
    step = grid_interval_seconds(spec, 120.0, 4, 80)
    assert snap_time(0.14, step) == 0.125
    assert snap_time(0.20, step) == 0.25


def test_triplet_shortens_interval():
    straight = grid_interval_seconds(GridSpec(kind="fixed", fixed_key="1/4"), 120.0, 4, 80)
    trip = grid_interval_seconds(
        GridSpec(kind="fixed", fixed_key="1/4", triplet=True), 120.0, 4, 80
    )
    assert straight == 0.5
    assert abs(trip - (0.5 * 2.0 / 3.0)) < 1e-9


def test_adaptive_picks_coarser_when_zoomed_out():
    # At 120 BPM a quarter note is 0.5s. 8 px/sec → 4 px per beat, so adaptive
    # should pick something coarser than a beat to satisfy a 20 px minimum.
    beats = adaptive_interval_beats(120.0, 4, pps=8.0, min_px=20.0)
    assert beats >= 4.0  # at least a bar


def test_adaptive_picks_finer_when_zoomed_in():
    beats = adaptive_interval_beats(120.0, 4, pps=400.0, min_px=20.0)
    assert beats <= 0.5


def test_fixed_interval_beats_table():
    assert fixed_interval_beats("off", 4) is None
    assert fixed_interval_beats("1/8", 4) == 0.5
    assert fixed_interval_beats("2bars", 4) == 8.0
