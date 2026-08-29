"""Arrangement loop wrap plan (no audio device)."""

from fantasia_core.engine.playback import loop_render_plan


def test_no_loop_is_a_single_piece():
    pieces, cursor = loop_render_plan(100, 64, False, 0, 1000)
    assert pieces == [(100, 64)]
    assert cursor == 164


def test_loop_wraps_inside_one_block():
    # Loop [0, 50); cursor 40; 20 frames → 10 then wrap 10.
    pieces, cursor = loop_render_plan(40, 20, True, 0, 50)
    assert pieces == [(40, 10), (0, 10)]
    assert cursor == 10


def test_loop_does_not_wrap_before_the_brace():
    pieces, cursor = loop_render_plan(0, 8, True, 20, 40)
    assert pieces == [(0, 8)]
    assert cursor == 8


def test_cursor_past_loop_end_plays_through():
    """Play always starts at the locator. Past the brace, do not snap in."""
    pieces, cursor = loop_render_plan(80, 10, True, 10, 30)
    assert pieces == [(80, 10)]
    assert cursor == 90


def test_play_before_the_loop_enters_then_wraps():
    # Locator at 0, loop [20, 40). A 50-frame block crosses the end and wraps.
    pieces, cursor = loop_render_plan(0, 50, True, 20, 40)
    assert pieces == [(0, 40), (20, 10)]
    assert cursor == 30


def test_play_inside_the_loop_wraps_at_the_end():
    pieces, cursor = loop_render_plan(25, 20, True, 20, 40)
    assert pieces == [(25, 15), (20, 5)]
    assert cursor == 25
