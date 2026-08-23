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


def test_cursor_past_loop_end_snaps_in():
    pieces, cursor = loop_render_plan(80, 10, True, 10, 30)
    assert 10 <= pieces[0][0] < 30
    assert sum(n for _s, n in pieces) == 10
    assert 10 <= cursor < 30
