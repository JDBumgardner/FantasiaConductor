"""Playhead updates must not repaint the world.

The playhead moves ~33 times a second. Repainting the whole viewport for it —
every clip, every note lane, the waveform — costs a third of the UI thread and
starves the audio callback, which is heard as popping. Full repaints belong to
scrolling, where the viewport-pinned ruler and key column would otherwise ghost.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QGraphicsView   # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _record_updates(view):
    calls = []
    view.viewport().update = lambda *a, **k: calls.append(a)
    return calls


def test_timeline_playhead_updates_only_a_strip(qapp):
    from ui.timeline_view import TimelineView

    v = TimelineView()
    v.resize(1200, 600)
    calls = _record_updates(v)
    v.set_playhead(2.0)
    assert calls, "the playhead must invalidate something"
    assert all(len(c) == 4 for c in calls), f"expected rects, got {calls}"


def test_piano_roll_playhead_updates_only_a_strip(qapp):
    from ui.piano_roll import PianoRollView

    v = PianoRollView()
    v.resize(1200, 600)
    calls = _record_updates(v)
    v.set_playhead(2.0)
    assert calls
    assert all(len(c) == 4 for c in calls)


def test_playhead_repaints_where_it_left_and_where_it_arrived(qapp):
    """Only invalidating the new position leaves the old line drawn behind."""
    from ui.timeline_view import TimelineView

    v = TimelineView()
    v.resize(1200, 600)
    v.set_playhead(1.0)
    calls = _record_updates(v)
    v.set_playhead(5.0)
    xs = sorted(c[0] for c in calls)
    assert len(xs) == 2 and xs[1] - xs[0] > 10, f"expected two separate strips: {xs}"


def test_scrolling_still_repaints_everything(qapp):
    """The ruler is painted in viewport coordinates, so a partial update on
    scroll leaves ghost copies of it."""
    from ui.timeline_view import TimelineView

    v = TimelineView()
    v.resize(1200, 600)
    v.show()
    v.setSceneRect(0, 0, 8000, 2000)      # something to actually scroll within
    qapp.processEvents()
    calls = _record_updates(v)
    v.horizontalScrollBar().setValue(150)
    qapp.processEvents()
    assert any(len(c) == 0 for c in calls), "scroll must request a full repaint"


def test_views_are_not_on_full_update_mode(qapp):
    from ui.piano_roll import PianoRollView
    from ui.timeline_view import TimelineView

    for view in (TimelineView(), PianoRollView()):
        assert view.viewportUpdateMode() != QGraphicsView.FullViewportUpdate
