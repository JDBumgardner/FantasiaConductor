"""Timeline interval selection and clip multi-select helpers."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fantasia_core.document import Project  # noqa: E402
from ui.timeline_view import ClipItem, TimelineView  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _clip_items(view: TimelineView) -> dict[str, ClipItem]:
    return {item.clip.id: item for item in view._scene.items() if isinstance(item, ClipItem)}


def _click_clip(view: TimelineView, item: ClipItem, modifiers=Qt.NoModifier) -> None:
    center = item.sceneBoundingRect().center()
    vp = view.mapFromScene(center)
    QTest.mouseClick(view.viewport(), Qt.LeftButton, modifiers, vp)
    QApplication.processEvents()


def test_range_span_normalizes_and_clears(qapp):  # noqa: ARG001
    view = TimelineView()
    p = Project()
    t = p.add_track("A")
    view.set_project(p)
    view.range_select = (t.id, 2.0, 0.5)
    assert view.range_span() == (t.id, 0.5, 1.5)
    view.clear_range_select()
    assert view.range_span() is None


def test_zero_width_range_is_ignored(qapp):  # noqa: ARG001
    view = TimelineView()
    p = Project()
    t = p.add_track("A")
    view.set_project(p)
    view.range_select = (t.id, 1.0, 1.0)
    assert view.range_span() is None


def test_shift_click_keeps_multiple_midi_clips_selected(qapp):  # noqa: ARG001
    view = TimelineView()
    p = Project()
    t = p.add_track("A")
    a = p.add_clip(t.id, 0.0, 1.0, "a", content_type="midi")
    b = p.add_clip(t.id, 2.0, 1.0, "b", content_type="midi")
    view.set_project(p)
    view.rebuild()
    view.resize(900, 280)
    view.show()
    QApplication.processEvents()
    items = _clip_items(view)
    _click_clip(view, items[a.id])
    assert set(view.selected_clip_ids()) == {a.id}
    _click_clip(view, items[b.id], Qt.ShiftModifier)
    assert set(view.selected_clip_ids()) == {a.id, b.id}


def test_clicking_unselected_midi_clip_does_not_move_locator(qapp):  # noqa: ARG001
    view = TimelineView()
    p = Project()
    t = p.add_track("A")
    clip = p.add_clip(t.id, 1.0, 2.0, "a", content_type="midi")
    view.set_project(p)
    view.rebuild()
    view.resize(900, 280)
    view.show()
    QApplication.processEvents()
    view.start_position = 0.0
    view.playhead = 0.0
    item = _clip_items(view)[clip.id]
    _click_clip(view, item)
    assert set(view.selected_clip_ids()) == {clip.id}
    assert view.start_position == 0.0


def test_clicking_selected_midi_clip_locates_to_click(qapp):  # noqa: ARG001
    view = TimelineView()
    p = Project()
    t = p.add_track("A")
    clip = p.add_clip(t.id, 0.0, 2.0, "a", content_type="midi")
    view.set_project(p)
    view.rebuild()
    view.resize(900, 280)
    view.show()
    QApplication.processEvents()
    item = _clip_items(view)[clip.id]
    _click_clip(view, item)
    view.start_position = 0.0
    _click_clip(view, item)
    assert clip.start < view.start_position < clip.end
    assert abs(view.start_position - view.snap(view.start_position)) < 1e-6


def test_reveal_locator_pans_when_offscreen(qapp):  # noqa: ARG001
    view = TimelineView()
    p = Project()
    p.add_track("A")
    view.set_project(p)
    view.rebuild()
    view.resize(400, 240)
    view.show()
    QApplication.processEvents()
    view.start_position = 12.0  # 12s × 80 pps is past a 400px viewport
    assert view.mapFromScene(view.start_position * view.pps, 0).x() > view.viewport().width()
    view.reveal_locator()
    loc_x = view.mapFromScene(view.start_position * view.pps, 0).x()
    assert 0 <= loc_x <= view.viewport().width()


def test_arrow_keys_nudge_locator_even_when_a_clip_is_selected(qapp):  # noqa: ARG001
    view = TimelineView()
    p = Project(tempo=120.0)
    t = p.add_track("A")
    clip = p.add_clip(t.id, 0.0, 2.0, "a", content_type="midi")
    view.set_project(p)
    view.rebuild()
    view.resize(900, 280)
    view.show()
    QApplication.processEvents()
    view.select_clips([clip.id])
    view.start_position = 1.0
    scroll_before = view.horizontalScrollBar().value()
    view.setFocus()
    QTest.keyClick(view, Qt.Key_Right)
    QApplication.processEvents()
    step = view._grid_seconds()
    assert step is not None and step > 0
    assert view.start_position == pytest.approx(1.0 + step)
    assert view.horizontalScrollBar().value() == scroll_before
    QTest.keyClick(view, Qt.Key_Left)
    QApplication.processEvents()
    assert view.start_position == pytest.approx(1.0)


def test_zoom_keeps_locator_at_the_same_viewport_x(qapp):  # noqa: ARG001
    view = TimelineView()
    p = Project()
    p.add_track("A")
    view.set_project(p)
    view.rebuild()
    view.resize(800, 240)
    view.show()
    QApplication.processEvents()
    view.start_position = 8.0
    view.horizontalScrollBar().setValue(int(4.0 * view.pps))
    QApplication.processEvents()
    before = view.mapFromScene(view.start_position * view.pps, 0).x()
    old_pps = view.pps
    view.zoom_in()
    QApplication.processEvents()
    after = view.mapFromScene(view.start_position * view.pps, 0).x()
    assert view.pps == pytest.approx(old_pps * 1.25)
    assert after == pytest.approx(before, abs=2)


def test_clip_color_inherits_track_until_overridden(qapp):  # noqa: ARG001
    view = TimelineView()
    p = Project()
    t = p.add_track("A")
    t.color = "#25e6d5"
    clip = p.add_clip(t.id, 0.0, 1.0, "a", content_type="midi")
    view.set_project(p)
    assert view.clip_color(clip.id) == "#25e6d5"
    clip.color = "#ffd76b"
    assert view.clip_color(clip.id) == "#ffd76b"
