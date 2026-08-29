"""Track header: double-click (or F2) to rename, single-click only selects."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fantasia_core.document.model import Project  # noqa: E402
from ui.main_window import export_default_filename  # noqa: E402
from ui.track_header import TrackHeader, TrackHeaderPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _header(qapp):  # noqa: ARG001
    track = Project().add_track("Drums")
    header = TrackHeader(track)
    header.show()
    header.activateWindow()
    QApplication.processEvents()
    return header


def _focused(header):
    """Whether the name field holds focus *within its own window*.

    ``hasFocus()`` also requires that window to be the active one, which in a
    headless run depends on whatever else Qt has shown — so asserting it
    directly fails intermittently under load.
    """
    return header.focusWidget() is header.name_edit


def test_name_starts_read_only(qapp):
    header = _header(qapp)
    assert header.name_edit.isReadOnly()
    assert header.name_edit.focusPolicy() == Qt.NoFocus


def test_single_click_does_not_rename(qapp):
    header = _header(qapp)
    selected = []
    header.clicked.connect(selected.append)
    QTest.mouseClick(header.name_edit, Qt.LeftButton)
    QApplication.processEvents()
    assert header.name_edit.isReadOnly()
    assert selected == [header.track_id]
    assert not _focused(header)


def test_double_click_starts_rename(qapp):
    header = _header(qapp)
    QTest.mouseDClick(header.name_edit, Qt.LeftButton)
    QApplication.processEvents()
    assert not header.name_edit.isReadOnly()
    assert _focused(header)


def test_escape_cancels_rename(qapp):
    header = _header(qapp)
    names = []
    header.renamed.connect(lambda _tid, name: names.append(name))
    header.begin_rename()
    QTest.keyClicks(header.name_edit, "Lead")
    QTest.keyClick(header.name_edit, Qt.Key_Escape)
    QApplication.processEvents()
    assert header.name_edit.isReadOnly()
    assert header.name_edit.text() == "Drums"
    assert names == []


def test_enter_commits_rename(qapp):
    header = _header(qapp)
    names = []
    header.renamed.connect(lambda _tid, name: names.append(name))
    header.begin_rename()
    header.name_edit.setText("Lead")
    header.name_edit.editingFinished.emit()
    QApplication.processEvents()
    assert header.name_edit.isReadOnly()
    assert names == ["Lead"]


def test_header_has_fader_readouts_and_meter(qapp):
    header = _header(qapp)
    assert header.fader_db.text() == "+0"
    assert header.pan_db.text() == "C"
    assert header.out_db.text() == "—  —"
    header.set_meter(0.5, playing=True)
    assert "−6" in header.out_db.text() or "-6" in header.out_db.text()
    header.reset_meter()
    assert header.out_db.text() == "—  —"


def test_current_gain_readout_holds_across_fast_ticks(qapp):
    header = _header(qapp)
    header.set_meter(0.5, playing=True)
    first = header.out_db.text()
    header.set_meter(1.0, playing=True)
    # The current number is held; max may rise. Current half (-6) stays.
    assert first.split()[0] == header.out_db.text().split()[0]


def test_typed_gain_and_pan_commit(qapp):
    header = _header(qapp)
    gains = []
    pans = []
    header.gain_changed.connect(lambda _tid, v: gains.append(v))
    header.pan_changed.connect(lambda _tid, v: pans.append(v))
    assert header._commit_gain("-12")
    assert header.vol.value() == -12
    assert gains[-1] == -12.0
    assert header._commit_pan("25R")
    assert header.pan_db.text() == "25R"
    assert pans[-1] == pytest.approx(0.25)


def test_header_panel_selects_multiple_tracks(qapp):
    p = Project()
    a = p.add_track("A")
    b = p.add_track("B")
    panel = TrackHeaderPanel()
    panel.rebuild(p)
    panel.set_selected([a.id, b.id])
    assert panel._headers[a.id]._selected
    assert panel._headers[b.id]._selected
    panel.set_selected(a.id)
    assert panel._headers[a.id]._selected
    assert not panel._headers[b.id]._selected


def test_export_default_filename_prefers_saved_project():
    assert export_default_filename("/tmp/Neon Nights.fcp", "Scratch", "wav") == "Neon Nights.wav"
    assert export_default_filename(None, "Demo", "mp3") == "Demo.mp3"
    assert export_default_filename(None, "", "wav") == "mix.wav"
