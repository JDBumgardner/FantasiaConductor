"""Track header: double-click (or F2) to rename, single-click only selects."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fantasia_core.document.model import Project  # noqa: E402
from ui.track_header import TrackHeader  # noqa: E402


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
