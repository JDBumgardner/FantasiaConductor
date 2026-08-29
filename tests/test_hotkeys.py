"""Hotkeys dialog lists menu shortcuts and contextual widget keys."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6.QtGui import QAction  # noqa: E402
from PySide6.QtWidgets import QApplication, QMainWindow  # noqa: E402

from ui.hotkeys import CONTEXTUAL_HOTKEYS, HotkeysDialog, all_hotkey_rows  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_contextual_covers_mute_solo_and_piano_roll(qapp):  # noqa: ARG001
    names = {action for _section, action, _keys in CONTEXTUAL_HOTKEYS}
    assert "Mute selected track(s)" in names
    assert "Solo selected track(s)" in names
    assert "Draw mode" in names
    assert "Remove selected node (auto-reconnect)" in names


def test_dialog_lists_menu_and_contextual_keys(qapp):
    window = QMainWindow()
    view = window.menuBar().addMenu("&View")
    hotkeys = QAction("&Hotkeys…", window, shortcut="F1")
    view.addAction(hotkeys)
    play = QAction("Play/Pause", window, shortcut="Space")
    window.menuBar().addMenu("&Transport").addAction(play)

    dlg = HotkeysDialog(window)
    text = dlg.listing_text()
    assert "Hotkeys" in text
    assert "F1" in text
    assert "Play/Pause" in text
    assert "Mute selected track(s)" in text
    assert "0" in text
    rows = all_hotkey_rows(window)
    assert any(row[1] == "Hotkeys…" and "F1" in row[2] for row in rows)
    # NativeText may render Space as "Space"
    assert any(row[1] == "Play/Pause" for row in rows)


def test_hotkeys_tree_uses_readable_contrast(qapp):
    from PySide6.QtGui import QColor, QPalette

    from ui import theme

    dlg = HotkeysDialog()
    pal = dlg.tree.palette()
    # Alternate rows stay on the dark palette, never system grey.
    assert pal.color(QPalette.AlternateBase).name() == QColor(theme.BG_ELEVATED).name()
    # Selection is the dusty-pink chip with dark ink (dark-on-light).
    assert pal.color(QPalette.HighlightedText).name() == QColor(theme.BUTTON_CHECKED_FG).name()
    sheet = dlg.tree.styleSheet()
    assert theme.BUTTON_CHECKED_FG in sheet
    assert "item:alternate" in sheet
