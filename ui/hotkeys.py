"""Hotkeys listing — menu shortcuts plus contextual keys that live on widgets."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui import theme

# (section, action, keys) for shortcuts that are not QActions on the menu bar.
# Menu-bar actions are collected from the window so the dialog cannot drift.
CONTEXTUAL_HOTKEYS: Sequence[Tuple[str, str, str]] = (
    ("Arrangement", "Mute selected track(s)", "0"),
    ("Arrangement", "Solo selected track(s)", "S"),
    ("Arrangement", "Previous / next track", "Up / Down"),
    ("Arrangement", "Nudge playhead (when no clip is selected)", "Left / Right"),
    ("Piano Roll", "Draw mode", "B"),
    ("Piano Roll", "Fold unused pitches", "F"),
    ("Piano Roll", "Zoom time", "+ / −"),
    ("Piano Roll", "Nudge notes / length (Shift) / octave (Shift+Up/Down)", "Arrows"),
    ("Piano Roll", "Select all notes", "Ctrl+A"),
    ("Piano Roll", "Invert note selection", "Ctrl+I"),
    ("Piano Roll", "Cut / copy / paste notes", "Ctrl+X / Ctrl+C / Ctrl+V"),
    ("Piano Roll", "Duplicate notes", "Ctrl+D"),
    ("Piano Roll", "Split notes at playhead", "Ctrl+E"),
    ("Piano Roll", "Legato (notes selected)", "Ctrl+L"),
    ("Piano Roll", "Quantize to grid", "Ctrl+U"),
    ("Piano Roll", "Piano-roll grid 1/16 … 1/2", "Ctrl+1 … Ctrl+5"),
    ("Piano Roll", "Nudge velocity", "Ctrl+Up / Ctrl+Down"),
    ("Piano Roll", "Cancel draw / clear selection", "Escape"),
    ("Piano Roll", "Delete selected notes", "Delete"),
    ("Signal Graph", "Remove selected node (auto-reconnect)", "Delete"),
    ("Signal Graph", "Zoom", "Ctrl+Wheel"),
    ("Signal Graph", "Break a cable", "Right-click the cable"),
    ("EQ", "Adjust band Q", "Mouse wheel on a handle"),
    ("EQ", "Toggle band enable", "Double-click a handle"),
)


def _plain(text: str) -> str:
    return text.replace("&", "")


def menu_hotkeys(window: QWidget) -> List[Tuple[str, str, str]]:
    """Shortcuts attached to the window's menu bar, grouped by top-level menu."""
    rows: List[Tuple[str, str, str]] = []
    menubar = window.menuBar() if hasattr(window, "menuBar") else None
    if menubar is None:
        return rows
    for top in menubar.actions():
        menu = top.menu()
        if menu is None:
            continue
        _walk_menu(menu, _plain(top.text()), rows)
    return rows


def _walk_menu(menu, section: str, rows: List[Tuple[str, str, str]], prefix: str = "") -> None:
    for act in menu.actions():
        if act.isSeparator():
            continue
        sub = act.menu()
        if sub is not None:
            _walk_menu(sub, section, rows, prefix=f"{prefix}{_plain(act.text())} ▸ ")
            continue
        keys = act.shortcut().toString(QKeySequence.NativeText)
        if not keys:
            continue
        rows.append((section, prefix + _plain(act.text()), keys))


def all_hotkey_rows(window: Optional[QWidget] = None) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    if window is not None:
        rows.extend(menu_hotkeys(window))
    rows.extend(CONTEXTUAL_HOTKEYS)
    return rows


class HotkeysDialog(QDialog):
    """Read-only listing of every shortcut the app actually handles."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Hotkeys")
        self.resize(640, 560)

        layout = QVBoxLayout(self)
        hint = QLabel("Shortcuts work unless you are typing in a text field.")
        hint.setStyleSheet(f"color:{theme.FG_DIM};")
        layout.addWidget(hint)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Action", "Keys"])
        self.tree.setRootIsDecorated(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setStyleSheet(
            f"QTreeWidget {{ background:{theme.BG_DEEP}; color:{theme.FG};"
            f" border:1px solid {theme.BORDER}; }}"
            f"QHeaderView::section {{ background:{theme.BG_ELEVATED}; color:{theme.FG_BRIGHT};"
            f" padding:4px 8px; border:none; border-bottom:1px solid {theme.BORDER}; }}"
        )
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        layout.addWidget(self.tree, 1)

        self._fill(all_hotkey_rows(parent))

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def _fill(self, rows: Iterable[Tuple[str, str, str]]) -> None:
        groups: dict[str, QTreeWidgetItem] = {}
        for section, action, keys in rows:
            group = groups.get(section)
            if group is None:
                group = QTreeWidgetItem([section, ""])
                group.setFirstColumnSpanned(True)
                self.tree.addTopLevelItem(group)
                groups[section] = group
            QTreeWidgetItem(group, [action, keys])
        self.tree.expandAll()

    def listing_text(self) -> str:
        """Flattened 'action<TAB>keys' lines — used by tests."""
        lines: list[str] = []
        for i in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(i)
            for j in range(group.childCount()):
                child = group.child(j)
                lines.append(f"{child.text(0)}\t{child.text(1)}")
        return "\n".join(lines)
