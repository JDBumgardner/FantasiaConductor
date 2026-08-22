"""Combined bottom editor — Piano Roll / Synth, hosted in the main vertical splitter."""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ui import theme
from ui.piano_roll import PianoRollPanel
from ui.synth_panel import SynthPanel


class EditorDock(QWidget):
    """Bottom clip/synth editor. Lives in a QSplitter so the user can grow it."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(0)
        self._split: Optional[QSplitter] = None
        self._saved_sizes: List[int] = []

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        bar = QWidget()
        bar.setObjectName("editorModeBar")
        bar.setStyleSheet(
            f"QWidget#editorModeBar {{ background:{theme.BG_PANEL}; }}"
            f"QWidget#editorModeBar QPushButton {{"
            f"  background: transparent; color: {theme.FG_DIM};"
            f"  border: none; border-bottom: 2px solid transparent;"
            f"  border-radius: 0; padding: 4px 12px; font-weight: 600;"
            f"}}"
            f"QWidget#editorModeBar QPushButton:hover {{"
            f"  color: {theme.FG_BRIGHT}; background: {theme.BG_HOVER};"
            f"}}"
            f"QWidget#editorModeBar QPushButton:checked {{"
            f"  color: {theme.FG_BRIGHT}; background: {theme.BG_SELECTED};"
            f"  border-bottom: 2px solid {theme.ACCENT}; font-weight: 700;"
            f"}}"
        )
        row = QHBoxLayout(bar)
        row.setContentsMargins(6, 4, 6, 4)
        row.setSpacing(6)
        self.btn_piano = QPushButton("🎹 Piano Roll")
        self.btn_synth = QPushButton("🎛 Synth")
        group = QButtonGroup(self)
        group.setExclusive(True)
        for b in (self.btn_piano, self.btn_synth):
            b.setCheckable(True)
            b.setMinimumWidth(110)
            group.addButton(b)
        self.btn_piano.setChecked(True)
        row.addWidget(self.btn_piano)
        row.addWidget(self.btn_synth)
        row.addStretch(1)

        self.stack = QStackedWidget()
        self.piano = PianoRollPanel()
        self.synth = SynthPanel()
        self.stack.addWidget(self.piano)   # index 0
        self.stack.addWidget(self.synth)   # index 1
        self.view = self.piano.view        # convenience alias used by the window

        v.addWidget(bar)
        v.addWidget(self.stack, 1)

        self.btn_piano.clicked.connect(lambda: self.switch_to_piano_mode())
        self.btn_synth.clicked.connect(lambda: self._set_mode(1))
        self.hide()

    def attach_splitter(self, split: QSplitter) -> None:
        self._split = split
        split.splitterMoved.connect(self._remember_sizes)

    def _remember_sizes(self) -> None:
        if self._split is None or not self.isVisible():
            return
        sizes = self._split.sizes()
        if len(sizes) == 2 and sizes[1] > 80:
            self._saved_sizes = sizes

    def _reveal(self) -> None:
        self.setMinimumHeight(140)
        self.show()
        if self._split is None:
            return
        total = max(sum(self._split.sizes()), self._split.height(), 400)
        if self._saved_sizes and self._saved_sizes[1] >= 140:
            self._split.setSizes(self._saved_sizes)
        elif self._split.sizes()[-1] < 140:
            self._split.setSizes([int(total * 0.52), int(total * 0.48)])

    def _set_mode(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)
        self.btn_piano.setChecked(idx == 0)
        self.btn_synth.setChecked(idx == 1)

    def switch_to_piano_mode(self) -> None:
        """Flip to piano-roll mode without forcing the editor open."""
        self._set_mode(0)

    def show_piano_roll(self) -> None:
        self._set_mode(0)
        self._reveal()
        self.piano.view.setFocus(Qt.OtherFocusReason)

    def show_synth(self, track) -> None:  # noqa: ANN001
        self.synth.set_track(track)
        self._set_mode(1)
        self._reveal()
