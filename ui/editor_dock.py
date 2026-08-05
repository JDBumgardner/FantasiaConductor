"""Combined bottom editor — one dock with a mode switch between the Piano Roll
and the Synth panel, instead of two separate/tabbed docks."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QDockWidget,
    QHBoxLayout,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ui import theme
from ui.piano_roll import PianoRollPanel
from ui.synth_panel import SynthPanel


class EditorDock(QDockWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Editor", parent)
        self.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)

        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        bar = QWidget()
        bar.setStyleSheet(f"background:{theme.BG_PANEL};")
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
        self.setWidget(body)
        self.setMinimumHeight(240)

        self.btn_piano.clicked.connect(lambda: self.switch_to_piano_mode())
        self.btn_synth.clicked.connect(lambda: self._set_mode(1))

    def _set_mode(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)
        self.btn_piano.setChecked(idx == 0)
        self.btn_synth.setChecked(idx == 1)

    def switch_to_piano_mode(self) -> None:
        """Flip to piano-roll mode without forcing the dock open."""
        self._set_mode(0)

    def show_piano_roll(self) -> None:
        self._set_mode(0)
        self.show()
        self.raise_()

    def show_synth(self, track) -> None:  # noqa: ANN001
        self.synth.set_track(track)
        self._set_mode(1)
        self.show()
        self.raise_()
