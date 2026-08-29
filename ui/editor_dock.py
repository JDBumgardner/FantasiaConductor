"""Combined bottom editor — Piano / Chain / Graph / Synth / EQ, in the splitter."""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ui import theme
from ui.eq_curve import EqEditor
from ui.fx_graph import FxGraphEditor
from ui.piano_roll import PianoRollPanel
from ui.signal_chain import SignalChainView
from ui.synth_panel import SynthPanel

# Stack indices — keep piano/synth/eq at 0/1/2 so existing tests hold.
MODE_PIANO = 0
MODE_SYNTH = 1
MODE_EQ = 2
MODE_CHAIN = 3
MODE_GRAPH = 4

# Shift-E cycle (synth/EQ are extra tabs, not in this loop).
CYCLE = ("piano", "chain", "graph", "off")


class EditorDock(QWidget):
    """Bottom clip/synth/chain editor. Lives in a QSplitter so the user can grow it."""

    mode_changed = Signal(str)  # piano / synth / eq / chain / graph / off

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
        tab_btn = (
            f"QPushButton {{"
            f"  background: transparent; color: {theme.FG_DIM};"
            f"  border: none; border-bottom: 2px solid transparent;"
            f"  border-radius: 0; padding: 4px 12px; font-weight: 600;"
            f"}}"
            f"QPushButton:hover {{"
            f"  color: {theme.FG_BRIGHT}; background: {theme.BG_HOVER};"
            f"}}"
            f"QPushButton:checked {{"
            f"  color: {theme.FG_BRIGHT}; background: {theme.BG_SELECTED};"
            f"  border: none; border-bottom: 2px solid {theme.ACCENT}; font-weight: 700;"
            f"}}"
        )
        bar.setStyleSheet(f"QWidget#editorModeBar {{ background:{theme.BG_PANEL}; }}")
        row = QHBoxLayout(bar)
        row.setContentsMargins(6, 4, 6, 4)
        row.setSpacing(6)
        self.btn_piano = QPushButton("🎹 Piano Roll")
        self.btn_chain = QPushButton("🔗 Chain")
        self.btn_graph = QPushButton("◇ Graph")
        self.btn_synth = QPushButton("🎛 Synth")
        self.btn_eq = QPushButton("📈 EQ")
        group = QButtonGroup(self)
        group.setExclusive(True)
        for b in (self.btn_piano, self.btn_chain, self.btn_graph, self.btn_synth, self.btn_eq):
            b.setCheckable(True)
            b.setMinimumWidth(96)
            b.setStyleSheet(tab_btn)
            group.addButton(b)
        self.btn_piano.setChecked(True)
        row.addWidget(self.btn_piano)
        row.addWidget(self.btn_chain)
        row.addWidget(self.btn_graph)
        row.addWidget(self.btn_synth)
        row.addWidget(self.btn_eq)
        row.addStretch(1)
        self.btn_close = QPushButton("✕")
        self.btn_close.setToolTip("Close the editor panel")
        self.btn_close.setFixedWidth(30)
        self.btn_close.setCheckable(False)
        row.addWidget(self.btn_close)

        self.stack = QStackedWidget()
        self.piano = PianoRollPanel()
        self.synth = SynthPanel()
        synth_scroll = QScrollArea()
        synth_scroll.setWidget(self.synth)
        synth_scroll.setWidgetResizable(True)
        synth_scroll.setFrameShape(QScrollArea.NoFrame)
        synth_scroll.setMinimumHeight(60)
        synth_scroll.setStyleSheet(
            f"QScrollArea {{ background: {theme.BG_PANEL}; border: none; }}"
            f"QScrollArea > QWidget {{ background: {theme.BG_PANEL}; }}"
        )
        synth_scroll.viewport().setStyleSheet(f"background: {theme.BG_PANEL};")
        self.stack.setStyleSheet(f"QStackedWidget {{ background: {theme.BG_PANEL}; }}")
        self.piano.setMinimumHeight(60)
        self.eq = EqEditor()
        self.chain = SignalChainView()
        self.graph = FxGraphEditor()
        self.stack.addWidget(self.piano)         # 0
        self.stack.addWidget(synth_scroll)       # 1
        self.stack.addWidget(self.eq)            # 2
        self.stack.addWidget(self.chain)         # 3
        self.stack.addWidget(self.graph)         # 4
        self.stack.setMinimumHeight(60)
        self.view = self.piano.view

        v.addWidget(bar)
        v.addWidget(self.stack, 1)

        self.btn_piano.clicked.connect(lambda: self._set_mode(MODE_PIANO))
        self.btn_synth.clicked.connect(lambda: self._set_mode(MODE_SYNTH))
        self.btn_eq.clicked.connect(lambda: self._set_mode(MODE_EQ))
        self.btn_chain.clicked.connect(lambda: self._set_mode(MODE_CHAIN))
        self.btn_graph.clicked.connect(lambda: self._set_mode(MODE_GRAPH))
        self.btn_close.clicked.connect(self.close_panel)
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

    def is_open(self) -> bool:
        return self.isVisible()

    def is_piano_open(self) -> bool:
        return self.isVisible() and self.stack.currentIndex() == MODE_PIANO

    def is_chain_open(self) -> bool:
        return self.isVisible() and self.stack.currentIndex() == MODE_CHAIN

    def is_graph_open(self) -> bool:
        return self.isVisible() and self.stack.currentIndex() == MODE_GRAPH

    def current_cycle_name(self) -> str:
        if not self.isVisible():
            return "off"
        idx = self.stack.currentIndex()
        if idx == MODE_CHAIN:
            return "chain"
        if idx == MODE_GRAPH:
            return "graph"
        if idx == MODE_PIANO:
            return "piano"
        return "piano"

    def collapse(self) -> None:
        """Hide the editor and give the arrangement the full splitter."""
        if self._split is not None and self.isVisible():
            sizes = self._split.sizes()
            if len(sizes) == 2 and sizes[1] > 80:
                self._saved_sizes = sizes
        self.setMinimumHeight(0)
        self.hide()
        self.mode_changed.emit("off")

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

    def close_panel(self) -> None:
        """Hide the editor and hand its space back to the arrangement."""
        self._remember_sizes()
        self.hide()
        if self._split is not None:
            sizes = self._split.sizes()
            if len(sizes) == 2:
                self._split.setSizes([sum(sizes), 0])
        self.mode_changed.emit("off")

    def _set_mode(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)
        self.btn_piano.setChecked(idx == MODE_PIANO)
        self.btn_synth.setChecked(idx == MODE_SYNTH)
        self.btn_eq.setChecked(idx == MODE_EQ)
        self.btn_chain.setChecked(idx == MODE_CHAIN)
        self.btn_graph.setChecked(idx == MODE_GRAPH)
        names = {MODE_PIANO: "piano", MODE_SYNTH: "synth", MODE_EQ: "eq",
                 MODE_CHAIN: "chain", MODE_GRAPH: "graph"}
        self.mode_changed.emit(names.get(idx, "piano"))

    def switch_to_piano_mode(self) -> None:
        self._set_mode(MODE_PIANO)

    def show_piano_roll(self) -> None:
        self._set_mode(MODE_PIANO)
        self._reveal()
        self.piano.view.setFocus(Qt.OtherFocusReason)

    def show_synth(self, track, reveal: bool = True) -> None:  # noqa: ANN001
        self.synth.set_track(track)
        self._set_mode(MODE_SYNTH)
        if reveal:
            self._reveal()

    def show_eq(self, specs, sr: int = 44100, title: str = "") -> None:
        self.eq.set_chain(specs, sr, title)
        self._set_mode(MODE_EQ)
        self._reveal()

    def show_chain(self, track=None) -> None:  # noqa: ANN001
        if track is not None:
            self.chain.set_track(track)
        self._set_mode(MODE_CHAIN)
        self._reveal()

    def show_graph(self, track=None) -> None:  # noqa: ANN001
        if track is not None:
            self.graph.set_track(track)
        self._set_mode(MODE_GRAPH)
        self._reveal()

    def next_cycle_action(self) -> str:
        """Name of the next Shift-E state: piano, chain, graph, or off."""
        if not self.is_open():
            return "piano"
        name = self.current_cycle_name()
        if self.stack.currentIndex() in (MODE_SYNTH, MODE_EQ):
            return "chain"
        nxt = {"piano": "chain", "chain": "graph", "graph": "off"}
        return nxt.get(name, "chain")
