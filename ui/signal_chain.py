"""Left-to-right selected-track signal chain (instrument + serial FX)."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fantasia_core.document.fx_insert import (
    device_label,
    insert_bypassed,
    insert_id,
    insert_type,
    linear_order,
)
from fantasia_core.document.model import MASTER_ID, Track
from ui import theme

_CARD = (
    f"QFrame#fxCard {{"
    f"  background: {theme.BG_ELEVATED}; border: 1px solid {theme.BORDER};"
    f"  border-radius: 8px;"
    f"}}"
    f"QLabel#fxTitle {{ color: {theme.FG_BRIGHT}; font-weight: 700; }}"
    f"QLabel#fxSub {{ color: {theme.FG_DIM}; font-size: 10px; }}"
    f"QPushButton#fxX {{"
    f"  background: transparent; color: {theme.FG_DIM}; border: none;"
    f"  font-weight: 700; padding: 0 4px;"
    f"}}"
    f"QPushButton#fxX:hover {{ color: {theme.RED}; }}"
)


def _instrument_label(track: Track) -> str:
    if track.id == MASTER_ID or getattr(track, "is_master", False):
        return "Mix bus"
    plugin = getattr(track, "plugin", "") or ""
    if plugin:
        return plugin
    if getattr(track, "is_synth", False):
        return "Built-in synth"
    if getattr(track, "is_drum", False):
        return "Drum kit"
    return "Soundfont / clips"


class _Card(QFrame):
    def __init__(self, title: str, subtitle: str, removable: bool = False,
                 insert_id: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("fxCard")
        self.insert_id = insert_id
        self.setFixedHeight(88)
        self.setMinimumWidth(128)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 8, 8)
        lay.setSpacing(2)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        name = QLabel(title)
        name.setObjectName("fxTitle")
        top.addWidget(name, 1)
        self.btn_x: Optional[QPushButton] = None
        if removable:
            btn = QPushButton("✕")
            btn.setObjectName("fxX")
            btn.setFixedSize(18, 18)
            btn.setToolTip("Remove from chain")
            top.addWidget(btn)
            self.btn_x = btn
        lay.addLayout(top)
        sub = QLabel(subtitle)
        sub.setObjectName("fxSub")
        lay.addWidget(sub)
        lay.addStretch(1)


class _Arrow(QLabel):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("→", parent)
        self.setStyleSheet(f"color: {theme.CYAN}; font-size: 18px; font-weight: 700;")
        self.setAlignment(Qt.AlignCenter)
        self.setFixedWidth(28)


class SignalChainView(QWidget):
    """Instrument → FX… → Out, for the selected track."""

    add_requested = Signal()
    remove_requested = Signal(str)
    device_activated = Signal(str, str)  # insert_id, type  ("" / "instrument")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._track: Optional[Track] = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        bar = QWidget()
        bar.setStyleSheet(f"background: {theme.BG_PANEL};")
        brow = QHBoxLayout(bar)
        brow.setContentsMargins(12, 8, 12, 4)
        self._title = QLabel("Signal chain")
        self._title.setStyleSheet(f"color: {theme.FG}; font-weight: 700;")
        brow.addWidget(self._title)
        brow.addStretch(1)
        hint = QLabel("Click a device to edit · ✕ removes it · + adds FX")
        hint.setStyleSheet(f"color: {theme.FG_DIM}; font-size: 10px;")
        brow.addWidget(hint)
        outer.addWidget(bar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: {theme.TIMELINE_BG}; border: none; }}"
        )
        self._row_host = QWidget()
        self._row_host.setStyleSheet(
            f"background: {theme.TIMELINE_BG}; {_CARD}"
        )
        self._row = QHBoxLayout(self._row_host)
        self._row.setContentsMargins(16, 16, 16, 16)
        self._row.setSpacing(4)
        self._row.addStretch(1)
        scroll.setWidget(self._row_host)
        outer.addWidget(scroll, 1)

    def set_track(self, track: Optional[Track]) -> None:
        self._track = track
        self._rebuild()

    def _clear_row(self) -> None:
        while self._row.count():
            item = self._row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _rebuild(self) -> None:
        self._clear_row()
        track = self._track
        if track is None:
            self._title.setText("Signal chain — no track selected")
            empty = QLabel("Select a track (or a clip) to see its instrument and FX.")
            empty.setStyleSheet(f"color: {theme.FG_DIM};")
            self._row.addWidget(empty)
            self._row.addStretch(1)
            return
        name = "Master" if getattr(track, "is_master", False) else track.name
        self._title.setText(f"Signal chain — {name}")

        inst = _Card(_instrument_label(track), "Source", removable=False, insert_id="")
        inst.mousePressEvent = lambda e, t="instrument": self._activate("", t)  # noqa: ARG005
        self._row.addWidget(inst)
        self._row.addWidget(_Arrow())

        order = linear_order(track.fx, getattr(track, "fx_wires", None))
        by_id = {insert_id(e): e for e in track.fx if insert_id(e)}
        for nid in order:
            spec = by_id.get(nid)
            if spec is None:
                continue
            sub = "bypassed" if insert_bypassed(spec) else insert_type(spec)
            card = _Card(device_label(spec), sub, removable=True, insert_id=nid)
            if card.btn_x is not None:
                card.btn_x.clicked.connect(lambda _=False, i=nid: self.remove_requested.emit(i))
            kind = insert_type(spec)
            card.mousePressEvent = lambda e, i=nid, k=kind: self._on_card(e, i, k)
            self._row.addWidget(card)
            self._row.addWidget(_Arrow())

        gain = getattr(track, "gain_db", 0.0)
        out_title = "Master" if getattr(track, "is_master", False) else "Channel out"
        out = _Card(out_title, f"{gain:+.1f} dB  → mix")
        self._row.addWidget(out)

        plus = QToolButton()
        plus.setText("+")
        plus.setToolTip("Add FX")
        plus.setFixedSize(36, 36)
        plus.setStyleSheet(
            f"QToolButton {{ background: {theme.BG_ELEVATED}; color: {theme.CYAN};"
            f" border: 1px solid {theme.CYAN}; border-radius: 18px; font-size: 18px; }}"
            f"QToolButton:hover {{ background: {theme.BG_HOVER}; }}"
        )
        plus.clicked.connect(self.add_requested.emit)
        self._row.addSpacing(12)
        self._row.addWidget(plus, 0, Qt.AlignVCenter)
        self._row.addStretch(1)

    def _on_card(self, event, insert_id: str, kind: str) -> None:  # noqa: ANN001
        if event.button() == Qt.LeftButton:
            self.device_activated.emit(insert_id, kind)

    def _activate(self, insert_id: str, kind: str) -> None:
        self.device_activated.emit(insert_id, kind)
