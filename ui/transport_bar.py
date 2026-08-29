"""Transport bar: play / stop / loop / metronome, tempo, and time readout.

Buttons emit Qt signals so the audio engine can connect without this widget
knowing about playback internals.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from ui import theme

_BTN = 36


def _deck_icon(kind: str, color: str, glow: bool = False, size: int = 32) -> QIcon:
    """Angular cyberpunk-deck glyph. ``kind`` is play/pause/stop/loop/metro."""
    pm = QPixmap(size, size)
    pm.fill(QColor(0, 0, 0, 0))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    c = QColor(color)
    if glow:
        haze = QColor(c)
        haze.setAlpha(70)
        p.setPen(Qt.NoPen)
        p.setBrush(haze)
        p.drawEllipse(1, 1, size - 2, size - 2)
    p.setPen(QPen(c, 1.6, Qt.SolidLine, Qt.SquareCap, Qt.MiterJoin))
    p.setBrush(c)
    m = size * 0.22
    r = QRectF(m, m, size - 2 * m, size - 2 * m)
    if kind == "play":
        path = QPainterPath()
        path.moveTo(r.left() + 1, r.top())
        path.lineTo(r.right(), r.center().y())
        path.lineTo(r.left() + 1, r.bottom())
        path.closeSubpath()
        p.drawPath(path)
    elif kind == "pause":
        w = r.width() * 0.28
        gap = r.width() * 0.16
        p.fillRect(QRectF(r.left(), r.top(), w, r.height()), c)
        p.fillRect(QRectF(r.right() - w, r.top(), w, r.height()), c)
        _ = gap
    elif kind == "stop":
        p.fillRect(r.adjusted(1, 1, -1, -1), c)
    elif kind == "loop":
        # Clockwise circuit: up the left, across the top, down the right,
        # back along the bottom. Arrow sits on the right edge pointing down.
        p.setBrush(Qt.NoBrush)
        circuit = QPainterPath()
        circuit.moveTo(r.left(), r.bottom() - 3)
        circuit.lineTo(r.left(), r.top())
        circuit.lineTo(r.right(), r.top())
        circuit.lineTo(r.right(), r.bottom())
        circuit.lineTo(r.left() + 4, r.bottom())
        p.drawPath(circuit)
        arrow = QPainterPath()
        ax, ay = r.right(), r.top() + r.height() * 0.42
        arrow.moveTo(ax - 3.6, ay - 1)
        arrow.lineTo(ax + 3.6, ay - 1)
        arrow.lineTo(ax, ay + 5.5)
        arrow.closeSubpath()
        p.setBrush(c)
        p.drawPath(arrow)
    elif kind == "metro":
        p.setBrush(Qt.NoBrush)
        # Two pulse ticks + a base — a click track, not a treble clef.
        p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.bottom()))
        p.setBrush(c)
        p.drawRect(QRectF(r.left() + 1, r.top() + r.height() * 0.15, 3.2, r.height() * 0.7))
        p.drawRect(QRectF(r.center().x() + 1, r.top() + r.height() * 0.4, 3.2, r.height() * 0.45))
    p.end()
    return QIcon(pm)


_DECK_QSS = f"""
QPushButton#deckBtn {{
  background: {theme.BG_ELEVATED};
  color: {theme.FG};
  border: 1px solid {theme.BORDER};
  border-radius: 3px;
  padding: 0px;
}}
QPushButton#deckBtn:hover {{
  border-color: {theme.CYAN};
  background: {theme.BG_HOVER};
}}
QPushButton#deckBtn:checked {{
  border-color: {theme.NEON_ORANGE};
  background: #2a1810;
}}
QPushButton#deckBtn[live="true"] {{
  border-color: {theme.NEON_ORANGE};
  background: #32180c;
}}
QPushButton#deckBtn[tail="true"] {{
  border-color: {theme.NEON_ORANGE};
  background: #4a1c08;
}}
"""


class TransportBar(QWidget):
    """Top-of-window transport controls."""

    play_requested = Signal()
    stop_requested = Signal()
    loop_toggled = Signal(bool)
    metronome_toggled = Signal(bool)
    tempo_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._playing = False
        self._in_tail = False
        self._build()
        self.setStyleSheet(_DECK_QSS)
        self._refresh_icons()
        self._refresh_clock()

    def _mk_btn(self, tip: str, checkable: bool = False) -> QPushButton:
        btn = QPushButton()
        btn.setObjectName("deckBtn")
        btn.setToolTip(tip)
        btn.setFixedSize(_BTN, 28)
        btn.setCheckable(checkable)
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    def _build(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        self.play_btn = self._mk_btn("Play (Space)")
        self.play_btn.clicked.connect(self.play_requested.emit)

        self.stop_btn = self._mk_btn("Stop (Shift+Space)")
        self.stop_btn.clicked.connect(self.stop_requested.emit)

        self.loop_btn = self._mk_btn(
            "Loop (Ctrl+L) — drag the brace in the arrangement ruler", True)
        self.loop_btn.toggled.connect(self._on_loop_toggled)

        self.metro_btn = self._mk_btn("Metronome (Ctrl+M)", True)
        self.metro_btn.toggled.connect(self._on_metro_toggled)

        self.time_label = QLabel("00:00.000")
        self.time_label.setObjectName("timeReadout")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setMinimumWidth(96)
        self.time_label.setToolTip("Playhead — minutes:seconds")

        self._bars_sep = QLabel("▸")
        self._bars_sep.setAlignment(Qt.AlignCenter)
        self.bars_label = QLabel("001.1.00")
        self.bars_label.setObjectName("barsReadout")
        self.bars_label.setAlignment(Qt.AlignCenter)
        self.bars_label.setMinimumWidth(84)
        self.bars_label.setToolTip("Playhead — bar.beat (1-based)")
        self._seconds = 0.0
        self._bpb = 4
        self._time_color = ""

        tempo_label = QLabel("Tempo")
        self.tempo_spin = QDoubleSpinBox()
        self.tempo_spin.setRange(20.0, 300.0)
        self.tempo_spin.setValue(120.0)
        self.tempo_spin.setDecimals(1)
        self.tempo_spin.setSuffix(" BPM")
        self.tempo_spin.setFixedWidth(110)
        self.tempo_spin.valueChanged.connect(self.tempo_changed.emit)

        layout.addWidget(self.play_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.loop_btn)
        layout.addWidget(self.metro_btn)
        layout.addSpacing(12)
        layout.addWidget(self.time_label)
        layout.addWidget(self._bars_sep)
        layout.addWidget(self.bars_label)
        layout.addStretch(1)
        layout.addWidget(tempo_label)
        layout.addWidget(self.tempo_spin)

    def _on_loop_toggled(self, on: bool) -> None:
        self._refresh_icons()
        self.loop_toggled.emit(on)

    def _on_metro_toggled(self, on: bool) -> None:
        self._refresh_icons()
        self.metronome_toggled.emit(on)

    def set_playback_state(self, playing: bool, in_tail: bool = False) -> None:
        """Neon-orange liveness: hotter when audio is past the last clip."""
        playing = bool(playing)
        in_tail = bool(in_tail) and playing
        if playing == self._playing and in_tail == self._in_tail:
            return
        self._playing = playing
        self._in_tail = in_tail
        self.play_btn.setProperty("live", playing)
        self.play_btn.setProperty("tail", in_tail)
        self.stop_btn.setProperty("live", playing)
        self.play_btn.setToolTip("Pause (Space)" if playing else "Play (Space)")
        self._repolish(self.play_btn)
        self._repolish(self.stop_btn)
        self._refresh_icons()
        self._refresh_clock()

    def _repolish(self, widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _refresh_icons(self) -> None:
        idle = theme.CYAN
        live = theme.NEON_ORANGE
        play_color = live if self._playing else idle
        self.play_btn.setIcon(_deck_icon(
            "pause" if self._playing else "play", play_color, glow=self._playing))
        self.stop_btn.setIcon(_deck_icon("stop", live if self._playing else theme.FG_DIM,
                                         glow=self._in_tail))
        self.loop_btn.setIcon(_deck_icon(
            "loop", live if self.loop_btn.isChecked() else idle,
            glow=self.loop_btn.isChecked()))
        self.metro_btn.setIcon(_deck_icon(
            "metro", live if self.metro_btn.isChecked() else idle,
            glow=self.metro_btn.isChecked()))
        for btn in (self.play_btn, self.stop_btn, self.loop_btn, self.metro_btn):
            btn.setIconSize(btn.size() * 0.72)

    def set_loop(self, on: bool) -> None:
        """Sync the button without emitting loop_toggled."""
        blocked = self.loop_btn.blockSignals(True)
        self.loop_btn.setChecked(bool(on))
        self.loop_btn.blockSignals(blocked)
        self._refresh_icons()

    def set_metronome(self, on: bool) -> None:
        """Sync the button without emitting metronome_toggled."""
        blocked = self.metro_btn.blockSignals(True)
        self.metro_btn.setChecked(bool(on))
        self.metro_btn.blockSignals(blocked)
        self._refresh_icons()

    def set_tempo(self, bpm: float) -> None:
        """Update the tempo display without emitting tempo_changed (for syncing
        from the model on load / undo / agent edits)."""
        blocked = self.tempo_spin.blockSignals(True)
        self.tempo_spin.setValue(float(bpm))
        self.tempo_spin.blockSignals(blocked)
        self._refresh_clock()

    def set_time_signature(self, beats_per_bar: int) -> None:
        self._bpb = max(1, int(beats_per_bar))
        self._refresh_clock()

    def set_time(self, seconds: float) -> None:
        """Update the clock and bar.beat readout."""
        self._seconds = max(0.0, float(seconds))
        self._refresh_clock()

    def _refresh_clock(self) -> None:
        seconds = self._seconds
        minutes = int(seconds // 60)
        secs = seconds - minutes * 60
        clock = f"{minutes:02d}:{secs:06.3f}"
        if self.time_label.text() != clock:
            self.time_label.setText(clock)
        bars = _format_bar_beat(seconds, self.tempo_spin.value(), self._bpb)
        if self.bars_label.text() != bars:
            self.bars_label.setText(bars)
        if self._playing:
            color = theme.NEON_ORANGE if self._in_tail else theme.ORANGE
        else:
            color = theme.CYAN
        if color != self._time_color:
            self._time_color = color
            sheet = f"color:{color}; font-weight:700;"
            self.time_label.setStyleSheet(sheet)
            self.bars_label.setStyleSheet(sheet)
            self._bars_sep.setStyleSheet(f"color:{theme.PURPLE}; font-weight:700;")


def _format_bar_beat(seconds: float, tempo: float, beats_per_bar: int) -> str:
    """``001.1.00`` — bar.beat, both 1-based, beat to hundredths."""
    bpb = max(1, int(beats_per_bar))
    spb = 60.0 / tempo if tempo > 0 else 0.5
    total = max(0.0, float(seconds)) / spb
    bar = int(total // bpb) + 1
    beat = (total % bpb) + 1.0
    return f"{bar:03d}.{beat:04.2f}"
