"""Stock 8-band EQ editor — curve, spectrogram, and band inspector.

The numbered handles are *bands* (the industry term). Dragging moves frequency
(log x) and gain (y); for high/low cuts, vertical drag sets Q instead — gain is
meaningless on a cut. The mouse wheel (or the Q spinbox) also sets width. Type
and on/off live in the inspector so a human has the same surface as
``set_eq_band``.

The frequency response is computed analytically (RBJ biquads) on the UI thread
so a drag never touches pedalboard or the audio callback. The spectrogram is a
precomputed polyline the window feeds from a :class:`SpectrumTap` snapshot;
this widget never reads the playback engine itself.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from fantasia_core.engine.eq import (
    BAND_TYPES,
    CUT_TYPES,
    DB_MAX,
    DB_MIN,
    F_MAX,
    F_MIN,
    MAX_BANDS,
    bands_from_fx,
    format_freq,
    handle_gain,
    log_freqs,
    normalize_band,
    q_from_vertical_drag,
    response_db,
)
from ui import theme

_TYPE_LABELS = {
    "bell": "Bell",
    "low_shelf": "Low Shelf",
    "high_shelf": "High Shelf",
    "low_cut": "Low Cut",
    "high_cut": "High Cut",
    "notch": "Notch",
}

# Analyzer fill: dBFS mapped onto the plot. Visual only — not the EQ gain scale.
_SPEC_DB_LO, _SPEC_DB_HI = -80.0, 0.0

# Frequency ticks: 100 Hz below 1 kHz, 1 kHz above (labels on the readable ones).
_FREQ_TICKS = (
    [(f, "") for f in range(100, 1000, 100)]
    + [(f, "") for f in range(1000, 20001, 1000)]
)
_FREQ_LABELS = {
    100: "100", 200: "200", 500: "500",
    1000: "1k", 2000: "2k", 5000: "5k", 10000: "10k", 20000: "20k",
}

_INSPECTOR_QSS = (
    f"QWidget#eqInspector {{ background:{theme.BG_PANEL}; }}"
    f"QLabel {{ color:{theme.FG}; }}"
    f"QComboBox, QDoubleSpinBox {{"
    f"  background:{theme.BG_ELEVATED}; color:{theme.FG};"
    f"  border:1px solid {theme.BORDER}; border-radius:5px; padding:2px 6px;"
    f"  min-height:22px;"
    f"}}"
    f"QComboBox:hover, QDoubleSpinBox:hover {{ border-color:{theme.CYAN}; }}"
    f"QComboBox:focus, QDoubleSpinBox:focus {{ border-color:{theme.CYAN}; }}"
    f"QComboBox:disabled, QDoubleSpinBox:disabled {{ color:{theme.FG_DIM}; }}"
    f"QComboBox::drop-down {{ border:none; width:16px; }}"
    f"QComboBox QAbstractItemView {{"
    f"  background:{theme.BG_ELEVATED}; color:{theme.FG};"
    f"  selection-background-color:{theme.BG_SELECTED};"
    f"}}"
)


def _plot_rect(widget: QWidget) -> QRectF:
    return QRectF(widget.rect()).adjusted(40, 8, -10, -20)


class _BandPill(QAbstractButton):
    """On/off band chip: fill = enabled, cyan ring = selected.

    Clicking selects the band and toggles it. Same enable surface as the
    power control and a double-click of the handle.
    """

    def __init__(self, number: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._number = number
        self._enabled = False
        self._chosen = False
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(24, 24)
        self.setFocusPolicy(Qt.NoFocus)
        self.setToolTip(f"Toggle band {number}")

    def set_state(self, enabled: bool, chosen: bool) -> None:
        if self._enabled == enabled and self._chosen == chosen:
            return
        self._enabled = bool(enabled)
        self._chosen = bool(chosen)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        fill = QColor(theme.ACCENT if self._enabled else theme.BG_ELEVATED)
        if not self._enabled:
            fill.setAlpha(200)
        p.setBrush(fill)
        ring = QColor(theme.CYAN if self._chosen else theme.BORDER)
        p.setPen(QPen(ring, 2.2 if self._chosen else 1))
        p.drawEllipse(r)
        p.setPen(QColor(theme.FG_BRIGHT if self._enabled or self._chosen else theme.FG_DIM))
        p.setFont(theme.ui_font(8, bold=True))
        p.drawText(self.rect(), Qt.AlignCenter, str(self._number))


class _PowerToggle(QAbstractButton):
    """Compact on/off for the selected band — neon disc, not a checkbox."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(22, 22)
        self.setFocusPolicy(Qt.NoFocus)
        self.setToolTip("Enable this band")

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        on = self.isChecked()
        fill = QColor(theme.GREEN if on else theme.BG_ELEVATED)
        p.setBrush(fill)
        p.setPen(QPen(QColor(theme.GREEN if on else theme.BORDER), 1.6))
        p.drawEllipse(r)
        # Power-stem glyph.
        cx, cy = r.center().x(), r.center().y()
        p.setPen(QPen(QColor(theme.BG_DEEP if on else theme.FG_DIM), 1.8, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(QRectF(cx - 4.5, cy - 3.5, 9, 9), 50 * 16, 260 * 16)
        p.drawLine(QPointF(cx, cy - 5.5), QPointF(cx, cy - 0.5))


class _EqPlot(QWidget):
    """Painted curve + handles. Emits edits; does not talk to the command bus."""

    band_pressed = Signal(int)
    band_moved = Signal(int, float, float, float)  # index, freq, gain, q
    band_released = Signal()
    q_nudged = Signal(int, float)
    band_double_clicked = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(90)
        self.setMouseTracking(True)
        self._bands: List[dict] = []
        self._freqs = log_freqs(512)
        self._db = np.zeros(512)
        self._title = "EQ"
        self._selected = 0
        self._drag: Optional[int] = None
        self._drag_y0 = 0.0
        self._drag_q0 = 1.0
        self._hover: Optional[int] = None
        self._spec_path: Optional[QPainterPath] = None

    def set_bands(self, bands: Sequence[dict], sr: int = 44100, title: str = "") -> None:
        self._bands = [normalize_band(b) for b in bands[:MAX_BANDS]]
        self._db = response_db(self._bands, self._freqs, sr)
        if title:
            self._title = title
        self.update()

    def set_selected(self, index: int) -> None:
        self._selected = max(0, min(MAX_BANDS - 1, index))
        self.update()

    def set_spectrum(self, freqs: Optional[np.ndarray], db: Optional[np.ndarray]) -> None:
        """Precomputed analyzer polyline. ``None`` clears it."""
        if freqs is None or db is None or len(freqs) == 0:
            self._spec_path = None
            self.update()
            return
        rect = _plot_rect(self)
        if rect.width() < 8 or rect.height() < 8:
            return
        path = QPainterPath()
        n = max(int(rect.width()), 2)
        lo, hi = np.log10(F_MIN), np.log10(F_MAX)
        for i in range(n):
            t = i / (n - 1)
            f = 10 ** (lo + t * (hi - lo))
            idx = int(np.searchsorted(freqs, f))
            idx = max(0, min(len(db) - 1, idx))
            mag = float(np.clip(db[idx], _SPEC_DB_LO, _SPEC_DB_HI))
            y_t = (mag - _SPEC_DB_HI) / (_SPEC_DB_LO - _SPEC_DB_HI)
            y = rect.top() + y_t * rect.height()
            pt = QPointF(rect.left() + i, y)
            path.moveTo(pt) if i == 0 else path.lineTo(pt)
        fill = QPainterPath(path)
        fill.lineTo(rect.right(), rect.bottom())
        fill.lineTo(rect.left(), rect.bottom())
        fill.closeSubpath()
        self._spec_path = fill
        self.update()

    # ---- mapping ---------------------------------------------------------
    def _x(self, f: float, rect: QRectF) -> float:
        lo, hi = np.log10(F_MIN), np.log10(F_MAX)
        t = (np.log10(max(f, F_MIN)) - lo) / (hi - lo)
        return rect.left() + t * rect.width()

    def _y(self, db: float, rect: QRectF) -> float:
        t = (db - DB_MAX) / (DB_MIN - DB_MAX)
        return rect.top() + t * rect.height()

    def _freq_at(self, x: float, rect: QRectF) -> float:
        t = (x - rect.left()) / max(rect.width(), 1e-6)
        t = min(1.0, max(0.0, t))
        return float(10 ** (np.log10(F_MIN) + t * (np.log10(F_MAX) - np.log10(F_MIN))))

    def _db_at(self, y: float, rect: QRectF) -> float:
        t = (y - rect.top()) / max(rect.height(), 1e-6)
        t = min(1.0, max(0.0, t))
        return float(DB_MAX + t * (DB_MIN - DB_MAX))

    def _handle_pos(self, i: int, rect: QRectF) -> QPointF:
        b = self._bands[i]
        db_at = handle_gain(b, self._freqs, self._db)
        return QPointF(self._x(b["freq"], rect), self._y(float(np.clip(db_at, DB_MIN, DB_MAX)), rect))

    def _hit(self, pos, rect: QRectF) -> Optional[int]:
        best, dist = None, 16.0
        for i in range(len(self._bands)):
            p = self._handle_pos(i, rect)
            d = ((p.x() - pos.x()) ** 2 + (p.y() - pos.y()) ** 2) ** 0.5
            if d < dist:
                best, dist = i, d
        return best

    # ---- mouse -----------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            return
        rect = _plot_rect(self)
        hit = self._hit(event.position(), rect)
        if hit is None:
            return
        self._drag = hit
        self._drag_y0 = float(event.position().y())
        self._drag_q0 = float(self._bands[hit]["q"])
        self._selected = hit
        self.band_pressed.emit(hit)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        rect = _plot_rect(self)
        if self._drag is None:
            self._hover = self._hit(event.position(), rect)
            self.update()
            return
        freq = self._freq_at(event.position().x(), rect)
        b = self._bands[self._drag]
        if b["type"] in CUT_TYPES:
            dy_norm = (self._drag_y0 - float(event.position().y())) / max(rect.height(), 1.0)
            q = q_from_vertical_drag(self._drag_q0, dy_norm)
            self.band_moved.emit(self._drag, freq, b["gain"], q)
        else:
            gain = self._db_at(event.position().y(), rect)
            self.band_moved.emit(self._drag, freq, gain, b["q"])

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag is not None:
            self._drag = None
            self.band_released.emit()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        rect = _plot_rect(self)
        hit = self._hit(event.position(), rect)
        if hit is not None:
            self.band_double_clicked.emit(hit)

    def wheelEvent(self, event) -> None:  # noqa: N802
        rect = _plot_rect(self)
        hit = self._hit(event.position(), rect)
        if hit is None:
            hit = self._selected
        if hit is None or not (0 <= hit < len(self._bands)):
            return
        steps = event.angleDelta().y() / 120.0
        q = self._bands[hit]["q"] * (1.08 ** steps)
        self.q_nudged.emit(hit, q)
        event.accept()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = None
        self.update()

    # ---- painting --------------------------------------------------------
    def _draw_handle(self, p: QPainter, pos: QPointF, b: dict,
                     selected: bool, hover: bool) -> None:
        r = 9.0 if selected or hover else 7.0
        fill_c = QColor(theme.ACCENT if b["enabled"] else theme.BG_ELEVATED)
        if not b["enabled"]:
            fill_c.setAlpha(160)
        p.setBrush(fill_c)
        ring = QColor(theme.CYAN if selected else (theme.FG_BRIGHT if hover else theme.FG))
        p.setPen(QPen(ring, 2 if selected else 1))
        kind = b["type"]
        if kind in CUT_TYPES:
            if kind == "low_cut":
                tri = QPolygonF([
                    QPointF(pos.x() - r, pos.y() - r),
                    QPointF(pos.x() - r, pos.y() + r),
                    QPointF(pos.x() + r, pos.y()),
                ])
            else:
                tri = QPolygonF([
                    QPointF(pos.x() + r, pos.y() - r),
                    QPointF(pos.x() + r, pos.y() + r),
                    QPointF(pos.x() - r, pos.y()),
                ])
            p.drawPolygon(tri)
        elif "shelf" in kind:
            p.drawRoundedRect(QRectF(pos.x() - r, pos.y() - r, 2 * r, 2 * r), 3.5, 3.5)
        else:
            p.drawEllipse(pos, r, r)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = _plot_rect(self)
        p.fillRect(self.rect(), QColor(theme.TIMELINE_BG))

        p.setFont(theme.ui_font(9))
        for db in (-24, -12, 0, 12, 24):
            y = self._y(db, rect)
            p.setPen(QPen(QColor(*(theme.GRID_BAR if db == 0 else theme.GRID_BEAT)), 1))
            p.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
            p.setPen(QColor(theme.FG_DIM))
            p.drawText(4, int(y) + 3, f"{db:+d}" if db else " 0")

        for f, _blank in _FREQ_TICKS:
            x = self._x(f, rect)
            label = _FREQ_LABELS.get(f, "")
            p.setPen(QPen(QColor(*(theme.GRID_BAR if label else theme.GRID_BEAT)), 1))
            p.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            if label:
                p.setPen(QColor(theme.FG_DIM))
                p.setFont(theme.ui_font(8))
                p.drawText(int(x) + 2, int(rect.bottom()) + 13, label)

        if self._spec_path is not None:
            shade = QColor(theme.YELLOW)
            shade.setAlpha(50)
            p.fillPath(self._spec_path, shade)

        if not self._bands:
            p.setPen(QColor(theme.FG_DIM))
            p.setFont(theme.ui_font(9))
            p.drawText(rect, Qt.AlignCenter, "No track selected")
            return

        path = QPainterPath()
        n = max(int(rect.width()), 2)
        lo, hi = np.log10(F_MIN), np.log10(F_MAX)
        for i in range(n):
            t = i / (n - 1)
            f = 10 ** (lo + t * (hi - lo))
            idx = int(np.searchsorted(self._freqs, f))
            idx = max(0, min(len(self._db) - 1, idx))
            y = self._y(float(np.clip(self._db[idx], DB_MIN, DB_MAX)), rect)
            pt = QPointF(rect.left() + i, y)
            path.moveTo(pt) if i == 0 else path.lineTo(pt)

        fill = QPainterPath(path)
        fill.lineTo(rect.right(), self._y(0.0, rect))
        fill.lineTo(rect.left(), self._y(0.0, rect))
        fill.closeSubpath()
        shade = QColor(theme.CYAN)
        shade.setAlpha(38)
        p.fillPath(fill, shade)
        p.setPen(QPen(QColor(theme.CYAN), 2))
        p.drawPath(path)

        p.setFont(theme.ui_font(8, bold=True))
        for i, b in enumerate(self._bands):
            pos = self._handle_pos(i, rect)
            self._draw_handle(p, pos, b, i == self._selected, i == self._hover)
            p.setPen(QColor(theme.FG_BRIGHT if b["enabled"] else theme.FG_DIM))
            p.drawText(QRectF(pos.x() - 8, pos.y() - 8, 16, 16), Qt.AlignCenter, str(i + 1))

        p.setPen(QColor(theme.FG_DIM))
        p.setFont(theme.ui_font(8))
        p.drawText(int(rect.left()) + 4, int(rect.top()) + 11, self._title)


class EqEditor(QWidget):
    """Curve + inspector. Emits the full 8-band list for the window to commit."""

    bands_changed = Signal(list, bool)  # (bands, mergeable)
    status_message = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._bands: List[dict] = []
        self._sr = 44100
        self._selected = 0
        self._suspend = False

        self.plot = _EqPlot(self)
        self.plot.band_pressed.connect(self._select)
        self.plot.band_moved.connect(self._on_drag)
        self.plot.band_released.connect(self._on_release)
        self.plot.q_nudged.connect(self._on_q)
        self.plot.band_double_clicked.connect(self._on_toggle)

        strip = QWidget()
        strip.setObjectName("eqInspector")
        strip.setStyleSheet(_INSPECTOR_QSS)
        row = QHBoxLayout(strip)
        row.setContentsMargins(8, 6, 8, 8)
        row.setSpacing(6)

        self._band_btns: List[_BandPill] = []
        for i in range(MAX_BANDS):
            btn = _BandPill(i + 1)
            btn.clicked.connect(lambda _=False, idx=i: self._on_pill(idx))
            self._band_btns.append(btn)
            row.addWidget(btn)

        row.addSpacing(4)
        self._power = _PowerToggle()
        self._power.toggled.connect(self._on_power)
        row.addWidget(self._power)

        row.addSpacing(6)
        row.addWidget(QLabel("Type"))
        self._type = QComboBox()
        for key in BAND_TYPES:
            self._type.addItem(_TYPE_LABELS[key], key)
        self._type.currentIndexChanged.connect(self._on_type)
        self._type.setMinimumWidth(108)
        row.addWidget(self._type)

        self._freq = self._spin("Hz", 20.0, 20000.0, 1.0, 10.0, self._on_freq)
        self._gain = self._spin("dB", -24.0, 24.0, 1.0, 0.1, self._on_gain)
        self._q = self._spin("Q", 0.1, 18.0, 2.0, 0.05, self._on_q_spin)
        self._freq_lbl = QLabel("Freq")
        self._gain_lbl = QLabel("Gain")
        self._q_lbl = QLabel("Q")
        row.addWidget(self._freq_lbl)
        row.addWidget(self._freq)
        row.addWidget(self._gain_lbl)
        row.addWidget(self._gain)
        row.addWidget(self._q_lbl)
        row.addWidget(self._q)
        row.addStretch(1)

        self._hint = QLabel()
        self._hint.setStyleSheet(f"color:{theme.FG_DIM}; font-size:10px;")
        row.addWidget(self._hint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.plot, 1)
        layout.addWidget(strip)

    def _spin(self, suffix, lo, hi, decimals, step, slot) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setDecimals(decimals)
        box.setSingleStep(step)
        box.setSuffix(f" {suffix}" if suffix else "")
        box.valueChanged.connect(slot)
        box.setMinimumWidth(92)
        return box

    # ---- public ----------------------------------------------------------
    def set_chain(self, specs: List[dict], sr: int = 44100, title: str = "") -> None:
        self._sr = sr
        self._bands = bands_from_fx(specs)
        self.plot.set_bands(self._bands, sr, title)
        self._sync_inspector()

    def set_spectrum(self, freqs, db) -> None:  # noqa: ANN001
        self.plot.set_spectrum(freqs, db)

    def selected_index(self) -> int:
        return self._selected

    # ---- internals -------------------------------------------------------
    def _commit(self, mergeable: bool, status: str = "") -> None:
        self.plot.set_bands(self._bands, self._sr)
        self.plot.set_selected(self._selected)
        self.bands_changed.emit([dict(b) for b in self._bands], mergeable)
        if status:
            self.status_message.emit(status)

    def _select(self, index: int) -> None:
        self._selected = index
        self.plot.set_selected(index)
        self._sync_inspector()

    def _on_pill(self, index: int) -> None:
        if self._suspend or not self._bands:
            return
        self._on_toggle(index)

    def _on_drag(self, index: int, freq: float, gain: float, q: float) -> None:
        b = self._bands[index]
        b["freq"] = float(freq)
        if b["type"] in CUT_TYPES:
            b["q"] = float(q)
        elif b["type"] in ("bell", "low_shelf", "high_shelf", "notch"):
            b["gain"] = float(gain)
        if not b["enabled"]:
            b["enabled"] = True
        self._selected = index
        self._suspend = True
        self._sync_inspector()
        self._suspend = False
        if b["type"] in CUT_TYPES:
            msg = f"Band {index + 1}  {format_freq(b['freq'])}  Q {b['q']:.2f}"
        else:
            msg = (f"Band {index + 1}  {format_freq(b['freq'])}  "
                   f"{b['gain']:+.1f} dB  Q {b['q']:.2f}")
        self._commit(True, msg)

    def _on_release(self) -> None:
        # The last mergeable drag is already the undo step; a no-op commit
        # would duplicate it. Inspector values are current.
        pass

    def _on_q(self, index: int, q: float) -> None:
        self._bands[index]["q"] = normalize_band({**self._bands[index], "q": q})["q"]
        self._selected = index
        self._suspend = True
        self._q.blockSignals(True)
        self._q.setValue(self._bands[index]["q"])
        self._q.blockSignals(False)
        self._suspend = False
        self._commit(True, f"Band {index + 1}  Q {self._bands[index]['q']:.2f}")

    def _on_toggle(self, index: int) -> None:
        self._bands[index]["enabled"] = not self._bands[index]["enabled"]
        self._selected = index
        self._sync_inspector()
        state = "on" if self._bands[index]["enabled"] else "off"
        self._commit(False, f"Band {index + 1} {state}")

    def _on_power(self, on: bool) -> None:
        if self._suspend or not self._bands:
            return
        self._bands[self._selected]["enabled"] = bool(on)
        self._commit(False, f"Band {self._selected + 1} {'on' if on else 'off'}")

    def _on_type(self, _idx: int = 0) -> None:
        if self._suspend or not self._bands:
            return
        kind = self._type.currentData()
        b = self._bands[self._selected]
        b["type"] = kind
        b["enabled"] = True
        self._sync_inspector()
        self._commit(False, f"Band {self._selected + 1} → {_TYPE_LABELS.get(kind, kind)}")

    def _on_freq(self, value: float) -> None:
        if self._suspend or not self._bands:
            return
        self._bands[self._selected]["freq"] = float(value)
        self._commit(True)

    def _on_gain(self, value: float) -> None:
        if self._suspend or not self._bands:
            return
        self._bands[self._selected]["gain"] = float(value)
        self._commit(True)

    def _on_q_spin(self, value: float) -> None:
        if self._suspend or not self._bands:
            return
        self._bands[self._selected]["q"] = float(value)
        self._commit(True)

    def _sync_inspector(self) -> None:
        self._suspend = True
        for i, btn in enumerate(self._band_btns):
            on = self._bands[i]["enabled"] if i < len(self._bands) else False
            btn.set_state(on, i == self._selected)
        if not self._bands:
            self._suspend = False
            return
        b = self._bands[self._selected]
        self._power.blockSignals(True)
        self._power.setChecked(bool(b["enabled"]))
        self._power.blockSignals(False)
        idx = self._type.findData(b["type"])
        self._type.blockSignals(True)
        if idx >= 0:
            self._type.setCurrentIndex(idx)
        self._type.blockSignals(False)
        for box, val in ((self._freq, b["freq"]), (self._gain, b["gain"]), (self._q, b["q"])):
            box.blockSignals(True)
            box.setValue(val)
            box.blockSignals(False)
        is_cut = b["type"] in CUT_TYPES
        self._gain.setVisible(not is_cut)
        self._gain_lbl.setVisible(not is_cut)
        if is_cut:
            self._hint.setText("Drag up/down = Q · wheel = Q · double-click toggles")
        else:
            self._hint.setText("Drag a band · wheel = Q · double-click toggles")
        self.plot.set_selected(self._selected)
        self._suspend = False


# Back-compat alias used by a few tests / call sites.
EqCurveView = EqEditor
