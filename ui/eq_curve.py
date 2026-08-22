"""EQ response curve — what the track's filters actually do to the spectrum.

The curve is *measured*, not modelled: an impulse is pushed through the real
pedalboard chain and the FFT of the result is the frequency response. That keeps
the display honest — it can never drift from the audio the way a re-implemented
set of biquad formulas would, and it picks up new filter types for free.

Only linear filters are plotted. Non-linear devices (saturation, compression)
have no single frequency response, so including them would be a lie.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from ui import theme

F_MIN, F_MAX = 20.0, 20000.0
DB_MIN, DB_MAX = -24.0, 24.0
_N = 16384  # impulse length — sets the resolution at low frequencies

# Filters with a meaningful frequency response.
LINEAR_FX = {"eq_peak", "eq_low_shelf", "eq_high_shelf", "lowpass", "highpass"}


def measure_response(specs: List[dict], sr: int = 44100):
    """Return (freqs, gains_db) of the chain, or None if it has no filters."""
    bands = [s for s in specs or [] if s.get("type") in LINEAR_FX]
    if not bands:
        return None
    try:
        from fantasia_core.engine.fx import build_board

        board = build_board(bands)
        if board is None:
            return None
        imp = np.zeros((2, _N), dtype=np.float32)
        imp[:, 0] = 1.0
        out = board(imp, float(sr), reset=True)
        spec = np.fft.rfft(out[0].astype(np.float64))
        freqs = np.fft.rfftfreq(_N, 1.0 / sr)
        mag = np.abs(spec)
        keep = (freqs >= F_MIN) & (freqs <= F_MAX)
        db = 20.0 * np.log10(np.maximum(mag[keep], 1e-6))
        return freqs[keep], db
    except Exception:  # noqa: BLE001 — a display must never break audio
        return None


class EqCurveView(QWidget):
    """Read-only plot of the chain's response, log frequency vs dB."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(90)
        self._curve = None          # (freqs, db)
        self._bands: List[dict] = []
        self._title = "No track selected"

    def set_chain(self, specs: List[dict], sr: int = 44100, title: str = "") -> None:
        self._bands = [s for s in specs or [] if s.get("type") in LINEAR_FX]
        self._curve = measure_response(specs, sr)
        self._title = title or "EQ"
        self.update()

    # ---- mapping ---------------------------------------------------------
    def _x(self, f: float, rect: QRectF) -> float:
        lo, hi = np.log10(F_MIN), np.log10(F_MAX)
        t = (np.log10(max(f, F_MIN)) - lo) / (hi - lo)
        return rect.left() + t * rect.width()

    def _y(self, db: float, rect: QRectF) -> float:
        t = (db - DB_MAX) / (DB_MIN - DB_MAX)
        return rect.top() + t * rect.height()

    # ---- painting --------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(38, 8, -8, -18)
        p.fillRect(self.rect(), QColor(theme.TIMELINE_BG))

        # dB grid
        p.setFont(QFont("", 9))
        for db in (-24, -12, 0, 12, 24):
            y = self._y(db, rect)
            p.setPen(QPen(QColor(*(theme.GRID_BAR if db == 0 else theme.GRID_BEAT)), 1))
            p.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
            p.setPen(QColor(theme.FG_DIM))
            p.drawText(4, int(y) + 3, f"{db:+d}" if db else " 0")

        # frequency grid (decades + the readable in-between ticks)
        for f, label in ((30, ""), (50, ""), (100, "100"), (200, ""), (500, ""),
                         (1000, "1k"), (2000, ""), (5000, ""), (10000, "10k"), (20000, "")):
            x = self._x(f, rect)
            p.setPen(QPen(QColor(*(theme.GRID_BAR if label else theme.GRID_BEAT)), 1))
            p.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            if label:
                p.setPen(QColor(theme.FG_DIM))
                p.setFont(QFont("", 9))
                p.drawText(int(x) + 3, int(rect.bottom()) + 13, label)

        if self._curve is None:
            p.setPen(QColor(theme.FG_DIM))
            p.setFont(QFont("", 9))
            p.drawText(rect, Qt.AlignCenter,
                       "No EQ on this track — add one from the track's FX menu")
            return

        freqs, db = self._curve
        path = QPainterPath()
        # Sample the measured response at even pixel steps across the log axis.
        for i in range(int(rect.width())):
            t = i / max(rect.width() - 1, 1)
            f = 10 ** (np.log10(F_MIN) + t * (np.log10(F_MAX) - np.log10(F_MIN)))
            idx = int(np.searchsorted(freqs, f))
            idx = max(0, min(len(db) - 1, idx))
            y = self._y(float(np.clip(db[idx], DB_MIN, DB_MAX)), rect)
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

        # Band markers, numbered in chain order like a real EQ.
        p.setFont(QFont("", 8))
        for i, b in enumerate(self._bands, start=1):
            prm = b.get("params", {})
            f = prm.get("freq", prm.get("cutoff", 1000.0))
            if "gain" in prm:
                db_at = float(prm["gain"])          # bells/shelves: their own gain
            else:
                # High/low-pass have no gain — sit the handle on the measured
                # curve (the −3 dB corner) rather than misleadingly at 0 dB.
                db_at = float(db[int(np.argmin(np.abs(freqs - f)))])
            x, y = self._x(f, rect), self._y(float(np.clip(db_at, DB_MIN, DB_MAX)), rect)
            p.setBrush(QColor(theme.ACCENT))
            p.setPen(QPen(QColor(theme.FG_BRIGHT), 1))
            p.drawEllipse(QPointF(x, y), 7, 7)
            p.drawText(QRectF(x - 7, y - 7, 14, 14), Qt.AlignCenter, str(i))

        p.setPen(QColor(theme.FG_DIM))
        p.setFont(QFont("", 8))
        p.drawText(int(rect.left()) + 4, int(rect.top()) + 11, self._title)
