"""Circular parameter knob — the stock control for FX / synth values.

Drag vertically or use the wheel. The numeric value sits beside the dial so
the same widget can later host modulation / mapping indicators without a
second text field.
"""

from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from fantasia_core.document.fx_params import ParamSpec
from ui import theme

KNOB_D = 32
# Sweep: 7 o'clock → 5 o'clock (typical mixer knob).
_START = 225.0
_SPAN = 270.0


class ParamKnob(QWidget):
    """A single rotary float control. Emits ``changed`` on every drag sample."""

    changed = Signal(float)

    def __init__(self, spec: ParamSpec, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._spec = spec
        self._value = float(spec.default)
        self._drag_y: Optional[int] = None
        self._drag_val = 0.0
        self.setFixedSize(KNOB_D + 2, KNOB_D + 2)
        self.setCursor(Qt.SizeVerCursor)
        self.setFocusPolicy(Qt.ClickFocus)
        hint = "drag up for more" if getattr(spec, "invert", False) else "drag or scroll"
        self.setToolTip(f"{spec.label} — {hint}")

    def value(self) -> float:
        return self._value

    def set_value(self, value: float) -> None:
        lo, hi = self._spec.minimum, self._spec.maximum
        try:
            v = min(hi, max(lo, float(value)))
        except (TypeError, ValueError):
            return
        if abs(v - self._value) < 10 ** (-(self._spec.decimals + 2)):
            return
        self._value = v
        self.update()

    def format_value(self) -> str:
        spec = self._spec
        text = f"{self._value:.{spec.decimals}f}"
        return f"{text}{spec.suffix}" if spec.suffix else text

    def _norm(self) -> float:
        lo, hi = self._spec.minimum, self._spec.maximum
        if hi <= lo:
            return 0.0
        n = (self._value - lo) / (hi - lo)
        return 1.0 - n if getattr(self._spec, "invert", False) else n

    def paintEvent(self, event) -> None:  # noqa: N802, ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        d = KNOB_D
        pad = 2.0
        box = QRectF(pad, pad, d - 2, d - 2)
        painter.setPen(QPen(QColor(theme.BORDER), 1.2))
        painter.setBrush(QColor(theme.BG_ELEVATED))
        painter.drawEllipse(box)
        arc = box.adjusted(3, 3, -3, -3)
        painter.setPen(QPen(QColor(theme.BORDER_SOFT), 3.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawArc(arc, int(_START * 16), int(-_SPAN * 16))
        painter.setPen(QPen(QColor(theme.CYAN), 3.0))
        painter.drawArc(arc, int(_START * 16), int(-_SPAN * self._norm() * 16))
        # Pointer.
        painter.setPen(QPen(QColor(theme.FG_BRIGHT), 1.6))
        ang = math.radians(_START - _SPAN * self._norm())
        cx, cy = box.center().x(), box.center().y()
        r = box.width() * 0.32
        painter.drawLine(QPoint(int(cx), int(cy)),
                         QPoint(int(cx + r * math.cos(ang)), int(cy - r * math.sin(ang))))
        painter.setBrush(QColor(theme.CYAN))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPoint(int(cx), int(cy)), 2, 2)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag_y = event.pos().y()
            self._drag_val = self._value
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_y is None:
            super().mouseMoveEvent(event)
            return
        spec = self._spec
        span = spec.maximum - spec.minimum
        delta = (self._drag_y - event.pos().y()) / 90.0 * span
        if getattr(spec, "invert", False):
            delta = -delta
        self._apply(self._drag_val + delta)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_y = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._apply(float(self._spec.default))
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        spec = self._spec
        step = max(10 ** (-spec.decimals), (spec.maximum - spec.minimum) / 80.0)
        if getattr(spec, "invert", False):
            step = -step
        self._apply(self._value + (step if event.angleDelta().y() > 0 else -step))
        event.accept()

    def _apply(self, value: float) -> None:
        before = self._value
        self.set_value(value)
        if self._value != before:
            self.changed.emit(self._value)
