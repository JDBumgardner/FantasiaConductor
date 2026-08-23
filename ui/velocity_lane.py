"""Ableton-style MIDI velocity lane under the piano roll."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ui import midi_ops
from ui import theme

# Match piano_roll.KEY_W / ZOOM_STRIP_W so the gutter lines up with the keys.
ZOOM_STRIP_W = 16
KEY_W = ZOOM_STRIP_W + 76


class VelocityLane(QWidget):
    """Time-aligned velocity marks: a line per note with a handle at the start."""

    velocity_edited = Signal()
    _PAD = 6
    _HANDLE = 3.4

    def __init__(self, view, parent: Optional[QWidget] = None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self._view = view
        self._dragging = False
        self.setMinimumHeight(36)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.NoFocus)

    def _items(self):
        return getattr(self._view, "_items", [])

    def _x_for_time(self, t: float) -> float:
        return float(self._view.mapFromScene(self._view.time_to_x(t), 0).x() - KEY_W)

    def _y_for_velocity(self, velocity: int) -> float:
        h = max(1, self.height() - self._PAD)
        return 3 + (1.0 - max(1, min(127, velocity)) / 127.0) * h

    def _note_at(self, x: float, y: float):
        hit = None
        best = 1e9
        for item in self._items():
            x0 = self._x_for_time(item.note.start)
            x1 = max(x0 + 8.0, self._x_for_time(item.note.start + item.note.duration))
            y0 = self._y_for_velocity(item.note.velocity)
            if x < x0 - 8 or x > x1 + 4:
                continue
            # Prefer the left-end handle, then the stem.
            handle = (x - x0) ** 2 + (y - y0) ** 2
            stem = (y - y0) ** 2 if x0 <= x <= x1 else 1e9
            dist = min(handle, stem)
            if dist < best and dist < 14 ** 2:
                best, hit = dist, item
        return hit

    def _velocity_from_y(self, y: float) -> int:
        h = max(1, self.height() - self._PAD)
        rel = 1.0 - (y - 3) / h
        return int(max(1, min(127, round(rel * 126) + 1)))

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(theme.TIMELINE_BG))
        h = self.height()
        w = self.width()
        for frac, alpha in ((0.5, 50), (1.0, 80)):
            y = 3 + (1.0 - frac) * (h - self._PAD)
            col = QColor(*theme.GRID_BEAT)
            col.setAlpha(alpha)
            painter.setPen(QPen(col, 1))
            painter.drawLine(0, int(y), w, int(y))
        items = list(self._items())
        if not items:
            painter.end()
            return
        unselected = [i for i in items if not i.isSelected()]
        selected = [i for i in items if i.isSelected()]
        for item in unselected + selected:
            x0 = self._x_for_time(item.note.start)
            x1 = self._x_for_time(item.note.start + item.note.duration)
            if x1 < -8 or x0 > w + 8:
                continue
            y = self._y_for_velocity(item.note.velocity)
            if item.isSelected():
                color = QColor(theme.CYAN)
            else:
                color = QColor(theme.FG)
                color.setAlpha(150)
            painter.setPen(QPen(color, 1.4 if item.isSelected() else 1.1))
            painter.drawLine(QPointF(x0, y), QPointF(max(x1, x0 + 6.0), y))
            r = self._HANDLE + (0.6 if item.isSelected() else 0.0)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(x0, y), r, r)
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            return
        pos = event.position()
        item = self._note_at(pos.x(), pos.y())
        if item is None:
            return
        if not item.isSelected() and not (event.modifiers() & Qt.ShiftModifier):
            self._view._scene.clearSelection()
            item.setSelected(True)
        elif not item.isSelected():
            item.setSelected(True)
        self._dragging = True
        self._apply_y(event.position().y())
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self._dragging:
            return
        self._apply_y(event.position().y())
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if not self._dragging:
            return
        self._dragging = False
        self._view.commit()
        self.velocity_edited.emit()
        event.accept()

    def _apply_y(self, y: float) -> None:
        vel = self._velocity_from_y(y)
        targets = self._view.selected_note_objs()
        if not targets:
            return
        midi_ops.set_velocity(targets, vel)
        for item in self._view.selected_items():
            item.update()
        self.update()


class VelocitySection(QWidget):
    """Resizable velocity pane with a left-side hide control aligned to the keys."""

    hide_requested = Signal()

    def __init__(self, view, parent: Optional[QWidget] = None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.setMinimumHeight(48)
        self.setMaximumHeight(240)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        left = QWidget()
        left.setFixedWidth(KEY_W)
        left.setStyleSheet(f"background:{theme.BG_PANEL};")
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(1, 4, 2, 4)
        left_l.setSpacing(2)
        self.btn_hide = QPushButton("▾")
        self.btn_hide.setFixedWidth(ZOOM_STRIP_W + 2)
        self.btn_hide.setToolTip("Hide MIDI Velocity")
        self.btn_hide.setStyleSheet(
            f"QPushButton {{ background:{theme.BG_ELEVATED}; color:{theme.FG};"
            f" border:1px solid {theme.BORDER}; font-weight:700; padding:2px; }}"
            f"QPushButton:hover {{ color:{theme.GREEN}; border-color:{theme.GREEN}; }}"
        )
        self.btn_hide.clicked.connect(self.hide_requested.emit)
        title = QLabel("MIDI\nVelocity")
        title.setStyleSheet(f"color:{theme.FG_DIM}; background:transparent; font-size:10px;")
        title.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        left_l.addWidget(self.btn_hide, 0, Qt.AlignLeft)
        left_l.addWidget(title)
        left_l.addStretch(1)

        self.lane = VelocityLane(view)
        row.addWidget(left)
        row.addWidget(self.lane, 1)
