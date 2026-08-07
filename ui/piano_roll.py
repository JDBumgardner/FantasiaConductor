"""Piano-roll / drum-roll editor for a MIDI clip.

Double-clicking a MIDI clip opens this in a bottom dock. Vertical axis is pitch
(chromatic keyboard) or, for a drum track, a fixed set of **named GM drum lanes**
(the "drum roll"). Horizontal axis is time within the clip.

* click empty grid → add a note/hit
* drag a note → move; drag its right edge → change length
* select + Delete/Backspace → remove

Every edit emits ``notes_changed(clip_id, notes)``; the window routes it through
the CommandBus (undoable) and re-renders the clip's audio.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from fantasia_core.document.model import Note
from ui import theme

NOTE_H = 12          # px per semitone (chromatic)
DRUM_ROW_H = 22      # px per drum lane
KEY_W = 92           # left gutter (wider for drum names)
PR_PPS = 160.0
PITCH_LO = 21        # A0
PITCH_HI = 108       # C8
SUBDIV = 4           # snap = beat / SUBDIV
RESIZE_EDGE = 9
MIN_NOTE = 0.05
PR_RULER_H = 22      # bar/beat ruler pinned to the top of the roll
_BLACK_KEYS = {1, 3, 6, 8, 10}
_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(pitch: int) -> str:
    """MIDI number → scientific pitch name (60 → C4)."""
    return f"{_NAMES[int(pitch) % 12]}{int(pitch) // 12 - 1}"

# Drum lanes, top (cymbals) → bottom (kick). (GM note, label)
DRUM_LANES = [
    (49, "Crash"),
    (51, "Ride"),
    (46, "Open Hat"),
    (42, "Closed Hat"),
    (39, "Clap"),
    (38, "Snare"),
    (45, "Tom"),
    (36, "Kick"),
]


class NoteItem(QGraphicsRectItem):
    def __init__(self, note: Note, view: "PianoRollView") -> None:
        super().__init__()
        self.note = note
        self._view = view
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self._mode: Optional[str] = None
        self._grab_dx = 0.0
        self.refresh()

    def refresh(self) -> None:
        v = self._view
        self.setPos(v.time_to_x(self.note.start), v.pitch_to_y(self.note.pitch))
        self.setRect(0.0, 1.0, max(self.note.duration * v.pps, 3.0), v.row_h() - 2.0)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: N802
        painter.setRenderHint(QPainter.Antialiasing, True)
        c = QColor(theme.NOTE_FILL)
        c.setAlpha(110 + int(self.note.velocity / 127.0 * 145))  # brighter = louder
        painter.setBrush(QBrush(c))
        painter.setPen(QPen(theme.NOTE_SELECTED, 1.5) if self.isSelected()
                       else QPen(theme.NOTE_BORDER, 1))
        r = self.rect()
        painter.drawRoundedRect(r, 2, 2)
        # Label the note so you can read the pitch straight off the block.
        if not self._view.drum_mode and r.width() >= 26 and r.height() >= 9:
            painter.setPen(QPen(QColor(theme.FG_BRIGHT)))
            painter.setFont(QFont("", 7))
            painter.drawText(r.adjusted(3, 0, -2, 0), Qt.AlignVCenter | Qt.AlignLeft,
                             note_name(self.note.pitch))

    def hoverMoveEvent(self, event) -> None:  # noqa: N802
        near = event.pos().x() >= self.rect().width() - RESIZE_EDGE
        self.setCursor(Qt.SizeHorCursor if near else Qt.OpenHandCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.RightButton:  # right-click erases the note
            self._view.remove_note(self)
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.AltModifier:  # Alt-drag = velocity
                self._mode = "velocity"
                self._vel_y = event.scenePos().y()
                self._vel_start = self.note.velocity
            elif event.pos().x() >= self.rect().width() - RESIZE_EDGE:
                self._mode = "resize"
            else:
                self._mode = "move"
                self._grab_dx = event.scenePos().x() - self.pos().x()
                self._last_preview = self.note.pitch
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        v = self._view
        if self._mode == "velocity":
            dy = event.scenePos().y() - self._vel_y  # drag up (negative) → louder
            self.note.velocity = int(max(1, min(127, self._vel_start - dy)))
            self.update()
            event.accept()
            return
        if self._mode == "move":
            x = max(KEY_W, event.scenePos().x() - self._grab_dx)
            row = round(event.scenePos().y() / v.row_h())
            self.setPos(x, row * v.row_h())
            pitch = v.y_to_pitch(self.pos().y())  # audition the pitch as it changes
            if pitch != getattr(self, "_last_preview", None):
                self._last_preview = pitch
                v.preview.emit(pitch)
            v.info.emit(v.describe(pitch, max(0.0, v.snap_time(v.x_to_time(self.pos().x())))))
            event.accept()
            return
        if self._mode == "resize":
            w = max(MIN_NOTE * v.pps, event.scenePos().x() - self.pos().x())
            r = self.rect()
            self.setRect(0.0, r.y(), w, r.height())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._mode is not None:
            self._mode = None
            v = self._view
            self.note.start = max(0.0, v.snap_time(v.x_to_time(self.pos().x())))
            self.note.duration = max(MIN_NOTE, self.rect().width() / v.pps)
            self.note.pitch = v.y_to_pitch(self.pos().y())
            self.refresh()
            v.commit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class PianoRollView(QGraphicsView):
    notes_changed = Signal(str, list)  # (clip_id, list[Note])
    copy_requested = Signal()
    paste_requested = Signal()
    preview = Signal(int)  # pitch to audition
    info = Signal(str)     # live "C#4 · bar 2 beat 3.5" readout

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self._scene = QGraphicsScene()
        super().__init__(self._scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(QColor(theme.TIMELINE_BG))
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setFocusPolicy(Qt.StrongFocus)
        # Full repaint on scroll — the blit default ghosts the pinned key gutter.
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        self.clip_id: Optional[str] = None
        self.duration = 0.0
        self.pps = PR_PPS
        self.spb = 0.5
        self.bpb = 4
        self.drum_mode = False
        self._lane_pitch: List[int] = []       # row -> pitch (drum mode)
        self._pitch_row: Dict[int, int] = {}   # pitch -> row (drum mode)
        self._items: List[NoteItem] = []
        self._drawing: Optional[NoteItem] = None  # note being drawn (click-drag)
        self._draw_anchor = 0.0
        self._note_len_beats = 1.0  # default draw length (1 beat = 1/4 note); set by the picker
        self._snap_beats = 0.25     # snap grid in beats (1/16); None = off
        self._peaks = None          # (min,max) envelope of the clip's rendered audio
        self.clip_bar = 1           # song bar this clip starts on (for the ruler)
        # Shift-drag on empty space rubber-band-selects notes; plain drag draws.
        self.setDragMode(QGraphicsView.RubberBandDrag)

    def set_note_value(self, beats: float) -> None:
        self._note_len_beats = max(0.0625, float(beats))

    # ---- axis mapping ----------------------------------------------------
    def row_h(self) -> float:
        return DRUM_ROW_H if self.drum_mode else NOTE_H

    def _nearest_row(self, pitch: int) -> int:
        best, bd = 0, 999
        for i, lp in enumerate(self._lane_pitch):
            d = abs(lp - pitch)
            if d < bd:
                bd, best = d, i
        return best

    def pitch_to_y(self, p: int) -> float:
        if self.drum_mode:
            row = self._pitch_row.get(p, self._nearest_row(p))
        else:
            row = PITCH_HI - p
        return PR_RULER_H + row * self.row_h()

    def y_to_pitch(self, y: float) -> int:
        row = int(round((y - PR_RULER_H) / self.row_h()))
        if self.drum_mode:
            row = max(0, min(len(self._lane_pitch) - 1, row))
            return self._lane_pitch[row]
        return int(max(PITCH_LO, min(PITCH_HI, PITCH_HI - row)))

    def time_to_x(self, t: float) -> float:
        return KEY_W + t * self.pps

    def x_to_time(self, x: float) -> float:
        return (x - KEY_W) / self.pps

    def set_snap(self, beats) -> None:
        """Set snap resolution in beats (None = off)."""
        self._snap_beats = beats

    def snap_time(self, t: float) -> float:
        beats = getattr(self, "_snap_beats", 0.25)
        if beats is None or self.spb <= 0:
            return t
        grid = beats * self.spb
        return round(t / grid) * grid if grid > 0 else t

    def bar_beat(self, t: float) -> tuple:
        """Clip-relative seconds → (bar, beat) in song terms, both 1-based."""
        if self.spb <= 0:
            return (self.clip_bar, 1.0)
        total = t / self.spb                      # beats from the clip start
        bar = self.clip_bar + int(total // self.bpb)
        beat = total % self.bpb + 1.0
        return (bar, beat)

    def describe(self, pitch: int, t: float) -> str:
        bar, beat = self.bar_beat(t)
        label = note_name(pitch)
        if self.drum_mode:
            label = dict(DRUM_LANES).get(pitch, label)
        return f"{label} · bar {bar} beat {beat:g}"

    def set_waveform(self, samples, sr: int) -> None:
        """Store a min/max envelope of the clip's rendered audio to paint behind
        the notes (see the reference: waveform superimposed on the roll)."""
        try:
            import numpy as np

            if samples is None or len(samples) == 0:
                self._peaks = None
            else:
                mono = samples if samples.ndim == 1 else samples.mean(axis=1)
                buckets = 1400
                step = max(1, len(mono) // buckets)
                trimmed = mono[: step * (len(mono) // step)].reshape(-1, step)
                self._peaks = (trimmed.min(axis=1), trimmed.max(axis=1))
        except Exception:  # noqa: BLE001
            self._peaks = None
        self.viewport().update()

    def set_pps(self, pps: float) -> None:
        self.pps = max(40.0, min(600.0, pps))
        for item in self._items:
            item.refresh()
        width = KEY_W + max(self.duration + 2.0, 8.0) * self.pps
        self._scene.setSceneRect(0, 0, width,
                                 PR_RULER_H + self._rows() * self.row_h())
        self.viewport().update()

    def wheelEvent(self, event) -> None:  # noqa: N802
        if event.modifiers() & Qt.ControlModifier:  # Ctrl+scroll = horizontal zoom
            self.set_pps(self.pps * (1.25 if event.angleDelta().y() > 0 else 0.8))
            event.accept()
            return
        super().wheelEvent(event)

    def _rows(self) -> int:
        return len(self._lane_pitch) if self.drum_mode else (PITCH_HI - PITCH_LO + 1)

    # ---- binding ---------------------------------------------------------
    def set_clip(self, clip, spb: float, bpb: int, drum_mode: bool = False) -> None:  # noqa: ANN001
        self.clip_id = clip.id
        self.duration = clip.duration
        self.spb = spb
        self.bpb = bpb
        self.drum_mode = drum_mode
        if drum_mode:
            self._lane_pitch = [p for p, _ in DRUM_LANES]
            self._pitch_row = {p: i for i, p in enumerate(self._lane_pitch)}
        self._rebuild(clip.notes)
        if not drum_mode:
            avg = int(sum(n.pitch for n in clip.notes) / len(clip.notes)) if clip.notes else 60
            self.centerOn(KEY_W + 20, self.pitch_to_y(avg))
        else:
            self.centerOn(KEY_W + 20, self._rows() * self.row_h() / 2)

    def reload(self, clip) -> None:  # noqa: ANN001
        if clip is None or clip.id != self.clip_id:
            return
        self.duration = clip.duration
        self._rebuild(clip.notes)

    def _rebuild(self, notes) -> None:
        self._scene.clear()
        self._items = []
        for n in notes:
            item = NoteItem(Note(n.pitch, n.start, n.duration, n.velocity), self)
            self._scene.addItem(item)
            self._items.append(item)
        width = KEY_W + max(self.duration + 2.0, 8.0) * self.pps
        height = self._rows() * self.row_h()
        self._scene.setSceneRect(0, 0, width, height)
        self.viewport().update()

    def commit(self) -> None:
        if self.clip_id is None:
            return
        notes = [
            Note(i.note.pitch, i.note.start, i.note.duration, i.note.velocity)
            for i in self._items
        ]
        self.notes_changed.emit(self.clip_id, notes)

    # ---- interaction -----------------------------------------------------
    def remove_note(self, item: NoteItem) -> None:
        if item in self._items:
            self._scene.removeItem(item)
            self._items.remove(item)
            self.commit()

    def _default_len(self) -> float:
        if self.drum_mode:
            return 0.1
        return max(MIN_NOTE, self._note_len_beats * self.spb)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        item = self.itemAt(event.position().toPoint())
        scene_pos = self.mapToScene(event.position().toPoint())
        # Left-click-drag on empty lane space draws a new note (length from the
        # drag). Hold Shift to rubber-band-select instead (falls through to super).
        if (event.button() == Qt.LeftButton and not isinstance(item, NoteItem)
                and scene_pos.x() >= KEY_W and self.clip_id
                and not (event.modifiers() & Qt.ShiftModifier)):
            start = max(0.0, self.snap_time(self.x_to_time(scene_pos.x())))
            pitch = self.y_to_pitch(scene_pos.y())
            new_item = NoteItem(Note(pitch, start, self._default_len(), 100), self)
            self._scene.addItem(new_item)
            self._items.append(new_item)
            self._scene.clearSelection()
            new_item.setSelected(True)
            self._drawing = new_item
            self._draw_anchor = start
            self.preview.emit(pitch)
            self.info.emit(self.describe(pitch, start))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drawing is not None:  # dragging out the new note's length
            scene_x = self.mapToScene(event.position().toPoint()).x()
            end = self.snap_time(self.x_to_time(scene_x))
            self._drawing.note.duration = max(end - self._draw_anchor, MIN_NOTE)
            self._drawing.refresh()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drawing is not None:
            self._drawing = None
            self.commit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            removed = False
            for item in list(self._scene.selectedItems()):
                if isinstance(item, NoteItem):
                    self._scene.removeItem(item)
                    self._items.remove(item)
                    removed = True
            if removed:
                self.commit()
                event.accept()
                return
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_C:
                self.copy_requested.emit()
                event.accept()
                return
            if event.key() == Qt.Key_V:
                self.paste_requested.emit()
                event.accept()
                return
        super().keyPressEvent(event)

    def selected_notes(self) -> List[Note]:
        """Selected notes, normalized so the earliest starts at 0."""
        sel = [i.note for i in self._items if i.isSelected()]
        if not sel:
            return []
        base = min(n.start for n in sel)
        return [Note(n.pitch, n.start - base, n.duration, n.velocity) for n in sel]

    def paste_notes(self, anchor: float, notes: List[Note]) -> None:
        if not notes or self.clip_id is None:
            return
        self._scene.clearSelection()
        for n in notes:
            item = NoteItem(Note(n.pitch, max(0.0, anchor + n.start), n.duration, n.velocity), self)
            self._scene.addItem(item)
            self._items.append(item)
            item.setSelected(True)
        self.commit()

    # ---- background: rows + grid ----------------------------------------
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        super().drawBackground(painter, rect)
        rh = self.row_h()
        if self.drum_mode:
            for i in range(len(self._lane_pitch)):
                y = PR_RULER_H + i * rh
                shade = theme.LANE_EVEN if i % 2 == 0 else theme.LANE_ODD
                painter.fillRect(QRectF(rect.left(), y, rect.width(), rh), shade)
                painter.setPen(QColor(*theme.GRID_BEAT))
                painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
        else:
            for p in range(PITCH_LO, PITCH_HI + 1):
                y = self.pitch_to_y(p)
                shade = theme.LANE_ODD if (p % 12) in _BLACK_KEYS else theme.LANE_EVEN
                painter.fillRect(QRectF(rect.left(), y, rect.width(), rh), shade)
                if p % 12 == 0:
                    painter.setPen(QColor(*theme.GRID_BAR))
                    painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))

        beat_px = self.spb * self.pps
        if beat_px > 0:
            first = max(int((rect.left() - KEY_W) // beat_px), 0)
            last = int((rect.right() - KEY_W) // beat_px) + 1
            for b in range(first, last + 1):
                x = KEY_W + b * beat_px
                painter.setPen(QColor(*(theme.GRID_BAR if b % self.bpb == 0 else theme.GRID_BEAT)))
                painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))

        endx = self.time_to_x(self.duration)
        end_pen = QColor(theme.PLAYHEAD)
        end_pen.setAlpha(120)
        painter.setPen(QPen(end_pen, 1))
        painter.drawLine(int(endx), int(rect.top()), int(endx), int(rect.bottom()))
        self._paint_waveform(painter, rect)

    def _paint_waveform(self, painter: QPainter, rect: QRectF) -> None:
        """The clip's rendered audio, superimposed faintly across the roll."""
        if self._peaks is None or self.duration <= 0:
            return
        lo, hi = self._peaks
        n = len(lo)
        if n == 0:
            return
        top = PR_RULER_H
        height = self._rows() * self.row_h()
        mid = top + height / 2
        scale = height * 0.42
        col = QColor(theme.CYAN)
        col.setAlpha(46)                     # behind the notes, never competing
        painter.setPen(QPen(col, 1))
        x0, x1 = self.time_to_x(0.0), self.time_to_x(self.duration)
        step = (x1 - x0) / n
        first = max(0, int((rect.left() - x0) / step) - 1) if step > 0 else 0
        last = min(n, int((rect.right() - x0) / step) + 2) if step > 0 else n
        for i in range(first, last):
            x = x0 + i * step
            painter.drawLine(int(x), int(mid - hi[i] * scale),
                             int(x), int(mid - lo[i] * scale))

    # ---- foreground: left gutter pinned -------------------------------
    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        view_left = self.mapToScene(0, 0).x()
        rh = self.row_h()
        painter.fillRect(QRectF(view_left, rect.top(), KEY_W, rect.height()), QColor(theme.BG_PANEL))
        painter.setFont(QFont("", 8))
        if self.drum_mode:
            for i, (_, label) in enumerate(DRUM_LANES):
                y = PR_RULER_H + i * rh
                painter.setPen(QColor(*theme.GRID_BEAT))
                painter.drawLine(int(view_left), int(y), int(view_left + KEY_W), int(y))
                painter.setPen(QColor(theme.FG))
                painter.drawText(int(view_left) + 6, int(y + rh - 6), label)
        else:
            painter.setFont(QFont("", 7))
            for p in range(PITCH_LO, PITCH_HI + 1):
                if p % 12 == 0:
                    y = self.pitch_to_y(p)
                    painter.setPen(QColor(*theme.GRID_BAR))
                    painter.drawLine(int(view_left), int(y), int(view_left + KEY_W), int(y))
                    painter.setPen(QColor(theme.CYAN))
                    painter.drawText(int(view_left) + 4, int(y) + NOTE_H - 2, f"C{p // 12 - 1}")
        painter.setPen(theme.RULER_LINE)
        painter.drawLine(
            int(view_left + KEY_W), int(rect.top()),
            int(view_left + KEY_W), int(rect.bottom()),
        )
        self._paint_ruler(painter, rect)

    def _paint_ruler(self, painter: QPainter, rect: QRectF) -> None:
        """Bar/beat ruler pinned to the top of the viewport, numbered in SONG
        bars (so what you read here matches what you tell the agent)."""
        view_top = self.mapToScene(0, 0).y()
        view_left = self.mapToScene(0, 0).x()
        painter.fillRect(QRectF(rect.left(), view_top, rect.width(), PR_RULER_H),
                         theme.RULER_BG)
        painter.setPen(theme.RULER_LINE)
        painter.drawLine(int(rect.left()), int(view_top + PR_RULER_H),
                         int(rect.right()), int(view_top + PR_RULER_H))
        beat_px = self.spb * self.pps
        if beat_px > 0:
            n_beats = int(self.duration / self.spb) + 2
            for b in range(n_beats):
                x = KEY_W + b * beat_px
                if x < rect.left() - beat_px or x > rect.right():
                    continue
                bar = self.clip_bar + b // self.bpb
                beat = b % self.bpb + 1
                downbeat = beat == 1
                painter.setPen(theme.RULER_LINE if downbeat else QColor(*theme.GRID_BEAT))
                painter.drawLine(int(x), int(view_top + (4 if downbeat else 12)),
                                 int(x), int(view_top + PR_RULER_H))
                painter.setFont(QFont("", 8 if downbeat else 7))
                painter.setPen(QColor(theme.CYAN if downbeat else theme.FG_DIM))
                painter.drawText(int(x) + 3, int(view_top + 11),
                                 f"{bar}" if downbeat else f"{bar}.{beat}")
        # keep the gutter corner clean
        painter.fillRect(QRectF(view_left, view_top, KEY_W, PR_RULER_H),
                         QColor(theme.BG_PANEL))
        painter.setPen(QColor(theme.FG_DIM))
        painter.setFont(QFont("", 7))
        painter.drawText(int(view_left) + 6, int(view_top + 14), "bar.beat")


class PianoRollPanel(QWidget):
    notes_changed = Signal(str, list)
    copy_requested = Signal()
    paste_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.view = PianoRollView()

        top = QWidget()
        top.setStyleSheet(f"background:{theme.BG_PANEL};")
        top_row = QHBoxLayout(top)
        top_row.setContentsMargins(5, 3, 8, 3)
        self._title = QLabel("  No clip — double-click a MIDI clip to edit")
        self._title.setStyleSheet(f"color:{theme.CYAN}; background:transparent;")
        note_label = QLabel("Note")
        note_label.setStyleSheet(f"color:{theme.FG_DIM}; background:transparent;")
        self.note_value = QComboBox()
        for name, beats in [("1/1", 4.0), ("1/2", 2.0), ("1/2.", 3.0), ("1/4", 1.0),
                            ("1/4.", 1.5), ("1/4T", 2.0 / 3.0), ("1/8", 0.5),
                            ("1/8.", 0.75), ("1/8T", 1.0 / 3.0), ("1/16", 0.25),
                            ("1/16T", 1.0 / 6.0), ("1/32", 0.125)]:
            self.note_value.addItem(name, beats)
        self.note_value.setCurrentText("1/4")
        self.note_value.setToolTip("Default length of new notes ( . = dotted, T = triplet )")
        self.note_value.currentIndexChanged.connect(
            lambda _=0: self.view.set_note_value(self.note_value.currentData()))

        snap_label = QLabel("Snap")
        snap_label.setStyleSheet(f"color:{theme.FG_DIM}; background:transparent;")
        self.snap_value = QComboBox()
        for name, beats in [("1/4", 1.0), ("1/8", 0.5), ("1/16", 0.25), ("1/32", 0.125),
                            ("Off", None)]:
            self.snap_value.addItem(name, beats)
        self.snap_value.setCurrentText("1/16")
        self.snap_value.setToolTip("Grid that note positions snap to")
        self.snap_value.currentIndexChanged.connect(
            lambda _=0: self.view.set_snap(self.snap_value.currentData()))

        self.readout = QLabel("")
        self.readout.setMinimumWidth(150)
        self.readout.setStyleSheet(
            f"color:{theme.NOTE_BORDER.name()}; background:transparent; font-weight:600;")
        top_row.addWidget(self._title, 1)
        top_row.addWidget(self.readout)
        top_row.addWidget(note_label)
        top_row.addWidget(self.note_value)
        top_row.addWidget(snap_label)
        top_row.addWidget(self.snap_value)

        layout.addWidget(top)
        layout.addWidget(self.view, 1)
        self.view.notes_changed.connect(self.notes_changed.emit)
        self.view.copy_requested.connect(self.copy_requested.emit)
        self.view.paste_requested.connect(self.paste_requested.emit)
        self.view.info.connect(self.readout.setText)

    def edit_clip(self, clip, spb: float, bpb: int, drum_mode: bool = False,
                  clip_bar: int = 1) -> None:  # noqa: ANN001
        self.view.clip_bar = clip_bar        # ruler numbers in song bars
        self.view.set_clip(clip, spb, bpb, drum_mode)
        self.refresh_title(clip, drum_mode)

    def reload(self, clip) -> None:  # noqa: ANN001
        self.view.reload(clip)
        self.refresh_title(clip, self.view.drum_mode)

    _HINT = ("drag = draw · edge = length · right-click = delete · "
             "Alt-drag = velocity · Shift-drag = select")

    def refresh_title(self, clip, drum_mode: bool = False) -> None:  # noqa: ANN001
        if clip is not None and clip.id == self.view.clip_id:
            kind = "Drum Roll" if drum_mode else "Piano Roll"
            self._title.setText(
                f"  {kind} — {clip.name}  ({len(clip.notes)} notes)     {self._HINT}")
