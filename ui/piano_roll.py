"""Piano-roll / drum-roll editor — Ableton Live-style MIDI editing.

Pointer mode (default)
  * double-click empty cell → create a note
  * double-click a note → delete it
  * click-drag empty → rubber-band select (Shift adds to selection)
  * drag notes to move; Shift constrains to one axis; Ctrl duplicates
  * drag a note's right edge to change length (all selected)

Draw mode (B)
  * click-drag empty → draw a note
  * right-click a note → erase

Hotkeys (when the roll has focus) follow Live: arrows, Shift+arrows,
Ctrl+A/C/X/V/D/I/L/U, Ctrl+1–5 grid, Ctrl+Up/Down velocity, F fold, +/− zoom.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QEvent, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fantasia_core.document.model import Note
from ui import midi_ops
from ui import theme

NOTE_H = 14
DRUM_ROW_H = 22
KEY_W = 92
PR_PPS = 160.0
PITCH_LO = 21
PITCH_HI = 108
RESIZE_EDGE = 9
MIN_NOTE = midi_ops.MIN_NOTE
PR_RULER_H = 22      # bar/beat ruler pinned to the top of the roll
_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(pitch: int) -> str:
    """MIDI number → scientific pitch name (60 → C4)."""
    return f"{_NAMES[int(pitch) % 12]}{int(pitch) // 12 - 1}"
_BLACK_KEYS = {1, 3, 6, 8, 10}

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

_GRID_BEATS = [
    ("1/32", 0.125),
    ("1/16", 0.25),
    ("1/8", 0.5),
    ("1/4", 1.0),
    ("1/2", 2.0),
]


class NoteItem(QGraphicsRectItem):
    def __init__(self, note: Note, view: "PianoRollView") -> None:
        super().__init__()
        self.note = note
        self._view = view
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self._mode: Optional[str] = None
        self.refresh()

    def refresh(self) -> None:
        v = self._view
        self.setPos(v.time_to_x(self.note.start), v.pitch_to_y(self.note.pitch))
        self.setRect(0.0, 1.0, max(self.note.duration * v.pps, 3.0), v.row_h() - 2.0)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: N802
        painter.setRenderHint(QPainter.Antialiasing, True)
        c = QColor(theme.NOTE_FILL)
        c.setAlpha(110 + int(self.note.velocity / 127.0 * 145))
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
        x = event.pos().x()
        near = x <= RESIZE_EDGE or x >= self.rect().width() - RESIZE_EDGE
        self.setCursor(Qt.SizeHorCursor if near else Qt.OpenHandCursor)
        super().hoverMoveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._view.remove_note(self)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.RightButton:
            self._view.remove_note(self)
            event.accept()
            return
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        mods = event.modifiers()
        if mods & Qt.ShiftModifier:
            self.setSelected(not self.isSelected())
            event.accept()
            return
        if not self.isSelected():
            if not (mods & Qt.ControlModifier):
                self._view._scene.clearSelection()
            self.setSelected(True)
        if mods & Qt.AltModifier:
            self._mode = "velocity"
        elif event.pos().x() <= RESIZE_EDGE:
            self._mode = "resize_left"
        elif event.pos().x() >= self.rect().width() - RESIZE_EDGE:
            self._mode = "resize"
        else:
            self._mode = "move"
        self._view.begin_note_drag(
            self, self._mode, event.scenePos(),
            duplicate=bool(mods & Qt.ControlModifier) and self._mode == "move",
        )
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._mode:
            self._view.update_note_drag(event.scenePos(), event.modifiers())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._mode:
            self._mode = None
            self._view.end_note_drag()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class PianoRollView(QGraphicsView):
    notes_changed = Signal(str, list)
    copy_requested = Signal()
    cut_requested = Signal()
    paste_requested = Signal()
    split_requested = Signal()
    preview = Signal(int)
    status = Signal(str)   # transient action feedback ("Split 3 notes")
    info = Signal(str)     # live "C#4 · bar 2 beat 3.5" readout while editing

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self._scene = QGraphicsScene()
        super().__init__(self._scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(QColor(theme.TIMELINE_BG))
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setDragMode(QGraphicsView.RubberBandDrag)

        self.clip_id: Optional[str] = None
        self.duration = 0.0
        self.pps = PR_PPS
        self.spb = 0.5
        self.bpb = 4
        self.drum_mode = False
        self.draw_mode = False
        self.fold = False
        self._row_scale = 1.0
        self._lane_pitch: List[int] = []
        self._pitch_row: Dict[int, int] = {}
        self._items: List[NoteItem] = []
        self._drawing: Optional[NoteItem] = None
        self._draw_anchor = 0.0
        self._just_drew = False
        self._note_len_beats = 1.0
        self._peaks = None          # (min,max) envelope of the clip's rendered audio
        self.clip_bar = 1           # song bar this clip starts on (for the ruler)
        self._snap_beats: Optional[float] = 0.25
        self._drag_mode: Optional[str] = None
        self._drag_origin = None
        self._drag_snapshot: List[Tuple[NoteItem, float, int, float, int]] = []
        self._dup_pending = False
        self._drum_labels: Dict[int, str] = {}

    # ---- axis mapping ----------------------------------------------------
    def row_h(self) -> float:
        base = DRUM_ROW_H if self.drum_mode else NOTE_H
        return max(8.0, base * self._row_scale)

    def set_row_scale(self, scale: float) -> None:
        self._row_scale = max(0.55, min(2.4, scale))
        for item in self._items:
            item.refresh()
        self._update_scene_rect()
        self.viewport().update()

    def _nearest_row(self, pitch: int) -> int:
        best, bd = 0, 999
        for i, lp in enumerate(self._lane_pitch):
            d = abs(lp - pitch)
            if d < bd:
                bd, best = d, i
        return best

    def pitch_to_y(self, p: int) -> float:
        if self.drum_mode or self.fold:
            row = self._pitch_row.get(p, self._nearest_row(p) if self._lane_pitch else 0)
        else:
            row = PITCH_HI - p
        return PR_RULER_H + row * self.row_h()

    def y_to_pitch(self, y: float) -> int:
        row = int((y - PR_RULER_H) / self.row_h())
        if self.drum_mode or self.fold:
            if not self._lane_pitch:
                return 60
            row = max(0, min(len(self._lane_pitch) - 1, row))
            return self._lane_pitch[row]
        return int(max(PITCH_LO, min(PITCH_HI, PITCH_HI - row)))

    def time_to_x(self, t: float) -> float:
        return KEY_W + t * self.pps

    def x_to_time(self, x: float) -> float:
        return (x - KEY_W) / self.pps

    def set_note_value(self, beats: float) -> None:
        self._note_len_beats = max(0.0625, float(beats))

    def set_snap(self, beats) -> None:
        self._snap_beats = beats
        self.viewport().update()

    def grid_seconds(self) -> Optional[float]:
        if self._snap_beats is None or self.spb <= 0:
            return None
        return self._snap_beats * self.spb

    def snap_time(self, t: float) -> float:
        grid = self.grid_seconds()
        if grid is None or grid <= 0:
            return t
        return round(t / grid) * grid

    def set_pps(self, pps: float) -> None:
        self.pps = max(40.0, min(600.0, pps))
        for item in self._items:
            item.refresh()
        self._update_scene_rect()
        self.viewport().update()

    def set_draw_mode(self, on: bool) -> None:
        self.draw_mode = bool(on)
        self.setDragMode(QGraphicsView.NoDrag if self.draw_mode else QGraphicsView.RubberBandDrag)
        self.status.emit("Draw mode" if self.draw_mode else "Pointer mode")

    def set_fold(self, on: bool) -> None:
        self.fold = bool(on)
        self._apply_lanes()
        for item in self._items:
            item.refresh()
        self._update_scene_rect()
        self.viewport().update()
        self.status.emit("Fold on" if self.fold else "Fold off")

    def wheelEvent(self, event) -> None:  # noqa: N802
        mods = event.modifiers()
        dy = event.angleDelta().y()
        if mods & Qt.AltModifier:
            self.set_row_scale(self._row_scale * (1.12 if dy > 0 else 0.9))
            event.accept()
            return
        if mods & Qt.ControlModifier:
            self.set_pps(self.pps * (1.25 if dy > 0 else 0.8))
            event.accept()
            return
        if mods & Qt.ShiftModifier:
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() - dy)
            event.accept()
            return
        super().wheelEvent(event)

    def _rows(self) -> int:
        if self.drum_mode or self.fold:
            return max(len(self._lane_pitch), 1)
        return PITCH_HI - PITCH_LO + 1

    def _update_scene_rect(self) -> None:
        width = KEY_W + max(self.duration + 2.0, 8.0) * self.pps
        self._scene.setSceneRect(0, 0, width,
                                 PR_RULER_H + self._rows() * self.row_h())



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

    def _apply_lanes(self) -> None:
        if self.drum_mode:
            used = {n.note.pitch for n in self._items} if self.fold else None
            lanes = [(p, name) for p, name in DRUM_LANES if used is None or p in used]
            if not lanes:
                lanes = list(DRUM_LANES)
            self._lane_pitch = [p for p, _ in lanes]
            self._pitch_row = {p: i for i, p in enumerate(self._lane_pitch)}
            self._drum_labels = {p: name for p, name in lanes}
            return
        self._drum_labels = {}
        if self.fold:
            pitches = midi_ops.fold_rows([i.note.pitch for i in self._items], pad=1,
                                         lo=PITCH_LO, hi=PITCH_HI)
            self._lane_pitch = pitches
            self._pitch_row = {p: i for i, p in enumerate(pitches)}
        else:
            self._lane_pitch = []
            self._pitch_row = {}

    # ---- binding ---------------------------------------------------------
    def set_clip(self, clip, spb: float, bpb: int, drum_mode: bool = False) -> None:  # noqa: ANN001
        self.clip_id = clip.id
        self.duration = clip.duration
        self.spb = spb
        self.bpb = bpb
        self.drum_mode = drum_mode
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
        selected = {(i.note.pitch, round(i.note.start, 4)) for i in self._items if i.isSelected()}
        self._rebuild(clip.notes)
        for item in self._items:
            if (item.note.pitch, round(item.note.start, 4)) in selected:
                item.setSelected(True)

    def _rebuild(self, notes) -> None:
        self._scene.clear()
        self._items = []
        self._drawing = None
        for n in notes:
            item = NoteItem(Note(n.pitch, n.start, n.duration, n.velocity), self)
            self._scene.addItem(item)
            self._items.append(item)
        self._apply_lanes()
        for item in self._items:
            item.refresh()
        self._update_scene_rect()
        self.viewport().update()

    def commit(self) -> None:
        if self.clip_id is None:
            return
        notes = [
            Note(i.note.pitch, i.note.start, i.note.duration, i.note.velocity)
            for i in self._items
        ]
        self.notes_changed.emit(self.clip_id, notes)

    def selected_items(self) -> List[NoteItem]:
        return [i for i in self._items if i.isSelected()]

    def selected_note_objs(self) -> List[Note]:
        return [i.note for i in self.selected_items()]

    def remove_note(self, item: NoteItem) -> None:
        if item in self._items:
            self._scene.removeItem(item)
            self._items.remove(item)
            self.commit()

    def delete_selected(self) -> None:
        items = list(self.selected_items())
        if not items:
            return
        for item in items:
            self._scene.removeItem(item)
            self._items.remove(item)
        self.commit()

    def split_notes_at(self, at: float) -> None:
        targets = self.selected_items() or self._items
        created = midi_ops.split_at([i.note for i in targets], at)
        for n in created:
            item = NoteItem(n, self)
            self._scene.addItem(item)
            self._items.append(item)
            item.setSelected(True)
        for item in self._items:
            item.refresh()
        if created:
            self.commit()
            self.status.emit(f"Split {len(created)} note(s)")

    def _default_len(self) -> float:
        if self.drum_mode:
            return 0.1
        return max(MIN_NOTE, self._note_len_beats * self.spb)

    def _insert_note_at(self, scene_pos, length: Optional[float] = None) -> NoteItem:
        start = max(0.0, self.snap_time(self.x_to_time(scene_pos.x())))
        pitch = self.y_to_pitch(scene_pos.y())
        item = NoteItem(Note(pitch, start, length or self._default_len(), 100), self)
        self._scene.addItem(item)
        self._items.append(item)
        self._scene.clearSelection()
        item.setSelected(True)
        self.preview.emit(pitch)
        self.info.emit(self.describe(pitch, start))
        return item

    # ---- note drag (move / resize / velocity of the selection) -----------
    def begin_note_drag(self, source: NoteItem, mode: str, scene_pos, duplicate: bool = False) -> None:  # noqa: ANN001
        self._drag_mode = mode
        self._drag_origin = (scene_pos.x(), scene_pos.y())
        self._dup_pending = duplicate
        self._drag_snapshot = [
            (i, i.note.start, i.note.pitch, i.note.duration, i.note.velocity)
            for i in self.selected_items() or [source]
        ]
        self._last_preview = source.note.pitch

    def update_note_drag(self, scene_pos, modifiers) -> None:  # noqa: ANN001
        if not self._drag_mode or self._drag_origin is None:
            return
        dx = scene_pos.x() - self._drag_origin[0]
        dy = scene_pos.y() - self._drag_origin[1]
        if self._dup_pending and (abs(dx) + abs(dy) > 4):
            self._dup_pending = False
            self.duplicate_selected(select_copies=True)
            self._drag_snapshot = [
                (i, i.note.start, i.note.pitch, i.note.duration, i.note.velocity)
                for i in self.selected_items()
            ]
        if modifiers & Qt.ShiftModifier:
            dx, dy = midi_ops.constrain_delta(dx, dy)
        if self._drag_mode == "velocity":
            delta = int(-dy)
            for item, _s, _p, _d, vel in self._drag_snapshot:
                item.note.velocity = int(max(1, min(127, vel + delta)))
                item.update()
            return
        if self._drag_mode == "resize":
            dt = dx / self.pps
            for item, _s, _p, dur, _v in self._drag_snapshot:
                item.note.duration = max(MIN_NOTE, dur + dt)
                item.refresh()
            return
        if self._drag_mode == "resize_left":
            dt = dx / self.pps
            for item, start, _p, dur, _v in self._drag_snapshot:
                end = start + dur
                new_start = min(end - MIN_NOTE, max(0.0, start + dt))
                item.note.start = new_start
                item.note.duration = max(MIN_NOTE, end - new_start)
                item.refresh()
            return
        dt = dx / self.pps
        dp = -int(round(dy / self.row_h()))
        lo, hi = (min(self._lane_pitch or [0]), max(self._lane_pitch or [127])) if (
            self.drum_mode or self.fold) else (PITCH_LO, PITCH_HI)
        # Keep the whole selection in range / on the timeline.
        min_start = min(s for _i, s, _p, _d, _v in self._drag_snapshot)
        if min_start + dt < 0:
            dt = -min_start
        for item, start, pitch, _d, _v in self._drag_snapshot:
            item.note.start = max(0.0, start + dt)
            item.note.pitch = int(max(lo, min(hi, pitch + dp)))
            item.refresh()
        if self._drag_snapshot:
            lead = self._drag_snapshot[0][0].note
            new_p = lead.pitch
            if new_p != getattr(self, "_last_preview", None):
                self._last_preview = new_p
                self.preview.emit(new_p)          # audition the pitch as it changes
            self.info.emit(self.describe(new_p, lead.start))

    def end_note_drag(self) -> None:
        if not self._drag_snapshot:
            self._drag_mode = None
            self._dup_pending = False
            return
        grid = self.grid_seconds()
        changed = False
        for item, start, pitch, dur, vel in self._drag_snapshot:
            new_start = max(0.0, self.snap_time(item.note.start))
            new_dur = item.note.duration
            if self._drag_mode in ("resize", "resize_left") and grid:
                end = self.snap_time(new_start + item.note.duration)
                new_dur = max(MIN_NOTE, end - new_start)
            if (new_start != start or item.note.pitch != pitch
                    or new_dur != dur or item.note.velocity != vel):
                changed = True
            item.note.start = new_start
            item.note.duration = new_dur
            item.refresh()
        self._drag_mode = None
        self._drag_snapshot = []
        self._dup_pending = False
        if changed:
            self.commit()

    def duplicate_selected(self, select_copies: bool = False) -> None:
        sel = self.selected_note_objs()
        if not sel:
            return
        copies = midi_ops.clone_notes(sel) if select_copies else midi_ops.duplicate_after(sel)
        self._scene.clearSelection()
        new_items = []
        for n in copies:
            item = NoteItem(n, self)
            self._scene.addItem(item)
            self._items.append(item)
            item.setSelected(True)
            new_items.append(item)
        if not select_copies:
            self.commit()
            self.status.emit(f"Duplicated {len(new_items)} note(s)")

    # ---- mouse -----------------------------------------------------------
    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton or self.clip_id is None:
            super().mouseDoubleClickEvent(event)
            return
        if self._just_drew:
            self._just_drew = False
            event.accept()
            return
        item = self.itemAt(event.position().toPoint())
        if isinstance(item, NoteItem):
            self.remove_note(item)
            event.accept()
            return
        scene_pos = self.mapToScene(event.position().toPoint())
        if scene_pos.x() >= KEY_W:
            self._insert_note_at(scene_pos)
            self.commit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        item = self.itemAt(event.position().toPoint())
        scene_pos = self.mapToScene(event.position().toPoint())
        if event.button() == Qt.LeftButton and scene_pos.x() < KEY_W and self.clip_id:
            pitch = self.y_to_pitch(scene_pos.y())
            self.preview.emit(pitch)
            if not (event.modifiers() & Qt.ShiftModifier):
                self._scene.clearSelection()
            for note_item in self._items:
                if note_item.note.pitch == pitch:
                    note_item.setSelected(True)
            event.accept()
            return
        if (self.draw_mode and event.button() == Qt.LeftButton
                and not isinstance(item, NoteItem) and scene_pos.x() >= KEY_W
                and self.clip_id):
            start = max(0.0, self.snap_time(self.x_to_time(scene_pos.x())))
            item = self._insert_note_at(scene_pos)
            self._drawing = item
            self._draw_anchor = start
            self._just_drew = True
            event.accept()
            return
        if (not self.draw_mode and event.button() == Qt.LeftButton
                and not isinstance(item, NoteItem)
                and not (event.modifiers() & Qt.ShiftModifier)):
            self._scene.clearSelection()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drawing is not None:
            scene_x = self.mapToScene(event.position().toPoint()).x()
            end = self.snap_time(self.x_to_time(scene_x))
            self._drawing.note.start = min(self._draw_anchor, end)
            self._drawing.note.duration = max(abs(end - self._draw_anchor), MIN_NOTE)
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

    # Steal Live shortcuts from window-wide QActions while the roll is focused.
    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ShortcutOverride and self._handles_key(event):
            event.accept()
            return True
        return super().event(event)

    def _handles_key(self, event) -> bool:
        key = event.key()
        mods = event.modifiers()
        ctrl = bool(mods & Qt.ControlModifier)
        if key in (Qt.Key_Delete, Qt.Key_Backspace, Qt.Key_B, Qt.Key_F,
                   Qt.Key_Plus, Qt.Key_Minus, Qt.Key_Equal, Qt.Key_Escape,
                   Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
            return True
        if ctrl and key in (Qt.Key_A, Qt.Key_C, Qt.Key_X, Qt.Key_V, Qt.Key_D,
                            Qt.Key_I, Qt.Key_L, Qt.Key_U, Qt.Key_E,
                            Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5):
            return True
        return False

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        mods = event.modifiers()
        ctrl = bool(mods & Qt.ControlModifier)
        shift = bool(mods & Qt.ShiftModifier)
        sel = self.selected_note_objs()

        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
            event.accept()
            return
        if key == Qt.Key_Escape:
            if self.draw_mode:
                self.set_draw_mode(False)
            else:
                self._scene.clearSelection()
            event.accept()
            return
        if key == Qt.Key_B and not ctrl:
            self.set_draw_mode(not self.draw_mode)
            event.accept()
            return
        if key == Qt.Key_F and not ctrl:
            self.set_fold(not self.fold)
            event.accept()
            return
        if key in (Qt.Key_Plus, Qt.Key_Equal) and not ctrl:
            self.set_pps(self.pps * 1.25)
            event.accept()
            return
        if key == Qt.Key_Minus and not ctrl:
            self.set_pps(self.pps / 1.25)
            event.accept()
            return

        if ctrl:
            if key == Qt.Key_A:
                for i in self._items:
                    i.setSelected(True)
                event.accept()
                return
            if key == Qt.Key_I:
                for i in self._items:
                    i.setSelected(not i.isSelected())
                event.accept()
                return
            if key == Qt.Key_C:
                self.copy_requested.emit()
                event.accept()
                return
            if key == Qt.Key_X:
                self.cut_requested.emit()
                event.accept()
                return
            if key == Qt.Key_V:
                self.paste_requested.emit()
                event.accept()
                return
            if key == Qt.Key_D:
                self.duplicate_selected()
                event.accept()
                return
            if key == Qt.Key_E:
                self.split_requested.emit()
                event.accept()
                return
            if key == Qt.Key_L and sel:
                midi_ops.legato(sel)
                for i in self._items:
                    i.refresh()
                self.commit()
                event.accept()
                return
            if key == Qt.Key_U and sel:
                midi_ops.quantize(sel, self.grid_seconds())
                for i in self._items:
                    i.refresh()
                self.commit()
                event.accept()
                return
            if key in (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5):
                idx = key - Qt.Key_1
                _name, beats = _GRID_BEATS[idx]
                self.set_snap(beats)
                self.status.emit(f"Grid {_name}")
                event.accept()
                return
            if key == Qt.Key_Up and sel:
                midi_ops.change_velocity(sel, 8 if not shift else 1)
                for i in self._items:
                    i.update()
                self.commit()
                event.accept()
                return
            if key == Qt.Key_Down and sel:
                midi_ops.change_velocity(sel, -8 if not shift else -1)
                for i in self._items:
                    i.update()
                self.commit()
                event.accept()
                return

        if key in (Qt.Key_Up, Qt.Key_Down) and sel and not ctrl:
            semis = (12 if shift else 1) * (1 if key == Qt.Key_Up else -1)
            lo, hi = (min(self._lane_pitch or [0]), max(self._lane_pitch or [127])) if (
                self.drum_mode or self.fold) else (PITCH_LO, PITCH_HI)
            midi_ops.transpose(sel, semis, lo, hi)
            for i in self._items:
                i.refresh()
            self.commit()
            event.accept()
            return
        if key in (Qt.Key_Left, Qt.Key_Right) and sel and not ctrl:
            grid = self.grid_seconds() or (self.spb * 0.25)
            step = grid * (4 if shift else 1)
            midi_ops.nudge_time(sel, step if key == Qt.Key_Right else -step)
            for i in self.selected_items():
                if not shift:
                    i.note.start = self.snap_time(i.note.start)
                i.refresh()
            self.commit()
            event.accept()
            return
        super().keyPressEvent(event)

    def selected_notes(self) -> List[Note]:
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

    # ---- background / gutter --------------------------------------------
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        super().drawBackground(painter, rect)
        rh = self.row_h()
        if self.drum_mode or self.fold:
            pitches = self._lane_pitch
            for i, p in enumerate(pitches):
                y = PR_RULER_H + i * rh
                shade = theme.LANE_ODD if (p % 12) in _BLACK_KEYS else theme.LANE_EVEN
                if self.drum_mode:
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

        grid = self.grid_seconds()
        step = grid if grid and grid > 0 else self.spb
        if step and step > 0:
            first = max(int((rect.left() - KEY_W) / (step * self.pps)), 0)
            last = int((rect.right() - KEY_W) / (step * self.pps)) + 1
            bar = self.spb * self.bpb
            for i in range(first, last + 1):
                t = i * step
                x = KEY_W + t * self.pps
                on_bar = bar > 0 and abs((t / bar) - round(t / bar)) < 1e-4
                on_beat = self.spb > 0 and abs((t / self.spb) - round(t / self.spb)) < 1e-4
                color = theme.GRID_BAR if on_bar else (theme.GRID_BEAT if on_beat else theme.GRID_SUBDIV)
                painter.setPen(QColor(*color))
                painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))

        self._paint_waveform(painter, rect)

        endx = self.time_to_x(self.duration)
        end_pen = QColor(theme.PLAYHEAD)
        end_pen.setAlpha(120)
        painter.setPen(QPen(end_pen, 1))
        painter.drawLine(int(endx), int(rect.top()), int(endx), int(rect.bottom()))

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        self._paint_ruler(painter, rect)
        view_left = self.mapToScene(0, 0).x()
        rh = self.row_h()
        painter.fillRect(QRectF(view_left, rect.top(), KEY_W, rect.height()), QColor(theme.BG_PANEL))
        painter.setFont(QFont("", 8))
        if self.drum_mode:
            labels = getattr(self, "_drum_labels", {p: n for p, n in DRUM_LANES})
            for i, p in enumerate(self._lane_pitch):
                y = PR_RULER_H + i * rh
                painter.setPen(QColor(*theme.GRID_BEAT))
                painter.drawLine(int(view_left), int(y), int(view_left + KEY_W), int(y))
                painter.setPen(QColor(theme.FG))
                painter.drawText(int(view_left) + 6, int(y + rh - 6), labels.get(p, str(p)))
        else:
            painter.setFont(QFont("", 7))
            if self.fold:
                for i, p in enumerate(self._lane_pitch):
                    y = PR_RULER_H + i * rh
                    if p % 12 == 0:
                        painter.setPen(QColor(theme.CYAN))
                        painter.drawText(int(view_left) + 4, int(y + rh - 2), f"C{p // 12 - 1}")
                    else:
                        names = "C C# D D# E F F# G G# A A# B".split()
                        painter.setPen(QColor(theme.FG_DIM))
                        painter.drawText(int(view_left) + 4, int(y + rh - 2), names[p % 12])
            else:
                for p in range(PITCH_LO, PITCH_HI + 1):
                    if p % 12 == 0:
                        y = self.pitch_to_y(p)
                        painter.setPen(QColor(*theme.GRID_BAR))
                        painter.drawLine(int(view_left), int(y), int(view_left + KEY_W), int(y))
                        painter.setPen(QColor(theme.CYAN))
                        painter.drawText(int(view_left) + 4, int(y) + int(self.row_h()) - 2,
                                         f"C{p // 12 - 1}")
        painter.setPen(theme.RULER_LINE)
        painter.drawLine(
            int(view_left + KEY_W), int(rect.top()),
            int(view_left + KEY_W), int(rect.bottom()),
        )


class PianoRollPanel(QWidget):
    notes_changed = Signal(str, list)
    copy_requested = Signal()
    cut_requested = Signal()
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

        btn_style = (
            f"QPushButton {{ background:transparent; color:{theme.FG_DIM}; border:none;"
            f" padding:3px 8px; font-weight:600; }}"
            f"QPushButton:checked {{ color:{theme.FG_BRIGHT}; background:{theme.BG_SELECTED};"
            f" border-bottom:2px solid {theme.ACCENT}; }}"
        )
        self.btn_draw = QPushButton("Draw (B)")
        self.btn_draw.setCheckable(True)
        self.btn_draw.setToolTip("Draw mode — click-drag to create notes")
        self.btn_draw.setStyleSheet(btn_style)
        self.btn_draw.toggled.connect(self.view.set_draw_mode)
        self.btn_fold = QPushButton("Fold (F)")
        self.btn_fold.setCheckable(True)
        self.btn_fold.setToolTip("Show only pitches that have notes")
        self.btn_fold.setStyleSheet(btn_style)
        self.btn_fold.toggled.connect(self.view.set_fold)

        note_label = QLabel("Note")
        note_label.setStyleSheet(f"color:{theme.FG_DIM}; background:transparent;")
        self.note_value = QComboBox()
        for name, beats in [("1/1", 4.0), ("1/2", 2.0), ("1/2.", 3.0), ("1/4", 1.0),
                            ("1/4.", 1.5), ("1/4T", 2.0 / 3.0), ("1/8", 0.5),
                            ("1/8.", 0.75), ("1/8T", 1.0 / 3.0), ("1/16", 0.25),
                            ("1/16T", 1.0 / 6.0), ("1/32", 0.125)]:
            self.note_value.addItem(name, beats)
        self.note_value.setCurrentText("1/4")
        self.note_value.setToolTip("Default length of new notes")
        self.note_value.currentIndexChanged.connect(
            lambda _=0: self.view.set_note_value(self.note_value.currentData()))

        snap_label = QLabel("Snap")
        snap_label.setStyleSheet(f"color:{theme.FG_DIM}; background:transparent;")
        self.snap_value = QComboBox()
        for name, beats in [("1/4", 1.0), ("1/8", 0.5), ("1/16", 0.25), ("1/32", 0.125),
                            ("Off", None)]:
            self.snap_value.addItem(name, beats)
        self.snap_value.setCurrentText("1/16")
        self.snap_value.setToolTip("Grid — Ctrl+1…5 while the roll is focused")
        self.snap_value.currentIndexChanged.connect(
            lambda _=0: self.view.set_snap(self.snap_value.currentData()))

        top_row.addWidget(self._title, 1)
        top_row.addWidget(self.btn_draw)
        top_row.addWidget(self.btn_fold)
        self.readout = QLabel("")
        self.readout.setMinimumWidth(150)
        self.readout.setStyleSheet(
            f"color:{theme.NOTE_BORDER.name()}; background:transparent; font-weight:600;")
        top_row.addWidget(self.readout)
        top_row.addWidget(note_label)
        top_row.addWidget(self.note_value)
        top_row.addWidget(snap_label)
        top_row.addWidget(self.snap_value)

        layout.addWidget(top)
        layout.addWidget(self.view, 1)
        self.view.notes_changed.connect(self.notes_changed.emit)
        self.view.copy_requested.connect(self.copy_requested.emit)
        self.view.cut_requested.connect(self.cut_requested.emit)
        self.view.paste_requested.connect(self.paste_requested.emit)
        self.view.status.connect(self._on_status)
        self.view.info.connect(self.readout.setText)

    def _on_status(self, msg: str) -> None:
        # Keep Draw/Fold buttons matched when toggled from the keyboard.
        if self.btn_draw.isChecked() != self.view.draw_mode:
            blocked = self.btn_draw.blockSignals(True)
            self.btn_draw.setChecked(self.view.draw_mode)
            self.btn_draw.blockSignals(blocked)
        if self.btn_fold.isChecked() != self.view.fold:
            blocked = self.btn_fold.blockSignals(True)
            self.btn_fold.setChecked(self.view.fold)
            self.btn_fold.blockSignals(blocked)
        # Sync snap combo if the user hit Ctrl+1–5.
        beats = self.view._snap_beats
        for i in range(self.snap_value.count()):
            if self.snap_value.itemData(i) == beats:
                blocked = self.snap_value.blockSignals(True)
                self.snap_value.setCurrentIndex(i)
                self.snap_value.blockSignals(blocked)
                break

    def edit_clip(self, clip, spb: float, bpb: int, drum_mode: bool = False,
                  clip_bar: int = 1) -> None:  # noqa: ANN001
        self.view.clip_bar = clip_bar        # ruler numbers in song bars
        self.view.set_clip(clip, spb, bpb, drum_mode)
        self.refresh_title(clip, drum_mode)
        self.view.setFocus(Qt.OtherFocusReason)

    def reload(self, clip) -> None:  # noqa: ANN001
        self.view.reload(clip)
        self.refresh_title(clip, self.view.drum_mode)

    _HINT = ("dbl-click add/delete · B draw · F fold · arrows nudge · "
             "Ctrl+D duplicate · drag the split above to resize")

    def refresh_title(self, clip, drum_mode: bool = False) -> None:  # noqa: ANN001
        if clip is not None and clip.id == self.view.clip_id:
            kind = "Drum Roll" if drum_mode else "Piano Roll"
            self._title.setText(
                f"  {kind} — {clip.name}  ({len(clip.notes)} notes)     {self._HINT}")
