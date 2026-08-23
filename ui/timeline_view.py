"""The timeline canvas: ruler, track lanes, clips, playhead.

Rendering strategy
------------------
* Lanes + vertical beat/bar grid are painted in :meth:`drawBackground`.
* The ruler is painted in :meth:`drawForeground`, pinned to the top of the
  viewport so it stays visible while scrolling vertically.
* Clips are :class:`ClipItem` objects (movable + right-edge resizable). Because
  ``QGraphicsItem`` is not a ``QObject`` and can't emit signals, items call back
  into the view, which re-emits Qt signals for the rest of the app.

Coordinates: ``x = seconds * pixels_per_second``; a clip on track *i* sits at
``y = RULER_H + i * TRACK_H (+ CLIP_MARGIN)``.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QMenu,
    QWidget,
)

from fantasia_core.document.model import Clip, Project
from ui import theme
from ui.grid import GridSpec, grid_interval_seconds, seconds_per_beat, snap_time
from ui.metrics import (
    CLIP_MARGIN,
    PPS_DEFAULT,
    PPS_MAX,
    PPS_MIN,
    RESIZE_EDGE,
    RULER_H,
    TRACK_H,
)

MIN_CLIP_SECONDS = 0.05
LOOP_HANDLE_W = 7
LOOP_BAR_H = 11


def _lane_color(index: int) -> QColor:
    return theme.LANE_EVEN if index % 2 == 0 else theme.LANE_ODD


class ClipItem(QGraphicsRectItem):
    """A draggable / right-edge-resizable clip bound to a model :class:`Clip`."""

    def __init__(self, clip: Clip, view: "TimelineView") -> None:
        super().__init__()
        self.clip = clip
        self._view = view
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self._mode: Optional[str] = None  # "move" | "resize"
        self._grab_offset = 0.0  # scene-x offset within the clip when grabbed
        self.refresh_geometry()

    # ---- geometry <-> model ---------------------------------------------
    def refresh_geometry(self) -> None:
        pps = self._view.pps
        row = self._view.track_row(self.clip.id)
        y = RULER_H + row * TRACK_H + CLIP_MARGIN
        self.setPos(self.clip.start * pps, y)
        self.setRect(0.0, 0.0, max(self.clip.duration * pps, 1.0), TRACK_H - 2 * CLIP_MARGIN)

    def _commit(self, mode: str = "move") -> None:
        pps = self._view.pps
        if mode == "resize":
            # Resizing: keep the start put, snap the (dragged) right edge to grid.
            start = self.clip.start
            end = self._view.snap((self.pos().x() + self.rect().width()) / pps)
            duration = max(end - start, MIN_CLIP_SECONDS)
        else:
            # Moving: snap the start to grid, keep the length.
            start = max(0.0, self._view.snap(self.pos().x() / pps))
            duration = max(self.clip.duration, MIN_CLIP_SECONDS)
        # Skip no-op edits (e.g. a plain click / double-click) so they don't
        # push empty geometry commands onto the undo stack.
        if abs(start - self.clip.start) < 1e-6 and abs(duration - self.clip.duration) < 1e-6:
            self.refresh_geometry()
            return
        # Don't mutate the model here — emit the proposed geometry and let the
        # window route it through the CommandBus (so it's undoable). The item
        # re-aligns to the model in refresh_clip() after the command applies.
        self._view.clip_geometry_edited.emit(self.clip.id, start, duration)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self._view.clip_double_clicked.emit(self.clip.id)
        event.accept()

    # ---- painting --------------------------------------------------------
    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: N802
        rect = self.rect()
        color = QColor(self._view.track_color(self.clip.id))
        empty = not self.clip.source_path and not self.clip.is_midi
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Empty (unfilled) clips read as translucent, dashed "slots".
        body = QColor(color)
        body.setAlpha(80 if empty else 210)
        painter.setBrush(QBrush(body))
        if self.isSelected():
            painter.setPen(QPen(QColor(255, 255, 255), 2))
        elif empty:
            painter.setPen(QPen(color.lighter(140), 1, Qt.DashLine))
        else:
            painter.setPen(QPen(color.darker(160), 1))
        painter.drawRoundedRect(rect, 4, 4)

        pool = getattr(self._view, "audio_pool", None)
        if self.clip.is_midi:
            self._paint_midi(painter, rect)
        elif empty:
            painter.setPen(QColor(220, 220, 220, 120))
            painter.setFont(QFont("", 8))
            painter.drawText(rect, Qt.AlignCenter, "empty · right-click to fill")
        elif pool is not None:
            self._paint_waveform(painter, rect, pool)

        self._paint_fades(painter, rect)

        painter.setPen(QColor(245, 245, 245))
        painter.setFont(QFont("", 9))
        text_rect = rect.adjusted(6, 2, -6, -2)
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignTop, self.clip.name)

    def _paint_midi(self, painter: QPainter, rect) -> None:
        notes = self.clip.notes
        if not notes:
            painter.setPen(QColor(230, 230, 230, 140))
            painter.setFont(QFont("", 8))
            painter.drawText(rect, Qt.AlignCenter, "MIDI · empty")
            return
        pitches = [n.pitch for n in notes]
        lo, hi = min(pitches), max(pitches)
        span = max(hi - lo, 1)
        dur = self.clip.duration or 1.0
        pad = 4
        h = rect.height() - 2 * pad
        painter.setPen(Qt.NoPen)
        painter.setBrush(theme.MIDI_PREVIEW)
        for note in notes:
            x = rect.left() + (note.start / dur) * rect.width()
            w = max((note.duration / dur) * rect.width(), 2)
            y = rect.top() + pad + (1.0 - (note.pitch - lo) / span) * h
            painter.drawRoundedRect(QRectF(x, y - 1.5, w, 3.0), 1, 1)

    def _paint_fades(self, painter: QPainter, rect) -> None:
        if self.clip.fade_in <= 0 and self.clip.fade_out <= 0:
            return
        pps = self._view.pps
        painter.setPen(QPen(QColor(255, 255, 255, 160), 1))
        if self.clip.fade_in > 0:
            fw = min(self.clip.fade_in * pps, rect.width())
            painter.drawLine(
                int(rect.left()), int(rect.bottom()),
                int(rect.left() + fw), int(rect.top()),
            )
        if self.clip.fade_out > 0:
            fw = min(self.clip.fade_out * pps, rect.width())
            painter.drawLine(
                int(rect.right() - fw), int(rect.top()),
                int(rect.right()), int(rect.bottom()),
            )

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        menu = QMenu()
        imp = menu.addAction(
            "Replace Audio…" if self.clip.source_path else "Import Audio into Clip…"
        )
        write_midi = None
        if not self.clip.source_path and not self.clip.is_midi:
            write_midi = menu.addAction("Write MIDI")
        transcribe = None
        separate = None
        vfx_map = {}
        hum = None
        if self.clip.source_path and not self.clip.is_midi:
            hum = menu.addAction("Hum → Melody  (monophonic)")
            transcribe = menu.addAction("Transcribe to MIDI  (polyphonic)")
            separate = menu.addAction("Separate Stems (Demucs)")
            vfx = menu.addMenu("Vocal FX")
            vfx_map = {
                vfx.addAction("Auto-Tune…"): "vfx_autotune",
                vfx.addAction("Harmonize +3rd (new track)"): "vfx_harmony3",
                vfx.addAction("Harmonize +5th (new track)"): "vfx_harmony5",
                vfx.addAction("De-ess"): "vfx_deess",
                vfx.addAction("Double"): "vfx_double",
                vfx.addAction("Formant Up (brighter)"): "vfx_formant_up",
                vfx.addAction("Formant Down (darker)"): "vfx_formant_down",
            }
            st = menu.addMenu("Time Stretch")
            vfx_map.update({
                st.addAction("Half Speed (2×)"): "stretch_2",
                st.addAction("Double Speed (0.5×)"): "stretch_0.5",
                st.addAction("Stretch by factor…"): "stretch_custom",
                st.addAction("Fit to Bars…"): "stretch_bars",
            })
            lock = menu.addAction("Lock to Tempo")
            lock.setCheckable(True)
            lock.setChecked(self.clip.lock_tempo is not None)
            lock.setToolTip("Re-stretch this clip to stay in time when the tempo changes")
            vfx_map[lock] = "tempo_lock"
        generate = menu.addAction("Generate Audio…")
        speak = menu.addAction("Text to Speech…")
        sing = menu.addAction("Sing Lyrics…") if self.clip.is_midi else None
        menu.addSeparator()
        act_split = menu.addAction("Split at Playhead")
        act_rev = menu.addAction("Reverse")
        act_rev.setCheckable(True)
        act_rev.setChecked(self.clip.reversed)
        act_norm = menu.addAction("Normalize")
        menu.addSeparator()
        act_fin = menu.addAction("Fade In 0.25s")
        act_fout = menu.addAction("Fade Out 0.25s")
        act_clear = menu.addAction("Clear Fades")
        menu.addSeparator()
        act_gup = menu.addAction("Gain +3 dB")
        act_gdn = menu.addAction("Gain −3 dB")
        menu.addSeparator()
        act_pu = menu.addAction("Pitch +1 semitone")
        act_pd = menu.addAction("Pitch −1 semitone")
        act_pou = menu.addAction("Pitch +12 (octave up)")
        act_pod = menu.addAction("Pitch −12 (octave down)")
        act_preset = menu.addAction("Reset Pitch")

        actions = {
            imp: "import",
            act_split: "split",
            act_rev: "reverse",
            act_norm: "normalize",
            act_fin: "fade_in",
            act_fout: "fade_out",
            act_clear: "clear_fades",
            act_gup: "gain_up",
            act_gdn: "gain_down",
            act_pu: "pitch_up",
            act_pd: "pitch_down",
            act_pou: "pitch_oct_up",
            act_pod: "pitch_oct_down",
            act_preset: "pitch_reset",
        }
        if write_midi is not None:
            actions[write_midi] = "write_midi"
        if transcribe is not None:
            actions[transcribe] = "transcribe"
        if hum is not None:
            actions[hum] = "hum"
        if separate is not None:
            actions[separate] = "separate"
        actions[generate] = "generate"
        actions[speak] = "tts"
        if sing is not None:
            actions[sing] = "sing"
        actions.update(vfx_map)
        name = actions.get(menu.exec(event.screenPos()))
        if name == "import":
            self._view.import_into_clip_requested.emit(self.clip.id)
        elif name is not None:
            self._view.clip_action_requested.emit(self.clip.id, name)
        event.accept()

    def _paint_waveform(self, painter: QPainter, rect, pool) -> None:
        buckets = max(1, min(int(rect.width()), 2000))
        from fantasia_core.document.tempo import source_span

        mins, maxs = pool.peaks(
            self.clip.source_path, self.clip.source_offset, source_span(self.clip), buckets
        )
        n = len(mins)
        if n == 0:
            return
        # Reflect the edits that change the sound: reverse flips the time axis,
        # clip gain scales the amplitude. This gives visual feedback for the
        # right-click operations.
        if self.clip.reversed:
            mins = mins[::-1]
            maxs = maxs[::-1]
        gain = 10.0 ** (self.clip.gain_db / 20.0)

        cy = rect.center().y()
        half = rect.height() / 2 - 3
        step = rect.width() / n

        def _y(sample: float) -> int:
            v = max(-1.0, min(1.0, sample * gain))
            return int(cy - v * half)

        painter.setPen(QPen(QColor(255, 255, 255, 120), 1))
        for i in range(n):
            x = int(rect.left() + i * step)
            painter.drawLine(x, _y(float(maxs[i])), x, _y(float(mins[i])))

    # ---- interaction -----------------------------------------------------
    def hoverMoveEvent(self, event) -> None:  # noqa: N802
        if event.pos().x() >= self.rect().width() - RESIZE_EDGE:
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.OpenHandCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            if event.pos().x() >= self.rect().width() - RESIZE_EDGE:
                self._mode = "resize"
            else:
                self._mode = "move"
                self._grab_offset = event.scenePos().x() - self.pos().x()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._mode == "move":
            new_x = max(0.0, event.scenePos().x() - self._grab_offset)
            self.setPos(new_x, self.pos().y())
            event.accept()
            return
        if self._mode == "resize":
            new_w = max(MIN_CLIP_SECONDS * self._view.pps,
                        event.scenePos().x() - self.pos().x())
            r = self.rect()
            self.setRect(0.0, 0.0, new_w, r.height())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._mode is not None:
            mode = self._mode
            self._mode = None
            self._commit(mode)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class TimelineView(QGraphicsView):
    """Arrange-area canvas rendering a :class:`Project`."""

    clip_selected = Signal(str)  # clip_id, or "" when cleared
    clip_geometry_edited = Signal(str, float, float)  # clip_id, start, duration
    playhead_moved = Signal(float)  # seconds
    track_selected = Signal(str)  # track_id (clicking an empty lane)
    import_into_clip_requested = Signal(str)  # clip_id (fill via context menu)
    clip_action_requested = Signal(str, str)  # clip_id, action name
    clip_double_clicked = Signal(str)  # clip_id (open editor)
    delete_requested = Signal()  # Delete/Backspace while the timeline is focused
    copy_requested = Signal()  # Ctrl+C
    paste_requested = Signal()  # Ctrl+V
    duplicate_requested = Signal()  # Ctrl+D
    loop_toggle_requested = Signal()  # Ctrl+L
    loop_region_changed = Signal(float, float)
    loop_enabled_changed = Signal(bool)
    grid_menu_requested = Signal(object)  # QPoint — empty-lane context menu

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self._scene = QGraphicsScene()
        super().__init__(self._scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(QColor(theme.TIMELINE_BG))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        # Anchor the scene top-left; otherwise Qt centers a short scene in the
        # viewport and clips drift away from their (top-aligned) track headers.
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setFocusPolicy(Qt.StrongFocus)  # receive Delete/Backspace when focused
        # Repaint the whole viewport on scroll. The default (blit + repaint the
        # newly-exposed strip) leaves ghost copies of the viewport-pinned ruler.
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        self._project: Optional[Project] = None
        self.audio_pool = None  # set via set_audio_pool(); used to draw waveforms
        self.selected_track_id: Optional[str] = None
        self.pps = PPS_DEFAULT
        self.grid = GridSpec()
        self.playhead = 0.0
        self.start_position = 0.0  # locator spacebar-play returns to
        self.playback_active = False

        self._clip_items: dict[str, ClipItem] = {}
        self._loop_drag: Optional[str] = None  # "start" | "end" | "move" | "toggle"
        self._loop_drag_origin_x = 0.0
        self._loop_drag_start = 0.0
        self._loop_drag_end = 0.0
        self._loop_press = QPoint()
        self._scene.selectionChanged.connect(self._on_selection_changed)

    # ---- project binding -------------------------------------------------
    def set_project(self, project: Project) -> None:
        self._project = project
        self.rebuild()

    def set_audio_pool(self, pool) -> None:
        self.audio_pool = pool
        self.viewport().update()

    def set_selected_track(self, track_id: Optional[str]) -> None:
        self.selected_track_id = track_id
        self.viewport().update()  # repaint lane tint

    def track_row(self, clip_id: str) -> int:
        if self._project is None:
            return 0
        for i, track in enumerate(self._project.tracks):
            if track.clip_by_id(clip_id) is not None:
                return i
        return 0

    def track_color(self, clip_id: str) -> str:
        if self._project is not None:
            for track in self._project.tracks:
                if track.clip_by_id(clip_id) is not None:
                    return track.color
        return theme.BLUE

    # ---- grid / snapping -------------------------------------------------
    @property
    def snap_enabled(self) -> bool:
        return self.grid.kind != "off"

    def _grid_seconds(self) -> Optional[float]:
        if self._project is None:
            return seconds_per_beat(120.0)
        return grid_interval_seconds(
            self.grid, self._project.tempo, self._project.beats_per_bar, self.pps
        )

    def snap(self, seconds: float) -> float:
        return snap_time(seconds, self._grid_seconds())

    def locate(self, seconds: float) -> None:
        """Set the play-start locator (snapped). Moves the cursor when stopped."""
        t = max(0.0, self.snap(seconds))
        self.start_position = t
        if not self.playback_active:
            self.set_playhead(t)

    # ---- rebuild ---------------------------------------------------------
    def rebuild(self) -> None:
        self._scene.clear()
        self._clip_items.clear()
        if self._project is None:
            return

        for track in self._project.tracks:
            for clip in track.clips:
                item = ClipItem(clip, self)
                self._scene.addItem(item)
                self._clip_items[clip.id] = item

        self._update_scene_rect()
        self.viewport().update()

    def _update_scene_rect(self) -> None:
        if self._project is None:
            return
        n = max(len(self._project.tracks), 1)
        content_seconds = max(self._project.duration + 8.0, 30.0)
        width = content_seconds * self.pps
        height = RULER_H + n * TRACK_H + TRACK_H  # trailing empty lane
        self._scene.setSceneRect(0, 0, width, height)

    def refresh_clip(self, clip_id: str) -> None:
        item = self._clip_items.get(clip_id)
        if item is not None:
            item.refresh_geometry()

    def refresh_geometries(self) -> None:
        """Re-place every clip after a tempo rescale (no scene rebuild)."""
        for item in self._clip_items.values():
            item.refresh_geometry()
        self._update_scene_rect()

    # ---- zoom ------------------------------------------------------------
    def set_pps(self, pps: float) -> None:
        self.pps = max(PPS_MIN, min(PPS_MAX, pps))
        for item in self._clip_items.values():
            item.refresh_geometry()
        self._update_scene_rect()
        self.viewport().update()

    def zoom_in(self) -> None:
        self.set_pps(self.pps * 1.25)

    def zoom_out(self) -> None:
        self.set_pps(self.pps / 1.25)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if event.modifiers() & Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
            return
        super().wheelEvent(event)

    # ---- playhead --------------------------------------------------------
    def set_playhead(self, seconds: float) -> None:
        self.playhead = max(0.0, seconds)
        self.viewport().update()
        self.playhead_moved.emit(self.playhead)

    # ---- selection -------------------------------------------------------
    def _on_selection_changed(self) -> None:
        items = self._scene.selectedItems()
        if items and isinstance(items[0], ClipItem):
            self.clip_selected.emit(items[0].clip.id)
        else:
            self.clip_selected.emit("")

    def selected_clip_id(self) -> Optional[str]:
        ids = self.selected_clip_ids()
        return ids[0] if ids else None

    def selected_clip_ids(self) -> list[str]:
        return [item.clip.id for item in self._scene.selectedItems()
                if isinstance(item, ClipItem)]

    def select_clips(self, clip_ids) -> None:  # noqa: ANN001
        wanted = set(clip_ids)
        self._scene.clearSelection()
        for cid, item in self._clip_items.items():
            if cid in wanted:
                item.setSelected(True)

    def selection_time_span(self) -> Optional[tuple[float, float]]:
        items = [i for i in self._scene.selectedItems() if isinstance(i, ClipItem)]
        if not items:
            return None
        start = min(i.clip.start for i in items)
        end = max(i.clip.start + i.clip.duration for i in items)
        if end <= start:
            return None
        return start, end

    def _loop_times(self) -> tuple[float, float]:
        if self._project is None:
            return 0.0, 8.0
        return self._project.loop_bounds()

    def _loop_hit(self, scene_x: float, scene_y: float, view_top: float) -> Optional[str]:
        bar_top = view_top + RULER_H - LOOP_BAR_H - 1
        if scene_y < bar_top or scene_y > view_top + RULER_H:
            return None
        start, end = self._loop_times()
        x0, x1 = start * self.pps, end * self.pps
        if abs(scene_x - x0) <= LOOP_HANDLE_W:
            return "start"
        if abs(scene_x - x1) <= LOOP_HANDLE_W:
            return "end"
        if x0 <= scene_x <= x1:
            return "move"
        return None

    def keyPressEvent(self, event) -> None:  # noqa: N802
        # Accept both the Mac delete key (Backspace) and forward-delete.
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_requested.emit()
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

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        item = self.itemAt(event.pos())
        if isinstance(item, ClipItem):
            super().contextMenuEvent(event)
            return
        self.grid_menu_requested.emit(event.globalPos())
        event.accept()

    # ---- mouse: ruler / empty-lane locate --------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        scene_pos = self.mapToScene(event.position().toPoint())
        view_top = self.mapToScene(0, 0).y()
        if event.button() == Qt.LeftButton and view_top <= scene_pos.y() <= view_top + RULER_H:
            hit = self._loop_hit(scene_pos.x(), scene_pos.y(), view_top)
            if hit is not None:
                start, end = self._loop_times()
                self._loop_drag = hit
                self._loop_drag_origin_x = scene_pos.x()
                self._loop_drag_start = start
                self._loop_drag_end = end
                self._loop_press = event.position().toPoint()
                event.accept()
                return
            self.locate(max(0.0, scene_pos.x() / self.pps))
            event.accept()
            return

        item = self.itemAt(event.position().toPoint())
        if event.button() == Qt.LeftButton and not isinstance(item, ClipItem):
            # Empty arrangement click sets the play-start locator (snapped).
            self.locate(max(0.0, scene_pos.x() / self.pps))
            if self._project is not None:
                row = int((scene_pos.y() - RULER_H) // TRACK_H)
                if 0 <= row < len(self._project.tracks):
                    self.track_selected.emit(self._project.tracks[row].id)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._loop_drag in ("start", "end", "move"):
            scene_x = self.mapToScene(event.position().toPoint()).x()
            dt = (scene_x - self._loop_drag_origin_x) / self.pps
            start, end = self._loop_drag_start, self._loop_drag_end
            if self._loop_drag == "start":
                start = max(0.0, self.snap(self._loop_drag_start + dt))
                start = min(start, end - MIN_CLIP_SECONDS)
            elif self._loop_drag == "end":
                end = max(start + MIN_CLIP_SECONDS, self.snap(self._loop_drag_end + dt))
            else:
                length = end - start
                start = max(0.0, self.snap(self._loop_drag_start + dt))
                end = start + length
            if self._project is not None:
                self._project.loop_start = start
                self._project.loop_end = end
            self.loop_region_changed.emit(start, end)
            self.viewport().update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._loop_drag is not None:
            moved = (event.position().toPoint() - self._loop_press).manhattanLength() > 4
            mode = self._loop_drag
            self._loop_drag = None
            if not moved and mode in ("move", "start", "end") and self._project is not None:
                self.loop_enabled_changed.emit(not self._project.loop_enabled)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ---- background: lanes + grid ---------------------------------------
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        super().drawBackground(painter, rect)
        if self._project is None:
            return

        # Track lanes (selected lane tinted so the add/import target is obvious).
        n = len(self._project.tracks)
        for i in range(n):
            y0 = RULER_H + i * TRACK_H
            if self._project.tracks[i].id == self.selected_track_id:
                color = QColor(theme.BG_SELECTED)
            else:
                color = _lane_color(i)
            painter.fillRect(QRectF(rect.left(), y0, rect.width(), TRACK_H), color)

        start, end = self._loop_times()
        lx0, lx1 = start * self.pps, end * self.pps
        lane_tint = theme.LOOP_LANE_ON if self._project.loop_enabled else theme.LOOP_LANE_OFF
        painter.fillRect(
            QRectF(lx0, RULER_H, max(1.0, lx1 - lx0), n * TRACK_H + TRACK_H),
            lane_tint,
        )

        # Vertical grid: lines at the snap interval, weighted bar > beat > subdiv.
        # Off still draws faint bar lines so the arrangement isn't a blank field.
        spb = self._project.seconds_per_beat()
        bpb = max(self._project.beats_per_bar, 1)
        bar_sec = spb * bpb
        interval = self._grid_seconds()
        beat_pen = QPen(QColor(*theme.GRID_BEAT), 1)
        bar_pen = QPen(QColor(*theme.GRID_BAR), 1)
        subdiv_pen = QPen(QColor(*theme.GRID_SUBDIV), 1)
        y0, y1 = int(rect.top()), int(rect.bottom())

        def _on_period(t: float, period: float) -> bool:
            if period <= 0:
                return False
            q = t / period
            return abs(q - round(q)) < 1e-4

        def _draw_step(step_sec: float, classify: bool) -> None:
            step_px = step_sec * self.pps
            if step_px < 6.0:
                return
            first = max(int(rect.left() // step_px), 0)
            last = int(rect.right() // step_px) + 1
            for i in range(first, last + 1):
                t = i * step_sec
                if classify and _on_period(t, bar_sec):
                    painter.setPen(bar_pen)
                elif classify and _on_period(t, spb):
                    painter.setPen(beat_pen)
                elif classify:
                    painter.setPen(subdiv_pen)
                else:
                    painter.setPen(bar_pen)
                painter.drawLine(int(i * step_px), y0, int(i * step_px), y1)

        if interval is None:
            _draw_step(bar_sec, classify=False)
        else:
            _draw_step(interval, classify=True)
            # Triplet (and other) intervals may miss bar lines — keep bars visible.
            if bar_sec > 0 and not _on_period(interval, bar_sec) and not _on_period(bar_sec, interval):
                _draw_step(bar_sec, classify=False)

    # ---- foreground: ruler pinned to viewport top -----------------------
    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        if self._project is None:
            return
        view_top = self.mapToScene(0, 0).y()

        painter.fillRect(
            QRectF(rect.left(), view_top, rect.width(), RULER_H), theme.RULER_BG
        )
        painter.setPen(theme.RULER_LINE)
        painter.drawLine(
            int(rect.left()), int(view_top + RULER_H),
            int(rect.right()), int(view_top + RULER_H),
        )

        spb = self._project.seconds_per_beat()
        bpb = max(self._project.beats_per_bar, 1)
        bar_px = spb * bpb * self.pps
        if bar_px > 0:
            painter.setPen(theme.RULER_TEXT)
            painter.setFont(QFont("", 8))
            first_bar = max(int(rect.left() // bar_px), 0)
            last_bar = int(rect.right() // bar_px) + 1
            for bar in range(first_bar, last_bar + 1):
                x = bar * bar_px
                painter.drawLine(int(x), int(view_top), int(x), int(view_top + RULER_H))
                painter.drawText(int(x) + 4, int(view_top + 14), str(bar + 1))

        start, end = self._loop_times()
        x0, x1 = start * self.pps, end * self.pps
        bar_top = view_top + RULER_H - LOOP_BAR_H - 1
        on = bool(self._project.loop_enabled)
        fill = QColor(theme.LOOP_ON if on else theme.LOOP_OFF)
        painter.fillRect(QRectF(x0, bar_top, max(2.0, x1 - x0), LOOP_BAR_H), fill)
        handle = QColor(theme.CYAN if on else theme.FG_DIM)
        handle.setAlpha(230 if on else 90)
        painter.setPen(Qt.NoPen)
        painter.setBrush(handle)
        painter.drawRect(QRectF(x0 - 2, bar_top, 4, LOOP_BAR_H))
        painter.drawRect(QRectF(x1 - 2, bar_top, 4, LOOP_BAR_H))
        painter.setPen(QColor(theme.FG_BRIGHT if on else theme.FG_DIM))
        painter.setFont(QFont("", 7, QFont.Bold if on else QFont.Normal))
        painter.drawText(int(x0) + 6, int(bar_top + 9), "LOOP" if on else "loop")

        # Playhead (full height + ruler marker).
        px = self.playhead * self.pps
        painter.setPen(QPen(QColor(theme.PLAYHEAD), 2))
        painter.drawLine(int(px), int(rect.top()), int(px), int(rect.bottom()))
