"""Track header widgets — the left column aligned with timeline lanes.

Each :class:`TrackHeader` shows a track's name, mute/solo, volume and pan, and
emits Qt signals when the user changes them. The window routes those signals
through the CommandBus (M2) so every change is undoable.

Important: widget states are initialised from the model *before* signals are
connected, so rebuilding the panel after an undo/redo never emits spurious
change signals (which would push bogus commands).
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from fantasia_core.engine.levels import amp_to_db

from fantasia_core.document.fx_insert import insert_type
from fantasia_core.document.model import MASTER_ID, Track
from ui import theme
from ui.gm_instruments import DRUM_KITS, GM_FAMILIES, gm_name
from ui.metrics import RULER_H, TRACK_H

_FADER_STYLE = (
    f"QSlider#gainFader::groove:horizontal {{"
    f"  height: 7px; background: {theme.BG_DEEP};"
    f"  border: 1px solid {theme.BORDER}; border-radius: 3px;"
    f"}}"
    f"QSlider#gainFader::sub-page:horizontal {{"
    f"  background: {theme.CYAN}; border-radius: 3px;"
    f"}}"
    f"QSlider#gainFader::handle:horizontal {{"
    f"  background: {theme.FG_BRIGHT}; border: 1px solid {theme.CYAN};"
    f"  width: 9px; margin: -5px 0; border-radius: 1px;"
    f"}}"
)

# Widget-local sheet: a parent QWidget background (the header / scroll
# content) otherwise swallows QPushButton:checked fills from the window sheet.
# Qt only paints a button background when the same sheet also sets a border.
_MS_BUTTON_STYLE = (
    f"QPushButton {{"
    f"  background-color: {theme.BG_ELEVATED}; color: {theme.FG};"
    f"  border: 1px solid {theme.BORDER}; border-radius: 4px;"
    f"  padding: 2px 0px; font-weight: 700;"
    f"}}"
    f"QPushButton:hover {{"
    f"  background-color: {theme.BG_HOVER}; border-color: {theme.PURPLE};"
    f"}}"
    f"QPushButton:checked,"
    f"QPushButton:checked:hover,"
    f"QPushButton:checked:pressed {{"
    f"  background-color: {theme.BUTTON_CHECKED};"
    f"  color: {theme.BUTTON_CHECKED_FG};"
    f"  border: 1px solid {theme.PINK};"
    f"}}"
)


class LevelMeter(QWidget):
    """Horizontal peak meter. ``level`` / ``held`` are 0..1 linear amplitudes."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(7)
        self.setMinimumWidth(48)
        self.level = 0.0
        self.held = 0.0

    def set_levels(self, current: float, held: float) -> None:
        self.level = max(0.0, min(1.0, float(current)))
        self.held = max(self.level, min(1.0, float(held)))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802, ARG002
        p = QPainter(self)
        r = self.rect().adjusted(0, 1, -1, -1)
        p.fillRect(r, QColor(theme.BG_DEEP))
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.drawRect(r)
        inner = r.adjusted(1, 1, -1, -1)
        if self.level > 0:
            fill = QColor(theme.CYAN if self.level < 0.89 else theme.NEON_ORANGE)
            if self.level >= 0.99:
                fill = QColor(theme.RED)
            p.fillRect(
                inner.x(), inner.y(),
                max(1, int(inner.width() * self.level)), inner.height(), fill)
        if self.held > 0.02:
            x = inner.x() + max(0, int(inner.width() * self.held) - 1)
            p.fillRect(x, inner.y(), 2, inner.height(), QColor(theme.FG_BRIGHT))
        p.end()


class TrackHeader(QWidget):
    """Header for a single track."""

    renamed = Signal(str, str)  # (track_id, new_name)
    mute_toggled = Signal(str, bool)
    solo_toggled = Signal(str, bool)
    gain_changed = Signal(str, float)  # dB
    pan_changed = Signal(str, float)  # -1..1
    clicked = Signal(str)  # track_id (selection)
    fx_action = Signal(str, str)  # (track_id, action) from the FX context menu

    def __init__(self, track: Track, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.track_id = track.id
        self.setFixedHeight(TRACK_H)
        self.setObjectName("trackHeader")
        self._selected = False
        self._build(track)
        self._install_selection_filter()

    def _build(self, track: Track) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        self._fx_types = [insert_type(e) for e in track.fx]
        self._is_drum = getattr(track, "is_drum", False)
        self._is_synth = getattr(track, "is_synth", False)
        self._instrument = getattr(track, "instrument", 0)
        self._plugin = getattr(track, "plugin", "") or ""
        self._is_master = track.id == MASTER_ID or getattr(track, "is_master", False)
        self._color = getattr(track, "color", theme.MAGENTA) or theme.MAGENTA

        # --- create widgets and set state from the model (no signals yet) ---
        self.name_edit = QLineEdit(track.name)
        self.name_edit.setObjectName("trackNameEdit")
        self.name_edit.setFrame(False)
        self.name_edit.setReadOnly(True)
        self.name_edit.setFocusPolicy(Qt.NoFocus)
        self.name_edit.setCursor(Qt.ArrowCursor)
        self.name_edit.setToolTip("Double-click or press F2 to rename")
        self.name_edit.setStyleSheet(
            f"color:{theme.FG_BRIGHT}; background:transparent; font-weight:700; font-size:12px;")
        self._renaming = False
        self._name_before_edit = track.name
        self.fx_badge = QLabel()
        self.fx_badge.setStyleSheet(f"color:{theme.CYAN}; font-size:10px; font-weight:600;")
        parts = []
        if self._is_master:
            parts.append("MASTER")
            self.setToolTip("Master mix bus — FX here apply to the whole mix")
        elif self._plugin:
            # A plugin track has no GM program, so the instrument name shown
            # here would be meaningless — name the plugin instead.
            parts.append(self._plugin)
            self.setToolTip(self._plugin)
        elif self._is_synth:
            parts.append("SYNTH")
            self.setToolTip("Built-in synth")
        elif self._is_drum:
            parts.append("DRUMS")
            self.setToolTip("Drum Kit")
        else:
            nm = gm_name(self._instrument)
            parts.append(nm if len(nm) <= 16 else nm[:15] + "…")
            self.setToolTip(nm)
        if track.fx:
            parts.append(f"FX·{len(track.fx)}")
        self.fx_badge.setText("  ".join(parts))

        self.mute_btn = QPushButton("M")
        self.mute_btn.setObjectName("trackMuteBtn")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setFixedSize(24, 22)
        self.mute_btn.setStyleSheet(_MS_BUTTON_STYLE)
        self.mute_btn.setChecked(track.mute)
        self.mute_btn.setToolTip("Mute")

        self.solo_btn = QPushButton("S")
        self.solo_btn.setObjectName("trackSoloBtn")
        self.solo_btn.setCheckable(True)
        self.solo_btn.setFixedSize(24, 22)
        self.solo_btn.setStyleSheet(_MS_BUTTON_STYLE)
        self.solo_btn.setChecked(track.solo)
        self.solo_btn.setToolTip("Solo")
        if self._is_master:
            self.solo_btn.setEnabled(False)
            self.solo_btn.setToolTip("Solo is per-track; mute the Master to silence the mix")
            self.setStyleSheet(
                f"QWidget#trackHeader {{ border-top: 1px solid {theme.CYAN}; }}"
            )

        self.vol = QSlider(Qt.Horizontal)
        self.vol.setObjectName("gainFader")
        self.vol.setStyleSheet(_FADER_STYLE)
        self.vol.setRange(-60, 12)  # linear dB ≈ equal-loudness steps
        self.vol.setValue(int(round(track.gain_db)))
        self.vol.setToolTip("Volume (dB) — linear in dB, matching perceived loudness")
        self.meter = LevelMeter()
        self._meter_amp = 0.0
        self._meter_max = 0.0
        self.fader_db = QLabel(self._fmt_set(track.gain_db))
        self.fader_db.setFixedWidth(34)
        self.fader_db.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.fader_db.setStyleSheet(
            f"color:{theme.FG_BRIGHT}; font-size:10px; font-weight:700;")
        self.out_db = QLabel("—  —")
        self.out_db.setFixedWidth(52)
        self.out_db.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.out_db.setStyleSheet(f"color:{theme.FG_DIM}; font-size:9px;")
        self.out_db.setToolTip("Current / max output (dBFS) while playing")

        self.pan = QSlider(Qt.Horizontal)
        self.pan.setRange(-100, 100)
        self.pan.setValue(int(track.pan * 100))
        self.pan.setFixedWidth(56)
        self.pan.setToolTip("Pan (L/R)")

        # --- lay out ---
        name_row = QHBoxLayout()
        name_row.setContentsMargins(0, 0, 0, 0)
        if not self._is_master:
            self._swatch = QLabel()
            self._swatch.setFixedSize(12, 12)
            self._swatch.setToolTip("Track color — right-click to change")
            self._swatch.setStyleSheet(
                f"background:{self._color}; border:1px solid {theme.BORDER};"
                f" border-radius:2px;"
            )
            name_row.addWidget(self._swatch)
        name_row.addWidget(self.name_edit, 1)
        name_row.addWidget(self.fx_badge)
        outer.addLayout(name_row)
        vol_col = QVBoxLayout()
        vol_col.setContentsMargins(0, 0, 0, 0)
        vol_col.setSpacing(1)
        vol_col.addWidget(self.vol)
        vol_col.addWidget(self.meter)
        readouts = QVBoxLayout()
        readouts.setContentsMargins(0, 0, 0, 0)
        readouts.setSpacing(0)
        readouts.addWidget(self.fader_db)
        readouts.addWidget(self.out_db)
        controls = QHBoxLayout()
        controls.setSpacing(4)
        controls.addWidget(self.mute_btn)
        controls.addWidget(self.solo_btn)
        controls.addLayout(vol_col, 1)
        controls.addLayout(readouts)
        controls.addWidget(QLabel("Pan"))
        controls.addWidget(self.pan)
        outer.addLayout(controls)

        # --- connect signals last, so setup above never emits ---
        self.name_edit.editingFinished.connect(self._on_name_editing_finished)
        self.mute_btn.toggled.connect(
            lambda on: self.mute_toggled.emit(self.track_id, on)
        )
        self.solo_btn.toggled.connect(
            lambda on: self.solo_toggled.emit(self.track_id, on)
        )
        self.vol.valueChanged.connect(self._on_vol_changed)
        self.pan.valueChanged.connect(
            lambda v: self.pan_changed.emit(self.track_id, v / 100.0)
        )

    def _fmt_set(self, db: float) -> str:
        return f"{db:+.0f}"

    def _fmt_out(self, db: float) -> str:
        if db <= -59.5:
            return "−∞"
        return f"{db:+.0f}"

    def _on_vol_changed(self, v: int) -> None:
        self.fader_db.setText(self._fmt_set(float(v)))
        self.gain_changed.emit(self.track_id, float(v))

    def set_meter(self, amp: float, playing: bool) -> None:
        """``amp`` is linear peak for this tick. Decays visually when quiet."""
        if not playing:
            self.reset_meter()
            return
        self._meter_amp = max(float(amp), self._meter_amp * 0.72)
        self._meter_max = max(self._meter_max, float(amp))
        self.meter.set_levels(self._meter_amp, self._meter_max)
        self.out_db.setText(
            f"{self._fmt_out(amp_to_db(self._meter_amp))}  "
            f"{self._fmt_out(amp_to_db(self._meter_max))}"
        )
        self.out_db.setStyleSheet(f"color:{theme.CYAN}; font-size:9px;")

    def reset_meter(self) -> None:
        self._meter_amp = 0.0
        self._meter_max = 0.0
        self.meter.set_levels(0.0, 0.0)
        self.out_db.setText("—  —")
        self.out_db.setStyleSheet(f"color:{theme.FG_DIM}; font-size:9px;")

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)

    def sync_mute_solo(self, track: Track) -> None:
        """Match M/S buttons to the model without emitting toggle commands."""
        self.mute_btn.blockSignals(True)
        self.mute_btn.setChecked(bool(track.mute))
        self.mute_btn.blockSignals(False)
        self.solo_btn.blockSignals(True)
        self.solo_btn.setChecked(bool(track.solo))
        self.solo_btn.blockSignals(False)

    def _install_selection_filter(self) -> None:
        # Catch presses on the header *and* its child controls, so any click
        # selects the track — without consuming the event (controls still work).
        self.installEventFilter(self)
        for child in self.findChildren(QWidget):
            child.installEventFilter(self)

    def begin_rename(self) -> None:
        """Enter in-place rename (double-click on the name, or F2)."""
        if self._renaming:
            self.name_edit.setFocus()
            self.name_edit.selectAll()
            return
        self._name_before_edit = self.name_edit.text()
        self._renaming = True
        self.name_edit.setReadOnly(False)
        self.name_edit.setFocusPolicy(Qt.StrongFocus)
        self.name_edit.setCursor(Qt.IBeamCursor)
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    def _end_rename(self, commit: bool) -> None:
        if not self._renaming:
            return
        self._renaming = False
        text = self.name_edit.text().strip() or self._name_before_edit
        if not commit:
            text = self._name_before_edit
        self.name_edit.blockSignals(True)
        self.name_edit.setText(text)
        self.name_edit.setReadOnly(True)
        self.name_edit.setFocusPolicy(Qt.NoFocus)
        self.name_edit.setCursor(Qt.ArrowCursor)
        self.name_edit.deselect()
        self.name_edit.blockSignals(False)
        self.name_edit.clearFocus()
        if commit:
            self.renamed.emit(self.track_id, text)

    def _on_name_editing_finished(self) -> None:
        self._end_rename(commit=True)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        et = event.type()
        if obj is self.name_edit and et == QEvent.MouseButtonDblClick:
            self.clicked.emit(self.track_id)
            self.begin_rename()
            return True
        if obj is self.name_edit and et == QEvent.KeyPress:
            if event.key() == Qt.Key_Escape:
                self._end_rename(commit=False)
                return True
        if et == QEvent.MouseButtonPress:
            self.clicked.emit(self.track_id)
        elif et == QEvent.ContextMenu:
            self._show_fx_menu(event.globalPos())
            return True  # consume; show the FX menu instead
        return False  # otherwise don't consume; let the control handle it too

    def _show_fx_menu(self, global_pos) -> None:
        self.clicked.emit(self.track_id)  # select the track first
        menu = QMenu()
        mapping = {}

        if self._is_master:
            head = menu.addAction("Master mix bus")
            head.setEnabled(False)
            menu.addSeparator()
        else:
            synth = menu.addAction("Synth Voice (built-in)")
            synth.setCheckable(True)
            synth.setChecked(self._is_synth)
            mapping[synth] = "toggle_synth"

            drum = menu.addAction("Drum Kit (percussion)")
            drum.setCheckable(True)
            drum.setChecked(self._is_drum)
            mapping[drum] = "toggle_drum"

            if self._is_drum:  # drum-kit picker
                kit_menu = menu.addMenu("Drum Kit")
                for prog, name in DRUM_KITS:
                    act = kit_menu.addAction(name)
                    act.setCheckable(True)
                    act.setChecked(prog == self._instrument)
                    mapping[act] = f"instrument:{prog}"
            elif not self._is_synth:  # instrument picker (soundfont tracks)
                inst_menu = menu.addMenu("Instrument")
                for family, insts in GM_FAMILIES:
                    sub = inst_menu.addMenu(family)
                    for prog, name in insts:
                        act = sub.addAction(name)
                        act.setCheckable(True)
                        act.setChecked(prog == self._instrument)
                        mapping[act] = f"instrument:{prog}"

            # A hosted plugin replaces the built-in engines entirely, so it sits
            # with them rather than in the FX chain below.
            plug = menu.addAction("Plugin Instrument…"
                                  + (f"  [{self._plugin}]" if self._plugin else ""))
            mapping[plug] = "plugin_instrument"
            if self._plugin:
                mapping[menu.addAction(f"Open {self._plugin} Interface…")] = "plugin_editor"

            menu.addSeparator()
            color_menu = menu.addMenu("Set Color")
            for name, hex_color in theme.TRACK_PALETTE:
                act = color_menu.addAction(theme.swatch_icon(hex_color), name)
                act.setCheckable(True)
                act.setChecked(self._color.lower() == hex_color.lower())
                mapping[act] = f"color:{hex_color}"

            menu.addSeparator()
        if self._fx_types:
            head = menu.addAction("FX: " + ", ".join(self._fx_types))
            head.setEnabled(False)
            menu.addSeparator()
        # Mixing chain, in the order you'd normally use it: EQ -> colour -> dynamics.
        eq = menu.addMenu("EQ")
        mapping[eq.addAction("Stock EQ (8-band)")] = "add_eq"
        for label, name in [
            ("High-pass 120 Hz  (remove rumble)", "add_highpass"),
            ("Low-pass 1.2 kHz  (darken)", "add_lowpass"),
            ("Low Shelf…  (boost/cut bass)", "add_eq_low_shelf"),
            ("Bell / Peak…  (shape a band)", "add_eq_peak"),
            ("High Shelf…  (air / brightness)", "add_eq_high_shelf"),
        ]:
            mapping[eq.addAction(label)] = name

        col = menu.addMenu("Colour")
        for label, name in [
            ("Saturator…  (warmth, harmonics)", "add_saturator"),
            ("Distortion  (hard drive)", "add_distortion"),
        ]:
            mapping[col.addAction(label)] = name

        dyn = menu.addMenu("Dynamics")
        for label, name in [
            ("Compressor…  (even out levels)", "add_compressor"),
            ("Limiter  (catch peaks)", "add_limiter"),
            ("Noise Gate  (cut silence/bleed)", "add_gate"),
        ]:
            mapping[dyn.addAction(label)] = name

        space = menu.addMenu("Space")
        for label, name in [("Reverb", "add_reverb"), ("Delay", "add_delay")]:
            mapping[space.addAction(label)] = name

        menu.addSeparator()
        mapping[menu.addAction("Clear FX")] = "clear_fx"
        if not self._is_master:
            menu.addSeparator()
            mapping[menu.addAction("Remove Track")] = "remove_track"

        action = mapping.get(menu.exec(global_pos))
        if action:
            self.fx_action.emit(self.track_id, action)


class TrackHeaderPanel(QScrollArea):
    """Vertical stack of :class:`TrackHeader` widgets, scroll-synced with the
    timeline. A top spacer of height ``RULER_H`` keeps rows aligned with the
    timeline lanes (which start below the ruler)."""

    header_clicked = Signal(str)
    renamed = Signal(str, str)
    mute_toggled = Signal(str, bool)
    solo_toggled = Signal(str, bool)
    gain_changed = Signal(str, float)
    pan_changed = Signal(str, float)
    fx_action = Signal(str, str)
    track_step_requested = Signal(int)  # -1 previous, +1 next

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # Resizable rather than fixed: the splitter can widen this to read long
        # track names, or narrow it to give the timeline room.
        self.setMinimumWidth(150)
        self.setMaximumWidth(460)
        self.resize(240, self.height())
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # driven by sync
        self.setStyleSheet(f"QScrollArea {{ background:{theme.BG_DEEP}; border:none; }}")

        self._content = QWidget()
        self._content.setObjectName("trackHeaderList")
        # Selector-scoped so this fill does not leak onto M/S buttons.
        self._content.setStyleSheet(
            f"QWidget#trackHeaderList {{ background:{theme.BG_DEEP}; }}"
        )
        self._layout = QVBoxLayout(self._content)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        self._spacer = QWidget()
        self._spacer.setFixedHeight(RULER_H)
        self._layout.addWidget(self._spacer)
        # A stretch absorbs any extra vertical space at the BOTTOM so the headers
        # stay packed at the top (TRACK_H apart, aligned with the timeline lanes).
        # Scroll range is matched to the timeline via _content.setMinimumHeight()
        # in rebuild() — not a fixed spacer, which would spread the headers out.
        self._layout.addStretch(1)

        self.setWidget(self._content)
        self._headers: dict[str, TrackHeader] = {}

    def rebuild(self, project) -> None:
        for header in self._headers.values():
            header.setParent(None)
            header.deleteLater()
        self._headers.clear()

        for track in project.tracks:
            header = TrackHeader(track)
            header.clicked.connect(self.header_clicked.emit)
            header.renamed.connect(self.renamed.emit)
            header.mute_toggled.connect(self.mute_toggled.emit)
            header.solo_toggled.connect(self.solo_toggled.emit)
            header.gain_changed.connect(self.gain_changed.emit)
            header.pan_changed.connect(self.pan_changed.emit)
            header.fx_action.connect(self.fx_action.emit)
            # Insert before the trailing stretch.
            self._layout.insertWidget(self._layout.count() - 1, header)
            self._headers[track.id] = header

        # Match the timeline scene height (RULER + n lanes + a trailing lane, plus
        # a scrollbar allowance) so the header scrolls exactly in step with the
        # timeline. The stretch keeps headers top-packed until this forces scroll.
        n = len(project.tracks)
        self._content.setMinimumHeight(RULER_H + n * TRACK_H + TRACK_H + 24)

    def set_selected(self, track_id: Optional[str]) -> None:
        for tid, header in self._headers.items():
            header.set_selected(tid == track_id)

    def begin_rename(self, track_id: str) -> bool:
        header = self._headers.get(track_id)
        if header is None:
            return False
        header.begin_rename()
        return True

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Up, Qt.Key_Down) and not (
            event.modifiers() & Qt.ControlModifier
        ):
            self.track_step_requested.emit(-1 if event.key() == Qt.Key_Up else 1)
            event.accept()
            return
        super().keyPressEvent(event)

    def set_meters(self, peaks: dict, playing: bool) -> None:  # noqa: ANN001
        for tid, header in self._headers.items():
            header.set_meter(float(peaks.get(tid, 0.0)), playing)

    def reset_meters(self) -> None:
        for header in self._headers.values():
            header.reset_meter()

    def sync_mute_solo(self, project) -> None:  # noqa: ANN001
        for track in project.tracks:
            header = self._headers.get(track.id)
            if header is not None:
                header.sync_mute_solo(track)
