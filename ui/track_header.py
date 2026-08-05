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

from fantasia_core.document.model import Track
from ui import theme
from ui.gm_instruments import DRUM_KITS, GM_FAMILIES, gm_name
from ui.metrics import RULER_H, TRACK_H


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

        self._fx_types = [e.get("type") for e in track.fx]
        self._is_drum = getattr(track, "is_drum", False)
        self._is_synth = getattr(track, "is_synth", False)
        self._instrument = getattr(track, "instrument", 0)

        # --- create widgets and set state from the model (no signals yet) ---
        self.name_edit = QLineEdit(track.name)
        self.name_edit.setFrame(False)
        self.name_edit.setStyleSheet(
            f"color:{theme.FG_BRIGHT}; background:transparent; font-weight:700; font-size:12px;")
        self.fx_badge = QLabel()
        self.fx_badge.setStyleSheet(f"color:{theme.CYAN}; font-size:10px; font-weight:600;")
        parts = []
        if self._is_synth:
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
        self.mute_btn.setCheckable(True)
        self.mute_btn.setChecked(track.mute)
        self.mute_btn.setFixedWidth(24)
        self.mute_btn.setToolTip("Mute")

        self.solo_btn = QPushButton("S")
        self.solo_btn.setCheckable(True)
        self.solo_btn.setChecked(track.solo)
        self.solo_btn.setFixedWidth(24)
        self.solo_btn.setToolTip("Solo")

        self.vol = QSlider(Qt.Horizontal)
        self.vol.setRange(-60, 12)  # dB
        self.vol.setValue(int(track.gain_db))
        self.vol.setToolTip("Volume (dB)")

        self.pan = QSlider(Qt.Horizontal)
        self.pan.setRange(-100, 100)
        self.pan.setValue(int(track.pan * 100))
        self.pan.setFixedWidth(56)
        self.pan.setToolTip("Pan (L/R)")

        # --- lay out ---
        name_row = QHBoxLayout()
        name_row.setContentsMargins(0, 0, 0, 0)
        name_row.addWidget(self.name_edit, 1)
        name_row.addWidget(self.fx_badge)
        outer.addLayout(name_row)
        controls = QHBoxLayout()
        controls.setSpacing(4)
        controls.addWidget(self.mute_btn)
        controls.addWidget(self.solo_btn)
        controls.addWidget(QLabel("Vol"))
        controls.addWidget(self.vol, 1)
        controls.addWidget(QLabel("Pan"))
        controls.addWidget(self.pan)
        outer.addLayout(controls)

        # --- connect signals last, so setup above never emits ---
        self.name_edit.editingFinished.connect(
            lambda: self.renamed.emit(self.track_id, self.name_edit.text())
        )
        self.mute_btn.toggled.connect(
            lambda on: self.mute_toggled.emit(self.track_id, on)
        )
        self.solo_btn.toggled.connect(
            lambda on: self.solo_toggled.emit(self.track_id, on)
        )
        self.vol.valueChanged.connect(
            lambda v: self.gain_changed.emit(self.track_id, float(v))
        )
        self.pan.valueChanged.connect(
            lambda v: self.pan_changed.emit(self.track_id, v / 100.0)
        )

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)

    def _install_selection_filter(self) -> None:
        # Catch presses on the header *and* its child controls, so any click
        # selects the track — without consuming the event (controls still work).
        self.installEventFilter(self)
        for child in self.findChildren(QWidget):
            child.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        et = event.type()
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

        menu.addSeparator()
        if self._fx_types:
            head = menu.addAction("FX: " + ", ".join(self._fx_types))
            head.setEnabled(False)
            menu.addSeparator()
        for label, name in [("Add Reverb", "add_reverb"), ("Add Delay", "add_delay"),
                            ("Add Low-pass", "add_lowpass"), ("Add High-pass", "add_highpass")]:
            mapping[menu.addAction(label)] = name
        menu.addSeparator()
        mapping[menu.addAction("Clear FX")] = "clear_fx"
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

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(240)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # driven by sync
        self.setStyleSheet(f"QScrollArea {{ background:{theme.BG_DEEP}; border:none; }}")

        self._content = QWidget()
        self._content.setStyleSheet(f"background:{theme.BG_DEEP};")
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
