"""Synth panel — knobs for a synth track's patch.

Shown (bottom dock) when a synth track is selected. Emits
``param_changed(track_id, key, value)`` as the user moves a control; the window
routes that through the CommandBus (mergeable → one undo per drag) and re-renders.

Controls are populated from ``{**DEFAULT_PATCH, **track.synth}`` with signals
blocked, so populating never emits (same discipline as the track headers).
"""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QGridLayout,
    QLabel,
    QSlider,
    QWidget,
)

from fantasia_core.engine import DEFAULT_PATCH, WAVEFORMS
from ui import theme

# (key, label, min, max, unit) — sliders use an int 0..1000 mapped onto [min, max].
_SLIDERS = [
    ("mix", "Osc Mix", 0.0, 1.0, ""),
    ("detune", "Detune", 0.0, 1.0, " st"),
    ("attack", "Attack", 0.0, 2.0, " s"),
    ("decay", "Decay", 0.0, 2.0, " s"),
    ("sustain", "Sustain", 0.0, 1.0, ""),
    ("release", "Release", 0.0, 2.0, " s"),
    ("cutoff", "Cutoff", 100.0, 12000.0, " Hz"),
    ("resonance", "Resonance", 0.0, 1.0, ""),
    ("env_amount", "Env→Cutoff", 0.0, 8000.0, " Hz"),
    ("gain", "Gain", 0.0, 1.0, ""),
]
_STEPS = 1000


class SynthPanel(QWidget):
    param_changed = Signal(str, str, object)  # (track_id, key, value)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.track_id: Optional[str] = None

        self.setObjectName("synthBody")
        self.setStyleSheet(
            f"QWidget#synthBody {{ background: {theme.BG_PANEL}; }}"
            f" QLabel {{ color: {theme.FG}; background: transparent; }}")
        grid = QGridLayout(self)
        grid.setContentsMargins(10, 8, 10, 8)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)

        # Oscillator waveform selectors.
        self.osc1 = QComboBox()
        self.osc1.addItems(WAVEFORMS)
        self.osc2 = QComboBox()
        self.osc2.addItems(WAVEFORMS)
        grid.addWidget(QLabel("Osc 1"), 0, 0)
        grid.addWidget(self.osc1, 0, 1, 1, 2)
        grid.addWidget(QLabel("Osc 2"), 0, 3)
        grid.addWidget(self.osc2, 0, 4, 1, 2)
        self.osc1.currentTextChanged.connect(lambda v: self._emit("osc1", v))
        self.osc2.currentTextChanged.connect(lambda v: self._emit("osc2", v))

        # Sliders in two column-groups: [label, slider, value].
        self._sliders: Dict[str, QSlider] = {}
        self._value_labels: Dict[str, QLabel] = {}
        self._specs = {s[0]: s for s in _SLIDERS}
        per_col = (len(_SLIDERS) + 1) // 2
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(4, 1)
        for i, (key, label, lo, hi, unit) in enumerate(_SLIDERS):
            row = 1 + (i % per_col)
            base = 0 if i < per_col else 3
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, _STEPS)
            vlabel = QLabel()
            vlabel.setMinimumWidth(56)
            vlabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(QLabel(label), row, base)
            grid.addWidget(slider, row, base + 1)
            grid.addWidget(vlabel, row, base + 2)
            self._sliders[key] = slider
            self._value_labels[key] = vlabel
            slider.valueChanged.connect(lambda v, k=key: self._on_slider(k, v))

    # ---- mapping ---------------------------------------------------------
    def _to_value(self, key: str, sval: int) -> float:
        _, _, lo, hi, _ = self._specs[key]
        return lo + (sval / _STEPS) * (hi - lo)

    def _to_slider(self, key: str, value: float) -> int:
        _, _, lo, hi, _ = self._specs[key]
        if hi == lo:
            return 0
        return int(round((float(value) - lo) / (hi - lo) * _STEPS))

    def _fmt(self, key: str, value: float) -> str:
        _, _, _, hi, unit = self._specs[key]
        if hi >= 100:  # Hz-scale
            return f"{value:.0f}{unit}"
        return f"{value:.2f}{unit}"

    # ---- populate / emit -------------------------------------------------
    def set_track(self, track) -> None:  # noqa: ANN001
        self.track_id = track.id
        patch = {**DEFAULT_PATCH, **(track.synth or {})}
        for combo, key in ((self.osc1, "osc1"), (self.osc2, "osc2")):
            combo.blockSignals(True)
            combo.setCurrentText(str(patch[key]))
            combo.blockSignals(False)
        for key, slider in self._sliders.items():
            slider.blockSignals(True)
            slider.setValue(self._to_slider(key, patch[key]))
            slider.blockSignals(False)
            self._value_labels[key].setText(self._fmt(key, float(patch[key])))
        self.setWindowTitle(f"Synth — {track.name}")

    def _on_slider(self, key: str, sval: int) -> None:
        value = self._to_value(key, sval)
        self._value_labels[key].setText(self._fmt(key, value))
        self._emit(key, value)

    def _emit(self, key: str, value) -> None:
        if self.track_id is not None:
            self.param_changed.emit(self.track_id, key, value)
