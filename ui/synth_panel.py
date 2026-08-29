"""Synth panel — Tokyo-nights modules for a synth track's patch.

Shown in the bottom dock when a synth track is selected. Emits
``param_changed(track_id, key, value)`` as the user moves a control; the window
routes that through the CommandBus (mergeable → one undo per drag) and re-renders.

Controls are populated from ``{**DEFAULT_PATCH, **track.synth}`` with signals
blocked, so populating never emits (same discipline as the track headers).
The engine is unchanged besides the third oscillator: this file is layout only.
"""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from fantasia_core.engine import DEFAULT_PATCH, WAVEFORMS
from ui import theme
from ui.numeric_popup import bind_double_click_edit, parse_number

_WAVE_LABELS = {"sine": "SIN", "saw": "SAW", "square": "SQR", "triangle": "TRI"}

# (key, label, min, max, unit)
_SLIDERS = [
    ("mix", "Stack", 0.0, 1.0, ""),
    ("detune", "Detune", 0.0, 1.0, " st"),
    ("attack", "Attack", 0.0, 2.0, " s"),
    ("decay", "Decay", 0.0, 2.0, " s"),
    ("sustain", "Sustain", 0.0, 1.0, ""),
    ("release", "Release", 0.0, 2.0, " s"),
    ("cutoff", "Cutoff", 100.0, 12000.0, " Hz"),
    ("resonance", "Reso", 0.0, 1.0, ""),
    ("env_amount", "Env", 0.0, 8000.0, " Hz"),
    ("gain", "Gain", 0.0, 1.0, ""),
]
_STEPS = 1000


def _block(title: str, accent: str) -> tuple[QFrame, QVBoxLayout]:
    """A colour-blocked module: neon top bar, elevated fill."""
    frame = QFrame()
    frame.setObjectName("synthBlock")
    frame.setStyleSheet(
        f"QFrame#synthBlock {{"
        f"  background: {theme.BG_ELEVATED};"
        f"  border: 1px solid {theme.BORDER};"
        f"  border-top: 3px solid {accent};"
        f"  border-radius: 6px;"
        f"}}"
        f"QFrame#synthBlock QLabel {{ color: {theme.FG}; background: transparent; }}"
        f"QFrame#synthBlock QLabel#synthBlockTitle {{"
        f"  color: {accent}; font-weight: 800; letter-spacing: 1.4px; font-size: 11px;"
        f"}}"
    )
    inner = QVBoxLayout(frame)
    inner.setContentsMargins(10, 8, 10, 10)
    inner.setSpacing(6)
    head = QLabel(title)
    head.setObjectName("synthBlockTitle")
    inner.addWidget(head)
    return frame, inner


class _WaveStrip(QWidget):
    """Exclusive waveform buttons for one oscillator."""

    changed = Signal(str)

    def __init__(self, caption: str, accent: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        tag = QLabel(caption)
        tag.setFixedWidth(40)
        tag.setStyleSheet(f"color:{accent}; font-weight:800; font-size:11px;")
        row.addWidget(tag)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: Dict[str, QPushButton] = {}
        for kind in WAVEFORMS:
            btn = QPushButton(_WAVE_LABELS[kind])
            btn.setCheckable(True)
            btn.setFixedHeight(24)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(
                f"QPushButton {{ background:{theme.BG_DEEP}; color:{theme.FG};"
                f" border:1px solid {theme.BORDER}; border-radius:3px; font-weight:700;"
                f" font-size:10px; padding:2px 6px; }}"
                f"QPushButton:hover {{ border-color:{accent}; color:{theme.FG_BRIGHT}; }}"
                f"QPushButton:checked {{ background:{accent}; color:{theme.BG_DEEP};"
                f" border-color:{accent}; }}"
            )
            self._group.addButton(btn)
            self._buttons[kind] = btn
            row.addWidget(btn, 1)
            btn.clicked.connect(lambda _=False, k=kind: self.changed.emit(k))

    def set_wave(self, kind: str) -> None:
        btn = self._buttons.get(kind) or self._buttons["saw"]
        btn.blockSignals(True)
        btn.setChecked(True)
        btn.blockSignals(False)


class SynthPanel(QWidget):
    param_changed = Signal(str, str, object)  # (track_id, key, value)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.track_id: Optional[str] = None
        self._specs = {s[0]: s for s in _SLIDERS}
        self._sliders: Dict[str, QSlider] = {}
        self._value_labels: Dict[str, QLabel] = {}

        self.setObjectName("synthBody")
        self.setStyleSheet(
            f"QWidget#synthBody {{ background: {theme.BG_DEEP}; }}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)

        banner = QLabel("STOCK SYNTH")
        banner.setStyleSheet(
            f"color:{theme.CYAN}; font-weight:800; letter-spacing:3px; font-size:12px;"
        )
        self._name = QLabel("")
        self._name.setStyleSheet(f"color:{theme.FG_DIM}; font-size:11px;")
        head = QHBoxLayout()
        head.addWidget(banner)
        head.addStretch(1)
        head.addWidget(self._name)
        root.addLayout(head)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        root.addLayout(grid, 1)

        osc, osc_l = _block("OSCILLATORS", theme.CYAN)
        self.osc1 = _WaveStrip("OSC 1", theme.CYAN)
        self.osc2 = _WaveStrip("OSC 2", theme.MAGENTA)
        self.osc3 = _WaveStrip("OSC 3", theme.PURPLE)
        for strip, key in ((self.osc1, "osc1"), (self.osc2, "osc2"), (self.osc3, "osc3")):
            osc_l.addWidget(strip)
            strip.changed.connect(lambda v, k=key: self._emit(k, v))
        mix_row = QHBoxLayout()
        mix_row.addLayout(self._slider_col("mix"), 2)
        mix_row.addLayout(self._slider_col("detune"), 2)
        osc_l.addLayout(mix_row)
        grid.addWidget(osc, 0, 0, 1, 2)

        filt, filt_l = _block("FILTER", theme.PURPLE)
        for key in ("cutoff", "resonance", "env_amount"):
            filt_l.addLayout(self._slider_col(key))
        grid.addWidget(filt, 1, 0)

        env, env_l = _block("ENVELOPE", theme.MAGENTA)
        for key in ("attack", "decay", "sustain", "release"):
            env_l.addLayout(self._slider_col(key))
        grid.addWidget(env, 1, 1)

        out, out_l = _block("OUTPUT", theme.GREEN)
        out_l.addLayout(self._slider_col("gain"))
        hint = QLabel("Stack 0 = one oscillator · 1 = detuned trio through the filter")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{theme.FG_DIM}; font-size:10px;")
        out_l.addWidget(hint)
        out_l.addStretch(1)
        grid.addWidget(out, 0, 2, 2, 1)

        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(2, 2)

        self._sliders["mix"].setToolTip("0 = centred osc 1 only; 1 = equal mix of three detuned oscillators")
        self._sliders["detune"].setToolTip("Osc 2 sharp / osc 3 flat, in semitones")
        self._sliders["cutoff"].setToolTip("Low-pass frequency — lower is darker")

    def _slider_col(self, key: str) -> QVBoxLayout:
        _, label, _lo, _hi, _unit = self._specs[key]
        col = QVBoxLayout()
        col.setSpacing(2)
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel(label))
        vlabel = QLabel()
        vlabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        vlabel.setStyleSheet(f"color:{theme.FG_BRIGHT}; font-weight:700;")
        vlabel.setMinimumWidth(52)
        name_row.addWidget(vlabel)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, _STEPS)
        col.addLayout(name_row)
        col.addWidget(slider)
        self._sliders[key] = slider
        self._value_labels[key] = vlabel
        slider.valueChanged.connect(lambda v, k=key: self._on_slider(k, v))
        bind_double_click_edit(
            slider,
            getter=lambda k=key: self._value_labels[k].text(),
            commit=lambda text, k=key: self._commit_typed(k, text),
        )
        bind_double_click_edit(
            vlabel,
            getter=lambda k=key: self._value_labels[k].text(),
            commit=lambda text, k=key: self._commit_typed(k, text),
        )
        return col

    def _commit_typed(self, key: str, text: str) -> bool:
        value = parse_number(text)
        if value is None:
            return False
        _, _, lo, hi, _ = self._specs[key]
        self._sliders[key].setValue(self._to_slider(key, max(lo, min(hi, value))))
        return True

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
        if hi >= 100:
            return f"{value:.0f}{unit}"
        return f"{value:.2f}{unit}"

    def set_track(self, track) -> None:  # noqa: ANN001
        self.track_id = track.id
        patch = {**DEFAULT_PATCH, **(track.synth or {})}
        self._name.setText(track.name)
        for strip, key in ((self.osc1, "osc1"), (self.osc2, "osc2"), (self.osc3, "osc3")):
            strip.set_wave(str(patch.get(key, "saw")))
        for key, slider in self._sliders.items():
            slider.blockSignals(True)
            slider.setValue(self._to_slider(key, float(patch[key])))
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
