"""Detailed inspector for one stock FX insert — the EQ-tab counterpart."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from fantasia_core.document.fx_insert import device_label, insert_bypassed, insert_type
from fantasia_core.document.fx_params import ParamSpec, read_param, specs_for
from ui import theme

_STYLE = f"""
QWidget#deviceRoot {{ background: {theme.BG_PANEL}; }}
QLabel#deviceTitle {{ color: {theme.FG_BRIGHT}; font-size: 16px; font-weight: 700; }}
QLabel#deviceSub {{ color: {theme.FG_DIM}; font-size: 11px; }}
QLabel#knobLabel {{ color: {theme.FG_DIM}; font-size: 11px; }}
QLabel#knobValue {{ color: {theme.CYAN}; font-size: 18px; font-weight: 700; }}
QSlider::groove:horizontal {{
  height: 8px; background: {theme.BG_ELEVATED}; border-radius: 4px;
}}
QSlider::handle:horizontal {{
  width: 16px; margin: -5px 0; background: {theme.CYAN}; border-radius: 8px;
}}
QDoubleSpinBox, QComboBox {{
  background: {theme.BG_ELEVATED}; color: {theme.FG_BRIGHT};
  border: 1px solid {theme.BORDER}; border-radius: 3px; padding: 2px 6px;
}}
QCheckBox {{ color: {theme.FG_BRIGHT}; }}
QPushButton#backBtn {{
  background: {theme.BG_ELEVATED}; color: {theme.FG};
  border: 1px solid {theme.BORDER}; border-radius: 4px; padding: 4px 10px;
}}
"""


class DevicePanel(QWidget):
    """Large-parameter view for a stock insert. VST hosts keep their own window."""

    param_changed = Signal(str, str, object)  # insert_id, key, value
    bypass_changed = Signal(str, bool)
    back_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("deviceRoot")
        self.setStyleSheet(_STYLE)
        self._insert_id = ""
        self._kind = ""
        self._specs: tuple[ParamSpec, ...] = ()
        self._controls: dict[str, QWidget] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(10)

        head = QHBoxLayout()
        self._back = QPushButton("← Graph")
        self._back.setObjectName("backBtn")
        self._back.clicked.connect(self.back_requested.emit)
        head.addWidget(self._back)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        self._title = QLabel("No device")
        self._title.setObjectName("deviceTitle")
        self._sub = QLabel("Double-click a node in the signal graph to open it here.")
        self._sub.setObjectName("deviceSub")
        self._sub.setWordWrap(True)
        titles.addWidget(self._title)
        titles.addWidget(self._sub)
        head.addLayout(titles, 1)
        self._bypass = QCheckBox("Bypass")
        self._bypass.toggled.connect(self._on_bypass)
        head.addWidget(self._bypass, 0, Qt.AlignTop)
        outer.addLayout(head)

        self._hint = QLabel("")
        self._hint.setObjectName("deviceSub")
        self._hint.setWordWrap(True)
        outer.addWidget(self._hint)

        self._wet_box = QWidget()
        wet_l = QVBoxLayout(self._wet_box)
        wet_l.setContentsMargins(0, 8, 0, 8)
        wet_l.setSpacing(4)
        row = QHBoxLayout()
        dry_l = QLabel("Dry")
        dry_l.setObjectName("knobLabel")
        wet_l_lbl = QLabel("Wet")
        wet_l_lbl.setObjectName("knobLabel")
        self._wet_value = QLabel("50%")
        self._wet_value.setObjectName("knobValue")
        self._wet_value.setAlignment(Qt.AlignCenter)
        row.addWidget(dry_l)
        row.addWidget(self._wet_value, 1)
        row.addWidget(wet_l_lbl)
        wet_l.addLayout(row)
        self._wet_slider = QSlider(Qt.Horizontal)
        self._wet_slider.setRange(0, 100)
        self._wet_slider.setValue(50)
        self._wet_slider.valueChanged.connect(self._on_wet)
        wet_l.addWidget(self._wet_slider)
        self._wet_caption = QLabel(
            "The Dry input is the first branch; Wet is the parallel chain. "
            "0% is fully dry, 100% is fully the wet path."
        )
        self._wet_caption.setObjectName("deviceSub")
        self._wet_caption.setWordWrap(True)
        wet_l.addWidget(self._wet_caption)
        outer.addWidget(self._wet_box)
        self._wet_box.hide()

        self._form_host = QWidget()
        self._form = QFormLayout(self._form_host)
        self._form.setContentsMargins(0, 4, 0, 0)
        self._form.setSpacing(8)
        outer.addWidget(self._form_host, 1)
        outer.addStretch(1)

    def set_insert(self, insert, track_name: str = "") -> None:  # noqa: ANN001
        if insert is None:
            self._insert_id = ""
            self._title.setText("No device")
            self._sub.setText("Double-click a node in the signal graph to open it here.")
            self._hint.setText("")
            self._bypass.setEnabled(False)
            self._clear_form()
            self._wet_box.hide()
            return
        self._insert_id = getattr(insert, "id", "") or ""
        self._kind = insert_type(insert)
        params = getattr(insert, "params", None) or {}
        self._specs = specs_for(self._kind, params)
        name = device_label(insert)
        where = f" on {track_name}" if track_name else ""
        self._title.setText(name)
        self._sub.setText(f"{self._kind}{where}  ·  id {self._insert_id}")
        self._bypass.setEnabled(True)
        blocked = self._bypass.blockSignals(True)
        self._bypass.setChecked(insert_bypassed(insert))
        self._bypass.blockSignals(blocked)
        if self._kind == "mix":
            self._hint.setText(
                "This join blends two parallel chains. Wire the dry path to the "
                "D input and the wet path to the W input, then set the mix."
            )
            self._wet_box.show()
            wet = float(params.get("wet", 0.5))
            blocked = self._wet_slider.blockSignals(True)
            self._wet_slider.setValue(int(round(wet * 100)))
            self._wet_slider.blockSignals(blocked)
            self._wet_value.setText(f"{int(round(wet * 100))}%")
            dry_src = params.get("dry_src") or "—"
            wet_src = params.get("wet_src") or "—"
            self._wet_caption.setText(
                f"Dry from {dry_src}  ·  Wet from {wet_src}. "
                "0% is fully dry, 100% is fully the wet path."
            )
        else:
            self._hint.setText("Tweaks apply live. Close this view or press ← Graph to return.")
            self._wet_box.hide()
        self._rebuild_form(params)

    def _clear_form(self) -> None:
        while self._form.rowCount():
            self._form.removeRow(0)
        self._controls = {}

    def _rebuild_form(self, params: dict) -> None:
        self._clear_form()
        specs = [s for s in self._specs if not (self._kind == "mix" and s.key == "wet")]
        for spec in specs:
            ctrl = self._make_control(spec)
            self._controls[spec.key] = ctrl
            self._form.addRow(spec.label, ctrl)
            self._load_one(spec, params)

    def _make_control(self, spec: ParamSpec) -> QWidget:
        if spec.kind == "choice":
            box = QComboBox()
            for c in spec.choices:
                box.addItem(c)
            box.currentTextChanged.connect(
                lambda text, key=spec.key: self._emit(key, text)
            )
            return box
        if spec.kind == "bool":
            box = QCheckBox()
            box.toggled.connect(lambda on, key=spec.key: self._emit(key, bool(on)))
            return box
        spin = QDoubleSpinBox()
        spin.setRange(spec.minimum, spec.maximum)
        spin.setDecimals(spec.decimals)
        spin.setSingleStep(max(10 ** (-spec.decimals), (spec.maximum - spec.minimum) / 200.0))
        if spec.suffix:
            spin.setSuffix(spec.suffix)
        spin.valueChanged.connect(lambda val, key=spec.key: self._emit(key, float(val)))
        return spin

    def _load_one(self, spec: ParamSpec, params: dict) -> None:
        ctrl = self._controls.get(spec.key)
        if ctrl is None:
            return
        val = read_param(params, spec)
        blocked = ctrl.blockSignals(True)
        if isinstance(ctrl, QComboBox):
            idx = ctrl.findText(str(val))
            if idx >= 0:
                ctrl.setCurrentIndex(idx)
        elif isinstance(ctrl, QCheckBox):
            ctrl.setChecked(bool(val))
        elif isinstance(ctrl, QDoubleSpinBox):
            try:
                ctrl.setValue(float(val))
            except (TypeError, ValueError):
                pass
        ctrl.blockSignals(blocked)

    def _emit(self, key: str, value) -> None:
        if self._insert_id:
            self.param_changed.emit(self._insert_id, key, value)

    def _on_wet(self, value: int) -> None:
        self._wet_value.setText(f"{value}%")
        self._emit("wet", value / 100.0)

    def _on_bypass(self, on: bool) -> None:
        if self._insert_id:
            self.bypass_changed.emit(self._insert_id, bool(on))
