"""Blender-style node editor for a track's directed FX graph."""

from __future__ import annotations

from math import ceil
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsProxyWidget,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fantasia_core.document.fx_insert import (
    OUT,
    SOURCE,
    device_label,
    effective_wires,
    insert_bypassed,
    insert_id,
    insert_type,
    is_wired,
    linear_order,
)
from fantasia_core.document.fx_params import (
    SYNTH_PARAM_SPECS,
    ParamSpec,
    read_param,
    specs_for,
)
from fantasia_core.document.model import MASTER_ID, Track
from fantasia_core.engine.levels import fx_meter_key
from fantasia_core.engine.synth import DEFAULT_PATCH
from ui import theme
from ui.param_knob import ParamKnob

NODE_W, NODE_H = 220.0, 72.0
KNOB_CELL_W = 108.0
KNOB_CELL_H = 56.0
PORT_R = 9.0
PORT_HIT = 26.0          # click/drag slop around a port (viewport px)
PORT_OUTSET = 11.0       # sit the circle outside the node so the proxy cannot eat it
METER_W = 6.0
METER_GAP = 5.0          # gap between the I/O meter and the port circle
COL_GAP, ROW_GAP = 268.0, 260.0
HEADER_H = 26.0
# Parameters spill into more columns rather than into a scrollbar. A node may
# grow tall, but it must never hide a control behind a scroll area.
TWO_COL_AT = 4
THREE_COL_AT = 10
FOUR_COL_AT = 20


def _param_columns(n_specs: int) -> int:
    if n_specs >= FOUR_COL_AT:
        return 4
    if n_specs >= THREE_COL_AT:
        return 3
    if n_specs >= TWO_COL_AT:
        return 2
    return 1


def _param_geometry(n_specs: int) -> tuple[int, float, float]:
    """Return (columns, node_width, body_height) for an inline param panel."""
    if n_specs <= 0:
        return 1, NODE_W, 0.0
    cols = _param_columns(n_specs)
    width = max(NODE_W, cols * KNOB_CELL_W + 10)
    rows = ceil(n_specs / cols)
    return cols, width, rows * KNOB_CELL_H


def node_size(n_specs: int, kind: str = "") -> tuple[float, float]:
    """Outer size of a node with ``n_specs`` inline parameters."""
    _, width, body = _param_geometry(n_specs)
    height = HEADER_H + body + 8 if n_specs else NODE_H
    if kind == "mix":
        height = max(height, 96.0)
    return width, height

_PARAM_STYLE = (
    f"QWidget {{ background: transparent; color: {theme.FG}; }}"
    f"QLabel {{ color: {theme.FG_DIM}; font-size: 10px; }}"
    f"QLabel#knobValue {{ color: {theme.CYAN}; font-size: 10px; font-weight: 700; }}"
    f"QComboBox {{"
    f"  background: {theme.BG_PANEL}; color: {theme.FG_BRIGHT};"
    f"  border: 1px solid {theme.BORDER}; border-radius: 2px;"
    f"  padding: 0px 2px; font-size: 10px; min-height: 16px; }}"
    f"QCheckBox {{ color: {theme.FG_BRIGHT}; spacing: 0px; }}"
)


def _instrument_label(track: Track) -> str:
    if track.id == MASTER_ID or getattr(track, "is_master", False):
        return "Mix bus"
    plugin = getattr(track, "plugin", "") or ""
    if plugin:
        return plugin
    if getattr(track, "is_synth", False):
        return "Built-in synth"
    if getattr(track, "is_drum", False):
        return "Drum kit"
    return "Soundfont / clips"


class _ParamPanel(QWidget):
    """Rotary knobs for stock devices. Never scrolls."""

    changed = Signal(str, object)  # key, value

    def __init__(self, specs: tuple[ParamSpec, ...], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._specs = specs
        self._controls: dict[str, QWidget] = {}
        self._value_labels: dict[str, QLabel] = {}
        self.setStyleSheet(_PARAM_STYLE)
        cols, width, body_h = _param_geometry(len(specs))
        self._body_h = body_h
        self._node_w = width
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        body = QWidget()
        grid = QGridLayout(body)
        grid.setContentsMargins(4, 2, 4, 4)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(2)
        rows = max(ceil(len(specs) / cols), 1)
        for i, spec in enumerate(specs):
            r, c = i % rows, i // rows
            grid.addWidget(self._make_cell(spec), r, c)
        outer.addWidget(body)
        self.setFixedWidth(int(width) - 8)
        self.setFixedHeight(int(max(body_h, KNOB_CELL_H)))

    def _make_cell(self, spec: ParamSpec) -> QWidget:
        cell = QWidget()
        cell.setFixedSize(int(KNOB_CELL_W) - 4, int(KNOB_CELL_H) - 2)
        col = QVBoxLayout(cell)
        col.setContentsMargins(2, 0, 2, 0)
        col.setSpacing(0)
        name = QLabel(spec.label)
        name.setAlignment(Qt.AlignLeft)
        col.addWidget(name)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        ctrl = self._make_control(spec)
        h.addWidget(ctrl, 0, Qt.AlignVCenter)
        if spec.kind == "float":
            val = QLabel(ctrl.format_value() if isinstance(ctrl, ParamKnob) else "")
            val.setObjectName("knobValue")
            val.setMinimumWidth(44)
            h.addWidget(val, 1, Qt.AlignVCenter)
            self._value_labels[spec.key] = val
        else:
            h.addStretch(1)
        col.addWidget(row)
        self._controls[spec.key] = ctrl
        return cell

    def _make_control(self, spec: ParamSpec) -> QWidget:
        if spec.kind == "choice":
            box = QComboBox()
            for c in spec.choices:
                box.addItem(c)
            box.setFocusPolicy(Qt.ClickFocus)
            box.currentTextChanged.connect(
                lambda text, key=spec.key: self.changed.emit(key, text)
            )
            return box
        if spec.kind == "bool":
            box = QCheckBox()
            box.setFocusPolicy(Qt.ClickFocus)
            box.toggled.connect(lambda on, key=spec.key: self.changed.emit(key, bool(on)))
            return box
        knob = ParamKnob(spec)
        knob.changed.connect(lambda val, key=spec.key: self._on_knob(key, val))
        return knob

    def _on_knob(self, key: str, value: float) -> None:
        label = self._value_labels.get(key)
        ctrl = self._controls.get(key)
        if label is not None and isinstance(ctrl, ParamKnob):
            label.setText(ctrl.format_value())
        self.changed.emit(key, float(value))

    def load_values(self, params: dict) -> None:
        for spec in self._specs:
            ctrl = self._controls.get(spec.key)
            if ctrl is None:
                continue
            val = read_param(params, spec)
            blocked = ctrl.blockSignals(True)
            if isinstance(ctrl, QComboBox):
                idx = ctrl.findText(str(val))
                if idx >= 0:
                    ctrl.setCurrentIndex(idx)
            elif isinstance(ctrl, QCheckBox):
                ctrl.setChecked(bool(val))
            elif isinstance(ctrl, ParamKnob):
                try:
                    ctrl.set_value(float(val))
                except (TypeError, ValueError):
                    pass
                label = self._value_labels.get(spec.key)
                if label is not None:
                    label.setText(ctrl.format_value())
            ctrl.blockSignals(blocked)


class _Port(QGraphicsEllipseItem):
    def __init__(self, node: "_Node", incoming: bool, role: str = "") -> None:
        super().__init__(-PORT_R, -PORT_R, PORT_R * 2, PORT_R * 2)
        self.node = node
        self.incoming = incoming
        self.role = role or ("in" if incoming else "out")
        if self.role == "dry":
            color = QColor(theme.GREEN)
        elif self.role == "wet":
            color = QColor(theme.MAGENTA)
        else:
            color = QColor(theme.CYAN if not incoming else theme.MAGENTA)
        self._base = color
        self.setBrush(color)
        self.setPen(QPen(QColor(theme.FG_BRIGHT), 1.2))
        self.setZValue(20)
        self.setCursor(Qt.PointingHandCursor)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIgnoresParentOpacity, True)
        if self.role == "out":
            self.setToolTip("Drag to an input · drag again to fork a parallel chain")
        elif self.role == "dry":
            self.setToolTip("Dry input — drop a cable here")
        elif self.role == "wet":
            self.setToolTip("Wet input — drop the parallel chain here")
        else:
            self.setToolTip("Input — drop a cable here")

    def set_hot(self, on: bool) -> None:
        """Legacy two-state highlight; kept for callers that only know on/off."""
        self.set_state("snap" if on else "idle")

    def set_state(self, state: str) -> None:
        """``idle`` | ``candidate`` (a legal drop) | ``snap`` (will connect)."""
        if state == "snap":
            self.setBrush(QColor(theme.YELLOW))
            self.setPen(QPen(QColor(theme.FG_BRIGHT), 2.4))
            self.setRect(-PORT_R - 3, -PORT_R - 3, (PORT_R + 3) * 2, (PORT_R + 3) * 2)
            return
        if state == "candidate":
            self.setBrush(self._base.lighter(135))
            self.setPen(QPen(QColor(theme.YELLOW), 1.8))
            self.setRect(-PORT_R, -PORT_R, PORT_R * 2, PORT_R * 2)
            return
        self.setBrush(self._base)
        self.setPen(QPen(QColor(theme.FG_BRIGHT), 1.2))
        self.setRect(-PORT_R, -PORT_R, PORT_R * 2, PORT_R * 2)

    def hoverEnterEvent(self, event) -> None:  # noqa: N802
        self.setRect(-PORT_R - 2, -PORT_R - 2, (PORT_R + 2) * 2, (PORT_R + 2) * 2)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802
        self.setRect(-PORT_R, -PORT_R, PORT_R * 2, PORT_R * 2)
        super().hoverLeaveEvent(event)


class _IoMeter(QGraphicsRectItem):
    """Vertical peak bar beside a node port. Peak-hold only; no extra DSP."""

    def __init__(self, node: "_Node", side: str, y: float | None = None) -> None:
        height = max(28.0, node.rect().height() - 12.0)
        super().__init__(0, 0, METER_W, height)
        self.node = node
        self.side = side
        self._amp = 0.0
        self._held = 0.0
        self.setParentItem(node)
        self.setZValue(18)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setFlag(QGraphicsItem.ItemIgnoresParentOpacity, True)
        if side == "in":
            self.setPos(-PORT_OUTSET - METER_GAP - METER_W, 6.0 if y is None else y)
        else:
            self.setPos(node.rect().width() + PORT_OUTSET + METER_GAP,
                        6.0 if y is None else y)
        self.setToolTip("Input level" if side == "in" else "Output level")

    def set_amp(self, amp: float, playing: bool) -> None:
        if not playing:
            if self._amp or self._held:
                self._amp = 0.0
                self._held = 0.0
                self.update()
            return
        self._amp = max(float(amp), self._amp * 0.72)
        self._held = max(self._held, float(amp))
        self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: N802, ARG002
        painter.setRenderHint(QPainter.Antialiasing, False)
        r = self.rect()
        painter.fillRect(r, QColor(theme.BG_DEEP))
        painter.setPen(QPen(QColor(theme.BORDER), 1))
        painter.drawRect(r)
        inner = r.adjusted(1, 1, -1, -1)
        level = max(0.0, min(1.0, self._amp))
        if level > 0:
            fill = QColor(theme.CYAN if level < 0.89 else theme.NEON_ORANGE)
            if level >= 0.99:
                fill = QColor(theme.RED)
            h = max(1.0, inner.height() * level)
            painter.fillRect(
                inner.x(), inner.bottom() - h, inner.width(), h, fill)
        if self._held > 0.02:
            y = inner.bottom() - inner.height() * min(1.0, self._held)
            painter.fillRect(inner.x(), y, inner.width(), 2, QColor(theme.FG_BRIGHT))


class _BypassChip(QGraphicsRectItem):
    """Header control: click to bypass / enable this insert."""

    def __init__(self, node: "_Node") -> None:
        super().__init__(0, 0, 18, 16)
        self.node = node
        self.setParentItem(node)
        self.setZValue(22)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setCursor(Qt.PointingHandCursor)
        self.setFlag(QGraphicsItem.ItemIgnoresParentOpacity, True)
        self.setToolTip("Bypass this effect (0)")
        self.setPos(node.rect().width() - 24, 5)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: N802, ARG002
        painter.setRenderHint(QPainter.Antialiasing, True)
        on = self.node.bypassed
        painter.setBrush(QColor(theme.YELLOW if on else theme.BG_DEEP))
        painter.setPen(QPen(QColor(theme.FG_BRIGHT if on else theme.BORDER), 1.0))
        painter.drawRoundedRect(self.rect(), 3, 3)
        painter.setPen(QColor(theme.BG_DEEP if on else theme.FG_DIM))
        painter.setFont(theme.ui_font(7, bold=True))
        painter.drawText(self.rect(), Qt.AlignCenter, "B")


class _Node(QGraphicsRectItem):
    def __init__(self, nid: str, title: str, subtitle: str, kind: str,
                 specs: tuple[ParamSpec, ...] = (), wired: bool = True,
                 bypassed: bool = False) -> None:
        width, height = node_size(len(specs), kind)
        super().__init__(0, 0, width, height)
        self.nid = nid
        self.kind = kind
        self.wired = wired
        self.bypassed = bool(bypassed)
        self._panel: Optional[_ParamPanel] = None
        self._chip: Optional[_BypassChip] = None
        self.in_meters: list[_IoMeter] = []
        self.out_meter: Optional[_IoMeter] = None
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setZValue(2)
        self.setPen(QPen(QColor(theme.BORDER), 1.4))
        self.setBrush(QColor(theme.BG_ELEVATED))
        self._title = title
        self._subtitle = subtitle if wired else "Not wired — drag a port"
        self.in_ports: list[_Port] = []
        if nid == SOURCE:
            self.in_port = None
        elif kind == "mix":
            dry = _Port(self, True, "dry")
            wet = _Port(self, True, "wet")
            dry.setParentItem(self)
            wet.setParentItem(self)
            dry.setPos(-PORT_OUTSET, HEADER_H + 14)
            wet.setPos(-PORT_OUTSET, height - 16)
            self.in_ports = [dry, wet]
            self.in_port = dry
        elif nid == OUT:
            self.in_port = _Port(self, True)
            self.in_port.setParentItem(self)
            self.in_port.setPos(-PORT_OUTSET, height / 2)
            self.in_ports = [self.in_port]
        else:
            self.in_port = _Port(self, True)
            self.in_port.setParentItem(self)
            self.in_port.setPos(-PORT_OUTSET, height / 2)
            self.in_ports = [self.in_port]
        if nid == OUT:
            self.out_port = None
        else:
            self.out_port = _Port(self, False)
            self.out_port.setParentItem(self)
            self.out_port.setPos(width + PORT_OUTSET, height / 2)
        if specs:
            panel = _ParamPanel(specs)
            proxy = QGraphicsProxyWidget(self)
            proxy.setWidget(panel)
            proxy.setPos(4, HEADER_H + 2)
            # Knobs must not steal clicks meant for the ports on the rim.
            proxy.setZValue(0)
            self._panel = panel
        if nid not in (SOURCE, OUT):
            self._chip = _BypassChip(self)
        self._add_io_meters()
        self.set_bypassed(self.bypassed, quiet=True)

    def _add_io_meters(self) -> None:
        if self.nid != SOURCE:
            if self.kind == "mix" and len(self.in_ports) >= 2:
                h = max(22.0, (self.rect().height() - HEADER_H) / 2.0 - 8.0)
                dry = _IoMeter(self, "in", HEADER_H + 2)
                dry.setRect(0, 0, METER_W, h)
                wet = _IoMeter(self, "in", self.rect().height() - h - 6)
                wet.setRect(0, 0, METER_W, h)
                self.in_meters = [dry, wet]
            else:
                self.in_meters = [_IoMeter(self, "in")]
        if self.nid != OUT:
            self.out_meter = _IoMeter(self, "out")

    def set_meters(self, incoming: float, outgoing: float, playing: bool) -> None:
        for meter in self.in_meters:
            meter.set_amp(incoming, playing)
        if self.out_meter is not None:
            self.out_meter.set_amp(outgoing, playing)

    def set_bypassed(self, on: bool, quiet: bool = False) -> None:  # noqa: ARG002
        self.bypassed = bool(on)
        self.setOpacity(0.48 if self.bypassed else 1.0)
        if self._chip is not None:
            self._chip.update()
        self.update()

    def header_color(self) -> QColor:
        if self.nid == SOURCE:
            return QColor(theme.PURPLE)
        if self.nid == OUT:
            return QColor(theme.YELLOW)
        if self.kind == "eq":
            return QColor(theme.CYAN)
        if self.kind == "mix":
            return QColor(theme.GREEN)
        if self.kind == "vst":
            return QColor(theme.ORANGE)
        return QColor(theme.ACCENT)

    def incoming_ports(self) -> list[_Port]:
        return list(self.in_ports)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: N802
        painter.setRenderHint(QPainter.Antialiasing, True)
        r = self.rect()
        border = theme.YELLOW if not self.wired and self.nid not in (SOURCE, OUT) else (
            theme.CYAN if self.isSelected() else theme.BORDER
        )
        painter.setPen(QPen(
            QColor(border),
            2 if self.isSelected() or not self.wired else 1.2,
            Qt.DashLine if not self.wired and self.nid not in (SOURCE, OUT) else Qt.SolidLine,
        ))
        painter.setBrush(QColor(theme.BG_ELEVATED))
        painter.drawRoundedRect(r, 8, 8)
        head = QRectF(r.x(), r.y(), r.width(), HEADER_H)
        painter.setBrush(self.header_color())
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(head, 8, 8)
        painter.fillRect(QRectF(r.x(), r.y() + 12, r.width(), 10), self.header_color())
        painter.setPen(QColor(theme.BG_DEEP))
        painter.setFont(theme.ui_font(9, bold=True))
        title_box = head.adjusted(8, 0, -28 if self._chip is not None else -8, 0)
        painter.drawText(title_box, Qt.AlignVCenter | Qt.AlignLeft, self._title)
        if self.bypassed:
            painter.setPen(QColor(theme.BG_DEEP))
            painter.setFont(theme.ui_font(7, bold=True))
            painter.drawText(head.adjusted(8, 0, -28, 0),
                             Qt.AlignVCenter | Qt.AlignRight, "BYPASS")
        if self.kind == "mix":
            painter.setPen(QColor(theme.GREEN))
            painter.setFont(theme.ui_font(7, bold=True))
            painter.drawText(QRectF(r.x() + 10, HEADER_H + 4, 24, 14), Qt.AlignLeft, "D")
            painter.setPen(QColor(theme.MAGENTA))
            painter.drawText(QRectF(r.x() + 10, r.height() - 22, 24, 14), Qt.AlignLeft, "W")
        if self._panel is None:
            painter.setPen(QColor(theme.FG_DIM))
            painter.setFont(theme.ui_font(8))
            painter.drawText(QRectF(r.x() + 8, r.y() + 26, r.width() - 16, 30),
                             Qt.AlignLeft | Qt.AlignTop, self._subtitle)

    def out_scene(self) -> QPointF:
        if self.out_port is None:
            return self.scenePos() + QPointF(self.rect().width(), self.rect().height() / 2)
        return self.out_port.scenePos()

    def in_scene(self, role: str = "") -> QPointF:
        if role:
            for port in self.in_ports:
                if port.role == role:
                    return port.scenePos()
        if self.in_ports:
            return self.in_ports[0].scenePos()
        if self.in_port is None:
            return self.scenePos() + QPointF(0, self.rect().height() / 2)
        return self.in_port.scenePos()

    def itemChange(self, change, value):  # noqa: N802
        if change == QGraphicsItem.ItemPositionHasChanged and self.scene():
            view = self.scene().views()
            if view and hasattr(view[0], "_relayout_wires"):
                view[0]._relayout_wires()
        return super().itemChange(change, value)


class _WireItem(QGraphicsPathItem):
    def __init__(self, src: str, dst: str, dst_role: str = "") -> None:
        super().__init__()
        self.src = src
        self.dst = dst
        self.dst_role = dst_role
        self._hot = False
        self.setZValue(1)
        self.setPen(QPen(QColor(theme.CYAN), 2.2))
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)

    def set_ends(self, a: QPointF, b: QPointF) -> None:
        path = QPainterPath(a)
        dx = max(40.0, abs(b.x() - a.x()) * 0.45)
        path.cubicTo(QPointF(a.x() + dx, a.y()), QPointF(b.x() - dx, b.y()), b)
        self.setPath(path)

    def shape(self):
        stroker = QPainterPathStroker()
        stroker.setWidth(14)
        return stroker.createStroke(self.path())

    def paint(self, painter, option, widget=None) -> None:  # noqa: N802, ARG002
        painter.setRenderHint(QPainter.Antialiasing, True)
        hot = getattr(self, "_hot", False) or self.isSelected()
        color = QColor(theme.YELLOW if hot else theme.CYAN)
        painter.setPen(QPen(color, 3.4 if hot else 2.2))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(self.path())

    def set_hot(self, on: bool) -> None:
        self._hot = bool(on)
        self.update()


class FxGraphView(QGraphicsView):
    """Directed flowchart of the selected track's instrument + FX."""

    add_requested = Signal()
    remove_requested = Signal(str)
    connect_requested = Signal(str, str, str)  # src, dst, port
    disconnect_requested = Signal(str, str)
    splice_requested = Signal(str, str, str)  # node, edge_src, edge_dst
    bypass_requested = Signal(str, bool)
    device_activated = Signal(str, str)
    param_changed = Signal(str, str, object)
    position_changed = Signal(str, float, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._track: Optional[Track] = None
        self._nodes: Dict[str, _Node] = {}
        self._wires: List[_WireItem] = []
        self._drag_src: Optional[str] = None
        self._temp_wire: Optional[_WireItem] = None
        self._press_pos: Dict[str, QPointF] = {}
        self._last_sig = None
        self._hot_cable: Optional[_WireItem] = None
        self._pan_anchor = None
        self._pan_h = 0
        self._pan_v = 0
        self._scene = QGraphicsScene(self)
        self._scene.setBackgroundBrush(QColor(theme.TIMELINE_BG))
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        # RubberBandDrag steals the press that should start a cable.
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setFocusPolicy(Qt.StrongFocus)
        # Delete is deliberately *not* a QShortcut here: the main window owns a
        # window-level Delete QAction, and two shortcuts for one key race. The
        # window routes Delete here via has_selection()/delete_selection().
        home = QShortcut(QKeySequence(Qt.Key_Home), self)
        home.setContext(Qt.WidgetShortcut)
        home.activated.connect(self.fit_all)

    def set_track(self, track: Optional[Track]) -> None:
        """Refresh the scene for ``track``.

        Commands mutate the same Track instance the view already holds, so we
        must compare against a *cached* signature — recomputing from
        ``self._track`` after the mutation would always look unchanged and
        skip the rebuild (new FX invisible until you switch tracks).
        """
        sig = self._structure_sig(track) if track is not None else None
        live = set()
        if track is not None:
            live = {insert_id(e) for e in (track.fx or []) if insert_id(e)}
            live.update({SOURCE, OUT})
        same = (
            track is not None
            and sig == self._last_sig
            and set(self._nodes) >= {SOURCE, OUT}
            and live <= set(self._nodes)
        )
        self._track = track
        self._last_sig = sig
        if same:
            self._sync_params()
            return
        self._rebuild()

    def _structure_sig(self, track: Track) -> tuple:
        fx = tuple(
            (insert_id(e), insert_type(e), insert_bypassed(e))
            for e in (track.fx or [])
        )
        wires = tuple(sorted(
            (w.src, w.dst)
            for w in effective_wires(track.fx, getattr(track, "fx_wires", None))
        ))
        src = (
            bool(getattr(track, "is_synth", False)),
            bool(getattr(track, "is_drum", False)),
            str(getattr(track, "plugin", "") or ""),
            bool(getattr(track, "is_master", False)),
        )
        return (track.id, fx, wires, src)

    def _sync_params(self) -> None:
        track = self._track
        if track is None:
            return
        src = self._nodes.get(SOURCE)
        if src is not None and src._panel is not None:
            src._panel.load_values({**DEFAULT_PATCH, **(getattr(track, "synth", None) or {})})
        by_id = {insert_id(e): e for e in track.fx if insert_id(e)}
        for nid, node in self._nodes.items():
            if node._panel is None or nid in (SOURCE, OUT):
                continue
            spec = by_id.get(nid)
            if spec is None:
                continue
            node._panel.load_values(getattr(spec, "params", None) or {})
            node.set_bypassed(insert_bypassed(spec))
        self._sync_positions()

    def _sync_positions(self) -> None:
        """Re-apply stored node positions.

        The structure signature ignores x/y, so a move (or an *undo* of one)
        never triggers a rebuild. Without this, undoing "Move FX node" changed
        the document but left the node where the user dropped it.
        """
        track = self._track
        if track is None or self._drag_src is not None:
            return
        by_id = {insert_id(e): e for e in track.fx if insert_id(e)}
        moved = False
        for nid, node in self._nodes.items():
            if nid in (SOURCE, OUT):
                xy = (getattr(track, "fx_graph_pos", None) or {}).get(nid)
                if not isinstance(xy, (list, tuple)) or len(xy) < 2:
                    continue
                x, y = float(xy[0]), float(xy[1])
            else:
                spec = by_id.get(nid)
                if spec is None:
                    continue
                x = float(getattr(spec, "x", 0.0) or 0.0)
                y = float(getattr(spec, "y", 0.0) or 0.0)
                if not x and not y:
                    continue
            if abs(node.pos().x() - x) > 0.5 or abs(node.pos().y() - y) > 0.5:
                node.setPos(x, y)
                moved = True
        if moved:
            self._relayout_wires()

    def _rebuild(self) -> None:
        # clear() destroys the items a live drag points at.
        self._cancel_wire(rebuilding=True)
        self._hot_cable = None
        self._scene.clear()
        self._nodes = {}
        self._wires = []
        self._press_pos = {}
        track = self._track
        if track is None:
            self._scene.addText("Select a track (or a clip) to edit its signal graph.")
            return

        positions = self._layout(track)
        source_sub = _instrument_label(track)
        src_specs = (
            SYNTH_PARAM_SPECS
            if getattr(track, "is_synth", False) and not getattr(track, "plugin", "")
            and not getattr(track, "is_master", False)
            else ()
        )
        saved_io = getattr(track, "fx_graph_pos", None) or {}
        src_node = self._add_node(SOURCE, "Source", source_sub, "instrument",
                                  positions[SOURCE], src_specs)
        self._apply_saved_pos(src_node, saved_io.get(SOURCE))
        if src_node._panel is not None:
            src_node._panel.load_values({**DEFAULT_PATCH, **(getattr(track, "synth", None) or {})})
            src_node._panel.changed.connect(
                lambda key, val: self.param_changed.emit(SOURCE, key, val)
            )
        wires = effective_wires(track.fx, getattr(track, "fx_wires", None))
        order = list(linear_order(track.fx, getattr(track, "fx_wires", None)))
        by_id = {insert_id(e): e for e in track.fx if insert_id(e)}
        for nid in by_id:
            if nid not in order:
                order.append(nid)
        for nid in order:
            spec = by_id.get(nid)
            if spec is None:
                continue
            kind = insert_type(spec)
            params = getattr(spec, "params", None) or {}
            specs = specs_for(kind, params)
            bypassed = insert_bypassed(spec)
            sub = "bypassed" if bypassed else kind
            pos = positions.get(nid, (COL_GAP, ROW_GAP))
            wired = is_wired(nid, wires)
            node = self._add_node(
                nid, device_label(spec), sub, kind, pos, specs, wired, bypassed)
            if node._panel is not None:
                node._panel.load_values(params)
                node._panel.changed.connect(
                    lambda key, val, ident=nid: self.param_changed.emit(ident, key, val)
                )
            x = getattr(spec, "x", 0.0) or 0.0
            y = getattr(spec, "y", 0.0) or 0.0
            if x or y:
                node.setPos(x, y)
        out_sub = "Master mix" if getattr(track, "is_master", False) else "Fader → mix"
        out_node = self._add_node(OUT, "Out", out_sub, "out", positions[OUT], wired=True)
        self._apply_saved_pos(out_node, saved_io.get(OUT))

        for w in wires:
            dest = by_id.get(w.dst)
            role = ""
            if dest is not None and insert_type(dest) == "mix":
                params = getattr(dest, "params", None) or {}
                if params.get("dry_src") == w.src:
                    role = "dry"
                elif params.get("wet_src") == w.src:
                    role = "wet"
            self._add_wire(w.src, w.dst, role)
        self._relayout_wires()
        self._frame_scene()
        self.viewport().update()

    def _layout(self, track: Track) -> Dict[str, Tuple[float, float]]:
        wires = effective_wires(track.fx, getattr(track, "fx_wires", None))
        order = [SOURCE] + linear_order(track.fx, getattr(track, "fx_wires", None)) + [OUT]
        sizes = self._node_sizes(track)
        pos: Dict[str, Tuple[float, float]] = {}
        floating: list[str] = []
        # Columns are as wide as the widest node in them, so a 3-column EQ node
        # cannot overlap its neighbour.
        col_of: Dict[str, int] = {}
        col = 0
        for nid in order:
            if nid not in (SOURCE, OUT) and not is_wired(nid, wires):
                floating.append(nid)
                continue
            col_of[nid] = col
            col += 1
        col_w: Dict[int, float] = {}
        for nid, c in col_of.items():
            col_w[c] = max(col_w.get(c, NODE_W), sizes.get(nid, (NODE_W, NODE_H))[0])
        col_x: Dict[int, float] = {}
        x = 40.0
        for c in range(col):
            col_x[c] = x
            x += col_w.get(c, NODE_W) + (COL_GAP - NODE_W)
        indeg: Dict[str, int] = {}
        for w in wires:
            indeg[w.dst] = indeg.get(w.dst, 0) + 1
        used_y: Dict[int, int] = {}
        bottom = 80.0
        for nid, c in col_of.items():
            extra = max(0, indeg.get(nid, 1) - 1)
            slot = used_y.get(c, 0) + extra
            used_y[c] = slot + 1
            y = 80.0 + slot * ROW_GAP
            pos[nid] = (col_x[c], y)
            bottom = max(bottom, y + sizes.get(nid, (NODE_W, NODE_H))[1])
        # Park unwired nodes on a tray *below* the tallest wired node, so a new
        # insert never lands on top of the chain (insert x/y overrides this).
        tray_x = 40.0
        for nid in floating:
            pos[nid] = (tray_x, bottom + 48.0)
            tray_x += sizes.get(nid, (NODE_W, NODE_H))[0] + 28.0
        return pos

    @staticmethod
    def _node_sizes(track: Track) -> Dict[str, Tuple[float, float]]:
        sizes: Dict[str, Tuple[float, float]] = {
            SOURCE: node_size(0),
            OUT: node_size(0),
        }
        if (
            getattr(track, "is_synth", False)
            and not getattr(track, "plugin", "")
            and not getattr(track, "is_master", False)
        ):
            sizes[SOURCE] = node_size(len(SYNTH_PARAM_SPECS))
        for spec in track.fx or []:
            nid = insert_id(spec)
            if not nid:
                continue
            kind = insert_type(spec)
            n = len(specs_for(kind, getattr(spec, "params", None) or {}))
            sizes[nid] = node_size(n, kind)
        return sizes

    @staticmethod
    def _apply_saved_pos(node: _Node, xy) -> None:  # noqa: ANN001
        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            return
        try:
            x, y = float(xy[0]), float(xy[1])
        except (TypeError, ValueError):
            return
        if x or y:
            node.setPos(x, y)

    def _add_node(self, nid, title, subtitle, kind, pos, specs=(), wired=True,  # noqa: ANN001
                  bypassed=False) -> _Node:
        node = _Node(nid, title, subtitle, kind, specs, wired=wired, bypassed=bypassed)
        node.setPos(pos[0], pos[1])
        self._scene.addItem(node)
        self._nodes[nid] = node
        return node

    def _add_wire(self, src: str, dst: str, dst_role: str = "") -> None:
        item = _WireItem(src, dst, dst_role)
        self._scene.addItem(item)
        self._wires.append(item)

    def _relayout_wires(self) -> None:
        for item in self._wires:
            a = self._nodes.get(item.src)
            b = self._nodes.get(item.dst)
            if a is None or b is None:
                continue
            item.set_ends(a.out_scene(), b.in_scene(item.dst_role))

    def _set_ports_hot(self, on: bool) -> None:
        for node in self._nodes.values():
            for port in node.incoming_ports():
                port.set_hot(on)

    def _iter_ports(self) -> list[_Port]:
        ports: list[_Port] = []
        for node in self._nodes.values():
            ports.extend(node.incoming_ports())
            if node.out_port is not None:
                ports.append(node.out_port)
        return ports

    def _event_pos(self, event) -> object:  # noqa: ANN001
        if hasattr(event, "position"):
            return event.position().toPoint()
        return event.pos()

    def _port_at(self, view_pos, incoming: Optional[bool] = None) -> Optional[_Port]:  # noqa: ANN001
        """Nearest port within ``PORT_HIT`` viewport pixels. Ignores stacked widgets."""
        px = float(view_pos.x())
        py = float(view_pos.y())
        best: Optional[_Port] = None
        best_d = PORT_HIT + 1.0
        for port in self._iter_ports():
            if incoming is not None and port.incoming != incoming:
                continue
            mapped = self.mapFromScene(port.scenePos())
            dx = float(mapped.x()) - px
            dy = float(mapped.y()) - py
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_d:
                best_d = dist
                best = port
        return best

    def _hit_target(self, pos) -> Tuple[Optional[str], str]:  # noqa: ANN001
        port = self._port_at(pos, incoming=True)
        if port is not None:
            role = port.role if port.role in ("dry", "wet") else ""
            return port.node.nid, role
        scene_pt = self.mapToScene(pos)
        for item in self._scene.items(scene_pt):
            if isinstance(item, _Node) and item.nid != SOURCE:
                return item.nid, ""
        return None, ""

    def _begin_wire(self, src: str) -> None:
        self._cancel_wire()
        self._drag_src = src
        self._temp_wire = _WireItem(src, "")
        self._temp_wire.setPen(QPen(QColor(theme.MAGENTA), 2, Qt.DashLine))
        self._temp_wire.setZValue(15)
        self._scene.addItem(self._temp_wire)
        # Every legal landing spot lights up; the one under the cursor is hotter.
        for port in self._candidate_ports(src):
            port.set_state("candidate")
        self.setFocus(Qt.MouseFocusReason)

    def _candidate_ports(self, src: str) -> list[_Port]:
        out: list[_Port] = []
        for nid, node in self._nodes.items():
            if nid == src or nid == SOURCE:
                continue
            out.extend(node.incoming_ports())
        return out

    def _cancel_wire(self, rebuilding: bool = False) -> None:
        """Drop any in-flight cable and restore port styling. Always safe."""
        wire, self._temp_wire = self._temp_wire, None
        self._drag_src = None
        if wire is not None and not rebuilding:
            try:
                self._scene.removeItem(wire)
            except RuntimeError:  # already destroyed by a rebuild
                pass
        if not rebuilding:
            self._set_ports_hot(False)

    def hideEvent(self, event) -> None:  # noqa: N802
        # Switching tabs mid-drag must not leave a dangling cable.
        self._cancel_wire()
        super().hideEvent(event)

    def has_selection(self) -> bool:
        nids, cables = self._selection_targets()
        return bool(nids or cables)

    def _selection_targets(self) -> Tuple[list, list]:
        """(insert ids, (src, dst) pairs) that a Delete would act on."""
        nids: list[str] = []
        cables: list[Tuple[str, str]] = []
        for item in self._scene.selectedItems():
            if isinstance(item, _Node) and item.nid not in (SOURCE, OUT):
                nids.append(item.nid)
            elif isinstance(item, _WireItem):
                cables.append((item.src, item.dst))
        return nids, cables

    def is_editing_param(self) -> bool:
        """True when a knob inside a node currently has keyboard focus.

        Node parameter panels live in a ``QGraphicsProxyWidget``, so they are
        top-level widgets — ``isAncestorOf`` never sees them.
        """
        w = QApplication.focusWidget()
        while w is not None:
            if isinstance(w, _ParamPanel):
                return True
            w = w.parentWidget()
        return False

    def has_node(self, nid: str) -> bool:
        return nid in self._nodes

    def set_meters(self, peaks: dict, playing: bool, track_id: str = "") -> None:  # noqa: ANN001
        """Paint I/O peaks already computed in the audio callback."""
        if not track_id:
            return
        for nid, node in self._nodes.items():
            node.set_meters(
                float(peaks.get(fx_meter_key(track_id, nid, "in"), 0.0)),
                float(peaks.get(fx_meter_key(track_id, nid, "out"), 0.0)),
                playing,
            )

    def selected_insert_ids(self) -> list[str]:
        return [
            it.nid for it in self._scene.selectedItems()
            if isinstance(it, _Node) and it.nid not in (SOURCE, OUT)
        ]

    def toggle_selected_bypass(self) -> bool:
        """Bypass every selected insert, or re-enable them if all are bypassed."""
        nids = self.selected_insert_ids()
        if not nids:
            return False
        nodes = [self._nodes[n] for n in nids if n in self._nodes]
        turn_on = not all(n.bypassed for n in nodes)
        for node in nodes:
            self.bypass_requested.emit(node.nid, turn_on)
        return True

    def delete_selection(self) -> bool:
        """Remove selected wires / nodes. True when something was deleted.

        Targets are resolved before anything is emitted: each signal triggers a
        command that rebuilds the scene and destroys the very items we would
        otherwise still be iterating.
        """
        if self.is_editing_param():
            return False
        nids, cables = self._selection_targets()
        if not nids and not cables:
            return False
        for src, dst in cables:
            self.disconnect_requested.emit(src, dst)
        for nid in nids:
            self.remove_requested.emit(nid)
        return True

    def spawn_scene_pos(self, width: float = NODE_W, height: float = NODE_H) -> QPointF:
        """Top-left for a new node, chosen so the whole node lands on screen."""
        vr = self.viewport().rect()
        if vr.width() < 16 or vr.height() < 16:
            return QPointF(60.0, 110.0)
        vis = self.mapToScene(vr).boundingRect()
        margin = 24.0
        left = vis.left() + margin
        top = vis.top() + margin
        # Keep the node inside the viewport even when it is a tall 3-column EQ.
        right = max(left, vis.right() - width - margin)
        bottom = max(top, vis.bottom() - height - margin)
        x, y = left, min(max(vis.center().y() - height / 2.0, top), bottom)
        taken = [n.sceneBoundingRect() for n in self._nodes.values()]
        for _ in range(24):
            rect = QRectF(x, y, width, height)
            if not any(rect.intersects(t) for t in taken):
                break
            x += 36.0
            y += 28.0
            if x > right or y > bottom:
                x, y = left, top
                break
        return QPointF(min(x, right), min(y, bottom))

    def reveal_node(self, nid: str) -> None:
        node = self._nodes.get(nid)
        if node is None:
            return
        self._scene.clearSelection()
        node.setSelected(True)
        self.ensureVisible(node, 72, 72)
        self.setFocus(Qt.OtherFocusReason)

    def fit_all(self) -> None:
        if not self._nodes:
            return
        rect = self._scene.itemsBoundingRect().adjusted(-48, -48, 48, 48)
        if rect.isEmpty():
            return
        self._scene.setSceneRect(rect.adjusted(-40, -40, 40, 40))
        self.fitInView(rect, Qt.KeepAspectRatio)
        if self.transform().m11() > 1.15:
            self.resetTransform()
            self.centerOn(rect.center())

    def _frame_scene(self) -> None:
        br = self._scene.itemsBoundingRect().adjusted(-80, -80, 120, 120)
        self._scene.setSceneRect(br)
        vis = self.mapToScene(self.viewport().rect()).boundingRect()
        if vis.width() < 8 or vis.height() < 8:
            return
        if not any(vis.intersects(n.sceneBoundingRect()) for n in self._nodes.values()):
            self.fit_all()

    def focus_node(self, nid: str) -> None:
        self.reveal_node(nid)

    def _wire_at(self, view_pos, ignore_nids=()) -> Optional[_WireItem]:  # noqa: ANN001
        """Cable under ``view_pos``, including those sitting under a node."""
        pt = self.mapToScene(view_pos)
        ignore = set(ignore_nids)
        for item in self._scene.items(pt):
            if not isinstance(item, _WireItem) or item is self._temp_wire:
                continue
            if item.src in ignore or item.dst in ignore:
                continue
            return item
        return None

    def _set_hot_cable(self, cable: Optional[_WireItem]) -> None:
        if self._hot_cable is cable:
            return
        if self._hot_cable is not None:
            try:
                self._hot_cable.set_hot(False)
            except RuntimeError:
                pass
        self._hot_cable = cable
        if cable is not None:
            cable.set_hot(True)

    def _drop_splice_target(self, view_pos) -> Optional[Tuple[str, str, str]]:
        """If a dragged insert was dropped on a cable, return (nid, src, dst)."""
        moved = [
            nid for nid, old in self._press_pos.items()
            if nid not in (SOURCE, OUT)
            and self._nodes.get(nid) is not None
            and (self._nodes[nid].pos() - old).manhattanLength() > 8
        ]
        if not moved:
            return None
        cable = self._wire_at(view_pos, ignore_nids=moved)
        if cable is None:
            return None
        return moved[0], cable.src, cable.dst

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.setFocus(Qt.MouseFocusReason)
        pos = self._event_pos(event)
        if event.button() == Qt.MiddleButton:
            self._pan_anchor = pos
            self._pan_h = self.horizontalScrollBar().value()
            self._pan_v = self.verticalScrollBar().value()
            event.accept()
            return
        item = self.itemAt(pos)
        if isinstance(item, _BypassChip) and event.button() == Qt.LeftButton:
            self.bypass_requested.emit(item.node.nid, not item.node.bypassed)
            event.accept()
            return
        hit = item.node if isinstance(item, (_Port, _BypassChip)) else item
        if isinstance(hit, _Node) and event.button() == Qt.RightButton:
            self._node_menu(hit)
            event.accept()
            return
        out_port = self._port_at(pos, incoming=False)
        if out_port is not None and event.button() == Qt.LeftButton:
            self._begin_wire(out_port.node.nid)
            a = self._nodes.get(out_port.node.nid)
            if a is not None and self._temp_wire is not None:
                self._temp_wire.set_ends(a.out_scene(), self.mapToScene(pos))
            event.accept()
            return
        if isinstance(item, _WireItem) and event.button() == Qt.RightButton:
            self.disconnect_requested.emit(item.src, item.dst)
            event.accept()
            return
        self._press_pos = {}
        hit = item.node if isinstance(item, _Port) else item
        if isinstance(hit, _Node):
            self._press_pos[hit.nid] = hit.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._temp_wire is not None and self._drag_src:
            pos = self._event_pos(event)
            a = self._nodes.get(self._drag_src)
            snap = self._port_at(pos, incoming=True)
            if snap is not None and snap.node.nid in (self._drag_src, SOURCE):
                snap = None
            candidates = self._candidate_ports(self._drag_src)
            for node in self._nodes.values():
                for port in node.incoming_ports():
                    if port is snap:
                        port.set_state("snap")
                    elif port in candidates:
                        port.set_state("candidate")
                    else:
                        port.set_state("idle")
            end = snap.scenePos() if snap is not None else self.mapToScene(pos)
            if a is not None:
                self._temp_wire.set_ends(a.out_scene(), end)
            event.accept()
            return
        if getattr(self, "_pan_anchor", None) is not None:
            pos = self._event_pos(event)
            delta = pos - self._pan_anchor
            self.horizontalScrollBar().setValue(self._pan_h - delta.x())
            self.verticalScrollBar().setValue(self._pan_v - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)
        if self._press_pos:
            pos = self._event_pos(event)
            ignore = [nid for nid in self._press_pos if nid not in (SOURCE, OUT)]
            cable = self._wire_at(pos, ignore_nids=ignore) if ignore else None
            # Only light the cable once the node has actually moved — a click
            # that happens to sit on a cable must not look like a splice.
            if cable is not None:
                moved = any(
                    self._nodes.get(nid) is not None
                    and (self._nodes[nid].pos() - old).manhattanLength() > 8
                    for nid, old in self._press_pos.items()
                    if nid not in (SOURCE, OUT)
                )
                if not moved:
                    cable = None
            self._set_hot_cable(cable)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if getattr(self, "_pan_anchor", None) is not None:
            self._pan_anchor = None
            event.accept()
            return
        if self._temp_wire is not None and self._drag_src:
            pos = self._event_pos(event)
            dst, port = self._hit_target(pos)
            cable = self._wire_at(pos, ignore_nids=(self._drag_src,)) if not dst else None
            src = self._drag_src
            self._cancel_wire()
            if dst and dst != src:
                self.connect_requested.emit(src, dst, port)
            elif cable is not None and src not in (cable.src, cable.dst):
                self.splice_requested.emit(src, cable.src, cable.dst)
            event.accept()
            return
        pos = self._event_pos(event)
        splice = self._drop_splice_target(pos)
        self._set_hot_cable(None)
        super().mouseReleaseEvent(event)
        if splice is not None:
            nid, edge_src, edge_dst = splice
            node = self._nodes.get(nid)
            if node is not None:
                self.position_changed.emit(nid, node.pos().x(), node.pos().y())
            self.splice_requested.emit(nid, edge_src, edge_dst)
            self._press_pos = {}
            return
        for nid, old in self._press_pos.items():
            node = self._nodes.get(nid)
            if node is None:
                continue
            if (node.pos() - old).manhattanLength() > 2:
                self.position_changed.emit(nid, node.pos().x(), node.pos().y())
        self._press_pos = {}

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        pos = self._event_pos(event)
        port = self._port_at(pos)
        item = port.node if port is not None else self.itemAt(pos)
        node = item.node if isinstance(item, _Port) else item
        if isinstance(node, _Node):
            if node.nid not in (SOURCE, OUT):
                self.focus_node(node.nid)
            kind = "instrument" if node.nid == SOURCE else node.kind
            self.device_activated.emit(
                node.nid if node.nid not in (SOURCE, OUT) else "",
                kind,
            )
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape and self._temp_wire is not None:
            self._cancel_wire()
            event.accept()
            return
        if event.key() == Qt.Key_0:
            if self.toggle_selected_bypass():
                event.accept()
                return
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.delete_selection():
                event.accept()
                return
        super().keyPressEvent(event)

    def _node_menu(self, node: _Node) -> None:
        if node.nid in (SOURCE, OUT):
            return
        menu = QMenu(self)
        bypass = menu.addAction("Bypass")
        bypass.setCheckable(True)
        bypass.setChecked(node.bypassed)
        bypass.triggered.connect(
            lambda checked=False: self.bypass_requested.emit(node.nid, bool(checked))
        )
        menu.addSeparator()
        remove = menu.addAction("Remove")
        remove.triggered.connect(lambda: self.remove_requested.emit(node.nid))
        menu.exec(self.cursor().pos())

    def wheelEvent(self, event) -> None:  # noqa: N802
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)


class FxGraphEditor(QWidget):
    add_requested = Signal()
    remove_requested = Signal(str)
    connect_requested = Signal(str, str, str)
    disconnect_requested = Signal(str, str)
    splice_requested = Signal(str, str, str)
    bypass_requested = Signal(str, bool)
    device_activated = Signal(str, str)
    param_changed = Signal(str, str, object)
    position_changed = Signal(str, float, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        bar = QWidget()
        bar.setStyleSheet(f"background: {theme.BG_PANEL};")
        row = QHBoxLayout(bar)
        row.setContentsMargins(12, 8, 12, 4)
        self._title = QLabel("Plugin Graph")
        self._title.setStyleSheet(f"color: {theme.FG}; font-weight: 700;")
        row.addWidget(self._title)
        row.addStretch(1)
        hint = QLabel(
            "New FX land just before Out. Drag a port to rewire, or drop a node "
            "onto a cable to insert it. B on the header (or 0) bypasses. "
            "Right-click a node for Bypass / Remove. Side bars are live I/O levels. "
            "Home fits · middle-drag pans."
        )
        hint.setStyleSheet(f"color: {theme.FG_DIM}; font-size: 10px;")
        hint.setWordWrap(True)
        row.addWidget(hint)
        fit = QToolButton()
        fit.setText("Fit")
        fit.setToolTip("Show every node (Home)")
        fit.setStyleSheet(
            f"QToolButton {{ background: {theme.BG_ELEVATED}; color: {theme.FG};"
            f" border: 1px solid {theme.BORDER}; border-radius: 4px; padding: 4px 10px; }}"
        )
        fit.clicked.connect(lambda: self.view.fit_all())
        row.addWidget(fit)
        plus = QToolButton()
        plus.setText("+ FX")
        plus.setToolTip("Add an effect — lands just before Out")
        plus.setStyleSheet(
            f"QToolButton {{ background: {theme.BG_ELEVATED}; color: {theme.CYAN};"
            f" border: 1px solid {theme.CYAN}; border-radius: 4px; padding: 4px 10px; }}"
        )
        plus.clicked.connect(self.add_requested.emit)
        row.addWidget(plus)
        outer.addWidget(bar)
        self.view = FxGraphView()
        self.view.add_requested.connect(self.add_requested)
        self.view.remove_requested.connect(self.remove_requested)
        self.view.connect_requested.connect(self.connect_requested)
        self.view.disconnect_requested.connect(self.disconnect_requested)
        self.view.splice_requested.connect(self.splice_requested)
        self.view.bypass_requested.connect(self.bypass_requested)
        self.view.device_activated.connect(self.device_activated)
        self.view.param_changed.connect(self.param_changed)
        self.view.position_changed.connect(self.position_changed)
        outer.addWidget(self.view, 1)

    def set_track(self, track: Optional[Track]) -> None:
        if track is None:
            self._title.setText("Plugin Graph — no track selected")
        else:
            name = "Master" if getattr(track, "is_master", False) else track.name
            self._title.setText(f"Plugin Graph — {name}")
        self.view.set_track(track)
