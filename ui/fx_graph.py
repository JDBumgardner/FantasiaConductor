"""Blender-style node editor for a track's directed FX graph."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
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
    linear_order,
)
from fantasia_core.document.model import MASTER_ID, Track
from ui import theme

NODE_W, NODE_H = 160.0, 64.0
PORT_R = 6.5
COL_GAP, ROW_GAP = 220.0, 96.0


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


class _Port(QGraphicsEllipseItem):
    def __init__(self, node: "_Node", incoming: bool) -> None:
        super().__init__(-PORT_R, -PORT_R, PORT_R * 2, PORT_R * 2)
        self.node = node
        self.incoming = incoming
        self.setBrush(QColor(theme.CYAN if not incoming else theme.MAGENTA))
        self.setPen(QPen(QColor(theme.FG_BRIGHT), 1.2))
        self.setZValue(3)
        self.setCursor(Qt.CrossCursor)
        self.setAcceptedMouseButtons(Qt.LeftButton)


class _Node(QGraphicsRectItem):
    def __init__(self, nid: str, title: str, subtitle: str, kind: str) -> None:
        super().__init__(0, 0, NODE_W, NODE_H)
        self.nid = nid
        self.kind = kind
        self.setFlag(QGraphicsItem.ItemIsMovable, nid not in (SOURCE, OUT))
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setZValue(2)
        self.setPen(QPen(QColor(theme.BORDER), 1.4))
        self.setBrush(QColor(theme.BG_ELEVATED))
        self._title = title
        self._subtitle = subtitle
        self.in_port = _Port(self, True)
        self.out_port = _Port(self, False)
        self.in_port.setParentItem(self)
        self.out_port.setParentItem(self)
        self.in_port.setPos(0, NODE_H / 2)
        self.out_port.setPos(NODE_W, NODE_H / 2)

    def header_color(self) -> QColor:
        if self.nid == SOURCE:
            return QColor(theme.PURPLE)
        if self.nid == OUT:
            return QColor(theme.YELLOW)
        if self.kind == "eq":
            return QColor(theme.CYAN)
        return QColor(theme.ACCENT)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: N802
        painter.setRenderHint(QPainter.Antialiasing, True)
        r = self.rect()
        painter.setPen(QPen(
            QColor(theme.CYAN if self.isSelected() else theme.BORDER),
            2 if self.isSelected() else 1.2,
        ))
        painter.setBrush(QColor(theme.BG_ELEVATED))
        painter.drawRoundedRect(r, 8, 8)
        head = QRectF(r.x(), r.y(), r.width(), 22)
        painter.setBrush(self.header_color())
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(head, 8, 8)
        painter.fillRect(QRectF(r.x(), r.y() + 12, r.width(), 10), self.header_color())
        painter.setPen(QColor(theme.BG_DEEP))
        painter.setFont(theme.ui_font(9, bold=True))
        painter.drawText(head.adjusted(8, 0, -8, 0), Qt.AlignVCenter | Qt.AlignLeft, self._title)
        painter.setPen(QColor(theme.FG_DIM))
        painter.setFont(theme.ui_font(8))
        painter.drawText(QRectF(r.x() + 8, r.y() + 26, r.width() - 16, 30),
                         Qt.AlignLeft | Qt.AlignTop, self._subtitle)

    def out_scene(self) -> QPointF:
        return self.out_port.scenePos()

    def in_scene(self) -> QPointF:
        return self.in_port.scenePos()

    def itemChange(self, change, value):  # noqa: N802
        if change == QGraphicsItem.ItemPositionHasChanged and self.scene():
            view = self.scene().views()
            if view and hasattr(view[0], "_relayout_wires"):
                view[0]._relayout_wires()
        return super().itemChange(change, value)


class _WireItem(QGraphicsPathItem):
    def __init__(self, src: str, dst: str) -> None:
        super().__init__()
        self.src = src
        self.dst = dst
        self.setZValue(1)
        self.setPen(QPen(QColor(theme.CYAN), 2.2))
        self.setAcceptedMouseButtons(Qt.LeftButton)

    def set_ends(self, a: QPointF, b: QPointF) -> None:
        path = QPainterPath(a)
        dx = max(40.0, abs(b.x() - a.x()) * 0.45)
        path.cubicTo(QPointF(a.x() + dx, a.y()), QPointF(b.x() - dx, b.y()), b)
        self.setPath(path)


class FxGraphView(QGraphicsView):
    """Directed flowchart of the selected track's instrument + FX."""

    add_requested = Signal()
    remove_requested = Signal(str)
    connect_requested = Signal(str, str)
    disconnect_requested = Signal(str, str)
    device_activated = Signal(str, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._track: Optional[Track] = None
        self._nodes: Dict[str, _Node] = {}
        self._wires: List[_WireItem] = []
        self._drag_src: Optional[str] = None
        self._temp_wire: Optional[_WireItem] = None
        self._scene = QGraphicsScene(self)
        self._scene.setBackgroundBrush(QColor(theme.TIMELINE_BG))
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def set_track(self, track: Optional[Track]) -> None:
        self._track = track
        self._rebuild()

    def _rebuild(self) -> None:
        self._scene.clear()
        self._nodes = {}
        self._wires = []
        self._drag_src = None
        self._temp_wire = None
        track = self._track
        if track is None:
            self._scene.addText("Select a track (or a clip) to edit its signal graph.")
            return

        positions = self._layout(track)
        source_sub = _instrument_label(track)
        self._add_node(SOURCE, "Source", source_sub, "instrument", positions[SOURCE])
        order = linear_order(track.fx, getattr(track, "fx_wires", None))
        by_id = {insert_id(e): e for e in track.fx if insert_id(e)}
        for nid in order:
            spec = by_id.get(nid)
            if spec is None:
                continue
            sub = "bypassed" if insert_bypassed(spec) else insert_type(spec)
            pos = positions.get(nid, (COL_GAP, ROW_GAP))
            node = self._add_node(nid, device_label(spec), sub, insert_type(spec), pos)
            x = getattr(spec, "x", 0.0) or 0.0
            y = getattr(spec, "y", 0.0) or 0.0
            if x or y:
                node.setPos(x, y)
        out_sub = "Master mix" if getattr(track, "is_master", False) else "Fader → mix"
        self._add_node(OUT, "Out", out_sub, "out", positions[OUT])

        for w in effective_wires(track.fx, getattr(track, "fx_wires", None)):
            self._add_wire(w.src, w.dst)
        self._relayout_wires()
        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-40, -40, 80, 80))

    def _layout(self, track: Track) -> Dict[str, Tuple[float, float]]:
        order = [SOURCE] + linear_order(track.fx, getattr(track, "fx_wires", None)) + [OUT]
        pos: Dict[str, Tuple[float, float]] = {}
        for i, nid in enumerate(order):
            pos[nid] = (40.0 + i * COL_GAP, 80.0)
        # Simple branch offset: extra incoming → nudge down.
        wires = effective_wires(track.fx, getattr(track, "fx_wires", None))
        indeg: Dict[str, int] = {}
        for w in wires:
            indeg[w.dst] = indeg.get(w.dst, 0) + 1
        used_y: Dict[int, int] = {}
        for nid, (x, y) in list(pos.items()):
            col = int(round((x - 40.0) / COL_GAP))
            extra = max(0, indeg.get(nid, 1) - 1)
            slot = used_y.get(col, 0) + extra
            used_y[col] = slot + 1
            pos[nid] = (x, 80.0 + slot * ROW_GAP)
        return pos

    def _add_node(self, nid, title, subtitle, kind, pos) -> _Node:  # noqa: ANN001
        node = _Node(nid, title, subtitle, kind)
        node.setPos(pos[0], pos[1])
        self._scene.addItem(node)
        self._nodes[nid] = node
        return node

    def _add_wire(self, src: str, dst: str) -> None:
        item = _WireItem(src, dst)
        self._scene.addItem(item)
        self._wires.append(item)

    def _relayout_wires(self) -> None:
        for item in self._wires:
            a = self._nodes.get(item.src)
            b = self._nodes.get(item.dst)
            if a is None or b is None:
                continue
            item.set_ends(a.out_scene(), b.in_scene())
        if self._temp_wire is not None and self._drag_src:
            a = self._nodes.get(self._drag_src)
            if a is not None:
                # ends updated from mouseMove
                pass

    def mousePressEvent(self, event) -> None:  # noqa: N802
        item = self.itemAt(event.pos())
        if isinstance(item, _Port) and not item.incoming:
            self._drag_src = item.node.nid
            self._temp_wire = _WireItem(self._drag_src, "")
            self._temp_wire.setPen(QPen(QColor(theme.MAGENTA), 2, Qt.DashLine))
            self._scene.addItem(self._temp_wire)
            event.accept()
            return
        if isinstance(item, _WireItem) and event.button() == Qt.RightButton:
            self.disconnect_requested.emit(item.src, item.dst)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._temp_wire is not None and self._drag_src:
            a = self._nodes.get(self._drag_src)
            if a is not None:
                self._temp_wire.set_ends(a.out_scene(), self.mapToScene(event.pos()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._temp_wire is not None and self._drag_src:
            hit = self.itemAt(event.pos())
            dst = None
            if isinstance(hit, _Port) and hit.incoming:
                dst = hit.node.nid
            elif isinstance(hit, _Node):
                dst = hit.nid
            src = self._drag_src
            self._scene.removeItem(self._temp_wire)
            self._temp_wire = None
            self._drag_src = None
            if dst and dst != src:
                self.connect_requested.emit(src, dst)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        item = self.itemAt(event.pos())
        node = item.node if isinstance(item, _Port) else item
        if isinstance(node, _Node):
            self.device_activated.emit(node.nid if node.nid not in (SOURCE, OUT) else "",
                                       node.kind)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            for item in list(self._scene.selectedItems()):
                if isinstance(item, _Node) and item.nid not in (SOURCE, OUT):
                    self.remove_requested.emit(item.nid)
                elif isinstance(item, _WireItem):
                    self.disconnect_requested.emit(item.src, item.dst)
            event.accept()
            return
        super().keyPressEvent(event)

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
    connect_requested = Signal(str, str)
    disconnect_requested = Signal(str, str)
    device_activated = Signal(str, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        bar = QWidget()
        bar.setStyleSheet(f"background: {theme.BG_PANEL};")
        row = QHBoxLayout(bar)
        row.setContentsMargins(12, 8, 12, 4)
        self._title = QLabel("Signal graph")
        self._title.setStyleSheet(f"color: {theme.FG}; font-weight: 700;")
        row.addWidget(self._title)
        row.addStretch(1)
        hint = QLabel("Drag a port to wire · Delete removes a node (auto-reconnects) "
                      "· Right-click a cable to break it · Ctrl+wheel zoom")
        hint.setStyleSheet(f"color: {theme.FG_DIM}; font-size: 10px;")
        row.addWidget(hint)
        plus = QToolButton()
        plus.setText("+ FX")
        plus.setToolTip("Add an effect")
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
        self.view.device_activated.connect(self.device_activated)
        outer.addWidget(self.view, 1)

    def set_track(self, track: Optional[Track]) -> None:
        if track is None:
            self._title.setText("Signal graph — no track selected")
        else:
            name = "Master" if getattr(track, "is_master", False) else track.name
            self._title.setText(f"Signal graph — {name}")
        self.view.set_track(track)
