"""The bottom editor panel: opening, switching mode, and closing it again."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt                       # noqa: E402
from PySide6.QtWidgets import QApplication, QSplitter, QWidget  # noqa: E402

from ui.editor_dock import EditorDock               # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def panel(qapp):
    dock = EditorDock()
    split = QSplitter(Qt.Vertical)
    split.addWidget(QWidget())
    split.addWidget(dock)
    dock.attach_splitter(split)
    split.resize(800, 600)
    split.show()          # children only report visible once the parent is shown
    return dock, split


def test_close_hides_the_panel(panel):
    dock, _ = panel
    dock.show_piano_roll()
    assert dock.isVisible()
    dock.close_panel()
    assert not dock.isVisible()


def test_close_gives_the_height_back(panel):
    """The mode buttons only switch what is shown; without a close, the only way
    out is dragging the splitter shut."""
    dock, split = panel
    dock.show_piano_roll()
    dock.close_panel()
    assert split.sizes()[-1] == 0


def test_reopening_restores_the_previous_height(panel):
    dock, split = panel
    dock.show_piano_roll()
    split.setSizes([300, 300])
    dock._remember_sizes()
    dock.close_panel()
    dock.show_piano_roll()
    assert split.sizes()[-1] > 0


def test_close_button_exists_and_is_not_a_mode(panel):
    dock, _ = panel
    assert dock.btn_close.isCheckable() is False
    assert dock.btn_close.text() == "✕"


def test_eq_editor_shows_eight_bands(panel):
    from fantasia_core.engine.eq import MAX_BANDS

    dock, _ = panel
    dock.show_eq([], 44100, "EQ — Test")
    assert len(dock.eq._bands) == MAX_BANDS
    assert dock.stack.currentIndex() == 2


def test_eq_pills_select_and_cut_drag_sets_q(panel):
    from PySide6.QtWidgets import QCheckBox

    from fantasia_core.engine.eq import MAX_BANDS

    dock, _ = panel
    dock.show_eq([], 44100, "EQ — Test")
    eq = dock.eq
    assert len(eq._band_btns) == MAX_BANDS
    assert not isinstance(eq._band_btns[0], QCheckBox)

    q0 = eq._bands[0]["q"]
    gain0 = eq._bands[0]["gain"]
    assert eq._bands[0]["type"] == "low_cut"
    eq._on_drag(0, 90.0, 12.0, 2.4)
    assert eq._bands[0]["freq"] == 90.0
    assert eq._bands[0]["q"] == 2.4
    assert eq._bands[0]["gain"] == gain0
    assert eq._bands[0]["enabled"] is True
    assert eq._gain.isVisible() is False
    assert "Q" in eq._hint.text()

    eq._select(2)
    assert eq._bands[2]["type"] == "bell"
    eq._on_drag(2, 800.0, 4.0, q0)
    assert eq._bands[2]["gain"] == 4.0
    assert eq._gain.isVisible() is True

    assert eq._bands[2]["enabled"] is True
    eq._band_btns[2].click()
    assert eq.selected_index() == 2
    assert eq._bands[2]["enabled"] is False
    eq._band_btns[2].click()
    assert eq._bands[2]["enabled"] is True
    assert eq._bands[7]["enabled"] is False
    eq._band_btns[7].click()
    assert eq.selected_index() == 7
    assert eq._bands[7]["enabled"] is True


def test_closing_an_already_closed_panel_is_harmless(panel):
    dock, _ = panel
    dock.close_panel()
    dock.close_panel()
    assert not dock.isVisible()


def test_shift_e_cycle_order(panel):
    from ui.editor_dock import MODE_CHAIN, MODE_GRAPH, MODE_PIANO

    dock, _ = panel
    assert dock.next_cycle_action() == "piano"
    dock.show_piano_roll()
    assert dock.stack.currentIndex() == MODE_PIANO
    assert dock.next_cycle_action() == "graph"
    dock.show_graph()
    assert dock.stack.currentIndex() == MODE_GRAPH
    assert dock.is_graph_open()
    assert dock.next_cycle_action() == "chain"
    dock.show_chain()
    assert dock.stack.currentIndex() == MODE_CHAIN
    assert dock.next_cycle_action() == "off"
    dock.collapse()
    assert not dock.is_open()
    assert dock.next_cycle_action() == "piano"


def test_signal_chain_and_graph_follow_the_track(panel):
    from fantasia_core.document import Project
    from fantasia_core.document.fx_insert import SOURCE

    dock, _ = panel
    p = Project()
    t = p.add_track("Lead")
    t.fx = [p.new_insert("reverb"), p.new_insert("delay")]
    dock.show_chain(t)
    assert "Lead" in dock.chain._title.text()
    dock.show_graph(t)
    assert SOURCE in dock.graph.view._nodes
    assert "Lead" in dock.graph._title.text()


def test_graph_nodes_list_stock_params(panel):
    from PySide6.QtWidgets import QDoubleSpinBox

    from fantasia_core.document import Project
    from fantasia_core.document.fx_insert import SOURCE
    from fantasia_core.engine.eq import default_bands
    from fantasia_core.engine.synth import DEFAULT_PATCH

    dock, _ = panel
    p = Project()
    t = p.add_track("Lead")
    t.is_synth = True
    t.synth = dict(DEFAULT_PATCH)
    t.fx = [p.new_insert("reverb", {"wet": 0.4, "dry": 0.6, "room_size": 0.5}),
            p.new_insert("eq", {"bands": default_bands()})]
    dock.show_graph(t)
    src = dock.graph.view._nodes[SOURCE]
    assert src._panel is not None
    assert "cutoff" in src._panel._controls
    rev = dock.graph.view._nodes[t.fx[0].id]
    assert rev._panel is not None
    wet = rev._panel._controls["wet"]
    assert isinstance(wet, QDoubleSpinBox)
    assert abs(wet.value() - 0.4) < 1e-6
    eq = dock.graph.view._nodes[t.fx[1].id]
    assert eq._panel is not None
    assert "b0.freq" in eq._panel._controls
    assert "b7.gain" in eq._panel._controls


def test_device_panel_shows_mix_wet_knob(panel):
    from fantasia_core.document import Project

    dock, _ = panel
    p = Project()
    t = p.add_track("Lead")
    mix = p.new_insert("mix", {"wet": 0.3, "dry_src": "in", "wet_src": "fx1"})
    t.fx = [mix]
    dock.show_device(mix, t.name)
    assert dock.stack.currentIndex() == 5
    assert dock.device._wet_box.isVisible()
    assert dock.device._wet_slider.value() == 30
    assert dock.btn_device.isVisible()


def test_unwired_graph_node_is_marked(panel):
    from fantasia_core.commands import AddFxCommand, AddTrackCommand, CommandBus
    from fantasia_core.document import Project
    from fantasia_core.document.fx_insert import SOURCE

    dock, _ = panel
    bus = CommandBus(Project())
    t = bus.dispatch(AddTrackCommand("Lead")).created_track
    cmd = bus.dispatch(AddFxCommand(t.id, "reverb"))
    dock.show_graph(t)
    assert SOURCE in dock.graph.view._nodes
    node = dock.graph.view._nodes[cmd.insert_id]
    assert node.wired is False
    assert "Not wired" in node._subtitle


def test_set_track_rebuilds_when_the_same_track_object_gains_an_insert(panel):
    """Commands mutate the Track the view already holds.

    Comparing signatures by re-reading that object after the mutation made
    every add look like a no-op, so the new node stayed invisible until the
    user switched tracks.
    """
    from fantasia_core.commands import AddFxCommand, AddTrackCommand, CommandBus
    from fantasia_core.document import Project

    dock, _ = panel
    bus = CommandBus(Project())
    t = bus.dispatch(AddTrackCommand("Pads")).created_track
    dock.show_graph(t)
    view = dock.graph.view
    before = set(view._nodes)
    cmd = bus.dispatch(AddFxCommand(t.id, "compressor", x=90.0, y=90.0))
    dock.graph.set_track(t)
    assert cmd.insert_id in view._nodes
    assert view.has_node(cmd.insert_id)
    assert set(view._nodes) > before


def test_dropping_a_node_on_a_cable_emits_splice(panel):
    from PySide6.QtCore import QPointF

    from fantasia_core.commands import AddFxCommand, AddTrackCommand, CommandBus
    from fantasia_core.document import Project
    from fantasia_core.document.fx_insert import OUT, SOURCE

    dock, _ = panel
    bus = CommandBus(Project())
    t = bus.dispatch(AddTrackCommand("Pads")).created_track
    rev = bus.dispatch(AddFxCommand(t.id, "reverb", x=40.0, y=300.0))
    dock.show_graph(t)
    view = dock.graph.view
    node = view._nodes[rev.insert_id]
    wire = view._wires[0]
    mid = wire.path().pointAtPercent(0.5)
    spliced: list[tuple] = []
    view.splice_requested.connect(lambda n, a, b: spliced.append((n, a, b)))
    view._press_pos = {rev.insert_id: QPointF(40.0, 300.0)}
    node.setPos(mid.x() - 40.0, mid.y() - 20.0)
    vp = view.mapFromScene(mid)
    hit = view._drop_splice_target(vp)
    assert hit == (rev.insert_id, SOURCE, OUT)
    view.mouseReleaseEvent(_left_release(vp))
    assert spliced == [(rev.insert_id, SOURCE, OUT)]


def _left_release(pos):
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent

    p = QPointF(pos)
    return QMouseEvent(
        QEvent.MouseButtonRelease, p, p, Qt.LeftButton, Qt.NoButton, Qt.NoModifier,
    )


def test_plugin_graph_tab_follows_piano_roll(panel):
    dock, _ = panel
    labels = [dock.btn_piano.text(), dock.btn_graph.text(), dock.btn_chain.text()]
    assert labels == ["Piano Roll", "Plugin Graph", "Chain"]
    bar = dock.btn_piano.parentWidget()
    layout = bar.layout()
    widgets = [layout.itemAt(i).widget() for i in range(layout.count())
               if layout.itemAt(i).widget() is not None]
    assert widgets.index(dock.btn_piano) < widgets.index(dock.btn_graph)
    assert widgets.index(dock.btn_graph) < widgets.index(dock.btn_chain)


def test_graph_port_hit_finds_output_near_the_circle(panel):
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from fantasia_core.document import Project
    from fantasia_core.document.fx_insert import OUT, SOURCE

    dock, _ = panel
    p = Project()
    t = p.add_track("Pads")
    dock.show_graph(t)
    QApplication.processEvents()
    view = dock.graph.view
    src = view._nodes[SOURCE]
    assert src.out_port is not None
    vp = view.mapFromScene(src.out_port.scenePos())
    hit = view._port_at(vp, incoming=False)
    assert hit is src.out_port
    # Slop: a click a few pixels off the circle still starts a cable.
    near = QPoint(vp.x() + 8, vp.y() + 4)
    assert view._port_at(near, incoming=False) is src.out_port
    dest = view._nodes[OUT]
    seen: list[tuple] = []
    view.connect_requested.connect(lambda a, b, c: seen.append((a, b, c)))
    QTest.mousePress(view.viewport(), Qt.LeftButton, Qt.NoModifier, vp)
    assert view._drag_src == SOURCE
    out_vp = view.mapFromScene(dest.in_port.scenePos())
    QTest.mouseMove(view.viewport(), out_vp)
    QTest.mouseRelease(view.viewport(), Qt.LeftButton, Qt.NoModifier, out_vp)
    QApplication.processEvents()
    assert seen and seen[0][0] == SOURCE and seen[0][1] == OUT


def test_graph_delete_selection_breaks_a_cable(panel):
    from fantasia_core.document import Project
    from fantasia_core.document.fx_insert import OUT, SOURCE

    dock, _ = panel
    p = Project()
    t = p.add_track("Pads")
    dock.show_graph(t)
    view = dock.graph.view
    broken: list[tuple] = []
    view.disconnect_requested.connect(lambda a, b: broken.append((a, b)))
    wire = view._wires[0]
    wire.setSelected(True)
    assert view.has_selection()
    assert view.delete_selection() is True
    assert broken == [(SOURCE, OUT)]


def test_delete_resolves_targets_before_any_handler_rebuilds(panel):
    """Every selected item is acted on even though the first signal rebuilds.

    Each ``remove_requested`` dispatches a command that clears the scene, so the
    targets have to be read off the selection up front. Cables are also broken
    before nodes: removing a node bridges its neighbours, which would otherwise
    leave a co-selected cable referring to a pair that no longer exists.
    """
    from fantasia_core.commands import (
        AddFxCommand,
        AddTrackCommand,
        CommandBus,
        RemoveFxCommand,
    )
    from fantasia_core.document import Project

    dock, _ = panel
    bus = CommandBus(Project())
    t = bus.dispatch(AddTrackCommand("Pads")).created_track
    first = bus.dispatch(AddFxCommand(t.id, "reverb", connect=True))
    second = bus.dispatch(AddFxCommand(t.id, "delay", connect=True))
    dock.show_graph(t)
    view = dock.graph.view
    order: list[str] = []

    def rebuild_on_remove(nid):
        order.append(f"node:{nid}")
        bus.dispatch(RemoveFxCommand(t.id, nid))
        dock.graph.set_track(t)  # rebuilds, destroying the old scene items

    view.remove_requested.connect(rebuild_on_remove)
    view.disconnect_requested.connect(lambda a, b: order.append(f"cable:{a}->{b}"))
    view._nodes[first.insert_id].setSelected(True)
    view._nodes[second.insert_id].setSelected(True)
    view._wires[0].setSelected(True)
    assert view.delete_selection() is True
    assert order[0].startswith("cable:")
    assert sorted(o for o in order if o.startswith("node:")) == sorted(
        [f"node:{first.insert_id}", f"node:{second.insert_id}"]
    )


def test_escape_cancels_a_wire_drag_without_holding_the_mouse(panel):
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    from fantasia_core.document import Project
    from fantasia_core.document.fx_insert import SOURCE

    dock, _ = panel
    t = Project().add_track("Pads")
    dock.show_graph(t)
    view = dock.graph.view
    view._begin_wire(SOURCE)
    assert view._temp_wire is not None
    # A leaked grab would freeze mouse input for the whole application.
    assert QWidget.mouseGrabber() is None
    view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
    QApplication.processEvents()
    assert view._temp_wire is None
    assert view._drag_src is None


def test_hiding_the_graph_mid_drag_drops_the_cable(panel):
    from fantasia_core.document import Project
    from fantasia_core.document.fx_insert import SOURCE

    dock, _ = panel
    t = Project().add_track("Pads")
    dock.show_graph(t)
    view = dock.graph.view
    view._begin_wire(SOURCE)
    dock.show_piano_roll()
    assert view._temp_wire is None
    assert view._drag_src is None


def test_undoing_a_node_move_puts_the_node_back(panel):
    """Positions are outside the structure signature, so the view must resync."""
    from fantasia_core.commands import AddFxCommand, AddTrackCommand, CommandBus
    from fantasia_core.document import Project

    dock, _ = panel
    bus = CommandBus(Project())
    t = bus.dispatch(AddTrackCommand("Pads")).created_track
    fx = bus.dispatch(AddFxCommand(t.id, "reverb", x=100.0, y=100.0))
    dock.show_graph(t)
    node = dock.graph.view._nodes[fx.insert_id]
    assert (node.pos().x(), node.pos().y()) == (100.0, 100.0)

    from fantasia_core.commands import SetTrackFxCommand
    from fantasia_core.document.fx_insert import copy_insert

    moved = [copy_insert(e) for e in t.fx]
    moved[0].x, moved[0].y = 400.0, 260.0
    bus.dispatch(SetTrackFxCommand(t.id, moved, label="Move FX node"))
    dock.graph.set_track(t)
    assert (node.pos().x(), node.pos().y()) == (400.0, 260.0)

    bus.undo()
    dock.graph.set_track(t)
    assert (node.pos().x(), node.pos().y()) == (100.0, 100.0)


def test_eq_node_uses_columns_and_stays_reasonably_short(panel):
    from PySide6.QtWidgets import QScrollArea

    from fantasia_core.document import Project
    from fantasia_core.engine.eq import default_bands

    dock, _ = panel
    p = Project()
    t = p.add_track("Lead")
    t.fx = [p.new_insert("eq", {"bands": default_bands()})]
    dock.show_graph(t)
    eq = dock.graph.view._nodes[t.fx[0].id]
    assert eq._panel.findChildren(QScrollArea) == []
    # 32 params over 3 columns: wide, tall-ish, but every control is reachable.
    assert eq.rect().width() > 400
    assert eq.rect().height() < 300
    assert len(eq._panel._controls) == 32


def test_unwired_node_is_parked_below_the_tallest_wired_node(panel):
    from fantasia_core.commands import AddFxCommand, AddTrackCommand, CommandBus
    from fantasia_core.document import Project
    from fantasia_core.document.fx_insert import SOURCE
    from fantasia_core.engine.eq import default_bands

    dock, _ = panel
    bus = CommandBus(Project())
    t = bus.dispatch(AddTrackCommand("Lead")).created_track
    eq = bus.dispatch(AddFxCommand(t.id, "eq", {"bands": default_bands()}, connect=True))
    free = bus.dispatch(AddFxCommand(t.id, "reverb"))
    dock.show_graph(t)
    view = dock.graph.view
    tall = view._nodes[eq.insert_id]
    parked = view._nodes[free.insert_id]
    assert parked.pos().y() >= tall.pos().y() + tall.rect().height()
    assert not parked.sceneBoundingRect().intersects(view._nodes[SOURCE].sceneBoundingRect())


def test_spawn_point_keeps_a_tall_node_fully_on_screen(panel):
    from fantasia_core.document import Project
    from ui.fx_graph import node_size

    dock, _ = panel
    t = Project().add_track("Pads")
    dock.show_graph(t)
    view = dock.graph.view
    view.resize(700, 320)
    QApplication.processEvents()
    w, h = node_size(32, "eq")
    pt = view.spawn_scene_pos(w, h)
    vis = view.mapToScene(view.viewport().rect()).boundingRect()
    assert vis.left() - 1 <= pt.x()
    assert vis.top() - 1 <= pt.y()


def test_source_and_out_nodes_are_movable(panel):
    from PySide6.QtWidgets import QGraphicsItem

    from fantasia_core.document import Project
    from fantasia_core.document.fx_insert import OUT, SOURCE

    dock, _ = panel
    t = Project().add_track("Pads")
    dock.show_graph(t)
    view = dock.graph.view
    src, dest = view._nodes[SOURCE], view._nodes[OUT]
    assert src.flags() & QGraphicsItem.ItemIsMovable
    assert dest.flags() & QGraphicsItem.ItemIsMovable
    src.setPos(160, 70)
    dest.setPos(480, 90)
    assert src.pos().x() == 160
    assert dest.pos().x() == 480


def test_graph_param_panel_has_no_scrollbar(panel):
    from PySide6.QtWidgets import QScrollArea

    from fantasia_core.document import Project
    from fantasia_core.engine.eq import default_bands
    from fantasia_core.engine.synth import DEFAULT_PATCH

    dock, _ = panel
    p = Project()
    t = p.add_track("Lead")
    t.is_synth = True
    t.synth = dict(DEFAULT_PATCH)
    t.fx = [p.new_insert("eq", {"bands": default_bands()})]
    dock.show_graph(t)
    eq = dock.graph.view._nodes[t.fx[0].id]
    assert eq._panel is not None
    assert eq._panel.findChildren(QScrollArea) == []
    assert eq.rect().width() > 196  # two columns
    assert eq.rect().height() > 200  # taller, not clipped


def test_new_graph_node_spawns_in_the_viewport(panel):
    from fantasia_core.document import Project

    dock, _ = panel
    t = Project().add_track("Pads")
    dock.show_graph(t)
    view = dock.graph.view
    view.resize(640, 360)
    pt = view.spawn_scene_pos()
    vis = view.mapToScene(view.viewport().rect()).boundingRect()
    assert vis.adjusted(-8, -8, 8, 8).contains(pt)
    view.reveal_node("in")
    vis = view.mapToScene(view.viewport().rect()).boundingRect()
    assert vis.intersects(view._nodes["in"].sceneBoundingRect())


def test_synth_panel_has_three_oscillators(panel):
    from fantasia_core.commands import AddTrackCommand, CommandBus
    from fantasia_core.document import Project

    dock, _ = panel
    t = CommandBus(Project()).dispatch(AddTrackCommand("Lead")).created_track
    dock.show_synth(t)
    assert dock.stack.currentIndex() == 1
    assert dock.synth.osc1 is not None
    assert dock.synth.osc2 is not None
    assert dock.synth.osc3 is not None
    assert "cutoff" in dock.synth._sliders
    assert dock.synth._name.text() == "Lead"
