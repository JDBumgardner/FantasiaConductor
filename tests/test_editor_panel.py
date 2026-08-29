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
    assert dock.next_cycle_action() == "chain"
    dock.show_chain()
    assert dock.stack.currentIndex() == MODE_CHAIN
    assert dock.is_chain_open()
    assert dock.next_cycle_action() == "graph"
    dock.show_graph()
    assert dock.stack.currentIndex() == MODE_GRAPH
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
