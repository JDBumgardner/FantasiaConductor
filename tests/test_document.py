"""Document model + serialization round-trip (M1)."""

from __future__ import annotations

from fantasia_core.document import Project
from fantasia_core.document.serialize import (
    load_project,
    project_from_dict,
    project_to_dict,
    save_project,
)


def _sample_project() -> Project:
    project = Project(name="Demo", tempo=128.0)
    t1 = project.add_track("Drums")
    t2 = project.add_track("Bass")
    project.add_clip(t1.id, start=0.0, duration=2.0, name="Kick loop")
    project.add_clip(t1.id, start=2.0, duration=2.0, name="Kick loop 2")
    project.add_clip(t2.id, start=1.0, duration=3.5, name="Bassline", gain_db=-3.0)
    return project


def test_ids_are_unique_and_monotonic():
    p = Project()
    ids = {p.new_id("t"), p.new_id("c"), p.new_id("t")}
    assert ids == {"t1", "c2", "t3"}


def test_add_and_lookup():
    p = _sample_project()
    assert len(p.tracks) == 2
    track, clip = p.find_clip(p.tracks[1].clips[0].id)
    assert track is p.tracks[1]
    assert clip.name == "Bassline"
    assert clip.gain_db == -3.0


def test_project_duration():
    p = _sample_project()
    # Bassline ends at 1.0 + 3.5 = 4.5, the latest clip end.
    assert p.duration == 4.5


def test_playback_end_adds_four_buffer_bars():
    p = Project(tempo=120.0, beats_per_bar=4)
    assert p.playback_end() == 0.0
    t = p.add_track("A")
    p.add_clip(t.id, 0.0, 2.0, "c")
    # 4 bars × 4 beats × 0.5s = 8s of tail after the last clip.
    assert p.tail_seconds() == 8.0
    assert p.playback_end() == 10.0


def test_dict_round_trip():
    p = _sample_project()
    restored = project_from_dict(project_to_dict(p))
    assert project_to_dict(restored) == project_to_dict(p)
    # Counter preserved so new ids don't collide with loaded ones.
    assert restored._next_id == p._next_id


def test_midi_clip_round_trip():
    from fantasia_core.document import Note, default_midi_pattern

    p = Project(name="MIDI")
    t = p.add_track("Synth")
    t.instrument = 5
    c = p.add_clip(t.id, 0.0, 2.0, "m")
    c.content_type = "midi"
    c.notes = default_midi_pattern(2.0)
    restored = project_from_dict(project_to_dict(p))
    rc = restored.tracks[0].clips[0]
    assert restored.tracks[0].instrument == 5
    assert rc.is_midi and len(rc.notes) == len(c.notes)
    assert isinstance(rc.notes[0], Note)
    assert rc.notes[0].pitch == 60 and rc.notes[-1].pitch == 72


def test_midi_clips_do_not_overlap_on_add():
    p = Project()
    t = p.add_track("Lead")
    a = p.add_clip(t.id, 0.0, 2.0, "a", content_type="midi")
    b = p.add_clip(t.id, 1.0, 2.0, "b", content_type="midi")
    c = p.add_clip(t.id, 2.0, 1.0, "c", content_type="midi")
    assert a is not None and b is None and c is not None
    assert t.midi_overlaps(0.5, 0.5) is True
    assert t.midi_overlaps(2.0, 1.0, exclude_id=c.id) is False
    assert t.midi_overlaps(3.0, 1.0) is False


def test_new_tracks_get_unique_palette_colors():
    from fantasia_core.document.colors import TRACK_CYCLE

    p = Project()
    colors = [p.add_track(f"T{i}").color for i in range(len(TRACK_CYCLE))]
    assert colors == list(TRACK_CYCLE)
    assert p.add_track("again").color == TRACK_CYCLE[0]


def test_midi_clip_color_inherits_until_set():
    p = Project()
    t = p.add_track("Lead")
    c = p.add_clip(t.id, 0.0, 1.0, "m", content_type="midi")
    assert c is not None and c.color == ""
    c.color = "#25e6d5"
    restored = project_from_dict(project_to_dict(p))
    assert restored.tracks[0].clips[0].color == "#25e6d5"


def test_nearest_midi_start_snaps_adjacent_and_respects_origin():
    p = Project()
    t = p.add_track("Lead")
    p.add_clip(t.id, 0.0, 2.0, "a", content_type="midi")
    p.add_clip(t.id, 5.0, 2.0, "b", content_type="midi")
    b = t.clips[1]
    assert t.nearest_midi_start(1.0, 2.0, exclude_id=b.id) == 2.0
    a = t.clips[0]
    assert t.nearest_midi_start(4.0, 2.0, exclude_id=a.id) == 3.0
    # Left of the clip at t=0 is blocked.
    assert t.nearest_midi_start(0.5, 2.0, exclude_id=b.id) == 2.0


def test_fx_params_apply_nested_eq_band():
    from fantasia_core.document.fx_params import apply_param, read_param, specs_for

    params = apply_param({"bands": [{"freq": 1000.0, "gain": 0.0}]}, "b0.gain", 3.5)
    assert params["bands"][0]["gain"] == 3.5
    specs = specs_for("reverb", {})
    assert any(s.key == "wet" for s in specs)
    wet = next(s for s in specs if s.key == "wet")
    assert read_param({"wet": 0.2}, wet) == 0.2


def test_file_round_trip(tmp_path):
    p = _sample_project()
    path = tmp_path / "demo.fcp"
    save_project(p, path)
    restored = load_project(path)
    assert restored.name == "Demo"
    assert restored.tempo == 128.0
    assert len(restored.tracks) == 2
    assert restored.tracks[0].clips[1].name == "Kick loop 2"


def test_loop_region_round_trip():
    p = Project(name="L", loop_enabled=True, loop_start=1.5, loop_end=5.0)
    restored = project_from_dict(project_to_dict(p))
    assert restored.loop_enabled is True
    assert restored.loop_start == 1.5
    assert restored.loop_end == 5.0
    assert restored.loop_bounds() == (1.5, 5.0)


def test_fx_inserts_get_stable_ids_on_save_and_load():
    from fantasia_core.document import FxInsert

    p = Project()
    t = p.add_track("A")
    t.fx = [{"type": "reverb", "params": {"wet": 0.3}}]
    p.master.fx = [p.new_insert("eq", {"bands": []})]
    restored = project_from_dict(project_to_dict(p))
    ins = restored.tracks[0].fx[0]
    assert isinstance(ins, FxInsert)
    assert ins.id.startswith("fx") and ins.type == "reverb"
    assert ins.bypassed is False
    ch, found, idx = restored.find_insert(ins.id)
    assert ch is restored.tracks[0] and found.id == ins.id and idx == 0
    master_eq = restored.master.fx[0]
    assert master_eq.id == p.master.fx[0].id and master_eq.type == "eq"


def test_fx_graph_serial_by_default_and_rewires_on_remove():
    from fantasia_core.document.fx_insert import (
        OUT, SOURCE, FxWire, is_serial, rewire_remove, serial_wires, would_cycle,
    )

    p = Project()
    t = p.add_track("A")
    a = p.new_insert("reverb")
    b = p.new_insert("delay")
    c = p.new_insert("eq")
    t.fx = [a, b, c]
    assert is_serial(t.fx, t.fx_wires)
    wires = serial_wires(t.fx)
    assert wires[0].src == SOURCE and wires[-1].dst == OUT
    # Remove the middle node: a should connect straight to c.
    bridged = rewire_remove(wires, b.id)
    keys = {(w.src, w.dst) for w in bridged}
    assert (a.id, c.id) in keys
    assert not any(w.src == b.id or w.dst == b.id for w in bridged)
    # Branching: a → b and a → c, both to out. Cycle a←c from c→a is rejected.
    branched = [FxWire(SOURCE, a.id), FxWire(a.id, b.id), FxWire(a.id, c.id),
                FxWire(b.id, OUT), FxWire(c.id, OUT)]
    assert not is_serial(t.fx, branched)
    assert would_cycle(branched, c.id, a.id)
    t.fx_wires = branched
    restored = project_from_dict(project_to_dict(p))
    assert {(w.src, w.dst) for w in restored.tracks[0].fx_wires} == {
        (w.src, w.dst) for w in branched
    }
