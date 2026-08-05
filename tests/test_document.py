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


def test_file_round_trip(tmp_path):
    p = _sample_project()
    path = tmp_path / "demo.fcp"
    save_project(p, path)
    restored = load_project(path)
    assert restored.name == "Demo"
    assert restored.tempo == 128.0
    assert len(restored.tracks) == 2
    assert restored.tracks[0].clips[1].name == "Kick loop 2"
