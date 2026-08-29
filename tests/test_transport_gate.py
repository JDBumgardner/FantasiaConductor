"""The transport must not enter a block whose audio does not exist."""

from __future__ import annotations

import numpy as np
import pytest

from fantasia_core.engine.plugin_render import PluginRenderer, missing_in_window


class _Note:
    def __init__(self, pitch=60):
        self.pitch, self.start, self.duration, self.velocity = pitch, 0.0, 1.0, 100


class _Clip:
    def __init__(self, cid, start, duration):
        self.id, self.name = cid, cid
        self.start, self.duration = start, duration
        self.content_type = "midi"
        self.notes = [_Note()]


def _project(*clips):
    track = type("T", (), {"id": "t1", "plugin": "Vital", "plugin_state": "",
                           "clips": list(clips)})()
    return type("P", (), {"tracks": [track]})()


@pytest.fixture()
def renderer(monkeypatch):
    from fantasia_core import plugins as plg

    monkeypatch.setattr(plg, "resolve", lambda name: f"/fake/{name}.vst3")
    monkeypatch.setattr(plg, "_LOADED", {})
    monkeypatch.setattr(plg, "load",
                        lambda name, owner=None: plg._LOADED.setdefault(
                            (plg.resolve(name), owner), object()))
    monkeypatch.setattr(plg, "restore_preset", lambda p, d: True)
    monkeypatch.setattr(plg, "render_notes",
                        lambda p, n, d, sr, tail=1.0, off_main_thread=False:
                        np.ones((int((d + tail) * sr), 2), dtype=np.float32) * 0.5)

    def _chunked(plugin, notes, duration, sr, chunk, on_chunk, tail=1.0,
                 off_main_thread=False):
        pos, total = 0.0, float(duration) + float(tail)
        while pos < total:
            span = min(float(chunk), total - pos)
            on_chunk(int(round(pos * sr)),
                     np.ones((int(span * sr), 2), dtype=np.float32) * 0.5)
            pos += span
    monkeypatch.setattr(plg, "render_notes_chunked", _chunked)
    return PluginRenderer(1000)


def test_only_clips_overlapping_the_window_are_required(renderer):
    """A clip two minutes away must not hold up pressing play."""
    p = _project(_Clip("now", 32.0, 16.0), _Clip("later", 100.0, 16.0),
                 _Clip("past", 0.0, 8.0))
    got = [row[0].id for row in missing_in_window(renderer, p, 32.0, 34.0)]
    assert got == ["now"]


def test_a_clip_starting_inside_the_window_counts(renderer):
    p = _project(_Clip("starts_soon", 33.0, 8.0))
    assert [r[0].id for r in missing_in_window(renderer, p, 32.0, 34.0)] == ["starts_soon"]


def test_a_clip_ending_inside_the_window_counts(renderer):
    p = _project(_Clip("tail", 30.0, 3.0))
    assert [r[0].id for r in missing_in_window(renderer, p, 32.0, 34.0)] == ["tail"]


def test_a_clip_that_only_touches_the_edge_is_excluded(renderer):
    """Half-open: a clip ending exactly at the window start makes no sound in it."""
    p = _project(_Clip("ends_at_start", 24.0, 8.0), _Clip("starts_at_end", 34.0, 8.0))
    assert missing_in_window(renderer, p, 32.0, 34.0) == []


def test_a_rendered_clip_is_not_required_again(renderer):
    p = _project(_Clip("now", 32.0, 16.0))
    clip = p.tracks[0].clips[0]
    assert missing_in_window(renderer, p, 32.0, 34.0)
    renderer.render(clip, "Vital", "", "t1")
    assert missing_in_window(renderer, p, 32.0, 34.0) == []


def test_the_window_is_ordered_by_when_it_is_needed(renderer):
    p = _project(_Clip("second", 33.0, 4.0), _Clip("sounding", 31.0, 8.0))
    assert [r[0].id for r in missing_in_window(renderer, p, 32.0, 40.0)] == \
        ["sounding", "second"]


def test_prefetching_a_window_leaves_the_gate_nothing_to_do(renderer):
    """Moving the locator is a whole interaction before pressing play. Starting
    the render there is what makes the gate free in the common case."""
    p = _project(_Clip("a", 32.0, 16.0), _Clip("b", 32.0, 16.0))
    need = missing_in_window(renderer, p, 32.0, 36.0)
    assert len(need) == 2
    for clip, plugin, state, owner in need:          # what the prefetch renders
        renderer.render(clip, plugin, state, owner)
    assert missing_in_window(renderer, p, 32.0, 36.0) == []


def test_the_window_query_does_not_rehash_the_patch_every_call(renderer):
    """A Vital patch is ~230KB and this runs per clip on every cache lookup,
    including from the audio callback."""
    big = "A" * 300_000
    track = type("T", (), {"id": "t1", "plugin": "Vital", "plugin_state": big,
                           "clips": [_Clip("a", 0.0, 4.0)]})()
    p = type("P", (), {"tracks": [track]})()
    missing_in_window(renderer, p, 0.0, 4.0)
    seen = dict(renderer._digests)
    assert seen, "the digest should be memoised"
    missing_in_window(renderer, p, 0.0, 4.0)
    assert renderer._digests == seen, "recomputed a digest it already had"


# ---- chunked rendering: latency without a seam ---------------------------
def test_chunk_plan_starts_small_and_grows():
    """Latency wants a tiny first slice; throughput wants big ones after."""
    from fantasia_core.plugins import chunk_plan

    plan = chunk_plan(33.0, first=0.15, cap=8.0)
    assert plan[0] == pytest.approx(0.15)
    assert plan[1] > plan[0] and plan[2] > plan[1]
    assert max(plan) <= 8.0
    assert sum(plan) == pytest.approx(33.0)
    assert len(plan) < 20, "too many slices: each costs a plugin call"


def test_chunk_plan_handles_a_clip_shorter_than_one_slice():
    from fantasia_core.plugins import chunk_plan

    assert chunk_plan(0.1, first=0.15) == [pytest.approx(0.1)]


def test_a_note_spanning_a_slice_boundary_is_split_correctly():
    """The plugin carries the voice across; the slice that contains the
    note-on sends only that, and the slice with the note-off sends only that.
    Sending both again would retrigger and audibly restart the note."""
    from types import SimpleNamespace as NS

    from fantasia_core.plugins import notes_to_midi_span

    note = NS(pitch=60, start=0.1, duration=1.0, velocity=100)   # 0.1 -> 1.1
    first = notes_to_midi_span([note], 0.0, 0.5)
    second = notes_to_midi_span([note], 0.5, 0.5)
    third = notes_to_midi_span([note], 1.0, 0.5)

    assert [m[0][0] & 0xF0 for m in first] == [0x90]      # note-on only
    assert second == []                                   # still sounding
    assert [m[0][0] & 0xF0 for m in third] == [0x80]      # note-off only
    assert third[0][1] == pytest.approx(0.1)              # 1.1s, slice-relative


def test_a_note_entirely_inside_one_slice_gets_both_messages():
    from types import SimpleNamespace as NS

    from fantasia_core.plugins import notes_to_midi_span

    note = NS(pitch=64, start=0.1, duration=0.2, velocity=90)
    msgs = notes_to_midi_span([note], 0.0, 1.0)
    assert [m[0][0] & 0xF0 for m in msgs] == [0x90, 0x80]


def test_a_clean_instance_is_not_flushed():
    """The flush exists to clear the previous clip's tail. A freshly loaded and
    primed instance has none, and that is the state every worker is in when the
    play gate asks for its first clip."""
    from fantasia_core import plugins as plg

    calls = []

    class _Plug:
        def __call__(self, msgs, duration=0.0, sample_rate=0, reset=False):
            calls.append(round(float(duration), 3))
            import numpy as np
            return np.zeros((2, int(duration * sample_rate)), dtype="float32")

    inst = _Plug()
    plg.mark_clean(inst)
    plg.render_notes(inst, [type("N", (), {"pitch": 60, "start": 0.0,
                                           "duration": 0.1, "velocity": 100})()],
                     0.2, 1000, tail=0.0, off_main_thread=True)
    assert plg.FLUSH_SECONDS not in calls, "flushed a clean instance"

    calls.clear()
    plg.render_notes(inst, [type("N", (), {"pitch": 60, "start": 0.0,
                                           "duration": 0.1, "velocity": 100})()],
                     0.2, 1000, tail=0.0, off_main_thread=True)
    assert plg.FLUSH_SECONDS in calls, "did not flush after a render left a tail"
