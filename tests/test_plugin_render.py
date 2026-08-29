"""Rendering MIDI through a hosted plugin, and keeping it off the audio thread.

The split is the whole point: a plugin can take an unbounded amount of time and
holds the GIL while it works, so the callback must only ever read a buffer that
was rendered earlier.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from fantasia_core.engine.plugin_render import PluginRenderer


class _Clip:
    content_type = "midi"

    def __init__(self, notes, duration=2.0, cid="c1"):
        self.id, self.notes, self.duration = cid, notes, duration


class _Note:
    def __init__(self, pitch, start, duration, velocity=100):
        self.pitch, self.start, self.duration, self.velocity = pitch, start, duration, velocity


class _Track:
    _n = 0

    def __init__(self, clips, plugin="", state="", tid=None):
        _Track._n += 1
        self.id = tid or f"t{_Track._n}"
        self.clips, self.plugin, self.plugin_state = clips, plugin, state


class _Project:
    def __init__(self, tracks):
        self.tracks = tracks


@pytest.fixture()
def fake_plugin(monkeypatch):
    """Stand in for the pedalboard layer and record what it was asked for."""
    calls = {"render": 0, "restore": []}

    def render_notes(plugin, notes, duration, sr, tail=1.0, off_main_thread=False):
        calls.setdefault("off_thread", []).append(off_main_thread)
        calls["render"] += 1
        return np.ones((int((duration + tail) * sr), 2), dtype=np.float32) * 0.5

    import fantasia_core.plugins as plg

    # Stub resolution but keep the real instance_for: without a resolvable
    # path a render raises before reaching render_notes, silence comes back,
    # and any test asserting "something was cached" passes on the failure.
    monkeypatch.setattr(plg, "resolve",                      # idempotent, like the real one
                        lambda name: str(name) if str(name).startswith("/")
                        else f"/fake/{name}.vst3")
    monkeypatch.setattr(plg, "load",
                        lambda name, owner=None: plg._LOADED.setdefault(
                            (plg.resolve(name), owner), f"<{name}:{owner}>"))
    monkeypatch.setattr(plg, "render_notes", render_notes)
    monkeypatch.setattr(plg, "restore_preset",
                        lambda p, d: calls["restore"].append(d) or True)
    return calls


def test_cached_is_empty_until_rendered(fake_plugin):
    r = PluginRenderer(1000)
    clip = _Clip([_Note(60, 0, 1)])
    assert r.cached(clip, "Vital") is None
    r.render(clip, "Vital")
    assert r.cached(clip, "Vital") is not None


def test_cached_never_synthesises(fake_plugin):
    """This runs on the audio thread; calling a plugin there is what blows the
    deadline."""
    r = PluginRenderer(1000)
    clip = _Clip([_Note(60, 0, 1)])
    r.cached(clip, "Vital")
    r.cached(clip, "Vital")
    assert fake_plugin["render"] == 0


def test_render_is_cached_not_repeated(fake_plugin):
    r = PluginRenderer(1000)
    clip = _Clip([_Note(60, 0, 1)])
    r.render(clip, "Vital")
    r.render(clip, "Vital")
    assert fake_plugin["render"] == 1


def test_buffer_is_trimmed_to_the_clip(fake_plugin):
    """The render carries a release tail; the clip's slot on the timeline does
    not grow to fit it."""
    r = PluginRenderer(1000, tail=1.0)
    buf = r.render(_Clip([_Note(60, 0, 1)], duration=2.0), "Vital")
    assert buf.shape == (2000, 2)


def test_changing_the_plugin_state_re_renders(fake_plugin):
    """Moving a knob has to invalidate what depends on it, and only that."""
    r = PluginRenderer(1000)
    clip = _Clip([_Note(60, 0, 1)])
    r.render(clip, "Vital", state="AAAA")
    r.render(clip, "Vital", state="BBBB")
    assert fake_plugin["render"] == 2
    assert r.cached(clip, "Vital", "AAAA") is not None   # the old one survives


def test_state_is_restored_once_per_change(fake_plugin):
    r = PluginRenderer(1000)
    state = base64.b64encode(b"patch").decode()
    r.render(_Clip([_Note(60, 0, 1)], cid="a"), "Vital", state)
    r.render(_Clip([_Note(62, 0, 1)], cid="b"), "Vital", state)
    assert fake_plugin["restore"] == [b"patch"]


def test_different_notes_are_different_cache_entries(fake_plugin):
    r = PluginRenderer(1000)
    r.render(_Clip([_Note(60, 0, 1)]), "Vital")
    r.render(_Clip([_Note(67, 0, 1)]), "Vital")
    assert fake_plugin["render"] == 2


def test_a_missing_plugin_yields_silence_not_a_crash(monkeypatch):
    """A project referencing a plugin the machine does not have must still open
    and play."""
    import fantasia_core.plugins as plg

    monkeypatch.setattr(plg, "load", lambda name, owner=None: (_ for _ in ()).throw(
        FileNotFoundError("no such plugin")))
    r = PluginRenderer(1000)
    buf = r.render(_Clip([_Note(60, 0, 1)], duration=2.0), "Nope")
    assert buf.shape == (2000, 2)
    assert not buf.any()


def test_warm_only_touches_plugin_tracks(fake_plugin):
    r = PluginRenderer(1000)
    r.warm(_Project([_Track([_Clip([_Note(60, 0, 1)], cid="a")], plugin="Vital"),
                     _Track([_Clip([_Note(60, 0, 1)], cid="b")])]))
    assert fake_plugin["render"] == 1


def test_invalidate_clears_one_plugin_or_all(fake_plugin):
    r = PluginRenderer(1000)
    clip = _Clip([_Note(60, 0, 1)])
    r.render(clip, "Vital")
    r.render(clip, "Other")
    r.invalidate("Vital")
    assert r.cached(clip, "Vital") is None
    assert r.cached(clip, "Other") is not None
    r.invalidate()
    assert r.cached(clip, "Other") is None


def test_pending_lists_only_what_is_not_rendered(fake_plugin):
    """Rendering a clip through a plugin costs a few hundred milliseconds, so
    the caller needs to know what is left rather than blocking on all of it."""
    r = PluginRenderer(1000)
    a, b = _Clip([_Note(60, 0, 1)], cid="a"), _Clip([_Note(62, 0, 1)], cid="b")
    track = _Track([a, b], plugin="Vital")
    proj = _Project([track])
    assert len(r.pending(proj)) == 2
    r.render(a, "Vital", "", track.id)
    assert [row[0].id for row in r.pending(proj)] == ["b"]


def test_pending_ignores_tracks_without_a_plugin(fake_plugin):
    r = PluginRenderer(1000)
    proj = _Project([_Track([_Clip([_Note(60, 0, 1)], cid="a")]),
                     _Track([_Clip([_Note(60, 0, 1)], cid="b")], plugin="Vital")])
    assert [row[0].id for row in r.pending(proj)] == ["b"]


def test_pending_reflects_a_changed_patch(fake_plugin):
    """A new patch means the old renders are stale and have to be redone."""
    r = PluginRenderer(1000)
    clip = _Clip([_Note(60, 0, 1)])
    track = _Track([clip], plugin="Vital", state="AAAA")
    r.render(clip, "Vital", "AAAA", track.id)
    assert r.pending(_Project([track])) == []
    track.plugin_state = "BBBB"
    assert len(r.pending(_Project([track]))) == 1


def test_warm_still_renders_everything(fake_plugin):
    r = PluginRenderer(1000)
    proj = _Project([_Track([_Clip([_Note(60, 0, 1)], cid="a"),
                             _Clip([_Note(62, 0, 1)], cid="b")], plugin="Vital")])
    r.warm(proj)
    assert r.pending(proj) == []


def test_each_track_gets_its_own_instance_and_patch(fake_plugin):
    """One synth on several tracks must not share a patch — a kick and a pad
    cannot be the same object. The owner is part of the cache key."""
    r = PluginRenderer(1000)
    clip = _Clip([_Note(60, 0, 1)])
    kick, pad = "a2ljay1wYXRjaA==", "cGFkLXBhdGNo"      # real base64: a patch blob, not a label
    r.render(clip, "Vital", kick, owner="t1")
    r.render(clip, "Vital", pad, owner="t2")
    assert fake_plugin["render"] == 2
    assert r.cached(clip, "Vital", kick, "t1") is not None
    assert r.cached(clip, "Vital", kick, "t2") is None


def test_state_is_restored_per_track(fake_plugin):
    r = PluginRenderer(1000)
    clip = _Clip([_Note(60, 0, 1)])
    import base64
    a, b = base64.b64encode(b"one").decode(), base64.b64encode(b"two").decode()
    r.render(clip, "Vital", a, owner="t1")
    r.render(clip, "Vital", b, owner="t2")
    assert fake_plugin["restore"] == [b"one", b"two"]


def test_invalidate_can_be_narrowed_to_one_track(fake_plugin):
    """A dozen tracks on one synth means clearing them all re-renders every clip
    to reflect a change that touched one."""
    r = PluginRenderer(1000)
    clip = _Clip([_Note(60, 0, 1)])
    r.render(clip, "Vital", "", owner="t1")
    r.render(clip, "Vital", "", owner="t2")
    r.invalidate("Vital", owner="t1")
    assert r.cached(clip, "Vital", "", "t1") is None
    assert r.cached(clip, "Vital", "", "t2") is not None


def test_invalidate_without_an_owner_still_clears_the_plugin(fake_plugin):
    r = PluginRenderer(1000)
    clip = _Clip([_Note(60, 0, 1)])
    r.render(clip, "Vital", "", owner="t1")
    r.render(clip, "Other", "", owner="t1")
    r.invalidate("Vital")
    assert r.cached(clip, "Vital", "", "t1") is None
    assert r.cached(clip, "Other", "", "t1") is not None


# ---- instance lifetime follows the track --------------------------------
def test_deleting_a_track_frees_its_instance(fake_plugin, monkeypatch):
    """A live plugin handle cannot live on the Track — the document is written
    to JSON, snapshotted for undo and copied on duplicate. So ownership is by
    id, and the lifetime is reconciled by sweeping."""
    from fantasia_core import plugins as plg
    from fantasia_core.engine.plugin_render import prune

    held = {("/x/Vital.vst3", "t1"): object(), ("/x/Vital.vst3", "t2"): object()}
    monkeypatch.setattr(plg, "_LOADED", held)
    r = PluginRenderer(1000)
    clip = _Clip([_Note(60, 0, 1)])
    r.render(clip, "Vital", "", owner="t1")
    r.render(clip, "Vital", "", owner="t2")

    class _P:
        tracks = [type("T", (), {"id": "t1"})()]      # t2 has been deleted

    assert prune(_P(), r) == 1
    # the shared render slot belongs to no track, so it stays
    assert {k for k in held if k[1] != plg.RENDER_OWNER} == {("/x/Vital.vst3", "t1")}
    assert r.cached(clip, "Vital", "", "t1") is not None
    assert r.cached(clip, "Vital", "", "t2") is None


def test_pruning_keeps_instances_with_no_owner(fake_plugin, monkeypatch):
    """An instance loaded without a track (a one-off inspection) is not a leak
    tied to any track, so a sweep must not take it."""
    from fantasia_core import plugins as plg
    from fantasia_core.engine.plugin_render import prune

    held = {("/x/Vital.vst3", None): object()}
    monkeypatch.setattr(plg, "_LOADED", held)

    class _P:
        tracks = []

    assert prune(_P(), PluginRenderer(1000)) == 0
    assert held


def test_loading_another_song_does_not_carry_instances_over(fake_plugin, monkeypatch):
    """Track ids restart at t1 per project, so song B's t3 must not inherit
    song A's t3 — which has no saved patch to overwrite the stale sound."""
    from fantasia_core import plugins as plg
    from fantasia_core.engine.plugin_render import reset

    held = {("/x/Vital.vst3", "t3"): object(), ("/x/Vital.vst3", None): object()}
    monkeypatch.setattr(plg, "_LOADED", held)
    r = PluginRenderer(1000)
    clip = _Clip([_Note(60, 0, 1)])
    r.render(clip, "Vital", _b64("PATCH-A"), owner="t3")
    assert r.cached(clip, "Vital", _b64("PATCH-A"), "t3") is not None

    assert reset(r) == 1                      # the owned one goes
    assert ("/x/Vital.vst3", "t3") not in held
    assert ("/x/Vital.vst3", None) in held     # the unowned one stays
    assert r.cached(clip, "Vital", _b64("PATCH-A"), "t3") is None


# ---- one shared instance, patch swapped per track -----------------------
def _b64(text: str) -> str:
    """A patch blob is base64 in the document; render() decodes it, and a bad
    string fails the render silently — which would make these pass vacuously."""
    import base64

    return base64.b64encode(text.encode()).decode()


def _recording_loader(monkeypatch):
    """Record which slot each load asks for, without a real plugin."""
    from fantasia_core import plugins as plg

    asked = []
    monkeypatch.setattr(plg, "load",
                        lambda name, owner=None: asked.append(owner) or f"<{name}:{owner}>")
    return asked


def test_tracks_render_through_one_shared_instance(fake_plugin, monkeypatch):
    """Holding an instance per track costs ~160MB each; a swap costs ~20ms
    against a ~210ms render."""
    from fantasia_core import plugins as plg

    monkeypatch.setattr(plg, "_LOADED", {})
    asked = _recording_loader(monkeypatch)
    r = PluginRenderer(1000)
    for i, patch in enumerate(("PATCH-A", "PATCH-B", "PATCH-C")):
        out = r.render(_Clip([_Note(60 + i, 0, 1)]), "Vital", _b64(patch), owner=f"t{i}")
        assert out.any(), "render failed — the rest of this test proves nothing"

    assert set(asked) == {plg.RENDER_OWNER}          # never one per track
    assert fake_plugin["restore"] == [b"PATCH-A", b"PATCH-B", b"PATCH-C"]


def test_a_track_rendered_twice_running_does_not_reswap(fake_plugin, monkeypatch):
    """The queue is grouped by track, so a song swaps once per track."""
    from fantasia_core import plugins as plg

    monkeypatch.setattr(plg, "_LOADED", {})
    _recording_loader(monkeypatch)
    r = PluginRenderer(1000)
    for pitch in (60, 62, 64):
        out = r.render(_Clip([_Note(pitch, 0, 1)]), "Vital", _b64("PATCH-A"), owner="t1")
        assert out.any(), "render failed — the rest of this test proves nothing"
    assert fake_plugin["restore"] == [b"PATCH-A"]


def test_an_open_editor_keeps_its_own_instance(fake_plugin, monkeypatch):
    """Its patch cannot be swapped out from under a window being edited."""
    from fantasia_core import plugins as plg

    editing = object()
    monkeypatch.setattr(plg, "_LOADED", {(plg.resolve("Vital"), "t7"): editing})
    _recording_loader(monkeypatch)

    inst, slot = plg.instance_for("Vital", "t7")
    assert inst is editing and slot == "t7"          # the editor's own
    _other, slot2 = plg.instance_for("Vital", "t8")
    assert slot2 == plg.RENDER_OWNER                 # everyone else shares


def test_the_shared_instance_survives_a_prune(fake_plugin, monkeypatch):
    """It belongs to no track, so no track's deletion may take it."""
    from fantasia_core import plugins as plg

    path = plg.resolve("Vital")
    loaded = {(path, plg.RENDER_OWNER): object(), (path, "t1"): object()}
    monkeypatch.setattr(plg, "_LOADED", loaded)

    assert plg.owners() == {"t1"}                    # the shared one is not a track
    assert plg.prune(set()) == 1
    assert list(loaded) == [(path, plg.RENDER_OWNER)]


def test_loading_a_song_forgets_the_patch_in_the_shared_instance(
        fake_plugin, monkeypatch):
    """It holds the old song's patch; the next render must not assume so."""
    from fantasia_core import plugins as plg
    from fantasia_core.engine.plugin_render import reset

    monkeypatch.setattr(plg, "_LOADED", {})
    _recording_loader(monkeypatch)
    r = PluginRenderer(1000)
    r.render(_Clip([_Note(60, 0, 1)]), "Vital", _b64("OLD-SONG-PATCH"), owner="t3")
    assert r._states
    reset(r)
    assert not r._states


def test_capture_state_reads_the_instance_a_change_was_written_to(fake_plugin, monkeypatch):
    """load(owner=...) would mint a fresh instance and capture its factory
    default over the edit — and leave one instance per track behind, undoing
    the sharing."""
    from fantasia_core import plugins as plg
    from fantasia_core.engine.plugin_render import capture_state

    class Inst:
        def __init__(self, tag): self.tag = tag

    loaded = {}
    monkeypatch.setattr(plg, "_LOADED", loaded)
    monkeypatch.setattr(plg, "load",
                        lambda name, owner=None: loaded.setdefault((name, owner), Inst(f"{name}:{owner}")))
    monkeypatch.setattr(plg, "preset_bytes", lambda inst: inst.tag.encode())

    written_to, _slot = plg.instance_for("Vital", "t1")
    before = len(loaded)
    got = base64.b64decode(capture_state("Vital", "t1")).decode()

    assert got == written_to.tag
    assert len(loaded) == before, "capture_state created an extra instance"


def test_capture_state_reads_an_open_editors_own_instance(fake_plugin, monkeypatch):
    """A track being edited has a dedicated instance; its patch lives there."""
    from fantasia_core import plugins as plg
    from fantasia_core.engine.plugin_render import capture_state

    class Inst:
        def __init__(self, tag): self.tag = tag

    editing = Inst("the-editor-instance")
    loaded = {(plg.resolve("Vital"), "t7"): editing}
    monkeypatch.setattr(plg, "_LOADED", loaded)
    monkeypatch.setattr(plg, "load",
                        lambda name, owner=None: loaded.setdefault((name, owner), Inst("shared")))
    monkeypatch.setattr(plg, "preset_bytes", lambda inst: inst.tag.encode())

    assert base64.b64decode(capture_state("Vital", "t7")).decode() == "the-editor-instance"


# ---- queue order follows the playhead -----------------------------------
def _clip_at(start, dur=4.0, pitch=60):
    c = _Clip([_Note(pitch, 0, 1)])
    c.start, c.duration = start, dur
    return c


def test_pending_is_ordered_by_when_each_clip_is_needed(fake_plugin, monkeypatch):
    """Project order renders bar 1 of the last track before the clip under the
    playhead, so pressing play mid-song leaves what you are hearing silent."""
    from fantasia_core import plugins as plg

    monkeypatch.setattr(plg, "_LOADED", {})
    r = PluginRenderer(1000)

    class _P:
        tracks = [type("T", (), {"id": "t1", "plugin": "Vital", "plugin_state": "",
                                 "clips": [_clip_at(0.0), _clip_at(60.0),
                                           _clip_at(32.0), _clip_at(40.0)]})()]

    order = [row[0].start for row in r.pending(_P(), from_time=33.0)]
    assert order[0] == 32.0     # sounding right now
    assert order[1] == 40.0     # next to arrive
    assert order[2] == 60.0     # later
    assert order[3] == 0.0      # already passed


def test_pending_without_a_playhead_keeps_project_order(fake_plugin, monkeypatch):
    """Warming a freshly loaded project has no playhead to prioritise around."""
    from fantasia_core import plugins as plg

    monkeypatch.setattr(plg, "_LOADED", {})
    r = PluginRenderer(1000)

    class _P:
        tracks = [type("T", (), {"id": "t1", "plugin": "Vital", "plugin_state": "",
                                 "clips": [_clip_at(60.0), _clip_at(0.0)]})()]

    assert [row[0].start for row in r.pending(_P())] == [60.0, 0.0]


def test_preroll_skips_a_plugin_that_is_not_loaded_yet(fake_plugin, monkeypatch):
    """Loading an instance takes seconds. Doing it while the user waits for
    playback to start trades a short gap for a long freeze."""
    from fantasia_core import plugins as plg

    monkeypatch.setattr(plg, "_LOADED", {})
    assert plg.is_resident("Vital") is False
    monkeypatch.setattr(plg, "_LOADED", {(plg.resolve("Vital"), plg.RENDER_OWNER): object()})
    assert plg.is_resident("Vital") is True
