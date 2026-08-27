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
    def __init__(self, clips, plugin="", state=""):
        self.clips, self.plugin, self.plugin_state = clips, plugin, state


class _Project:
    def __init__(self, tracks):
        self.tracks = tracks


@pytest.fixture()
def fake_plugin(monkeypatch):
    """Stand in for the pedalboard layer and record what it was asked for."""
    calls = {"render": 0, "restore": []}

    def render_notes(plugin, notes, duration, sr, tail=1.0):
        calls["render"] += 1
        return np.ones((int((duration + tail) * sr), 2), dtype=np.float32) * 0.5

    import fantasia_core.plugins as plg

    monkeypatch.setattr(plg, "load", lambda name: f"<{name}>")
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

    monkeypatch.setattr(plg, "load", lambda name: (_ for _ in ()).throw(
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
    proj = _Project([_Track([a, b], plugin="Vital")])
    assert len(r.pending(proj)) == 2
    r.render(a, "Vital")
    assert [c.id for c, _p, _s in r.pending(proj)] == ["b"]


def test_pending_ignores_tracks_without_a_plugin(fake_plugin):
    r = PluginRenderer(1000)
    proj = _Project([_Track([_Clip([_Note(60, 0, 1)], cid="a")]),
                     _Track([_Clip([_Note(60, 0, 1)], cid="b")], plugin="Vital")])
    assert [c.id for c, _p, _s in r.pending(proj)] == ["b"]


def test_pending_reflects_a_changed_patch(fake_plugin):
    """A new patch means the old renders are stale and have to be redone."""
    r = PluginRenderer(1000)
    clip = _Clip([_Note(60, 0, 1)])
    track = _Track([clip], plugin="Vital", state="AAAA")
    r.render(clip, "Vital", "AAAA")
    assert r.pending(_Project([track])) == []
    track.plugin_state = "BBBB"
    assert len(r.pending(_Project([track]))) == 1


def test_warm_still_renders_everything(fake_plugin):
    r = PluginRenderer(1000)
    proj = _Project([_Track([_Clip([_Note(60, 0, 1)], cid="a"),
                             _Clip([_Note(62, 0, 1)], cid="b")], plugin="Vital")])
    r.warm(proj)
    assert r.pending(proj) == []
