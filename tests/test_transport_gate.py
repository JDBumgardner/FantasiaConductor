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
