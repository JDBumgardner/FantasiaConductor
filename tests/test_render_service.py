"""Background clip rendering: ordering, dedup, and not dying on a bad clip."""

from __future__ import annotations

import threading
import time

import pytest

from fantasia_core.engine.render_service import RenderService


class _Clip:
    def __init__(self, cid): self.id = cid


class _Renderer:
    """Records what was rendered, in order, and how it was called."""

    def __init__(self, delay=0.0, fail_on=()):
        self.seen, self.delay, self.fail_on = [], delay, set(fail_on)
        self.off_thread_flags, self.threads, self.slots = [], set(), set()
        self._lock = threading.Lock()

    def render(self, clip, plugin, state="", owner="", off_main_thread=False, slot=None):
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self.seen.append(clip.id)
            self.off_thread_flags.append(off_main_thread)
            self.threads.add(threading.current_thread().name)
            self.slots.add(slot)
        if clip.id in self.fail_on:
            raise RuntimeError("bad clip")


def _job(cid, owner="t1"):
    return (_Clip(cid), "Vital", "", owner)


def _drain(svc, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end and not svc.idle:
        time.sleep(0.01)
    return svc.idle


def test_renders_in_the_order_submitted():
    """Submission order is the playhead order the caller worked out."""
    r = _Renderer()
    svc = RenderService(r, workers=1)
    svc.start()
    svc.submit([_job("a"), _job("b"), _job("c")])
    assert _drain(svc)
    svc.stop()
    assert r.seen == ["a", "b", "c"]


def test_it_renders_off_the_calling_thread():
    # Slow enough that both workers must take a job; with instant renders one
    # worker can drain the queue before the other starts.
    r = _Renderer(delay=0.05)
    svc = RenderService(r, workers=2)
    svc.start()
    svc.submit([_job(str(i)) for i in range(6)])
    assert _drain(svc)
    svc.stop()
    assert threading.current_thread().name not in r.threads
    assert all(r.off_thread_flags), "workers must skip the cross-thread reset"
    # one instance per worker: pedalboard's process call is not re-entrant
    assert len(r.slots) == 2 and None not in r.slots


def test_resubmitting_the_same_clip_does_not_render_it_twice():
    """The playhead moves constantly; re-prioritising must be free."""
    r = _Renderer(delay=0.05)
    svc = RenderService(r, workers=1)
    svc.start()
    svc.submit([_job("a"), _job("b")])
    added = svc.submit([_job("a"), _job("b")])
    assert added == 0
    assert _drain(svc)
    svc.stop()
    assert sorted(r.seen) == ["a", "b"]


def test_the_same_clip_on_two_tracks_is_two_jobs():
    """Two tracks can share a clip id but not a patch."""
    r = _Renderer()
    svc = RenderService(r, workers=1)
    svc.start()
    assert svc.submit([_job("a", owner="t1"), _job("a", owner="t2")]) == 2
    assert _drain(svc)
    svc.stop()
    assert r.seen == ["a", "a"]


def test_a_failing_clip_does_not_kill_the_worker():
    r = _Renderer(fail_on={"b"})
    svc = RenderService(r, workers=1)
    svc.start()
    svc.submit([_job("a"), _job("b"), _job("c")])
    assert _drain(svc)
    svc.stop()
    assert r.seen == ["a", "b", "c"]
    assert svc.stats()["done"] == 3


def test_clear_drops_work_not_yet_started():
    r = _Renderer(delay=0.05)
    svc = RenderService(r, workers=1)
    svc.start()
    svc.submit([_job(str(i)) for i in range(40)])
    svc.clear()
    time.sleep(0.2)
    svc.stop()
    assert len(r.seen) < 40


def test_progress_is_reported():
    seen = []
    r = _Renderer()
    svc = RenderService(r, workers=1, on_progress=lambda d, s: seen.append((d, s)))
    svc.start()
    svc.submit([_job("a"), _job("b")])
    assert _drain(svc)
    svc.stop()
    assert seen and seen[-1][0] == 2


def test_two_workers_beat_one_when_rendering_is_slow():
    """The whole point: pedalboard releases the GIL, so this is real."""
    def elapsed(n):
        r = _Renderer(delay=0.05)
        svc = RenderService(r, workers=n)
        svc.start()
        t0 = time.time()
        svc.submit([_job(str(i)) for i in range(8)])
        assert _drain(svc)
        svc.stop()
        return time.time() - t0

    assert elapsed(2) < elapsed(1) * 0.75


def test_stop_is_safe_to_call_twice():
    svc = RenderService(_Renderer(), workers=1)
    svc.start()
    svc.stop()
    svc.stop()
    assert svc.idle


# ---- workers need their own, pre-loaded instances ------------------------
def test_each_worker_gets_a_distinct_slot():
    """pedalboard's process call is not re-entrant, so two workers sharing one
    instance is a data race."""
    from fantasia_core import plugins as plg

    slots = {plg.render_slot(i) for i in range(3)}
    assert len(slots) == 3
    assert plg.render_slot(0) == plg.RENDER_OWNER


def test_preload_creates_exactly_the_slots_workers_will_ask_for(monkeypatch):
    """The bug this guards: main-thread code loaded owner=None while workers
    looked up owner=RENDER_OWNER, so every worker tried to construct a plugin —
    which pedalboard refuses off the main thread — and cached silence."""
    from fantasia_core import plugins as plg

    loaded = {}
    monkeypatch.setattr(plg, "_LOADED", loaded)
    # Independent of whether a real plugin is installed on the machine.
    monkeypatch.setattr(plg, "resolve", lambda name: f"/fake/{name}.vst3")
    monkeypatch.setattr(plg, "load",
                        lambda name, owner=None: loaded.setdefault(
                            (plg.resolve(name), owner), object()))

    assert plg.preload_slots("Vital", 2) == 2
    for i in range(2):
        _inst, slot = plg.instance_for("Vital", None, slot=plg.render_slot(i))
        assert slot == plg.render_slot(i)
        assert ("/fake/Vital.vst3", slot) in loaded
    assert plg.preload_slots("Vital", 2) == 0     # idempotent


def test_a_render_that_fails_records_why_instead_of_only_going_silent():
    """Silence with no explanation reads as a mixing mistake, not a bug."""
    import numpy as np

    from fantasia_core.engine.plugin_render import PluginRenderer

    r = PluginRenderer(1000)

    class _Clip:
        id, name, duration = "c1", "Lead", 1.0
        notes = [type("N", (), {"pitch": 60, "start": 0.0, "duration": 0.5,
                                "velocity": 100})()]

    out = r.render(_Clip(), "NoSuchPlugin", "", "t1")
    assert not np.abs(out).max()          # still silent, playback survives
    assert r.errors == 1
    assert "Lead" in r.last_error
