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
        self.off_thread_flags, self.threads = [], set()
        self._lock = threading.Lock()

    def render(self, clip, plugin, state="", owner="", off_main_thread=False):
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self.seen.append(clip.id)
            self.off_thread_flags.append(off_main_thread)
            self.threads.add(threading.current_thread().name)
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
    r = _Renderer()
    svc = RenderService(r, workers=2)
    svc.start()
    svc.submit([_job(str(i)) for i in range(6)])
    assert _drain(svc)
    svc.stop()
    assert threading.current_thread().name not in r.threads
    assert all(r.off_thread_flags), "workers must skip the cross-thread reset"


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
