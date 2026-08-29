"""Background clip rendering, off the UI thread.

Rendering a plugin clip costs a few hundred milliseconds. Doing that on the UI
thread — one clip per timer tick — is why pressing play mid-song leaves the
parts under the playhead silent while the queue crawls: the work and the
interface compete for the same thread, so neither can be given priority.

Workers render instead, ordered by when each clip is needed. Two facts shape
the design, both measured rather than assumed:

* pedalboard releases the GIL during rendering, so worker threads genuinely run
  in parallel (~1.96x on two).
* pedalboard refuses to *reset* a plugin loaded on another thread. Instances are
  therefore loaded on the main thread and handed out, and each worker flushes
  the previous tail with silence instead of resetting. See
  ``plugins.render_notes(off_main_thread=True)``.

The renderer's cache is a plain dict keyed by clip: a worker's completed write
publishes atomically, and the audio callback only ever reads it.
"""

from __future__ import annotations

import threading
from queue import Empty, PriorityQueue
from typing import Callable, Optional


class RenderService:
    """A small pool of threads that fill a PluginRenderer's cache."""

    def __init__(self, renderer, workers: int = 2,
                 on_progress: Optional[Callable[[int, int], None]] = None) -> None:
        self.renderer = renderer
        self.n_workers = max(1, int(workers))
        self.on_progress = on_progress
        self._q: PriorityQueue = PriorityQueue()
        self._threads: list = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._queued: set = set()          # keys in flight, so a re-queue is free
        self._done = 0
        self._errors: list = []
        self._submitted = 0
        self._seq = 0                      # tie-break, and keeps tuples comparable

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        for i in range(self.n_workers):
            t = threading.Thread(target=self._run, args=(i,), name=f"render-{i}",
                                 daemon=True)
            t.start()
            self._threads.append(t)

    def prepare(self, plugins) -> int:
        """Load each worker's instance. Call from the main thread only.

        Workers cannot construct a plugin themselves, so anything not loaded
        here fails in the worker and leaves a silent track behind.
        """
        from fantasia_core import plugins as plg

        made = 0
        for name in {p for p in plugins if p}:
            try:
                made += plg.preload_slots(name, self.n_workers)
            except Exception as exc:  # noqa: BLE001
                self._errors.append(f"preload {name}: {exc!r}"[:200])
        return made

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        for _ in self._threads:
            self._q.put((-1.0, -1, None))   # wake each worker
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()

    # ---- submission ------------------------------------------------------
    def submit(self, jobs) -> int:
        """Queue ``(clip, plugin, state, owner)`` rows; earlier rows go first.

        Already-queued clips are skipped, so re-submitting on every playhead
        move costs nothing and simply re-prioritises what is left.
        """
        added = 0
        with self._lock:
            for priority, row in enumerate(jobs):
                key = self._key(row)
                if key in self._queued:
                    continue
                self._queued.add(key)
                self._seq += 1
                self._q.put((float(priority), self._seq, row))
                self._submitted += 1
                added += 1
        return added

    def clear(self) -> None:
        """Drop anything not yet started — the project changed under us."""
        with self._lock:
            try:
                while True:
                    self._q.get_nowait()
                    self._q.task_done()
            except Empty:
                pass
            self._queued.clear()

    # ---- state -----------------------------------------------------------
    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._queued)

    @property
    def idle(self) -> bool:
        return self.pending == 0

    def stats(self) -> dict:
        with self._lock:
            return {"workers": self.n_workers, "pending": len(self._queued),
                    "submitted": self._submitted, "done": self._done,
                    "failed": len(self._errors), "errors": self._errors[-5:]}

    # ---- internals -------------------------------------------------------
    @staticmethod
    def _key(row) -> tuple:
        clip, plugin, state, owner = row
        return (getattr(clip, "id", id(clip)), plugin, owner)

    def _run(self, index: int = 0) -> None:
        slot = None
        try:
            from fantasia_core import plugins as plg

            slot = plg.render_slot(index)
        except Exception:  # noqa: BLE001
            pass
        while not self._stop.is_set():
            try:
                _prio, _seq, row = self._q.get(timeout=0.2)
            except Empty:
                continue
            if row is None:
                self._q.task_done()
                break
            clip, plugin, state, owner = row
            try:
                self.renderer.render(clip, plugin, state, owner,
                                     off_main_thread=True, slot=slot)
            except Exception as exc:  # noqa: BLE001 — one bad clip must not kill a worker
                # Recorded rather than swallowed: a worker that silently fails
                # produces a track that is simply inaudible, with nothing
                # anywhere to say why.
                with self._lock:
                    self._errors.append(f"{getattr(clip, 'name', '?')}: {exc!r}"[:200])
            finally:
                with self._lock:
                    self._queued.discard(self._key(row))
                    self._done += 1
                    done, sub = self._done, self._submitted
                self._q.task_done()
                if self.on_progress is not None:
                    try:
                        self.on_progress(done, sub)
                    except Exception:  # noqa: BLE001
                        pass
