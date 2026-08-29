"""Why playback stuttered, recorded from the audio callback.

``PlaybackEngine.underruns`` says how many blocks were missed but nothing about
when or what was happening, which is not enough to chase an intermittent
stutter. This records one row per bad block so the pattern is visible: whether
it clusters at transport start, which song position it hit, how close the
callback came to its deadline, and whether the block was actually silent
because audio had not been rendered yet.

The audio callback may not allocate or take locks, so this is a preallocated
array plus a counter that only ever increases. Readers take a snapshot and
tolerate a torn row the way the spectrum tap does — losing one diagnostic row
is always better than stalling the callback that produces it.
"""

from __future__ import annotations

import numpy as np

# Why a block was flagged.
UNDERFLOW = 0   # PortAudio reported it could not get the block in time
SLOW = 1        # the callback ran long enough to be at risk
STARVED = 2     # a clip was silent because its audio was not rendered yet

KIND_NAMES = {UNDERFLOW: "underflow", SLOW: "slow", STARVED: "starved"}

_FIELDS = 6     # t, position, ms, budget_ms, kind, misses


class DropoutLog:
    """Fixed-size ring of flagged audio blocks."""

    def __init__(self, capacity: int = 512) -> None:
        self._cap = int(max(16, capacity))
        self._rows = np.zeros((self._cap, _FIELDS), dtype=np.float64)
        self._n = 0                  # total ever recorded; index = _n % cap
        self._started = 0.0          # perf_counter when playback last started

    # ---- audio thread ----------------------------------------------------
    def mark_start(self, now: float) -> None:
        self._started = now

    def record(self, now: float, position: float, ms: float,
               budget_ms: float, kind: int, misses: int = 0) -> None:
        """One flagged block. No allocation, no locks."""
        row = self._rows[self._n % self._cap]
        row[0] = now - self._started
        row[1] = position
        row[2] = ms
        row[3] = budget_ms
        row[4] = kind
        row[5] = misses
        self._n += 1

    # ---- UI thread -------------------------------------------------------
    @property
    def total(self) -> int:
        return self._n

    def reset(self) -> None:
        self._n = 0

    def snapshot(self, limit: int = 50) -> list:
        """The most recent flagged blocks, newest last."""
        if self._n == 0:
            return []
        keep = min(self._n, self._cap, int(limit))
        out = []
        for i in range(self._n - keep, self._n):
            t, pos, ms, budget, kind, misses = self._rows[i % self._cap]
            out.append({
                "since_start": round(float(t), 3),
                "position": round(float(pos), 3),
                "callback_ms": round(float(ms), 2),
                "budget_ms": round(float(budget), 1),
                "headroom_pct": round(100.0 * (1.0 - ms / budget), 1) if budget else 0.0,
                "kind": KIND_NAMES.get(int(kind), str(int(kind))),
                "unrendered_clips": int(misses),
            })
        return out

    def summary(self) -> dict:
        """Enough to say whether this clusters at the start of playback."""
        rows = self.snapshot(limit=self._cap)
        if not rows:
            return {"total": 0, "note": "no flagged blocks since playback started"}
        by_kind: dict = {}
        for r in rows:
            by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        first_2s = sum(1 for r in rows if r["since_start"] <= 2.0)
        worst = max(rows, key=lambda r: r["callback_ms"])
        return {
            "total": self._n,
            "by_kind": by_kind,
            "within_2s_of_start": first_2s,
            "share_at_start_pct": round(100.0 * first_2s / len(rows), 1),
            "worst_callback_ms": worst["callback_ms"],
            "budget_ms": worst["budget_ms"],
        }


class BlockStats:
    """Scratch counter render_block fills in; one instance, reused."""

    __slots__ = ("misses",)

    def __init__(self) -> None:
        self.misses = 0
