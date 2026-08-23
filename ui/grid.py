"""Arrangement grid: interval math used for snap + drawing.

UI menus (View / toolbar / timeline context) share the same option keys.
Interval is in seconds; ``None`` means Off (no snap).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Adaptive: coarsest interval whose on-screen spacing is at least this many px.
ADAPTIVE_LEVELS = (
    ("Widest", "widest", 80.0),
    ("Wide", "wide", 52.0),
    ("Medium", "medium", 32.0),
    ("Narrow", "narrow", 20.0),
    ("Narrowest", "narrowest", 12.0),
)

ADAPTIVE_MIN_PX = {key: px for _, key, px in ADAPTIVE_LEVELS}

# Fixed options. Bar keys scale with time signature; beat keys are in beats
# (1.0 = one quarter-note beat).
FIXED_OPTIONS = (
    ("8 Bars", "8bars"),
    ("4 Bars", "4bars"),
    ("2 Bars", "2bars"),
    ("1 Bar", "1bar"),
    ("1/2", "1/2"),
    ("1/4", "1/4"),
    ("1/8", "1/8"),
    ("1/16", "1/16"),
    ("1/32", "1/32"),
    ("Off", "off"),
)

_FIXED_BEATS = {
    "1/2": 2.0,
    "1/4": 1.0,
    "1/8": 0.5,
    "1/16": 0.25,
    "1/32": 0.125,
}

_BAR_MULTIPLES = {
    "8bars": 8.0,
    "4bars": 4.0,
    "2bars": 2.0,
    "1bar": 1.0,
}

# Coarse → fine candidates for adaptive (in beats, bars filled in at call time).
_ADAPTIVE_BEAT_STEPS = (2.0, 1.0, 0.5, 0.25, 0.125)


@dataclass
class GridSpec:
    kind: str = "fixed"  # "fixed" | "adaptive" | "off"
    fixed_key: str = "1/4"
    adaptive: str = "narrow"
    triplet: bool = False

    def action_key(self) -> str:
        if self.kind == "off":
            return "off"
        if self.kind == "adaptive":
            return f"adaptive:{self.adaptive}"
        return f"fixed:{self.fixed_key}"


def seconds_per_beat(tempo: float) -> float:
    return 60.0 / tempo if tempo > 0 else 0.5


def fixed_interval_beats(key: str, beats_per_bar: int) -> Optional[float]:
    if key == "off":
        return None
    if key in _BAR_MULTIPLES:
        return _BAR_MULTIPLES[key] * max(beats_per_bar, 1)
    return _FIXED_BEATS.get(key)


def adaptive_interval_beats(
    tempo: float,
    beats_per_bar: int,
    pps: float,
    min_px: float,
) -> float:
    spb = seconds_per_beat(tempo)
    bpb = max(beats_per_bar, 1)
    # Coarse → fine: keep the finest interval that still has ≥ min_px spacing.
    candidates = [m * bpb for m in (8.0, 4.0, 2.0, 1.0)] + list(_ADAPTIVE_BEAT_STEPS)
    chosen = candidates[0]
    for beats in candidates:
        if beats * spb * max(pps, 1e-9) >= min_px:
            chosen = beats
        else:
            break
    return chosen


def grid_interval_seconds(
    spec: GridSpec,
    tempo: float,
    beats_per_bar: int,
    pps: float,
) -> Optional[float]:
    """Snap/draw interval in seconds, or ``None`` when the grid is Off."""
    if spec.kind == "off":
        return None
    if spec.kind == "adaptive":
        beats = adaptive_interval_beats(
            tempo, beats_per_bar, pps, ADAPTIVE_MIN_PX.get(spec.adaptive, 20.0)
        )
    else:
        beats = fixed_interval_beats(spec.fixed_key, beats_per_bar)
        if beats is None:
            return None
    if spec.triplet:
        beats *= 2.0 / 3.0
    return beats * seconds_per_beat(tempo)


def snap_time(seconds: float, interval: Optional[float]) -> float:
    if interval is None or interval <= 0:
        return seconds
    return round(seconds / interval) * interval
