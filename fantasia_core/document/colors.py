"""Arrangement colour cycle — shared by the document (auto-assign) and the UI.

Hex values match ``ui.theme.TRACK_PALETTE`` (Tokyo 90s sci-fi, deuteranopia-safe).
"""

from __future__ import annotations

TRACK_CYCLE = (
    "#ff2e97",  # Magenta
    "#25e6d5",  # Cyan
    "#b46bff",  # Violet
    "#5a8bff",  # Blue
    "#ffd76b",  # Gold
    "#ff9e64",  # Orange
    "#ff6ac1",  # Pink
    "#a8b4ff",  # Ice
    "#3ecfff",  # Aqua
    "#ff7a8a",  # Coral
)


def next_track_color(existing: list[str]) -> str:
    """Least-used palette colour among current tracks (ties keep palette order)."""
    counts = {c: 0 for c in TRACK_CYCLE}
    order = {c: i for i, c in enumerate(TRACK_CYCLE)}
    for raw in existing:
        key = str(raw or "").lower()
        for c in TRACK_CYCLE:
            if c.lower() == key:
                counts[c] += 1
                break
    return min(TRACK_CYCLE, key=lambda c: (counts[c], order[c]))
