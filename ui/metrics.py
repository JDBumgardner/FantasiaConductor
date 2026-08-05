"""Shared layout metrics so the header panel and the timeline stay aligned."""

from __future__ import annotations

RULER_H = 28          # height of the time ruler strip (px)
TRACK_H = 72          # height of one track lane / header (px)
CLIP_MARGIN = 4       # vertical inset of a clip within its lane (px)
PPS_DEFAULT = 80.0    # pixels per second at 1.0 zoom
PPS_MIN = 8.0
PPS_MAX = 800.0
RESIZE_EDGE = 6       # px hot-zone at a clip's right edge for resizing
