"""Tokyo-nightlife theme — deep indigo night with neon accents.

One palette shared by the Qt stylesheet (`main_window`) and the custom-painted
widgets (timeline, piano roll, headers), so the whole app reads as one mood:
near-black indigo surfaces, lavender-white text, and hot neon magenta / cyan /
electric-purple accents that glow against the dark.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont

# UI typeface — Linux first so Fedora/WSL don't fall back to a serif.
FONT_FAMILIES = (
    "Noto Sans",
    "DejaVu Sans",
    "Liberation Sans",
    "SF Pro Text",
    "Helvetica Neue",
    "sans-serif",
)
FONT_CSS = ", ".join(f'"{n}"' if " " in n else n for n in FONT_FAMILIES)


def ui_font(point_size: int = 9, bold: bool = False) -> QFont:
    """Sans-serif used on painted surfaces (piano roll, timeline, EQ)."""
    font = QFont()
    font.setFamilies(list(FONT_FAMILIES[:-1]))
    font.setStyleHint(QFont.SansSerif)
    font.setPointSize(point_size)
    font.setBold(bold)
    return font

# ---- surfaces (deep night) ----------------------------------------------
BG_DEEP = "#0b0c16"      # window base
BG_PANEL = "#14162a"     # panels, headers, docks
BG_ELEVATED = "#1b1e38"  # inputs, menus, elevated controls
BG_HOVER = "#262a4d"
BG_SELECTED = "#2a2f57"
BORDER = "#2a2e52"
BORDER_SOFT = "#20233f"

# ---- text ----------------------------------------------------------------
FG = "#c6ccf5"           # lavender white
FG_DIM = "#6b719e"
FG_BRIGHT = "#eef1ff"

# ---- neon accents --------------------------------------------------------
MAGENTA = "#ff2e97"
PINK = "#ff6ac1"
CYAN = "#25e6d5"
BLUE = "#5a8bff"
PURPLE = "#b46bff"
GREEN = "#4ff2a6"
ORANGE = "#ff9e64"
YELLOW = "#ffd76b"
RED = "#ff4d6d"

ACCENT = MAGENTA         # primary interactive accent
ACCENT_DIM = "#b3216b"
# Active mute/solo (and other checked buttons): dusty pink so dark M/S stays readable.
BUTTON_CHECKED = "#edb4ce"
BUTTON_CHECKED_FG = "#1a1014"

# ---- painted surfaces ----------------------------------------------------
TIMELINE_BG = "#0c0d1c"
PLAYHEAD = MAGENTA

LANE_EVEN = QColor(0x18, 0x1b, 0x36)
LANE_ODD = QColor(0x13, 0x15, 0x2c)
# In-scale piano-roll rows (muted neon green — highlight only, not a fold).
SCALE_LANE = QColor(0x4f, 0xf2, 0xa6, 38)
SCALE_KEY = QColor(0x4f, 0xf2, 0xa6, 52)
RULER_BG = QColor(0x14, 0x16, 0x2a)
RULER_LINE = QColor(0x38, 0x3d, 0x6b)
RULER_TEXT = QColor(0x8b, 0x92, 0xc8)
# Arrangement loop brace — vivid when on, still visible when off.
LOOP_ON = QColor(0x25, 0xe6, 0xd5, 160)
LOOP_OFF = QColor(0x25, 0xe6, 0xd5, 42)
LOOP_LANE_ON = QColor(0x25, 0xe6, 0xd5, 24)
LOOP_LANE_OFF = QColor(0x25, 0xe6, 0xd5, 10)

# grid line colours as RGBA tuples (bright lavender, high enough alpha to pop)
GRID_SUBDIV = (220, 216, 255, 55)
GRID_BEAT = (230, 226, 255, 95)
GRID_BAR = (240, 236, 255, 160)

# MIDI note preview inside a timeline clip
MIDI_PREVIEW = QColor(0x25, 0xe6, 0xd5, 0xcc)  # cyan, glowy

# piano-roll note
NOTE_FILL = QColor(0xff, 0x2e, 0x97)     # neon magenta
NOTE_BORDER = QColor(0xff, 0x9a, 0xd0)
NOTE_SELECTED = QColor(0xff, 0xff, 0xff)
# Dark olive-gray — color-wheel complement of purple, readable on magenta notes.
NOTE_LABEL = QColor(0x1e, 0x1c, 0x12)

# ---- track colour cycle (neon) ------------------------------------------
TRACK_CYCLE = [MAGENTA, CYAN, PURPLE, BLUE, GREEN, ORANGE, PINK, YELLOW]
STEM_COLORS = {"drums": MAGENTA, "bass": GREEN, "vocals": YELLOW, "other": PURPLE}


def track_color(index: int) -> str:
    return TRACK_CYCLE[index % len(TRACK_CYCLE)]
