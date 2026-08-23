"""Document model — the project's source of truth."""

from fantasia_core.document.model import (
    Clip,
    Note,
    Project,
    Track,
    default_drum_pattern,
    default_midi_pattern,
)
from fantasia_core.document.tempo import scale_timeline, source_span

__all__ = [
    "Clip",
    "Note",
    "Track",
    "Project",
    "default_midi_pattern",
    "default_drum_pattern",
    "scale_timeline",
    "source_span",
]
