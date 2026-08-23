"""Command layer — every edit is a reversible Command dispatched on a CommandBus.

This is the architectural spine of the app: the UI and (from M6) the agent both
dispatch the *same* Command objects, so undo/redo and "AI hooks into the tools"
come from one mechanism instead of two.
"""

from fantasia_core.commands.base import Command, CommandBus
from fantasia_core.commands.clip_cmds import (
    AddClipCommand,
    DuplicateClipsCommand,
    MakeMidiClipCommand,
    RemoveClipCommand,
    SetClipAttrCommand,
    SetClipGeometryCommand,
    SetClipNotesCommand,
    SetClipSourceCommand,
    SplitClipCommand,
)
from fantasia_core.commands.track_cmds import (
    AddTrackCommand,
    RemoveTrackCommand,
    SetTempoCommand,
    SetTrackAttrCommand,
    SetTrackFxCommand,
    SetTrackSynthCommand,
    SetTrackSynthParamCommand,
)

__all__ = [
    "Command",
    "CommandBus",
    "AddTrackCommand",
    "RemoveTrackCommand",
    "SetTempoCommand",
    "SetTrackAttrCommand",
    "SetTrackFxCommand",
    "SetTrackSynthCommand",
    "SetTrackSynthParamCommand",
    "AddClipCommand",
    "DuplicateClipsCommand",
    "MakeMidiClipCommand",
    "RemoveClipCommand",
    "SetClipAttrCommand",
    "SetClipGeometryCommand",
    "SetClipNotesCommand",
    "SetClipSourceCommand",
    "SplitClipCommand",
]
