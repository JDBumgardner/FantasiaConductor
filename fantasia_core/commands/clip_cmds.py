"""Clip commands: add / remove / set-geometry (move + resize)."""

from __future__ import annotations

from typing import Optional

from fantasia_core.commands.base import _UNSET, Command
from fantasia_core.document.model import Clip


class AddClipCommand(Command):
    """Add a clip to a track. Clip created once; redo re-inserts the same object."""

    def __init__(
        self,
        track_id: str,
        start: float,
        duration: float,
        name: str = "Clip",
        **kwargs,
    ) -> None:
        self.track_id = track_id
        self.start = start
        self.duration = duration
        self.name = name
        self.kwargs = kwargs
        self._clip = None
        self._index: Optional[int] = None
        self.label = "Add clip"

    def do(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is None:
            return
        if self._clip is None:
            self._clip = project.add_clip(
                self.track_id, self.start, self.duration, self.name, **self.kwargs
            )
            self._index = len(track.clips) - 1
        else:
            track.clips.insert(self._index, self._clip)

    def undo(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is not None and self._clip in track.clips:
            track.clips.remove(self._clip)

    @property
    def created_clip(self):
        return self._clip


class RemoveClipCommand(Command):
    """Remove a clip, remembering its track and index for undo."""

    def __init__(self, clip_id: str) -> None:
        self.clip_id = clip_id
        self._track_id: Optional[str] = None
        self._index: Optional[int] = None
        self._clip = None
        self.label = "Delete clip"

    def do(self, project) -> None:  # noqa: ANN001
        if self._clip is None:
            track, clip = project.find_clip(self.clip_id)
            if clip is None:
                return
            self._track_id = track.id
            self._index = track.clips.index(clip)
            self._clip = clip
        track = project.track_by_id(self._track_id)
        if track is not None and self._clip in track.clips:
            track.clips.remove(self._clip)

    def undo(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self._track_id)
        if track is not None and self._clip is not None:
            track.clips.insert(self._index, self._clip)


class SetClipAttrCommand(Command):
    """Set a scalar clip attribute (gain_db / fade_in / fade_out / reversed).

    Continuous edits (gain, fade drags) pass ``mergeable=True`` to coalesce into
    one undo step, mirroring :class:`SetTrackAttrCommand`.
    """

    def __init__(self, clip_id: str, attr: str, value, mergeable: bool = False) -> None:
        self.clip_id = clip_id
        self.attr = attr
        self.value = value
        self.mergeable = mergeable
        self._before = _UNSET
        self.label = f"Set clip {attr}"

    def do(self, project) -> None:  # noqa: ANN001
        _, clip = project.find_clip(self.clip_id)
        if clip is None:
            return
        if self._before is _UNSET:
            self._before = getattr(clip, self.attr)
        setattr(clip, self.attr, self.value)

    def undo(self, project) -> None:  # noqa: ANN001
        _, clip = project.find_clip(self.clip_id)
        if clip is not None and self._before is not _UNSET:
            setattr(clip, self.attr, self._before)

    def merge_key(self):
        return ("clip_attr", self.clip_id, self.attr) if self.mergeable else None

    def merge(self, other: "SetClipAttrCommand") -> None:
        self.value = other.value


class SplitClipCommand(Command):
    """Split a clip at an absolute timeline position into two clips.

    The left part is the original (shrunk); a new clip holds the right part,
    referencing the same source at an advanced ``source_offset``. Undo removes
    the new clip and restores the original's duration/fade.
    """

    def __init__(self, clip_id: str, at_time: float) -> None:
        self.clip_id = clip_id
        self.at = at_time
        self._track_id = None
        self._index = None
        self._new_clip = None
        self._left_before = _UNSET
        self.label = "Split clip"

    def do(self, project) -> None:  # noqa: ANN001
        track, clip = project.find_clip(self.clip_id)
        if clip is None or not (clip.start < self.at < clip.end):
            return
        if self._new_clip is None:
            self._track_id = track.id
            self._index = track.clips.index(clip)
            self._left_before = (clip.duration, clip.fade_out, clip.source_duration)
            left_dur = self.at - clip.start
            span = float(getattr(clip, "source_duration", 0.0) or 0.0)
            if span > 0 and clip.duration > 0:
                left_src = left_dur * (span / clip.duration)
            else:
                left_src = left_dur
            self._new_clip = Clip(
                id=project.new_id("c"),
                name=clip.name,
                start=self.at,
                duration=clip.end - self.at,
                content_type=clip.content_type,
                source_path=clip.source_path,
                source_offset=clip.source_offset + left_src,
                source_duration=(span - left_src) if span > 0 else 0.0,
                gain_db=clip.gain_db,
                fade_in=0.0,
                fade_out=clip.fade_out,
                reversed=clip.reversed,
                pitch_semitones=clip.pitch_semitones,
            )
        clip.duration = self.at - clip.start
        clip.fade_out = 0.0
        if float(getattr(clip, "source_duration", 0.0) or 0.0) > 0 and self._new_clip is not None:
            clip.source_duration = float(self._left_before[2]) - float(self._new_clip.source_duration)
        track = project.track_by_id(self._track_id)
        if self._new_clip not in track.clips:
            track.clips.insert(self._index + 1, self._new_clip)

    def undo(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self._track_id)
        if track is not None and self._new_clip in track.clips:
            track.clips.remove(self._new_clip)
        _, clip = project.find_clip(self.clip_id)
        if clip is not None and self._left_before is not _UNSET:
            clip.duration, clip.fade_out, clip.source_duration = self._left_before


class MakeMidiClipCommand(Command):
    """Convert a clip into a MIDI clip (the 'write MIDI' fill on the container).

    Sets ``content_type='midi'`` and seeds the given notes; clears any audio
    source. Fully reversible.
    """

    def __init__(self, clip_id: str, notes: list) -> None:
        self.clip_id = clip_id
        self.notes = list(notes)
        self._before = _UNSET
        self.label = "Write MIDI"

    def do(self, project) -> None:  # noqa: ANN001
        _, clip = project.find_clip(self.clip_id)
        if clip is None:
            return
        if self._before is _UNSET:
            self._before = (
                clip.content_type,
                clip.source_path,
                clip.source_offset,
                list(clip.notes),
            )
        clip.content_type = "midi"
        clip.source_path = None
        clip.source_offset = 0.0
        clip.notes = list(self.notes)

    def undo(self, project) -> None:  # noqa: ANN001
        _, clip = project.find_clip(self.clip_id)
        if clip is not None and self._before is not _UNSET:
            ctype, src, off, notes = self._before
            clip.content_type = ctype
            clip.source_path = src
            clip.source_offset = off
            clip.notes = list(notes)


class SetClipNotesCommand(Command):
    """Replace a MIDI clip's note list (used by the piano-roll editor)."""

    def __init__(self, clip_id: str, notes: list, label: str = "Edit notes") -> None:
        self.clip_id = clip_id
        self.notes = list(notes)
        self._before = _UNSET
        self.label = label

    def do(self, project) -> None:  # noqa: ANN001
        _, clip = project.find_clip(self.clip_id)
        if clip is None:
            return
        if self._before is _UNSET:
            self._before = list(clip.notes)
        clip.notes = list(self.notes)

    def undo(self, project) -> None:  # noqa: ANN001
        _, clip = project.find_clip(self.clip_id)
        if clip is not None and self._before is not _UNSET:
            clip.notes = list(self._before)


class SetClipSourceCommand(Command):
    """Fill (or replace) a clip's audio content.

    This is the first "fill" operation on the clip-as-container model: a clip is
    a slot, and importing audio fills it. Future fills (write-MIDI, generate)
    will follow the same shape — a Command that swaps the clip's content and is
    fully undoable. Passing ``duration`` resizes the slot to the new content.
    """

    def __init__(
        self,
        clip_id: str,
        source_path: Optional[str],
        source_offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> None:
        self.clip_id = clip_id
        self.source_path = source_path
        self.source_offset = source_offset
        self.duration = duration
        self._before = _UNSET
        self.label = "Import into clip"

    def do(self, project) -> None:  # noqa: ANN001
        _, clip = project.find_clip(self.clip_id)
        if clip is None:
            return
        if self._before is _UNSET:
            self._before = (
                clip.content_type,
                clip.source_path,
                clip.source_offset,
                clip.source_duration,
                clip.duration,
                list(clip.notes),
            )
        clip.content_type = "audio"  # importing audio makes it an audio clip
        clip.notes = []
        clip.source_path = self.source_path
        clip.source_offset = self.source_offset
        if self.duration is not None:
            clip.duration = self.duration
        clip.source_duration = float(clip.duration)

    def undo(self, project) -> None:  # noqa: ANN001
        _, clip = project.find_clip(self.clip_id)
        if clip is not None and self._before is not _UNSET:
            ctype, src, off, src_dur, dur, notes = self._before
            clip.content_type = ctype
            clip.source_path = src
            clip.source_offset = off
            clip.source_duration = src_dur
            clip.duration = dur
            clip.notes = list(notes)


class SetClipGeometryCommand(Command):
    """Set a clip's ``start`` and ``duration`` (covers both move and resize)."""

    def __init__(self, clip_id: str, start: float, duration: float) -> None:
        self.clip_id = clip_id
        self.start = start
        self.duration = duration
        self._before = _UNSET
        self.label = "Move/resize clip"

    def do(self, project) -> None:  # noqa: ANN001
        _, clip = project.find_clip(self.clip_id)
        if clip is None:
            return
        if self._before is _UNSET:
            self._before = (clip.start, clip.duration)
        clip.start = self.start
        clip.duration = self.duration

    def undo(self, project) -> None:  # noqa: ANN001
        _, clip = project.find_clip(self.clip_id)
        if clip is not None and self._before is not _UNSET:
            clip.start, clip.duration = self._before
