"""Clip commands: add / remove / set-geometry (move + resize)."""

from __future__ import annotations

from typing import Optional

from fantasia_core.commands.base import _UNSET, Command
from fantasia_core.document.model import Clip, Note


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
            if self._clip is None:
                return
            self._index = track.clips.index(self._clip)
        else:
            if getattr(self._clip, "is_midi", False) and track.midi_overlaps(
                self._clip.start, self._clip.duration, exclude_id=self._clip.id
            ):
                return
            idx = self._index if self._index is not None else len(track.clips)
            track.clips.insert(idx, self._clip)

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
        self._left_notes = None
        self.label = "Split clip"

    def do(self, project) -> None:  # noqa: ANN001
        track, clip = project.find_clip(self.clip_id)
        if clip is None or not (clip.start < self.at < clip.end):
            return
        if self._new_clip is None:
            self._track_id = track.id
            self._index = track.clips.index(clip)
            self._left_before = (
                clip.duration, clip.fade_out, clip.source_duration, list(clip.notes),
            )
            left_dur = self.at - clip.start
            span = float(getattr(clip, "source_duration", 0.0) or 0.0)
            if span > 0 and clip.duration > 0:
                left_src = left_dur * (span / clip.duration)
            else:
                left_src = left_dur
            right_notes: list = []
            left_notes: list = list(clip.notes)
            if clip.is_midi:
                left_notes, right_notes = [], []
                for n in clip.notes:
                    n_end = n.start + n.duration
                    if n_end <= left_dur + 1e-9:
                        left_notes.append(n)
                    elif n.start >= left_dur - 1e-9:
                        right_notes.append(Note(
                            n.pitch, n.start - left_dur, n.duration, n.velocity))
                    else:
                        left_notes.append(Note(
                            n.pitch, n.start, left_dur - n.start, n.velocity))
                        right_notes.append(Note(
                            n.pitch, 0.0, n_end - left_dur, n.velocity))
            self._left_notes = left_notes
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
                color=getattr(clip, "color", "") or "",
                notes=right_notes,
            )
        clip.duration = self.at - clip.start
        clip.fade_out = 0.0
        if self._left_notes is not None and clip.is_midi:
            clip.notes = list(self._left_notes)
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
            clip.duration, clip.fade_out, clip.source_duration, notes = self._left_before
            clip.notes = list(notes)


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
        track, clip = project.find_clip(self.clip_id)
        if clip is None:
            return
        if not clip.is_midi and track is not None and track.midi_overlaps(
            clip.start, clip.duration, exclude_id=clip.id
        ):
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
            self._before = (clip.content_type, list(clip.notes))
        clip.content_type = "midi"
        clip.notes = list(self.notes)

    def undo(self, project) -> None:  # noqa: ANN001
        _, clip = project.find_clip(self.clip_id)
        if clip is not None and self._before is not _UNSET:
            if isinstance(self._before, tuple):
                clip.content_type, notes = self._before
                clip.notes = list(notes)
            else:
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


def _clip_copy_kwargs(clip: Clip) -> dict:
    return {
        "content_type": clip.content_type,
        "source_path": clip.source_path,
        "source_offset": clip.source_offset,
        "source_duration": clip.source_duration,
        "notes": [Note(n.pitch, n.start, n.duration, n.velocity) for n in clip.notes],
        "gain_db": clip.gain_db,
        "fade_in": clip.fade_in,
        "fade_out": clip.fade_out,
        "reversed": clip.reversed,
        "pitch_semitones": clip.pitch_semitones,
        "lock_tempo": clip.lock_tempo,
        "orig_source_path": clip.orig_source_path,
        "lock_base_dur": clip.lock_base_dur,
        "color": clip.color,
    }


class DuplicateClipsCommand(Command):
    """Copy selected clips onto the same tracks, immediately after the selection."""

    def __init__(self, clip_ids: list[str]) -> None:
        self.clip_ids = list(clip_ids)
        self._created: list[tuple[str, Clip]] = []
        self.label = "Duplicate clips"

    def do(self, project) -> None:  # noqa: ANN001
        if self._created:
            for tid, clip in self._created:
                track = project.track_by_id(tid)
                if track is not None and clip not in track.clips:
                    track.clips.append(clip)
            return
        found: list[tuple[object, Clip]] = []
        for cid in self.clip_ids:
            track, clip = project.find_clip(cid)
            if track is not None and clip is not None:
                found.append((track, clip))
        if not found:
            return
        starts = [c.start for _, c in found]
        ends = [c.start + c.duration for _, c in found]
        span = max(ends) - min(starts)
        if span <= 0:
            span = found[0][1].duration
        for track, clip in found:
            copy = project.add_clip(
                track.id, clip.start + span, clip.duration, clip.name,
                **_clip_copy_kwargs(clip),
            )
            if copy is not None:
                self._created.append((track.id, copy))

    def undo(self, project) -> None:  # noqa: ANN001
        for tid, clip in self._created:
            track = project.track_by_id(tid)
            if track is not None and clip in track.clips:
                track.clips.remove(clip)

    @property
    def created_ids(self) -> list[str]:
        return [c.id for _, c in self._created]


class SetClipGeometryCommand(Command):
    """Set a clip's ``start`` and ``duration`` (covers both move and resize)."""

    def __init__(self, clip_id: str, start: float, duration: float) -> None:
        self.clip_id = clip_id
        self.start = start
        self.duration = duration
        self._before = _UNSET
        self.label = "Move/resize clip"

    def do(self, project) -> None:  # noqa: ANN001
        track, clip = project.find_clip(self.clip_id)
        if clip is None:
            return
        if clip.is_midi and track is not None:
            if abs(self.duration - clip.duration) < 1e-9:
                snapped = track.nearest_midi_start(
                    self.start, self.duration, exclude_id=clip.id)
                if snapped is None:
                    return
                self.start = snapped
            else:
                self.duration = track.clamp_midi_duration(
                    self.start, self.duration, exclude_id=clip.id)
        if self._before is _UNSET:
            self._before = (clip.start, clip.duration, clip.source_duration)
        # First audio resize freezes the file-native span so changing duration
        # time-stretches (pitch-preserving) instead of trimming the file.
        if (clip.source_path and float(getattr(clip, "source_duration", 0.0) or 0.0) <= 0.0
                and abs(self.duration - clip.duration) > 1e-9):
            clip.source_duration = float(clip.duration)
        clip.start = self.start
        clip.duration = self.duration

    def undo(self, project) -> None:  # noqa: ANN001
        _, clip = project.find_clip(self.clip_id)
        if clip is not None and self._before is not _UNSET:
            clip.start, clip.duration, clip.source_duration = self._before


class JoinMidiClipsCommand(Command):
    """Merge selected MIDI clips on one track into a single spanning clip.

    The result covers ``min(start)…max(end)``, including empty gaps between
    the selected clips. Refuses if the clips sit on different tracks, or if
    an unselected MIDI clip occupies any part of that span.
    """

    def __init__(self, clip_ids: list[str]) -> None:
        self.clip_ids = list(clip_ids)
        self._track_id: Optional[str] = None
        self._removed: list[tuple[int, Clip]] = []
        self._joined: Optional[Clip] = None
        self.label = "Join MIDI clips"

    def do(self, project) -> None:  # noqa: ANN001
        if self._joined is not None:
            track = project.track_by_id(self._track_id)
            if track is None:
                return
            for _, clip in self._removed:
                if clip in track.clips:
                    track.clips.remove(clip)
            if self._joined not in track.clips:
                track.clips.append(self._joined)
            return
        found: list[tuple[object, Clip]] = []
        for cid in self.clip_ids:
            track, clip = project.find_clip(cid)
            if track is not None and clip is not None and clip.is_midi:
                found.append((track, clip))
        if len(found) < 2:
            return
        if len({track.id for track, _ in found}) != 1:
            return
        track = found[0][0]
        clips = sorted((c for _, c in found), key=lambda c: (c.start, c.id))
        start = min(c.start for c in clips)
        end = max(c.end for c in clips)
        selected = {c.id for c in clips}
        if track.midi_overlaps(start, end - start, exclude_ids=selected):
            return
        notes = []
        for clip in clips:
            offset = clip.start - start
            for n in clip.notes:
                notes.append(Note(n.pitch, n.start + offset, n.duration, n.velocity))
        notes.sort(key=lambda n: (n.start, n.pitch))
        self._track_id = track.id
        self._removed = [(track.clips.index(c), c) for c in clips]
        for clip in clips:
            if clip in track.clips:
                track.clips.remove(clip)
        extra = {}
        if clips[0].color:
            extra["color"] = clips[0].color
        self._joined = project.add_clip(
            track.id, start, end - start, name=clips[0].name,
            content_type="midi", notes=notes, gain_db=clips[0].gain_db, **extra,
        )

    def undo(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self._track_id)
        if track is None:
            return
        if self._joined is not None and self._joined in track.clips:
            track.clips.remove(self._joined)
        for idx, clip in sorted(self._removed, key=lambda item: item[0]):
            if clip not in track.clips:
                track.clips.insert(min(idx, len(track.clips)), clip)

    @property
    def created_clip(self):
        return self._joined
