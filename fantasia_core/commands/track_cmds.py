"""Track commands: add / remove / set-attribute."""

from __future__ import annotations

from typing import Optional

from fantasia_core.commands.base import _UNSET, Command
from fantasia_core.document.fx_insert import copy_insert, insert_id


class AddTrackCommand(Command):
    """Add a new track. Creates the ``Track`` once; redo re-inserts the *same*
    object at its original index so its id is stable across undo/redo."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name
        self._track = None
        self._index: Optional[int] = None
        self.label = "Add track"

    def do(self, project) -> None:  # noqa: ANN001
        if self._track is None:
            self._track = project.add_track(self.name)
            self._index = len(project.tracks) - 1
        else:
            project.insert_track(self._index, self._track)

    def undo(self, project) -> None:  # noqa: ANN001
        project.remove_track(self._track.id)

    @property
    def created_track(self):
        return self._track


class RemoveTrackCommand(Command):
    """Remove a track, remembering its index and contents for undo."""

    def __init__(self, track_id: str) -> None:
        self.track_id = track_id
        self._index: Optional[int] = None
        self._track = None
        self.label = "Delete track"

    def do(self, project) -> None:  # noqa: ANN001
        if self._track is None:
            idx = project.track_index(self.track_id)
            if idx is None:
                return
            self._index = idx
            self._track = project.tracks[idx]
        if self._track in project.tracks:
            project.tracks.remove(self._track)

    def undo(self, project) -> None:  # noqa: ANN001
        if self._track is not None and self._index is not None:
            project.insert_track(self._index, self._track)


class SetTrackAttrCommand(Command):
    """Set a scalar track attribute (mute/solo/gain_db/pan/name).

    Continuous controls (volume, pan) pass ``mergeable=True`` so a slider drag
    collapses to one undo entry; discrete toggles/renames don't.
    """

    def __init__(self, track_id: str, attr: str, value, mergeable: bool = False) -> None:
        self.track_id = track_id
        self.attr = attr
        self.value = value
        self.mergeable = mergeable
        self._before = _UNSET
        self.label = f"Set {attr}"

    def do(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is None:
            return
        if self._before is _UNSET:
            self._before = getattr(track, self.attr)
        setattr(track, self.attr, self.value)

    def undo(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is not None and self._before is not _UNSET:
            setattr(track, self.attr, self._before)

    def merge_key(self):
        if not self.mergeable:
            return None
        return ("track_attr", self.track_id, self.attr)

    def merge(self, other: "SetTrackAttrCommand") -> None:
        # ``other`` has already applied its value; keep our original pre-state
        # and adopt the latest value so redo reproduces the final position.
        self.value = other.value


class SetTrackFxCommand(Command):
    """Replace a track's insert graph (list of :class:`FxInsert`).

    The UI computes the new list (EQ-band write, clear) and dispatches this;
    the previous chain is captured for undo. Continuous edits (EQ-band drags)
    pass ``mergeable=True`` so one gesture is one undo step. Missing insert
    ids are minted on first ``do`` and reused on redo.
    """

    def __init__(self, track_id: str, fx_list: list, label: str = "Set track FX",
                 mergeable: bool = False) -> None:
        self.track_id = track_id
        self.fx_list = _copy_fx(fx_list)
        self._before = _UNSET
        self.label = label
        self.mergeable = mergeable

    def do(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is None:
            return
        if self._before is _UNSET:
            self._before = _copy_fx(track.fx)
        track.fx = _copy_fx(self.fx_list)
        project.ensure_inserts(track)
        # Capture minted ids so redo does not issue a fresh identity.
        self.fx_list = _copy_fx(track.fx)

    def undo(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is not None and self._before is not _UNSET:
            track.fx = _copy_fx(self._before)

    def merge_key(self):
        if not self.mergeable:
            return None
        return ("track_fx", self.track_id)

    def merge(self, other: "SetTrackFxCommand") -> None:
        self.fx_list = other.fx_list


def _copy_fx(fx_list) -> list:
    return [copy_insert(e) for e in (fx_list or [])]


class AddFxCommand(Command):
    """Append one insert with a stable id. Redo reinserts the same identity."""

    def __init__(self, track_id: str, kind: str, params: Optional[dict] = None,
                 bypassed: bool = False) -> None:
        self.track_id = track_id
        self.kind = str(kind)
        self.params = dict(params or {})
        bands = self.params.get("bands")
        if isinstance(bands, list):
            self.params["bands"] = [dict(b) for b in bands]
        self.bypassed = bool(bypassed)
        self._insert = None
        self.label = f"Add {self.kind}"

    def do(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is None:
            return
        if self._insert is None:
            self._insert = project.new_insert(self.kind, self.params, self.bypassed)
        track.fx = list(track.fx) + [copy_insert(self._insert)]

    def undo(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is None or self._insert is None:
            return
        iid = self._insert.id
        track.fx = [e for e in track.fx if insert_id(e) != iid]

    @property
    def insert_id(self) -> str:
        return self._insert.id if self._insert is not None else ""


class BypassFxCommand(Command):
    """Bypass (or un-bypass) one insert. Graph change — FxHost rebuilds."""

    def __init__(self, track_id: str, insert_id: str, bypassed: bool) -> None:
        self.track_id = track_id
        self.insert_id = insert_id
        self.bypassed = bool(bypassed)
        self._before = _UNSET
        self.label = "Bypass FX" if self.bypassed else "Enable FX"

    def do(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is None:
            return
        idx = track.fx_index(self.insert_id)
        if idx is None:
            return
        ins = copy_insert(track.fx[idx])
        if self._before is _UNSET:
            self._before = ins.bypassed
        ins.bypassed = self.bypassed
        chain = list(track.fx)
        chain[idx] = ins
        track.fx = chain

    def undo(self, project) -> None:  # noqa: ANN001
        if self._before is _UNSET:
            return
        track = project.track_by_id(self.track_id)
        if track is None:
            return
        idx = track.fx_index(self.insert_id)
        if idx is None:
            return
        ins = copy_insert(track.fx[idx])
        ins.bypassed = self._before
        chain = list(track.fx)
        chain[idx] = ins
        track.fx = chain


class MoveFxCommand(Command):
    """Reorder an insert on a channel. ``index`` is the destination slot."""

    def __init__(self, track_id: str, insert_id: str, index: int) -> None:
        self.track_id = track_id
        self.insert_id = insert_id
        self.index = int(index)
        self._before = _UNSET
        self.label = "Move FX"

    def do(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is None:
            return
        src = track.fx_index(self.insert_id)
        if src is None:
            return
        if self._before is _UNSET:
            self._before = src
        chain = list(track.fx)
        item = chain.pop(src)
        dest = max(0, min(self.index, len(chain)))
        chain.insert(dest, item)
        track.fx = chain

    def undo(self, project) -> None:  # noqa: ANN001
        if self._before is _UNSET:
            return
        track = project.track_by_id(self.track_id)
        if track is None:
            return
        src = track.fx_index(self.insert_id)
        if src is None:
            return
        chain = list(track.fx)
        item = chain.pop(src)
        dest = max(0, min(int(self._before), len(chain)))
        chain.insert(dest, item)
        track.fx = chain


class RemoveFxCommand(Command):
    """Remove one insert, remembering its slot for undo."""

    def __init__(self, track_id: str, insert_id: str) -> None:
        self.track_id = track_id
        self.insert_id = insert_id
        self._index: Optional[int] = None
        self._insert = None
        self.label = "Remove FX"

    def do(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is None:
            return
        idx = track.fx_index(self.insert_id)
        if idx is None:
            return
        if self._insert is None:
            self._index = idx
            self._insert = copy_insert(track.fx[idx])
        track.fx = [e for e in track.fx if insert_id(e) != self.insert_id]

    def undo(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is None or self._insert is None or self._index is None:
            return
        chain = list(track.fx)
        chain.insert(max(0, min(self._index, len(chain))), copy_insert(self._insert))
        track.fx = chain


class SetTrackSynthParamCommand(Command):
    """Set one parameter of a track's synth patch. Mergeable so a knob drag
    collapses to a single undo step."""

    def __init__(self, track_id: str, key: str, value) -> None:
        self.track_id = track_id
        self.key = key
        self.value = value
        self._before = _UNSET
        self.label = f"Synth {key}"

    def do(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is None:
            return
        if self._before is _UNSET:
            self._before = dict(track.synth)
        track.synth = {**track.synth, self.key: self.value}

    def undo(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is not None and self._before is not _UNSET:
            track.synth = dict(self._before)

    def merge_key(self):
        return ("synth", self.track_id, self.key)

    def merge(self, other: "SetTrackSynthParamCommand") -> None:
        self.value = other.value


class SetTrackSynthCommand(Command):
    """Replace a track's whole synth patch in one undoable step (sound design)."""

    def __init__(self, track_id: str, patch: dict, label: str = "Design synth patch") -> None:
        self.track_id = track_id
        self.patch = dict(patch)
        self._before = _UNSET
        self.label = label

    def do(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is None:
            return
        if self._before is _UNSET:
            self._before = dict(track.synth)
        track.synth = dict(self.patch)

    def undo(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is not None and self._before is not _UNSET:
            track.synth = dict(self._before)


class SetTempoCommand(Command):
    """Set the project tempo (BPM) and rescale the arrangement so the song
    plays faster/slower. Undoable; consecutive changes (a spinner drag or
    repeated agent edits) coalesce into one undo entry."""

    def __init__(self, bpm: float) -> None:
        self.bpm = float(bpm)
        self._old = _UNSET
        self.last_factor = 1.0  # timeline scale applied by the last do/undo
        self.label = "Set tempo"

    def do(self, project) -> None:  # noqa: ANN001
        from fantasia_core.document.tempo import scale_timeline

        if self._old is _UNSET:
            self._old = project.tempo
        old = float(project.tempo)
        new = self.bpm
        if old > 0 and new > 0 and abs(old - new) > 1e-9:
            self.last_factor = old / new
            scale_timeline(project, self.last_factor)
        else:
            self.last_factor = 1.0
        project.tempo = new

    def undo(self, project) -> None:  # noqa: ANN001
        from fantasia_core.document.tempo import scale_timeline

        if self._old is _UNSET:
            return
        current = float(project.tempo)
        target = float(self._old)
        if current > 0 and target > 0 and abs(current - target) > 1e-9:
            self.last_factor = current / target
            scale_timeline(project, self.last_factor)
        else:
            self.last_factor = 1.0
        project.tempo = target

    def merge_key(self):
        return ("set_tempo",)

    def merge(self, other: "SetTempoCommand") -> None:
        self.bpm = other.bpm  # keep the original _old; adopt the latest target
