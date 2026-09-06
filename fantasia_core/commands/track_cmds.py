"""Track commands: add / remove / set-attribute."""

from __future__ import annotations

from typing import Optional

from fantasia_core.commands.base import _UNSET, Command
from fantasia_core.document.fx_insert import (
    OUT,
    SOURCE,
    connect_wire,
    copy_insert,
    copy_wires,
    disconnect_wire,
    incoming_srcs,
    insert_id,
    insert_type,
    is_serial,
    rewire_remove,
    sanitize_wires,
    serial_wires,
    splice_before_out,
    splice_into_edge,
    would_cycle,
)


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
            from fantasia_core.engine.synth import DEFAULT_PATCH

            self._track = project.add_track(self.name)
            self._track.is_synth = True
            self._track.synth = dict(DEFAULT_PATCH)
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


class MoveTrackCommand(Command):
    """Move a track to a new index in the arrangement list.

    ``index`` is the desired final position after the move (0 = top).
    The master bus is not in ``project.tracks`` and cannot be reordered.
    """

    def __init__(self, track_id: str, index: int) -> None:
        self.track_id = track_id
        self.index = int(index)
        self._from = _UNSET
        self.label = "Reorder track"

    def do(self, project) -> None:  # noqa: ANN001
        from_idx = project.track_index(self.track_id)
        if from_idx is None:
            return
        dest = max(0, min(self.index, len(project.tracks) - 1))
        if dest == from_idx:
            return
        if self._from is _UNSET:
            self._from = from_idx
        track = project.tracks.pop(from_idx)
        dest = max(0, min(self.index, len(project.tracks)))
        project.tracks.insert(dest, track)

    def undo(self, project) -> None:  # noqa: ANN001
        if self._from is _UNSET:
            return
        cur = project.track_index(self.track_id)
        if cur is None:
            return
        track = project.tracks.pop(cur)
        dest = max(0, min(int(self._from), len(project.tracks)))
        project.tracks.insert(dest, track)


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
        self._before_wires = _UNSET
        self.label = label
        self.mergeable = mergeable

    def do(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is None:
            return
        if self._before is _UNSET:
            self._before = _copy_fx(track.fx)
            self._before_wires = copy_wires(getattr(track, "fx_wires", None))
        track.fx = _copy_fx(self.fx_list)
        project.ensure_inserts(track)
        # Capture minted ids so redo does not issue a fresh identity.
        self.fx_list = _copy_fx(track.fx)
        if not track.fx:
            track.fx_wires = []
        else:
            track.fx_wires = sanitize_wires(track.fx, getattr(track, "fx_wires", None) or [])

    def undo(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is not None and self._before is not _UNSET:
            track.fx = _copy_fx(self._before)
            if self._before_wires is not _UNSET:
                track.fx_wires = copy_wires(self._before_wires)

    def merge_key(self):
        if not self.mergeable:
            return None
        return ("track_fx", self.track_id)

    def merge(self, other: "SetTrackFxCommand") -> None:
        self.fx_list = other.fx_list


class SetTrackFxWiresCommand(Command):
    """Replace a track's FX *topology* without touching the inserts.

    The inserts are the nodes; this is the edges. An empty list restores the
    implicit serial graph ``in → fx[0] → … → out``, which is how a chain with
    no explicit wiring is defined.

    Wires naming a missing insert are dropped rather than rejected, matching
    ``sanitize_wires``: a graph that outlives one of its nodes should lose the
    dangling edge, not the whole edit.
    """

    def __init__(self, track_id: str, wires: list, label: str = "Set FX routing") -> None:
        self.track_id = track_id
        self.wires = copy_wires(wires)
        self._before = _UNSET
        self.label = label

    def do(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is None:
            return
        if self._before is _UNSET:
            self._before = copy_wires(getattr(track, "fx_wires", None))
        track.fx_wires = sanitize_wires(track.fx, self.wires)

    def undo(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is not None and self._before is not _UNSET:
            track.fx_wires = copy_wires(self._before)


def _copy_fx(fx_list) -> list:
    return [copy_insert(e) for e in (fx_list or [])]


class AddFxCommand(Command):
    """Append one insert with a stable id. Redo reinserts the same identity.

    ``connect=True`` (the default) lands the insert just before Out. An
    implicit serial chain stays implicit — the new node is simply last in
    ``track.fx``. An already-explicit graph is spliced: every feed into Out
    is reattached to the new device, then the device feeds Out. Parallel
    chains therefore stay intact and pass through the new effect.
    ``connect=False`` leaves the node floating for manual wiring.
    """

    def __init__(self, track_id: str, kind: str, params: Optional[dict] = None,
                 bypassed: bool = False, connect: bool = True,
                 x: float = 0.0, y: float = 0.0) -> None:
        self.track_id = track_id
        self.kind = str(kind)
        self.params = dict(params or {})
        self.x = float(x)
        self.y = float(y)
        bands = self.params.get("bands")
        if isinstance(bands, list):
            self.params["bands"] = [dict(b) for b in bands]
        self.bypassed = bool(bypassed)
        self.connect = bool(connect)
        self._insert = None
        self._wires_before = _UNSET
        self.label = f"Add {self.kind}"

    def do(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is None:
            return
        if self._insert is None:
            self._insert = project.new_insert(self.kind, self.params, self.bypassed)
            if self.x or self.y:
                self._insert.x = self.x
                self._insert.y = self.y
        if self._wires_before is _UNSET:
            self._wires_before = copy_wires(getattr(track, "fx_wires", None))
        existing = list(track.fx)
        track.fx = existing + [copy_insert(self._insert)]
        if self.connect:
            if self._wires_before:
                track.fx_wires = splice_before_out(self._wires_before, self._insert.id)
            return
        # Manual wiring: freeze the current path so the new node stays floating.
        if self._wires_before:
            track.fx_wires = copy_wires(self._wires_before)
        else:
            track.fx_wires = serial_wires(existing)

    def undo(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is None or self._insert is None:
            return
        iid = self._insert.id
        track.fx = [e for e in track.fx if insert_id(e) != iid]
        if self._wires_before is not _UNSET:
            track.fx_wires = copy_wires(self._wires_before)

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
        self._wires_before = _UNSET
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
            self._wires_before = copy_wires(getattr(track, "fx_wires", None))
        chain = list(track.fx)
        item = chain.pop(src)
        dest = max(0, min(self.index, len(chain)))
        chain.insert(dest, item)
        track.fx = chain
        if getattr(track, "fx_wires", None) and is_serial(chain, self._wires_before):
            track.fx_wires = serial_wires(chain)

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
        if self._wires_before is not _UNSET:
            track.fx_wires = copy_wires(self._wires_before)


class RemoveFxCommand(Command):
    """Remove one insert, remembering its slot for undo."""

    def __init__(self, track_id: str, insert_id: str) -> None:
        self.track_id = track_id
        self.insert_id = insert_id
        self._index: Optional[int] = None
        self._insert = None
        self._wires_before = _UNSET
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
        if self._wires_before is _UNSET:
            self._wires_before = copy_wires(getattr(track, "fx_wires", None))
        if getattr(track, "fx_wires", None):
            track.fx_wires = rewire_remove(track.fx_wires, self.insert_id)
        track.fx = [e for e in track.fx if insert_id(e) != self.insert_id]

    def undo(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is None or self._insert is None or self._index is None:
            return
        chain = list(track.fx)
        chain.insert(max(0, min(self._index, len(chain))), copy_insert(self._insert))
        track.fx = chain
        if self._wires_before is not _UNSET:
            track.fx_wires = copy_wires(self._wires_before)


class ConnectFxCommand(Command):
    """Add or remove one directed wire. Materializes implicit serial wires first.

    A second incoming edge on a non-mix node inserts a Dry/Wet Mix so the
    join is a blend rather than a loud sum. ``port`` is ``dry`` / ``wet``
    when the drop targeted a Mix input; otherwise roles are assigned in order.

    ``applied`` / ``reason`` let the caller tell the user why a drag did
    nothing instead of reporting a success that never happened.
    """

    def __init__(self, track_id: str, src: str, dst: str, connect: bool = True,
                 port: str = "") -> None:
        self.track_id = track_id
        self.src = str(src)
        self.dst = str(dst)
        self.connect = bool(connect)
        self.port = str(port or "")
        self._before = _UNSET
        self._fx_before = _UNSET
        self._mix = None
        self.applied = False
        self.reason = ""
        self.label = "Connect FX" if self.connect else "Disconnect FX"

    def do(self, project) -> None:  # noqa: ANN001
        self.applied = False
        self.reason = ""
        track = project.track_by_id(self.track_id)
        if track is None:
            self.reason = "that track is no longer in the project"
            return
        if self._before is _UNSET:
            self._before = copy_wires(getattr(track, "fx_wires", None))
            self._fx_before = _copy_fx(track.fx)
        current = list(track.fx_wires) or serial_wires(track.fx)
        if not self.connect:
            track.fx_wires = disconnect_wire(current, self.src, self.dst)
            self._clear_mix_role(track, self.dst, self.src)
            self.applied = True
            return
        if self.src == self.dst:
            self.reason = "a device cannot feed itself"
            return
        if would_cycle(current, self.src, self.dst):
            self.reason = "that would create a feedback loop"
            return
        dest = track.fx_by_id(self.dst) if self.dst not in (SOURCE, OUT) else None
        dest_kind = insert_type(dest) if dest is not None else ""
        incoming = incoming_srcs(current, self.dst)
        if dest_kind == "mix":
            nxt = connect_wire(current, self.src, self.dst)
            if nxt is None:
                self.reason = "that cable is not allowed here"
                return
            track.fx_wires = nxt
            self._assign_mix_role(track, dest, self.src, self.port)
            self.applied = True
            return
        if (
            incoming
            and self.src not in incoming
            and self.dst != SOURCE
        ):
            # Fork-merge: first existing feed is dry, the new feed is wet.
            dry_src = incoming[0]
            if self._mix is None:
                self._mix = project.new_insert("mix", {
                    "wet": 0.5,
                    "dry_src": dry_src,
                    "wet_src": self.src,
                })
            mix = copy_insert(self._mix)
            if track.fx_by_id(mix.id) is None:
                track.fx = list(track.fx) + [mix]
            for old in incoming:
                current = disconnect_wire(current, old, self.dst)
            nxt = current
            for old in incoming:
                nxt = connect_wire(nxt, old, mix.id) or nxt
            nxt = connect_wire(nxt, self.src, mix.id) or nxt
            nxt = connect_wire(nxt, mix.id, self.dst) or nxt
            track.fx_wires = nxt
            live = track.fx_by_id(mix.id)
            if live is not None:
                live.params = {
                    **dict(live.params or {}),
                    "wet": float((live.params or {}).get("wet", 0.5)),
                    "dry_src": dry_src,
                    "wet_src": self.src,
                }
            self.applied = True
            return
        nxt = connect_wire(current, self.src, self.dst)
        if nxt is None:
            self.reason = "that cable is not allowed here"
            return
        track.fx_wires = nxt
        self.applied = True

    def undo(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is None:
            return
        if self._fx_before is not _UNSET:
            track.fx = _copy_fx(self._fx_before)
        if self._before is not _UNSET:
            track.fx_wires = copy_wires(self._before)

    @staticmethod
    def _assign_mix_role(track, dest, src: str, port: str) -> None:  # noqa: ANN001
        if dest is None:
            return
        params = dict(dest.params or {})
        role = port if port in ("dry", "wet") else ""
        if not role:
            if not params.get("dry_src"):
                role = "dry"
            elif not params.get("wet_src") or params.get("wet_src") == src:
                role = "wet"
            else:
                role = "wet"
        key = "dry_src" if role == "dry" else "wet_src"
        params[key] = src
        dest.params = params
        idx = track.fx_index(dest.id)
        if idx is not None:
            ins = copy_insert(dest)
            ins.params = params
            chain = list(track.fx)
            chain[idx] = ins
            track.fx = chain

    @staticmethod
    def _clear_mix_role(track, dst: str, src: str) -> None:  # noqa: ANN001
        dest = track.fx_by_id(dst) if dst not in (SOURCE, OUT) else None
        if dest is None or insert_type(dest) != "mix":
            return
        params = dict(dest.params or {})
        changed = False
        for key in ("dry_src", "wet_src"):
            if params.get(key) == src:
                params.pop(key, None)
                changed = True
        if not changed:
            return
        dest.params = params
        idx = track.fx_index(dest.id)
        if idx is not None:
            ins = copy_insert(dest)
            ins.params = params
            chain = list(track.fx)
            chain[idx] = ins
            track.fx = chain


class SpliceFxCommand(Command):
    """Drop a node onto a cable: ``edge_src → node → edge_dst``.

    An already-wired node is detached from its current position first, so
    dropping it on a cable *moves* it into that spot rather than silently
    leaving a fork behind. ``applied`` reports whether anything changed.
    """

    def __init__(self, track_id: str, node_id: str, edge_src: str, edge_dst: str) -> None:
        self.track_id = track_id
        self.node_id = str(node_id)
        self.edge_src = str(edge_src)
        self.edge_dst = str(edge_dst)
        self._before = _UNSET
        self.applied = False
        self.reason = ""
        self.label = "Insert FX on cable"

    def do(self, project) -> None:  # noqa: ANN001
        self.applied = False
        self.reason = ""
        track = project.track_by_id(self.track_id)
        if track is None:
            self.reason = "that track is no longer in the project"
            return
        if self._before is _UNSET:
            self._before = copy_wires(getattr(track, "fx_wires", None))
        current = list(track.fx_wires) or serial_wires(track.fx)
        if self.node_id in (self.edge_src, self.edge_dst):
            self.reason = "that device is already on this cable"
            return
        # Lift the node out of its old slot, bridging its neighbours, so a
        # re-drop cannot leave the graph forked through the same device twice.
        detached = rewire_remove(current, self.node_id)
        nxt = splice_into_edge(detached, self.node_id, self.edge_src, self.edge_dst)
        if nxt is None:
            nxt = splice_into_edge(current, self.node_id, self.edge_src, self.edge_dst)
        if nxt is None:
            self.reason = "that would create a feedback loop"
            return
        track.fx_wires = nxt
        self.applied = True

    def undo(self, project) -> None:  # noqa: ANN001
        track = project.track_by_id(self.track_id)
        if track is None or self._before is _UNSET:
            return
        track.fx_wires = copy_wires(self._before)


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
