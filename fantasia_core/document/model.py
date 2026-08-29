"""The document model: ``Project`` → ``Track`` → ``Clip``.

This is the source of truth for the whole application. It is deliberately
plain Python (no Qt, no audio libs) so the headless core stays UI-agnostic and
trivially serializable/testable.

Design notes
------------
* **Time is measured in seconds** (float) on the timeline. Changing project
  tempo rescales clip/note times (``old_bpm / new_bpm``) so the song speeds up
  or slows down; audio is warped to the new clip length. Grid/snap use tempo
  as well.
* **IDs are stable strings** (``"t1"``, ``"c2"``, ``"fx12"``) from a single
  monotonic counter on the project. Commands and the agent address tracks,
  clips, and FX inserts by these ids, and they must survive save/load — so
  the counter is part of the serialized form. An FX insert is an
  :class:`FxInsert` (id, type, bypassed, params) on ``Track.fx``. Optional
  :class:`FxWire` entries on ``Track.fx_wires`` describe a directed graph
  (branching and merging); an empty wire list is the implicit serial chain
  ``in → fx[0] → … → out``. The numbered EQ *bands* live inside an ``eq``
  insert's params, not as their own graph nodes.
* Audio content is referenced by ``source_path`` and only *loaded* by the
  engine (M3); the model just records where a clip's audio comes from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from fantasia_core.document.fx_insert import (
    FxInsert,
    FxWire,
    as_insert,
    mint_missing_ids,
    sanitize_wires,
)

DEFAULT_SAMPLE_RATE = 44100
MASTER_ID = "master"


@dataclass
class Note:
    """A single MIDI note within a clip (times are seconds relative to the clip)."""

    pitch: int  # MIDI note number, 0-127 (60 = middle C)
    start: float  # seconds from the clip's start
    duration: float  # seconds
    velocity: int = 100  # 1-127

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class Clip:
    """A region on a track's timeline.

    In M1 a clip may have no audio (``source_path is None``) and simply occupies
    ``[start, start + duration)`` on the timeline. From M3, ``source_path`` /
    ``source_offset`` point into decoded audio, and ``gain_db`` / ``fade_in`` /
    ``fade_out`` / ``reversed`` describe non-destructive edits applied at render.
    """

    id: str
    name: str
    start: float  # timeline position, seconds
    duration: float  # seconds
    # Content: "audio" uses source_path/offset; "midi" uses notes. "empty" (no
    # source, no notes) is a placeholder slot to be filled.
    content_type: str = "audio"
    source_path: Optional[str] = None  # audio file backing this clip (M3+)
    source_offset: float = 0.0  # seconds into the source where playback starts
    source_duration: float = 0.0  # seconds of source to consume; 0 → equals duration
    # After a tempo change, duration is scaled but source_duration stays so the
    # mixer can warp the same source region into the new clip length.
    notes: List["Note"] = field(default_factory=list)  # for content_type == "midi"
    gain_db: float = 0.0
    fade_in: float = 0.0  # seconds
    fade_out: float = 0.0  # seconds
    reversed: bool = False
    pitch_semitones: float = 0.0  # transpose (length-preserving)
    # Tempo-lock: when lock_tempo is set, the clip's audio is (re)stretched from
    # orig_source_path so it stays in musical time as the project tempo changes.
    lock_tempo: Optional[float] = None       # BPM the original plays at 1×
    orig_source_path: Optional[str] = None   # un-stretched source to re-derive from
    lock_base_dur: float = 0.0               # clip duration at lock time (at lock_tempo)
    color: str = ""  # clip override; empty → inherit the track colour

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def is_midi(self) -> bool:
        return self.content_type == "midi"


@dataclass
class Track:
    """A horizontal lane of clips.

    MIDI clips on one track must not overlap (enforced by ``add_clip`` and
    geometry commands). Audio clips may stack; the engine mixes whatever is
    present.
    """

    id: str
    name: str
    clips: list[Clip] = field(default_factory=list)
    gain_db: float = 0.0
    pan: float = 0.0  # -1.0 (hard left) .. +1.0 (hard right)
    mute: bool = False
    solo: bool = False
    color: str = "#ff2e97"
    # Ordered insert *nodes*. Topology lives in ``fx_wires``; an empty wire
    # list means the implicit serial graph in → fx[0] → … → out.
    fx: list = field(default_factory=list)
    fx_wires: list = field(default_factory=list)  # list of :class:`FxWire`
    instrument: int = 0  # GM program / soundfont preset for MIDI clips
    is_drum: bool = False  # render MIDI on the GM percussion bank (drum kit)
    is_synth: bool = False  # render MIDI with the built-in subtractive synth
    synth: dict = field(default_factory=dict)  # synth patch (empty → engine defaults)
    plugin: str = ""  # VST3/AU instrument name; wins over is_synth when set
    plugin_state: str = ""  # the plugin's own preset blob, base64 (see serialize)

    def clip_by_id(self, clip_id: str) -> Optional[Clip]:
        for clip in self.clips:
            if clip.id == clip_id:
                return clip
        return None

    def midi_overlaps(
        self,
        start: float,
        duration: float,
        exclude_id: Optional[str] = None,
        exclude_ids: Optional[set] = None,
    ) -> bool:
        """True if a MIDI region ``[start, start+duration)`` hits another MIDI clip.

        Adjacent clips that only share a boundary (``end == start``) do not
        overlap. Audio clips are ignored — they may still stack on a lane.
        """
        skip = set(exclude_ids or ())
        if exclude_id:
            skip.add(exclude_id)
        end = float(start) + float(duration)
        if end <= start:
            return False
        for clip in self.clips:
            if not clip.is_midi or clip.id in skip:
                continue
            if start < clip.end - 1e-9 and clip.start < end - 1e-9:
                return True
        return False

    def nearest_midi_start(
        self,
        start: float,
        duration: float,
        exclude_id: Optional[str] = None,
    ) -> Optional[float]:
        """Legal MIDI start closest to ``start`` (adjacent if the drop overlaps).

        Returns ``None`` when no gap on this track can hold ``duration``.
        """
        start = max(0.0, float(start))
        duration = float(duration)
        if duration <= 0:
            return start
        if not self.midi_overlaps(start, duration, exclude_id=exclude_id):
            return start
        others = [c for c in self.clips if c.is_midi and c.id != exclude_id]
        overlapping = [
            c for c in others
            if start < c.end - 1e-9 and c.start < start + duration - 1e-9
        ]
        candidates: list[float] = []
        for clip in overlapping:
            left = clip.start - duration
            if left >= -1e-9:
                candidates.append(max(0.0, left))
            candidates.append(max(0.0, clip.end))
        valid = [
            s for s in candidates
            if s >= 0.0 and not self.midi_overlaps(s, duration, exclude_id=exclude_id)
        ]
        if not valid:
            return None
        return min(valid, key=lambda s: (abs(s - start), s))

    def clamp_midi_duration(
        self,
        start: float,
        duration: float,
        exclude_id: Optional[str] = None,
    ) -> float:
        """Shrink a right-edge resize so it does not overlap later MIDI clips."""
        start = float(start)
        duration = max(0.05, float(duration))
        end = start + duration
        limit = end
        for clip in self.clips:
            if not clip.is_midi or clip.id == exclude_id:
                continue
            if clip.start >= start - 1e-9 and start < clip.end and end > clip.start + 1e-9:
                limit = min(limit, clip.start)
        return max(0.05, limit - start)

    def fx_by_id(self, insert_id: str) -> Optional[FxInsert]:
        for ins in self.fx:
            if getattr(ins, "id", None) == insert_id or (
                isinstance(ins, dict) and ins.get("id") == insert_id
            ):
                return as_insert(ins)
        return None

    def fx_index(self, insert_id: str) -> Optional[int]:
        for i, ins in enumerate(self.fx):
            ident = getattr(ins, "id", None) or (ins.get("id") if isinstance(ins, dict) else "")
            if ident == insert_id:
                return i
        return None

    @property
    def is_master(self) -> bool:
        return self.id == MASTER_ID


def _default_master() -> "Track":
    return Track(id=MASTER_ID, name="Master", color="#ffd76b")


@dataclass
class Project:
    """Top-level container. Owns id generation and track lookup."""

    name: str = "Untitled"
    sample_rate: int = DEFAULT_SAMPLE_RATE
    tempo: float = 120.0  # BPM
    beats_per_bar: int = 4
    tracks: list[Track] = field(default_factory=list)
    # Mix bus: same Track shape (FX, gain, pan, mute) but never holds clips
    # and is mixed *after* every arrangement track is summed.
    master: Track = field(default_factory=_default_master)
    loop_enabled: bool = False
    loop_start: float = 0.0
    loop_end: float = 8.0  # 4 bars at the default 120 BPM
    _next_id: int = 1

    def loop_bounds(self) -> tuple[float, float]:
        """``(start, end)`` seconds, guaranteed ``end > start``."""
        start = max(0.0, float(self.loop_start))
        end = max(start + 0.05, float(self.loop_end))
        return start, end

    # ---- id generation ---------------------------------------------------
    def new_id(self, prefix: str) -> str:
        """Return a fresh, globally-unique id like ``"t1"`` / ``"c7"`` / ``"fx12"``."""
        ident = f"{prefix}{self._next_id}"
        self._next_id += 1
        return ident

    # ---- track/clip operations (thin; Commands wrap these in M2) ---------
    def add_track(self, name: Optional[str] = None) -> Track:
        from fantasia_core.document.colors import next_track_color

        track = Track(id=self.new_id("t"), name=name or f"Track {len(self.tracks) + 1}")
        track.color = next_track_color([t.color for t in self.tracks])
        self.tracks.append(track)
        return track

    def insert_track(self, index: int, track: Track) -> None:
        """Reinsert an existing track (used by undo of remove)."""
        self.tracks.insert(index, track)

    def remove_track(self, track_id: str) -> Optional[int]:
        """Remove a track by id; return the index it occupied (for undo).

        The master channel cannot be removed.
        """
        if track_id == MASTER_ID:
            return None
        for i, track in enumerate(self.tracks):
            if track.id == track_id:
                del self.tracks[i]
                return i
        return None

    def track_by_id(self, track_id: str) -> Optional[Track]:
        if track_id == MASTER_ID or (
            self.master is not None and track_id == self.master.id
        ):
            return self.master
        for track in self.tracks:
            if track.id == track_id:
                return track
        return None

    def channels(self) -> list[Track]:
        """Arrangement tracks plus the master bus (master last)."""
        return [*self.tracks, self.master]

    def track_index(self, track_id: str) -> Optional[int]:
        for i, track in enumerate(self.tracks):
            if track.id == track_id:
                return i
        return None

    def add_clip(
        self,
        track_id: str,
        start: float,
        duration: float,
        name: Optional[str] = None,
        **kwargs,
    ) -> Optional[Clip]:
        track = self.track_by_id(track_id)
        if track is None or getattr(track, "is_master", False):
            return None
        if str(kwargs.get("content_type") or "audio") == "midi":
            if track.midi_overlaps(start, duration):
                return None
        clip = Clip(
            id=self.new_id("c"),
            name=name or "Clip",
            start=start,
            duration=duration,
            **kwargs,
        )
        track.clips.append(clip)
        return clip

    def find_clip(self, clip_id: str) -> tuple[Optional[Track], Optional[Clip]]:
        """Locate a clip anywhere in the project, returning (track, clip)."""
        for track in self.tracks:
            clip = track.clip_by_id(clip_id)
            if clip is not None:
                return track, clip
        return None, None

    def new_insert(self, kind: str, params: Optional[dict] = None,
                   bypassed: bool = False) -> FxInsert:
        """Mint a new insert with a stable id (``fx12``)."""
        return FxInsert(
            id=self.new_id("fx"),
            type=str(kind),
            params=dict(params or {}),
            bypassed=bool(bypassed),
        )

    def ensure_inserts(self, track: Optional[Track] = None) -> None:
        """Coerce every FX entry to :class:`FxInsert` and fill missing ids."""
        channels = [track] if track is not None else self.channels()
        for ch in channels:
            if ch is None:
                continue
            ch.fx = mint_missing_ids(ch.fx, lambda: self.new_id("fx"))
            ch.fx_wires = sanitize_wires(ch.fx, getattr(ch, "fx_wires", None) or [])

    def find_insert(self, insert_id: str
                    ) -> tuple[Optional[Track], Optional[FxInsert], Optional[int]]:
        """Locate an insert on any channel, returning (track, insert, index)."""
        for track in self.channels():
            idx = track.fx_index(insert_id)
            if idx is not None:
                return track, as_insert(track.fx[idx]), idx
        return None, None, None

    # ---- derived ---------------------------------------------------------
    @property
    def duration(self) -> float:
        """Timeline length = end of the last clip (0.0 if empty)."""
        end = 0.0
        for track in self.tracks:
            for clip in track.clips:
                end = max(end, clip.end)
        return end

    def seconds_per_beat(self) -> float:
        return 60.0 / self.tempo if self.tempo > 0 else 0.5


# C major scale (semitone offsets from C) for seeding new MIDI clips.
_C_MAJOR = [0, 2, 4, 5, 7, 9, 11, 12]


def default_midi_pattern(duration: float, base_pitch: int = 60) -> List[Note]:
    """A simple ascending C-major scale filling ``duration`` — so a freshly
    'written' MIDI clip has something to see and hear before piano-roll editing."""
    n = len(_C_MAJOR)
    step = duration / n
    notes: List[Note] = []
    for i, semis in enumerate(_C_MAJOR):
        notes.append(
            Note(pitch=base_pitch + semis, start=i * step, duration=step * 0.9, velocity=100)
        )
    return notes


# General MIDI percussion notes.
DRUM_KICK, DRUM_SNARE, DRUM_HAT = 36, 38, 42


def default_drum_pattern(duration: float, seconds_per_beat: float) -> List[Note]:
    """A basic backbeat: kick on 1 & 3, snare on 2 & 4, hats on every 8th."""
    if seconds_per_beat <= 0:
        seconds_per_beat = 0.5
    beats = max(int(round(duration / seconds_per_beat)), 4)
    hit = seconds_per_beat * 0.25
    notes: List[Note] = []
    for b in range(beats):
        t = b * seconds_per_beat
        if b % 2 == 0:
            notes.append(Note(DRUM_KICK, t, hit, 110))
        else:
            notes.append(Note(DRUM_SNARE, t, hit, 105))
        notes.append(Note(DRUM_HAT, t, hit, 80))
        notes.append(Note(DRUM_HAT, t + seconds_per_beat / 2, hit, 70))
    return notes
