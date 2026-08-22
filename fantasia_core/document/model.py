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
* **IDs are stable strings** (``"t1"``, ``"c2"``) from a single monotonic
  counter on the project. Commands (M2) and the agent (M6) address tracks and
  clips by these ids, and they must survive save/load — so the counter is part
  of the serialized form.
* Audio content is referenced by ``source_path`` and only *loaded* by the
  engine (M3); the model just records where a clip's audio comes from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

DEFAULT_SAMPLE_RATE = 44100


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

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def is_midi(self) -> bool:
        return self.content_type == "midi"


@dataclass
class Track:
    """A horizontal lane holding non-overlapping clips (overlap allowed for now;
    the engine mixes whatever is present)."""

    id: str
    name: str
    clips: list[Clip] = field(default_factory=list)
    gain_db: float = 0.0
    pan: float = 0.0  # -1.0 (hard left) .. +1.0 (hard right)
    mute: bool = False
    solo: bool = False
    color: str = "#4a90d9"
    fx: list = field(default_factory=list)  # list of {"type": str, "params": dict}
    instrument: int = 0  # GM program / soundfont preset for MIDI clips
    is_drum: bool = False  # render MIDI on the GM percussion bank (drum kit)
    is_synth: bool = False  # render MIDI with the built-in subtractive synth
    synth: dict = field(default_factory=dict)  # synth patch (empty → engine defaults)

    def clip_by_id(self, clip_id: str) -> Optional[Clip]:
        for clip in self.clips:
            if clip.id == clip_id:
                return clip
        return None


@dataclass
class Project:
    """Top-level container. Owns id generation and track lookup."""

    name: str = "Untitled"
    sample_rate: int = DEFAULT_SAMPLE_RATE
    tempo: float = 120.0  # BPM
    beats_per_bar: int = 4
    tracks: list[Track] = field(default_factory=list)
    _next_id: int = 1

    # ---- id generation ---------------------------------------------------
    def new_id(self, prefix: str) -> str:
        """Return a fresh, globally-unique id like ``"t1"`` / ``"c7"``."""
        ident = f"{prefix}{self._next_id}"
        self._next_id += 1
        return ident

    # ---- track/clip operations (thin; Commands wrap these in M2) ---------
    def add_track(self, name: Optional[str] = None) -> Track:
        track = Track(id=self.new_id("t"), name=name or f"Track {len(self.tracks) + 1}")
        self.tracks.append(track)
        return track

    def insert_track(self, index: int, track: Track) -> None:
        """Reinsert an existing track (used by undo of remove)."""
        self.tracks.insert(index, track)

    def remove_track(self, track_id: str) -> Optional[int]:
        """Remove a track by id; return the index it occupied (for undo)."""
        for i, track in enumerate(self.tracks):
            if track.id == track_id:
                del self.tracks[i]
                return i
        return None

    def track_by_id(self, track_id: str) -> Optional[Track]:
        for track in self.tracks:
            if track.id == track_id:
                return track
        return None

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
        if track is None:
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
