"""JSON serialization for the document model.

Explicit (not ``dataclasses.asdict``) so the on-disk schema is versioned and can
evolve independently of the in-memory dataclasses. A project file is a plain
``.fcp`` JSON document.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fantasia_core.document.model import Clip, Note, Project, Track

FORMAT = "fantasia-project"
VERSION = 1


def clip_to_dict(clip: Clip) -> dict[str, Any]:
    return {
        "id": clip.id,
        "name": clip.name,
        "start": clip.start,
        "duration": clip.duration,
        "content_type": clip.content_type,
        "source_path": clip.source_path,
        "source_offset": clip.source_offset,
        "notes": [
            {"pitch": n.pitch, "start": n.start, "duration": n.duration, "velocity": n.velocity}
            for n in clip.notes
        ],
        "gain_db": clip.gain_db,
        "fade_in": clip.fade_in,
        "fade_out": clip.fade_out,
        "reversed": clip.reversed,
        "pitch_semitones": clip.pitch_semitones,
        "lock_tempo": clip.lock_tempo,
        "orig_source_path": clip.orig_source_path,
        "lock_base_dur": clip.lock_base_dur,
    }


def track_to_dict(track: Track) -> dict[str, Any]:
    return {
        "id": track.id,
        "name": track.name,
        "gain_db": track.gain_db,
        "pan": track.pan,
        "mute": track.mute,
        "solo": track.solo,
        "color": track.color,
        "fx": [dict(e) for e in track.fx],
        "instrument": track.instrument,
        "is_drum": track.is_drum,
        "is_synth": track.is_synth,
        "synth": dict(track.synth),
        "clips": [clip_to_dict(c) for c in track.clips],
    }


def project_to_dict(project: Project) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "version": VERSION,
        "name": project.name,
        "sample_rate": project.sample_rate,
        "tempo": project.tempo,
        "beats_per_bar": project.beats_per_bar,
        "next_id": project._next_id,
        "tracks": [track_to_dict(t) for t in project.tracks],
    }


def clip_from_dict(data: dict[str, Any]) -> Clip:
    return Clip(
        id=data["id"],
        name=data.get("name", "Clip"),
        start=float(data["start"]),
        duration=float(data["duration"]),
        content_type=data.get("content_type", "audio"),
        source_path=data.get("source_path"),
        source_offset=float(data.get("source_offset", 0.0)),
        notes=[
            Note(
                pitch=int(nd["pitch"]),
                start=float(nd["start"]),
                duration=float(nd["duration"]),
                velocity=int(nd.get("velocity", 100)),
            )
            for nd in data.get("notes", [])
        ],
        gain_db=float(data.get("gain_db", 0.0)),
        fade_in=float(data.get("fade_in", 0.0)),
        fade_out=float(data.get("fade_out", 0.0)),
        reversed=bool(data.get("reversed", False)),
        pitch_semitones=float(data.get("pitch_semitones", 0.0)),
        lock_tempo=data.get("lock_tempo"),
        orig_source_path=data.get("orig_source_path"),
        lock_base_dur=float(data.get("lock_base_dur", 0.0)),
    )


def track_from_dict(data: dict[str, Any]) -> Track:
    track = Track(
        id=data["id"],
        name=data.get("name", "Track"),
        gain_db=float(data.get("gain_db", 0.0)),
        pan=float(data.get("pan", 0.0)),
        mute=bool(data.get("mute", False)),
        solo=bool(data.get("solo", False)),
        color=data.get("color", "#4a90d9"),
    )
    track.fx = [dict(e) for e in data.get("fx", [])]
    track.instrument = int(data.get("instrument", 0))
    track.is_drum = bool(data.get("is_drum", False))
    track.is_synth = bool(data.get("is_synth", False))
    track.synth = dict(data.get("synth", {}))
    track.clips = [clip_from_dict(c) for c in data.get("clips", [])]
    return track


def project_from_dict(data: dict[str, Any]) -> Project:
    if data.get("format") != FORMAT:
        raise ValueError(f"Not a {FORMAT} document")
    project = Project(
        name=data.get("name", "Untitled"),
        sample_rate=int(data.get("sample_rate", 44100)),
        tempo=float(data.get("tempo", 120.0)),
        beats_per_bar=int(data.get("beats_per_bar", 4)),
    )
    project.tracks = [track_from_dict(t) for t in data.get("tracks", [])]
    project._next_id = int(data.get("next_id", 1))
    return project


def save_project(project: Project, path: str | Path) -> None:
    path = Path(path)
    path.write_text(json.dumps(project_to_dict(project), indent=2), encoding="utf-8")


def load_project(path: str | Path) -> Project:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return project_from_dict(data)
