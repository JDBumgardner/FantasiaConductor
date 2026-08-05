"""Claude tool definitions + dispatch for driving the DAW.

Each mutation tool is a thin wrapper that dispatches a Command on the bus — no
new logic — so agent edits go through undo/redo and behave exactly like UI
edits. Query tools let the model discover track/clip ids before mutating.

``AgentTools`` is headless; the UI injects a ``refresh`` callback that re-renders
audio and rebuilds the view after each mutation. ``execute`` must be called on
the UI thread (it mutates the model and touches the renderers).
"""

from __future__ import annotations

from typing import Callable, List, Optional

from fantasia_core.commands import (
    AddClipCommand,
    AddTrackCommand,
    MakeMidiClipCommand,
    RemoveClipCommand,
    RemoveTrackCommand,
    SetClipAttrCommand,
    SetClipNotesCommand,
    SetClipSourceCommand,
    SetTempoCommand,
    SetTrackAttrCommand,
    SetTrackFxCommand,
    SetTrackSynthCommand,
    SetTrackSynthParamCommand,
    SplitClipCommand,
)
from fantasia_core.document.model import Note
from fantasia_core.engine import DEFAULT_PATCH, WAVEFORMS

_SYNTH_NUMERIC = {"mix", "detune", "attack", "decay", "sustain", "release",
                  "cutoff", "resonance", "env_amount", "gain"}


def _clean_patch(patch: dict) -> dict:
    """Keep only valid synth params, coercing types."""
    out = {}
    for key, val in (patch or {}).items():
        if key in ("osc1", "osc2"):
            if val in WAVEFORMS:
                out[key] = val
        elif key in _SYNTH_NUMERIC:
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                pass
    return out

_NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "pitch": {"type": "integer", "description": "MIDI note 0-127 (60=middle C)"},
        "start": {"type": "number", "description": "seconds from the clip start"},
        "duration": {"type": "number", "description": "seconds"},
        "velocity": {"type": "integer", "description": "1-127 (loudness), default 100"},
    },
    "required": ["pitch", "start", "duration"],
}


def _notes(items) -> List[Note]:
    out = []
    for n in items or []:
        out.append(
            Note(
                pitch=int(n["pitch"]),
                start=float(n["start"]),
                duration=float(n.get("duration", 0.25)),
                velocity=int(n.get("velocity", 100)),
            )
        )
    return out


class AgentTools:
    def __init__(self, bus, refresh: Optional[Callable[[], None]] = None,
                 search=None) -> None:
        self.bus = bus
        self._refresh = refresh
        self.search = search  # SearchService or None

    @property
    def project(self):
        return self.bus.project

    # ---- tool schemas ----------------------------------------------------
    def definitions(self) -> list:
        defs = [
            {"name": "get_project", "description": "Get tempo, time signature, duration, playhead, and track count.",
             "input_schema": {"type": "object", "properties": {}}},
            {"name": "list_tracks", "description": "List all tracks with their id, name, mode (drum/synth), instrument, mute/solo, gain, pan, and clip count.",
             "input_schema": {"type": "object", "properties": {}}},
            {"name": "list_clips", "description": "List clips, optionally filtered to one track. Returns id, track, name, start, duration, content_type (audio/midi/empty), note count.",
             "input_schema": {"type": "object", "properties": {"track_id": {"type": "string"}}}},
            {"name": "get_clip_notes", "description": "Get the notes of a MIDI clip.",
             "input_schema": {"type": "object", "properties": {"clip_id": {"type": "string"}}, "required": ["clip_id"]}},
            {"name": "add_track", "description": "Add a new track. Returns its id.",
             "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}}},
            {"name": "remove_track", "description": "Delete a track by id.",
             "input_schema": {"type": "object", "properties": {"track_id": {"type": "string"}}, "required": ["track_id"]}},
            {"name": "set_track", "description": "Set track properties. is_drum renders MIDI as a drum kit; is_synth uses the built-in synth; instrument is a GM program 0-127 (or drum-kit program for drum tracks).",
             "input_schema": {"type": "object", "properties": {
                 "track_id": {"type": "string"}, "name": {"type": "string"},
                 "mute": {"type": "boolean"}, "solo": {"type": "boolean"},
                 "gain_db": {"type": "number"}, "pan": {"type": "number", "description": "-1 (L) to 1 (R)"},
                 "is_drum": {"type": "boolean"}, "is_synth": {"type": "boolean"},
                 "instrument": {"type": "integer"}}, "required": ["track_id"]}},
            {"name": "add_fx", "description": "Append an effect to a track's FX chain.",
             "input_schema": {"type": "object", "properties": {
                 "track_id": {"type": "string"},
                 "type": {"type": "string", "enum": ["reverb", "delay", "lowpass", "highpass"]},
                 "params": {"type": "object"}}, "required": ["track_id", "type"]}},
            {"name": "clear_fx", "description": "Remove all effects from a track.",
             "input_schema": {"type": "object", "properties": {"track_id": {"type": "string"}}, "required": ["track_id"]}},
            {"name": "set_synth_param", "description": "Set one synth patch parameter on a synth track (osc1/osc2: sine|saw|square|triangle; mix,sustain,resonance,gain 0-1; detune semitones; attack,decay,release seconds; cutoff,env_amount Hz).",
             "input_schema": {"type": "object", "properties": {
                 "track_id": {"type": "string"}, "key": {"type": "string"}, "value": {}},
                 "required": ["track_id", "key", "value"]}},
            {"name": "get_synth_patch", "description": "Get the full current synth patch of a synth track.",
             "input_schema": {"type": "object", "properties": {"track_id": {"type": "string"}}, "required": ["track_id"]}},
            {"name": "set_synth_patch", "description": "Design a sound: set the whole synth patch at once (merges with the current patch). Params: osc1/osc2 (sine|saw|square|triangle), mix/sustain/resonance/gain (0-1), detune (semitones), attack/decay/release (seconds), cutoff/env_amount (Hz). Use this to create a sound from a description; add_fx layers reverb/delay/filters on top.",
             "input_schema": {"type": "object", "properties": {
                 "track_id": {"type": "string"}, "patch": {"type": "object"}}, "required": ["track_id", "patch"]}},
            {"name": "add_clip", "description": "Add an empty clip to a track at a start time (seconds) with a duration. Returns its id. Fill it with write_midi.",
             "input_schema": {"type": "object", "properties": {
                 "track_id": {"type": "string"}, "start": {"type": "number"},
                 "duration": {"type": "number"}, "name": {"type": "string"}},
                 "required": ["track_id", "start", "duration"]}},
            {"name": "remove_clip", "description": "Delete a clip by id.",
             "input_schema": {"type": "object", "properties": {"clip_id": {"type": "string"}}, "required": ["clip_id"]}},
            {"name": "set_clip", "description": "Set clip properties: gain_db, fade_in/fade_out (seconds), reversed, pitch_semitones (audio only).",
             "input_schema": {"type": "object", "properties": {
                 "clip_id": {"type": "string"}, "gain_db": {"type": "number"},
                 "fade_in": {"type": "number"}, "fade_out": {"type": "number"},
                 "reversed": {"type": "boolean"}, "pitch_semitones": {"type": "number"}},
                 "required": ["clip_id"]}},
            {"name": "write_midi", "description": "Turn a clip into a MIDI clip with the given notes (times are seconds from the clip start). For a drum track use GM drum pitches (36 kick, 38 snare, 42 closed hat, 46 open hat, 49 crash).",
             "input_schema": {"type": "object", "properties": {
                 "clip_id": {"type": "string"}, "notes": {"type": "array", "items": _NOTE_SCHEMA}},
                 "required": ["clip_id", "notes"]}},
            {"name": "set_notes", "description": "Replace the notes of an existing MIDI clip.",
             "input_schema": {"type": "object", "properties": {
                 "clip_id": {"type": "string"}, "notes": {"type": "array", "items": _NOTE_SCHEMA}},
                 "required": ["clip_id", "notes"]}},
            {"name": "generate_audio", "description": "Generate a real audio waveform from a text prompt with MusicGen (drum hits, textures, risers, ambience — sounds the synth can't make) and drop it into a clip. Give clip_id to fill an existing clip, or track_id (+ optional start) to create a new clip. Generation is slow (tens of seconds).",
             "input_schema": {"type": "object", "properties": {
                 "prompt": {"type": "string"}, "duration": {"type": "number", "description": "seconds (default 4)"},
                 "clip_id": {"type": "string"}, "track_id": {"type": "string"}, "start": {"type": "number"}},
                 "required": ["prompt"]}},
            {"name": "separate_stems", "description": "Isolate/extract instruments from an audio clip with Demucs — splits it into 4 stems (drums, bass, vocals, other), each placed on its own new track. Works on any audio clip (imported or generated). Slow (runs off-thread).",
             "input_schema": {"type": "object", "properties": {"clip_id": {"type": "string"}}, "required": ["clip_id"]}},
            {"name": "speak", "description": "Text-to-speech: synthesize spoken voice (Kokoro) and drop it into a clip. Give clip_id to fill an existing clip, or track_id (+ optional start) for a new clip. Voices: af_heart/af_bella/am_michael/am_adam (American), bf_emma/bm_george (British). Fast (runs on the GPU).",
             "input_schema": {"type": "object", "properties": {
                 "text": {"type": "string"}, "voice": {"type": "string"}, "speed": {"type": "number", "description": "0.5-2.0, default 1"},
                 "clip_id": {"type": "string"}, "track_id": {"type": "string"}, "start": {"type": "number"}},
                 "required": ["text"]}},
            {"name": "vocal_fx", "description": "Apply a WORLD-vocoder vocal effect to an audio (vocal) clip: 'autotune' (snap pitch to key+scale), 'harmony' (add a harmony voice a given interval up/down on a NEW track), 'formant_up'/'formant_down' (brighter/darker character), 'deess' (tame sibilance), 'double' (thicken). Formant-preserving, so it sounds natural.",
             "input_schema": {"type": "object", "properties": {
                 "clip_id": {"type": "string"},
                 "effect": {"type": "string", "enum": ["autotune", "harmony", "formant_up", "formant_down", "deess", "double"]},
                 "key": {"type": "string", "description": "autotune key, e.g. C, F#, Bb"},
                 "scale": {"type": "string", "enum": ["major", "minor", "harmonic_minor", "pentatonic", "chromatic"]},
                 "strength": {"type": "number", "description": "autotune 0-1"},
                 "semitones": {"type": "number", "description": "harmony interval, e.g. 4 (3rd), 7 (5th), -5"}},
                 "required": ["clip_id", "effect"]}},
            {"name": "sing_melody", "description": "Compose-and-sing IN TIME with the song: you provide the vocal melody as notes timed in BEATS (grid-locked to the project tempo) plus lyrics, and it renders a sung vocal onto a new track at the right bar. Use this to add a vocal that syncs with the arrangement — first call get_project for the tempo/key context, pick notes in the song's key, one syllable per note. Times are in beats so it's always on the grid.",
             "input_schema": {"type": "object", "properties": {
                 "notes": {"type": "array", "items": {"type": "object", "properties": {
                     "pitch": {"type": "integer", "description": "MIDI pitch (60=middle C)"},
                     "beat": {"type": "number", "description": "start in beats from start_beat"},
                     "beats": {"type": "number", "description": "duration in beats"},
                     "velocity": {"type": "integer"}}, "required": ["pitch", "beat", "beats"]}},
                 "lyrics": {"type": "string", "description": "one syllable per note; hyphen-split words"},
                 "start_beat": {"type": "number", "description": "where the vocal begins, in beats from the song start (0 = bar 1). A bar = beats_per_bar beats."},
                 "voice": {"type": "string"}},
                 "required": ["notes", "lyrics"]}},
            {"name": "sing", "description": "Sing a melody: turn a MIDI clip's notes into a sung vocal using lyrics (one syllable per note; split words with hyphens, e.g. 'fan-ta-si-a'). Vocoder-style singing placed on a new vocal track. Draw/write the melody first, then sing it.",
             "input_schema": {"type": "object", "properties": {
                 "clip_id": {"type": "string"}, "lyrics": {"type": "string"}, "voice": {"type": "string"}},
                 "required": ["clip_id", "lyrics"]}},
            {"name": "split_clip", "description": "Split a clip at an absolute timeline position (seconds).",
             "input_schema": {"type": "object", "properties": {
                 "clip_id": {"type": "string"}, "at_time": {"type": "number"}}, "required": ["clip_id", "at_time"]}},
            {"name": "stretch_clip", "description": "Time-stretch an audio clip by a factor (duration multiplier), keeping pitch unchanged (Rubber Band). factor 2.0 = twice as long / half speed, 0.5 = half as long / double speed. The clip's timeline length changes to match.",
             "input_schema": {"type": "object", "properties": {
                 "clip_id": {"type": "string"}, "factor": {"type": "number"}}, "required": ["clip_id", "factor"]}},
            {"name": "stretch_clip_to_bars", "description": "Time-stretch an audio clip so it spans exactly N bars at the current tempo (pitch preserved). Use to lock a loop/sample to the grid.",
             "input_schema": {"type": "object", "properties": {
                 "clip_id": {"type": "string"}, "bars": {"type": "number"}}, "required": ["clip_id", "bars"]}},
            {"name": "duplicate_clip", "description": "Copy/paste a clip: create a copy of a clip (with all its audio/MIDI content and settings) at a new start time, optionally on a different track. Use this to repeat a loop or pattern.",
             "input_schema": {"type": "object", "properties": {
                 "clip_id": {"type": "string"}, "start": {"type": "number", "description": "new clip start (seconds); default = same as source"},
                 "track_id": {"type": "string", "description": "destination track; default = same track"}},
                 "required": ["clip_id"]}},
            {"name": "duplicate_track", "description": "Copy/paste a whole track: create a new track that copies the source track's settings (instrument, FX, synth, gain/pan) and all its clips.",
             "input_schema": {"type": "object", "properties": {
                 "track_id": {"type": "string"}, "name": {"type": "string"}}, "required": ["track_id"]}},
            {"name": "set_tempo", "description": "Set the project tempo in BPM.",
             "input_schema": {"type": "object", "properties": {"bpm": {"type": "number"}}, "required": ["bpm"]}},
            {"name": "undo", "description": "Undo the last edit.", "input_schema": {"type": "object", "properties": {}}},
            {"name": "redo", "description": "Redo.", "input_schema": {"type": "object", "properties": {}}},
        ]
        if self.search is not None:
            defs += [
                {"name": "find_sound", "description": "Semantic search of the sound library for existing audio files matching a description (e.g. 'warm analog pad', 'punchy kick'). Returns matches with a path, name, duration, tags, and score. Use add_sound to drop one onto a track. Prefer this over generate_audio when a fitting sample may already exist.",
                 "input_schema": {"type": "object", "properties": {
                     "query": {"type": "string"}, "k": {"type": "integer", "description": "how many results (default 8)"}},
                     "required": ["query"]}},
                {"name": "add_sound", "description": "Place a sound from find_sound onto a track as an audio clip. Pass the result's path and duration.",
                 "input_schema": {"type": "object", "properties": {
                     "path": {"type": "string"}, "duration": {"type": "number"},
                     "track_id": {"type": "string"}, "start": {"type": "number"}, "name": {"type": "string"}},
                     "required": ["path", "duration", "track_id"]}},
            ]
        return defs

    # ---- dispatch --------------------------------------------------------
    def execute(self, name: str, args: dict):
        args = args or {}
        result = self._dispatch(name, args)
        _READS = ("get_project", "list_tracks", "list_clips", "get_clip_notes", "find_sound")
        if name not in _READS and self._refresh:
            self._refresh()
        return result

    def _dispatch(self, name: str, a: dict):
        p = self.project
        if name == "get_project":
            return {"tempo": p.tempo, "beats_per_bar": p.beats_per_bar,
                    "duration": round(p.duration, 3), "num_tracks": len(p.tracks),
                    "seconds_per_beat": round(p.seconds_per_beat(), 4)}
        if name == "list_tracks":
            return [{"id": t.id, "name": t.name, "is_drum": t.is_drum, "is_synth": t.is_synth,
                     "instrument": t.instrument, "mute": t.mute, "solo": t.solo,
                     "gain_db": t.gain_db, "pan": t.pan, "num_clips": len(t.clips)} for t in p.tracks]
        if name == "list_clips":
            tid = a.get("track_id")
            rows = []
            for t in p.tracks:
                if tid and t.id != tid:
                    continue
                for c in t.clips:
                    ctype = "midi" if c.is_midi else ("audio" if c.source_path else "empty")
                    rows.append({"id": c.id, "track_id": t.id, "track_name": t.name, "name": c.name,
                                 "start": c.start, "duration": c.duration, "content_type": ctype,
                                 "num_notes": len(c.notes)})
            return rows
        if name == "get_clip_notes":
            _, c = p.find_clip(a["clip_id"])
            if c is None:
                return {"error": "clip not found"}
            return [{"pitch": n.pitch, "start": n.start, "duration": n.duration, "velocity": n.velocity}
                    for n in c.notes]

        if name == "add_track":
            cmd = self.bus.dispatch(AddTrackCommand(a.get("name")))
            return {"track_id": cmd.created_track.id, "name": cmd.created_track.name}
        if name == "remove_track":
            self.bus.dispatch(RemoveTrackCommand(a["track_id"]))
            return {"ok": True}
        if name == "set_track":
            for key in ("name", "mute", "solo", "gain_db", "pan", "is_drum", "is_synth", "instrument"):
                if key in a:
                    self.bus.dispatch(SetTrackAttrCommand(a["track_id"], key, a[key]))
            return {"ok": True}
        if name == "add_fx":
            t = p.track_by_id(a["track_id"])
            if t is None:
                return {"error": "track not found"}
            fx = list(t.fx) + [{"type": a["type"], "params": a.get("params", {})}]
            self.bus.dispatch(SetTrackFxCommand(a["track_id"], fx, label=f"Add {a['type']}"))
            return {"ok": True, "fx_count": len(fx)}
        if name == "clear_fx":
            self.bus.dispatch(SetTrackFxCommand(a["track_id"], [], label="Clear FX"))
            return {"ok": True}
        if name == "set_synth_param":
            self.bus.dispatch(SetTrackSynthParamCommand(a["track_id"], a["key"], a["value"]))
            return {"ok": True}
        if name == "get_synth_patch":
            t = p.track_by_id(a["track_id"])
            if t is None:
                return {"error": "track not found"}
            return {**DEFAULT_PATCH, **t.synth}
        if name == "set_synth_patch":
            t = p.track_by_id(a["track_id"])
            if t is None:
                return {"error": "track not found"}
            merged = {**t.synth, **_clean_patch(a["patch"])}
            self.bus.dispatch(SetTrackSynthCommand(a["track_id"], merged))
            return {"ok": True, "patch": {**DEFAULT_PATCH, **merged}}

        if name == "add_clip":
            cmd = self.bus.dispatch(AddClipCommand(a["track_id"], float(a["start"]),
                                                   float(a["duration"]), name=a.get("name", "Clip")))
            clip = cmd.created_clip
            return {"clip_id": clip.id} if clip else {"error": "track not found"}
        if name == "remove_clip":
            self.bus.dispatch(RemoveClipCommand(a["clip_id"]))
            return {"ok": True}
        if name == "set_clip":
            for key in ("gain_db", "fade_in", "fade_out", "reversed", "pitch_semitones"):
                if key in a:
                    self.bus.dispatch(SetClipAttrCommand(a["clip_id"], key, a[key]))
            return {"ok": True}
        if name == "write_midi":
            self.bus.dispatch(MakeMidiClipCommand(a["clip_id"], _notes(a["notes"])))
            return {"ok": True, "num_notes": len(a["notes"])}
        if name == "set_notes":
            self.bus.dispatch(SetClipNotesCommand(a["clip_id"], _notes(a["notes"])))
            return {"ok": True, "num_notes": len(a["notes"])}
        if name == "split_clip":
            self.bus.dispatch(SplitClipCommand(a["clip_id"], float(a["at_time"])))
            return {"ok": True}
        if name == "duplicate_clip":
            src_track, c = p.find_clip(a["clip_id"])
            if c is None:
                return {"error": "clip not found"}
            tid = a.get("track_id") or (src_track.id if src_track else None)
            if tid is None:
                return {"error": "no destination track"}
            notes = [Note(n.pitch, n.start, n.duration, n.velocity) for n in c.notes]
            cmd = self.bus.dispatch(AddClipCommand(
                tid, float(a.get("start", c.start)), c.duration, name=c.name,
                content_type=c.content_type, source_path=c.source_path,
                source_offset=c.source_offset, notes=notes, gain_db=c.gain_db,
                fade_in=c.fade_in, fade_out=c.fade_out, reversed=c.reversed,
                pitch_semitones=c.pitch_semitones))
            return {"clip_id": cmd.created_clip.id} if cmd.created_clip else {"error": "track not found"}
        if name == "duplicate_track":
            src = p.track_by_id(a["track_id"])
            if src is None:
                return {"error": "track not found"}
            cmd = self.bus.dispatch(AddTrackCommand(a.get("name") or f"{src.name} copy"))
            nt = cmd.created_track
            for attr in ("gain_db", "pan", "mute", "solo", "color", "instrument",
                         "is_drum", "is_synth"):
                setattr(nt, attr, getattr(src, attr))
            nt.fx = [dict(fx) for fx in src.fx]
            nt.synth = dict(getattr(src, "synth", {}) or {})
            for c in src.clips:
                notes = [Note(n.pitch, n.start, n.duration, n.velocity) for n in c.notes]
                self.bus.dispatch(AddClipCommand(
                    nt.id, c.start, c.duration, name=c.name, content_type=c.content_type,
                    source_path=c.source_path, source_offset=c.source_offset, notes=notes,
                    gain_db=c.gain_db, fade_in=c.fade_in, fade_out=c.fade_out,
                    reversed=c.reversed, pitch_semitones=c.pitch_semitones))
            return {"track_id": nt.id, "num_clips": len(src.clips)}
        if name == "generate_audio":
            # The heavy generation runs on the agent worker thread (see _AgentWorker);
            # if it reaches here it was mis-routed.
            return {"error": "generate_audio must be run off the UI thread"}
        if name == "separate_stems":
            return {"error": "separate_stems must be run off the UI thread"}
        if name == "speak":
            return {"error": "speak must be run off the UI thread"}
        if name == "sing":
            return {"error": "sing must be run off the UI thread"}
        if name == "sing_melody":
            return {"error": "sing_melody must be run off the UI thread"}
        if name == "vocal_fx":
            return {"error": "vocal_fx must be run off the UI thread"}
        if name in ("stretch_clip", "stretch_clip_to_bars"):
            return {"error": f"{name} must be run off the UI thread"}
        if name == "_fill_generated":
            clip_id = a.get("clip_id")
            if not clip_id:
                tid = a.get("track_id") or (p.tracks[0].id if p.tracks else None)
                if tid is None:
                    return {"error": "no track to add the clip to"}
                cmd = self.bus.dispatch(AddClipCommand(tid, float(a.get("start", 0.0)),
                                                       float(a["duration"]), name="Generated"))
                clip_id = cmd.created_clip.id if cmd.created_clip else None
            if clip_id is None:
                return {"error": "could not create clip"}
            self.bus.dispatch(SetClipSourceCommand(clip_id, a["path"], 0.0, float(a["duration"])))
            return {"ok": True, "clip_id": clip_id}
        if name == "find_sound":
            if self.search is None:
                return {"error": "sound search unavailable"}
            try:
                return {"results": self.search.search_text(a["query"], int(a.get("k", 8)))}
            except Exception as exc:  # noqa: BLE001
                return {"error": str(exc)}
        if name == "add_sound":
            import os

            tid = a.get("track_id") or (p.tracks[0].id if p.tracks else None)
            if tid is None:
                return {"error": "no track to add the sound to"}
            dur = float(a.get("duration") or 0.0)
            if dur <= 0:
                return {"error": "duration required"}
            name_ = a.get("name") or os.path.basename(a["path"])
            cmd = self.bus.dispatch(AddClipCommand(tid, float(a.get("start", 0.0)), dur, name=name_))
            cid = cmd.created_clip.id if cmd.created_clip else None
            if cid is None:
                return {"error": "could not create clip"}
            self.bus.dispatch(SetClipSourceCommand(cid, a["path"], 0.0, dur))
            return {"ok": True, "clip_id": cid}
        if name == "set_tempo":
            self.bus.dispatch(SetTempoCommand(float(a["bpm"])))  # undoable
            return {"ok": True, "tempo": p.tempo}
        if name == "undo":
            self.bus.undo()
            return {"ok": True}
        if name == "redo":
            self.bus.redo()
            return {"ok": True}
        return {"error": f"unknown tool {name}"}
