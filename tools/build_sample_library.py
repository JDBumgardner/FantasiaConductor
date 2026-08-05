"""Build a starter sound library for the search system.

Two sources (the user picked "Both"):
  * SOUNDFONT — render short, representative phrases for a curated spread of
    General MIDI instruments (plus drum hits) from the GeneralUser GS soundfont
    via FluidSynth. Fast and deterministic.
  * MUSICGEN  — generate a handful of texture/FX sounds (pads, risers, impacts,
    ambience) from text prompts. Slow (~30-60s each), so it's opt-out.

Every sample is written to ``.fantasia_cache/library/`` and ingested into the
same LanceDB the app uses (``.fantasia_cache/soundlib.lancedb``) with 10-15
pre-authored tags each, so text/audio search returns useful results immediately.

Usage:
    .venv/bin/python tools/build_sample_library.py               # both passes
    .venv/bin/python tools/build_sample_library.py --no-musicgen # soundfont only
    .venv/bin/python tools/build_sample_library.py --only-musicgen
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import soundfile as sf

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from fantasia_core.document.model import Clip, Note  # noqa: E402
from fantasia_core.engine.midi_render import MidiRenderer, default_soundfont  # noqa: E402
from fantasia_core.search import SearchService  # noqa: E402

SR = 44100
LIB_DIR = _ROOT / ".fantasia_cache" / "library"
DB_PATH = str(_ROOT / ".fantasia_cache" / "soundlib.lancedb")

# ---- phrase builders -----------------------------------------------------
_TRIAD = (0, 4, 7)


def _clip(notes, duration):
    return Clip(id="_", name="_", start=0.0, duration=duration, content_type="midi", notes=notes)


def chord(root: int, dur: float = 2.4):
    notes = [Note(root + iv, 0.0, dur - 0.2, 96) for iv in _TRIAD]
    return _clip(notes, dur)


def arp(root: int, dur: float = 2.4):
    seq = [root, root + 4, root + 7, root + 12, root + 7, root + 4]
    step = dur / len(seq)
    notes = [Note(p, i * step, step * 1.6, 100) for i, p in enumerate(seq)]
    return _clip(notes, dur)


def sustained(root: int, dur: float = 2.4):
    return _clip([Note(root, 0.0, dur - 0.2, 100)], dur)


def bass_riff(root: int, dur: float = 2.4):
    seq = [(root, 0.0), (root, 0.5), (root + 7, 1.0), (root + 5, 1.5), (root, 2.0)]
    notes = [Note(p, t, 0.45, 110) for p, t in seq]
    return _clip(notes, dur)


_PHRASES = {"chord": chord, "arp": arp, "note": sustained, "riff": bass_riff}

# ---- curated GM instruments: (program, key, root, phrase, tags) ----------
INSTRUMENTS = [
    (0, "grand_piano", 60, "chord", ["acoustic piano", "grand piano", "piano", "keys", "keyboard", "bright", "percussive", "classical", "melodic", "warm", "natural", "chord"]),
    (4, "electric_piano", 60, "chord", ["electric piano", "rhodes", "keys", "keyboard", "warm", "mellow", "soft", "vintage", "soul", "jazz", "melodic", "chord"]),
    (6, "harpsichord", 60, "arp", ["harpsichord", "keys", "baroque", "classical", "plucked", "bright", "vintage", "ornate", "melodic", "arpeggio", "acoustic"]),
    (11, "vibraphone", 72, "arp", ["vibraphone", "vibes", "mallet", "chromatic percussion", "metallic", "bright", "jazz", "shimmering", "melodic", "arpeggio", "bell"]),
    (12, "marimba", 60, "arp", ["marimba", "mallet", "wooden", "percussion", "warm", "bright", "melodic", "acoustic", "arpeggio", "tuned percussion"]),
    (9, "glockenspiel", 84, "arp", ["glockenspiel", "bell", "mallet", "metallic", "bright", "shimmering", "high", "chromatic percussion", "melodic", "twinkle", "arpeggio"]),
    (16, "drawbar_organ", 48, "chord", ["organ", "drawbar organ", "hammond", "keys", "sustained", "warm", "vintage", "rock", "gospel", "chord", "electric"]),
    (19, "church_organ", 48, "chord", ["church organ", "pipe organ", "organ", "sustained", "cinematic", "sacred", "huge", "classical", "chord", "dramatic"]),
    (24, "nylon_guitar", 52, "arp", ["nylon guitar", "classical guitar", "acoustic guitar", "guitar", "plucked", "warm", "mellow", "fingerstyle", "melodic", "arpeggio", "spanish"]),
    (25, "steel_guitar", 52, "arp", ["steel string guitar", "acoustic guitar", "guitar", "plucked", "bright", "folk", "strummy", "melodic", "arpeggio", "organic"]),
    (27, "clean_guitar", 52, "arp", ["clean electric guitar", "electric guitar", "guitar", "plucked", "bright", "funk", "pop", "melodic", "arpeggio", "clean tone"]),
    (30, "distortion_guitar", 40, "chord", ["distortion guitar", "electric guitar", "guitar", "rock", "metal", "heavy", "aggressive", "power chord", "gritty", "loud", "chord"]),
    (33, "finger_bass", 36, "riff", ["fingered bass", "electric bass", "bass guitar", "bass", "low", "groovy", "warm", "funk", "rhythmic", "riff", "round"]),
    (34, "pick_bass", 36, "riff", ["picked bass", "electric bass", "bass guitar", "bass", "low", "bright", "attack", "rock", "rhythmic", "riff", "punchy"]),
    (38, "synth_bass", 36, "riff", ["synth bass", "bass", "electronic", "low", "sub", "punchy", "edm", "analog", "rhythmic", "riff", "808"]),
    (40, "violin", 67, "note", ["violin", "solo strings", "strings", "bowed", "expressive", "singing", "classical", "warm", "melodic", "orchestral", "sustained"]),
    (42, "cello", 48, "note", ["cello", "solo strings", "strings", "bowed", "deep", "warm", "rich", "classical", "melodic", "orchestral", "sustained"]),
    (45, "pizzicato", 55, "arp", ["pizzicato strings", "strings", "plucked", "staccato", "short", "playful", "classical", "orchestral", "melodic", "arpeggio"]),
    (46, "harp", 60, "arp", ["harp", "plucked", "strings", "glissando", "delicate", "dreamy", "classical", "shimmering", "melodic", "arpeggio", "ethereal"]),
    (48, "string_ensemble", 55, "chord", ["string ensemble", "strings", "orchestral", "lush", "warm", "cinematic", "sustained", "sweeping", "classical", "chord", "pad"]),
    (56, "trumpet", 60, "note", ["trumpet", "brass", "bright", "bold", "fanfare", "jazz", "orchestral", "punchy", "melodic", "sustained", "loud"]),
    (57, "trombone", 48, "note", ["trombone", "brass", "warm", "low", "bold", "jazz", "orchestral", "melodic", "sustained", "smooth"]),
    (61, "brass_section", 52, "chord", ["brass section", "brass", "horns", "punchy", "funk", "bold", "stabs", "energetic", "chord", "loud", "ensemble"]),
    (65, "alto_sax", 60, "note", ["alto saxophone", "sax", "reed", "woodwind", "smooth", "jazz", "warm", "expressive", "melodic", "sustained", "soulful"]),
    (68, "oboe", 67, "note", ["oboe", "reed", "woodwind", "nasal", "expressive", "classical", "orchestral", "melodic", "sustained", "reedy"]),
    (71, "clarinet", 55, "note", ["clarinet", "reed", "woodwind", "warm", "smooth", "classical", "jazz", "melodic", "sustained", "mellow"]),
    (73, "flute", 72, "note", ["flute", "woodwind", "airy", "breathy", "light", "bright", "classical", "melodic", "sustained", "soft", "high"]),
    (75, "pan_flute", 67, "note", ["pan flute", "woodwind", "breathy", "airy", "ethnic", "folk", "mellow", "melodic", "sustained", "dreamy"]),
    (80, "square_lead", 60, "arp", ["square lead", "synth lead", "synth", "electronic", "chiptune", "retro", "bright", "8-bit", "melodic", "arpeggio", "video game"]),
    (81, "saw_lead", 60, "arp", ["saw lead", "synth lead", "synth", "electronic", "bright", "cutting", "edm", "analog", "melodic", "arpeggio", "buzzy"]),
    (88, "new_age_pad", 52, "chord", ["new age pad", "synth pad", "pad", "synth", "warm", "ambient", "dreamy", "atmospheric", "lush", "sustained", "chord", "ethereal"]),
    (89, "warm_pad", 52, "chord", ["warm pad", "synth pad", "pad", "synth", "analog", "warm", "ambient", "atmospheric", "lush", "sustained", "chord", "soft"]),
    (94, "halo_pad", 52, "chord", ["halo pad", "synth pad", "pad", "synth", "ethereal", "airy", "ambient", "atmospheric", "choir-like", "sustained", "chord", "spacey"]),
    (104, "sitar", 52, "arp", ["sitar", "ethnic", "plucked", "indian", "exotic", "drone", "twangy", "world", "melodic", "arpeggio", "raga"]),
    (108, "kalimba", 60, "arp", ["kalimba", "thumb piano", "mallet", "plucked", "warm", "gentle", "ethnic", "melodic", "arpeggio", "wooden", "africa"]),
    (114, "steel_drums", 60, "arp", ["steel drums", "steelpan", "tuned percussion", "caribbean", "bright", "metallic", "tropical", "melodic", "arpeggio", "happy"]),
]

# ---- drums ---------------------------------------------------------------
_DRUM_HITS = [
    ("kick", 36, ["kick drum", "kick", "bass drum", "drum", "percussion", "low", "punchy", "boom", "one-shot", "acoustic", "beat"]),
    ("snare", 38, ["snare drum", "snare", "drum", "percussion", "crack", "backbeat", "sharp", "one-shot", "acoustic", "beat"]),
    ("closed_hat", 42, ["closed hi-hat", "hi-hat", "hat", "cymbal", "drum", "percussion", "tick", "crisp", "one-shot", "beat"]),
    ("open_hat", 46, ["open hi-hat", "hi-hat", "hat", "cymbal", "drum", "percussion", "sizzle", "sustained", "one-shot", "beat"]),
    ("crash", 49, ["crash cymbal", "cymbal", "crash", "drum", "percussion", "bright", "wash", "accent", "one-shot", "loud"]),
    ("clap", 39, ["hand clap", "clap", "percussion", "drum", "sharp", "backbeat", "electronic", "one-shot", "beat"]),
    ("tom", 45, ["tom", "tom-tom", "drum", "percussion", "low", "round", "fill", "one-shot", "acoustic", "beat"]),
    ("rimshot", 37, ["rimshot", "rim", "snare", "drum", "percussion", "click", "sharp", "one-shot", "beat"]),
]


def drum_beat(dur: float = 2.0):
    notes = []
    for t in (0.0, 1.0):
        notes.append(Note(36, t, 0.2, 120))       # kick on 1 & 3
    for t in (0.5, 1.5):
        notes.append(Note(38, t, 0.2, 110))       # snare on 2 & 4
    t = 0.0
    while t < dur:
        notes.append(Note(42, t, 0.1, 80))         # eighth-note hats
        t += 0.25
    return _clip(notes, dur)


def _write_wav(path: pathlib.Path, buf: np.ndarray) -> None:
    peak = float(np.max(np.abs(buf))) if buf.size else 0.0
    if peak > 1e-6:
        buf = buf * (0.9 / peak)  # peak-normalize
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), buf.astype(np.float32), SR)


def build_soundfont(items: list) -> None:
    sfont = default_soundfont()
    renderer = MidiRenderer(sfont, SR)
    if not renderer.available():
        print(f"!! soundfont/FluidSynth unavailable (sf={sfont}); skipping instrument render")
        return
    inst_dir = LIB_DIR / "instruments"
    print(f"Rendering {len(INSTRUMENTS)} instruments + drums from {pathlib.Path(sfont).name}")
    for prog, key, root, phrase, tags in INSTRUMENTS:
        clip = _PHRASES[phrase](root)
        buf = renderer.render(clip, prog, is_drum=False)
        path = inst_dir / f"{key}.wav"
        _write_wav(path, buf)
        items.append({"path": str(path), "name": key.replace("_", " "), "tags": tags})
    # drum beat + individual hits (drum bank)
    beat = drum_beat()
    bpath = inst_dir / "drum_beat.wav"
    _write_wav(bpath, renderer.render(beat, 0, is_drum=True))
    items.append({"path": str(bpath), "name": "drum beat",
                  "tags": ["drum beat", "drums", "beat", "groove", "percussion", "rhythm", "kick", "snare", "hi-hat", "loop", "acoustic kit"]})
    for key, pitch, tags in _DRUM_HITS:
        clip = _clip([Note(pitch, 0.0, 0.6, 120)], 1.0)
        path = inst_dir / f"drum_{key}.wav"
        _write_wav(path, renderer.render(clip, 0, is_drum=True))
        items.append({"path": str(path), "name": key.replace("_", " "), "tags": tags})
    renderer.close()
    print(f"  wrote {len(INSTRUMENTS) + 1 + len(_DRUM_HITS)} soundfont samples")


# ---- MusicGen texture/FX prompts ----------------------------------------
TEXTURES = [
    ("warm_analog_pad", "a warm evolving analog synth pad, lush and dreamy", ["warm pad", "analog", "synth pad", "pad", "ambient", "lush", "dreamy", "evolving", "atmospheric", "texture", "soft", "generated"]),
    ("dark_drone", "a dark cinematic drone, ominous and deep", ["drone", "dark", "cinematic", "ominous", "deep", "ambient", "tension", "texture", "atmospheric", "low", "generated"]),
    ("bright_riser", "a bright shimmering synth riser building tension", ["riser", "uplifter", "sweep", "build-up", "tension", "bright", "shimmering", "transition", "fx", "electronic", "generated"]),
    ("sub_rumble", "a deep sub bass rumble, cinematic low end", ["sub bass", "rumble", "low", "deep", "cinematic", "bass", "boom", "texture", "impact", "generated"]),
    ("choir_texture", "an airy ethereal vocal choir texture, ambient", ["choir", "vocal", "airy", "ethereal", "ambient", "angelic", "pad", "texture", "atmospheric", "voices", "generated"]),
    ("rain_ambience", "gentle rain with distant thunder ambience", ["rain", "thunder", "ambience", "field recording", "nature", "atmospheric", "weather", "texture", "calm", "background", "generated"]),
    ("ocean_waves", "ocean waves on a beach, field recording", ["ocean", "waves", "sea", "beach", "ambience", "field recording", "nature", "water", "texture", "calm", "generated"]),
    ("metal_impact", "a heavy metallic impact hit, cinematic", ["impact", "metallic", "hit", "cinematic", "trailer", "boom", "percussive", "fx", "hard", "accent", "generated"]),
    ("scifi_zap", "a sci-fi laser zap sound effect", ["laser", "zap", "sci-fi", "sound effect", "fx", "electronic", "futuristic", "blip", "synthetic", "generated"]),
    ("vinyl_crackle", "warm vinyl crackle and hiss texture", ["vinyl", "crackle", "hiss", "noise", "lo-fi", "texture", "vintage", "background", "warm", "analog", "generated"]),
    ("wind_howl", "howling wind ambience, cold and desolate", ["wind", "howl", "ambience", "cold", "desolate", "nature", "atmospheric", "texture", "eerie", "background", "generated"]),
    ("ambient_chime", "soft bell-like ambient chimes, peaceful", ["chime", "bell", "ambient", "peaceful", "soft", "shimmering", "meditative", "texture", "melodic", "gentle", "generated"]),
]


def build_musicgen(items: list, duration: float = 4.0) -> None:
    from fantasia_core import generate as gen

    if not gen.available():
        print("!! MusicGen unavailable (need torch+transformers); skipping texture generation")
        return
    tex_dir = LIB_DIR / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(TEXTURES)} MusicGen textures (~{duration:.0f}s each — this is slow)")
    for i, (key, prompt, tags) in enumerate(TEXTURES, 1):
        path = tex_dir / f"{key}.wav"
        if path.exists():
            print(f"  [{i}/{len(TEXTURES)}] {key} (exists, skip generate)")
        else:
            print(f"  [{i}/{len(TEXTURES)}] generating {key!r}…", flush=True)
            try:
                gen.generate_to_file(prompt, duration, SR, str(path))
            except Exception as exc:  # noqa: BLE001
                print(f"      failed: {exc}")
                continue
        items.append({"path": str(path), "name": key.replace("_", " "), "tags": tags})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-musicgen", action="store_true", help="skip the slow MusicGen pass")
    ap.add_argument("--only-musicgen", action="store_true", help="skip the soundfont pass")
    ap.add_argument("--duration", type=float, default=4.0, help="MusicGen clip length (s)")
    args = ap.parse_args()

    items: list = []
    if not args.only_musicgen:
        build_soundfont(items)
    if not args.no_musicgen:
        build_musicgen(items, args.duration)

    if not items:
        print("Nothing to ingest.")
        return 1

    print(f"Ingesting {len(items)} samples into {DB_PATH} (embedding with CLAP)…")
    svc = SearchService(DB_PATH)
    if not svc.available():
        print("!! CLAP unavailable (need torch+transformers) — samples written but not embedded.")
        return 1

    def _progress(done, total):
        if done % 8 == 0 or done == total:
            print(f"  embedded {done}/{total}")

    added = svc.ingest_tagged(items, progress=_progress)
    print(f"Done. Added {added} new samples. Library now holds {svc.count()} sounds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
