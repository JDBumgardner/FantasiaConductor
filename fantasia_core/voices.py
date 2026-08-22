"""Reference-voice catalog — the pool of timbres the clone backend draws from.

A *reference voice* is a short clip of someone speaking. The engine conditions
on it and then speaks new text in that timbre, zero-shot — no training.

A transcript (``ref_text``) is stored alongside but is optional: Chatterbox, the
current engine, works from audio alone. Transcript-conditioned engines
(Fish Speech, VoxCPM) clone noticeably better when they have one, so it is worth
filling in — that is the only reason the field exists.

Two ways to fill the catalog:

* :func:`build_builtin` renders each Kokoro voice speaking :data:`REFERENCE_LINE`.
  Those are *synthetic* voices, so the starter catalog carries no likeness of a
  real person; it exists so there is something to pick from on a fresh install.
* :func:`add_voice` takes any WAV — your own mic recordings being the obvious
  source.

Clips are stored under ``.fantasia_cache/voices/`` as mono 44.1kHz WAV beside a
JSON sidecar. Headless: no Qt in here.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import shutil
from typing import List, Optional

import numpy as np

# Phonetically broad: cloning works better from a clip covering many phonemes
# than from a longer but repetitive one.
REFERENCE_LINE = (
    "The quick brown fox jumps over the lazy dog. "
    "She sells sea shells by the shore, and I owe you a huge favour. "
    "Bright vowels, soft consonants, a question? Then a firm answer."
)

# The project rate. Engines resample as needed; storing one rate keeps the
# catalog engine-agnostic.
REF_SR = 44100
# Past ~15s the reference stops helping and just costs prompt tokens.
MAX_REF_SECONDS = 15.0


def catalog_dir() -> pathlib.Path:
    d = pathlib.Path(os.environ.get("FANTASIA_VOICES", "")) if os.environ.get(
        "FANTASIA_VOICES") else pathlib.Path.cwd() / ".fantasia_cache" / "voices"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclasses.dataclass
class RefVoice:
    slug: str
    name: str
    ref_text: str
    path: str
    source: str = "custom"      # "builtin" (synthetic) | "recording" | "custom"
    tags: List[str] = dataclasses.field(default_factory=list)
    seconds: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.name}  ({self.seconds:.0f}s)" if self.seconds else self.name


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return s or "voice"


def _sidecar(slug: str) -> pathlib.Path:
    return catalog_dir() / f"{slug}.json"


def list_voices() -> List[RefVoice]:
    """Every reference voice in the catalog, alphabetical by display name."""
    out = []
    for js in sorted(catalog_dir().glob("*.json")):
        try:
            d = json.loads(js.read_text())
            wav = catalog_dir() / f"{js.stem}.wav"
            if not wav.exists():
                continue                      # sidecar without audio: skip
            out.append(RefVoice(slug=js.stem, name=d.get("name", js.stem),
                                ref_text=d.get("ref_text", ""), path=str(wav),
                                source=d.get("source", "custom"),
                                tags=list(d.get("tags", [])),
                                seconds=float(d.get("seconds", 0.0))))
        except Exception:                     # noqa: BLE001 — a bad sidecar shouldn't
            continue                          # take down the whole catalog
    return sorted(out, key=lambda v: v.name.lower())


def get(slug: str) -> Optional[RefVoice]:
    for v in list_voices():
        if v.slug == slug:
            return v
    return None


def remove(slug: str) -> bool:
    hit = False
    for p in (catalog_dir() / f"{slug}.wav", _sidecar(slug)):
        if p.exists():
            p.unlink()
            hit = True
    return hit


def add_voice(name: str, wav_path: str, ref_text: str = "", *, source: str = "custom",
              tags: Optional[List[str]] = None, slug: Optional[str] = None) -> RefVoice:
    """Import ``wav_path`` as a reference voice: mono, 44.1kHz, <=15s, normalized.

    ``ref_text``, if given, must be what is actually *said* in the clip — a wrong
    transcript is worse than none, since transcript-conditioned engines align the
    audio against it.
    """
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)                            # to mono
    if sr != REF_SR and len(audio):
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=REF_SR)
    audio = audio[: int(MAX_REF_SECONDS * REF_SR)].astype(np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio = (audio / peak * 0.95).astype(np.float32)

    slug = slug or slugify(name)
    dest = catalog_dir() / f"{slug}.wav"
    sf.write(str(dest), audio, REF_SR, subtype="PCM_16")
    seconds = len(audio) / REF_SR
    _sidecar(slug).write_text(json.dumps(
        {"name": name, "ref_text": ref_text.strip(), "source": source,
         "tags": tags or [], "seconds": round(seconds, 2)}, indent=2))
    return RefVoice(slug, name, ref_text.strip(), str(dest), source,
                    list(tags or []), seconds)


def load_ref(voice: RefVoice) -> np.ndarray:
    """The reference clip as mono float32 at :data:`REF_SR`."""
    import soundfile as sf

    audio, sr = sf.read(voice.path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != REF_SR and len(audio):
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=REF_SR)
    return audio.astype(np.float32)


def build_builtin(progress=None, overwrite: bool = False) -> List[RefVoice]:
    """Seed the catalog by having each Kokoro voice read :data:`REFERENCE_LINE`.

    Kokoro is small and fast, so this takes well under a minute and needs no
    network beyond the Kokoro weights already used for TTS.
    """
    from fantasia_core import tts

    if not tts.available():
        raise RuntimeError("Kokoro (mlx-audio + misaki) is not installed")

    made = []
    tmp = catalog_dir() / "_tmp_builtin.wav"
    for i, (vid, label) in enumerate(tts.VOICES):
        slug = f"builtin_{vid}"
        if not overwrite and get(slug) is not None:
            made.append(get(slug))
            continue
        if progress:
            progress(i, len(tts.VOICES), label)
        tts.synthesize_to_file(REFERENCE_LINE, str(tmp), voice=vid,
                               sr_out=REF_SR, backend="kokoro")
        made.append(add_voice(label, str(tmp), REFERENCE_LINE, source="builtin",
                              tags=["synthetic", "kokoro"], slug=slug))
    if tmp.exists():
        tmp.unlink()
    if progress:
        progress(len(tts.VOICES), len(tts.VOICES), "done")
    return made


def import_recording(path: str, name: str, ref_text: str = "") -> RefVoice:
    """Add one of your own mic takes to the catalog."""
    return add_voice(name, path, ref_text, source="recording", tags=["recorded"])
