"""Render MIDI clips to audio buffers with FluidSynth (offline).

A MIDI clip's notes are synthesised to a stereo float32 buffer and cached, so
the mixer treats a MIDI clip exactly like an audio clip (same slicing, gain,
fades, FX, export). FluidSynth is used purely as an offline renderer via
``get_samples`` — no audio driver — and rendering happens only off the audio
thread (:meth:`render` / :meth:`warm`); the audio callback only reads the cache
(:meth:`cached`), which keeps the non-thread-safe synth off the callback thread.
"""

from __future__ import annotations

import glob
import os
import pathlib
from typing import Dict, Optional, Tuple

import numpy as np

DRUM_BANK = 128  # General MIDI percussion bank

_REPO_SF = str(
    pathlib.Path(__file__).resolve().parents[2] / "assets" / "soundfonts" / "*.sf2"
)
# Priority: repo-local GM soundfont (GeneralUser GS) → brew's bundled font → OS.
_SF_GLOBS = [
    _REPO_SF,
    "/opt/homebrew/opt/fluid-synth/share/fluid-synth/sf2/*.sf2",
    "/opt/homebrew/Cellar/fluid-synth/*/share/fluid-synth/sf2/*.sf2",
    "/usr/share/sounds/sf2/*.sf2",
]


def default_soundfont() -> Optional[str]:
    """Locate a usable ``.sf2`` (env override → repo-local GM font → brew → OS)."""
    env = os.environ.get("FANTASIA_SOUNDFONT")
    if env and os.path.isfile(env):
        return env
    for pattern in _SF_GLOBS:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


class MidiRenderer:
    def __init__(self, soundfont: Optional[str], sample_rate: int = 44100) -> None:
        self.soundfont = soundfont
        self.sr = sample_rate
        self._fs = None
        self._sfid = None
        self._cache: Dict[tuple, np.ndarray] = {}

    # ---- availability / lifecycle ---------------------------------------
    def available(self) -> bool:
        if not self.soundfont or not os.path.isfile(self.soundfont):
            return False
        try:
            import fluidsynth  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    def _ensure(self) -> bool:
        if self._fs is not None:
            return True
        if not self.available():
            return False
        import fluidsynth

        self._fs = fluidsynth.Synth(samplerate=float(self.sr))
        self._sfid = self._fs.sfload(self.soundfont)
        return True

    def close(self) -> None:
        if self._fs is not None:
            try:
                self._fs.delete()
            except Exception:  # noqa: BLE001
                pass
            self._fs = None

    # ---- keys / cache ----------------------------------------------------
    def _key(self, clip, instrument: int, is_drum: bool) -> Tuple:  # noqa: ANN001
        notes = tuple(
            (n.pitch, round(n.start, 4), round(n.duration, 4), n.velocity)
            for n in clip.notes
        )
        return (int(instrument), bool(is_drum), round(clip.duration, 4), notes)

    def cached(self, clip, instrument: int, is_drum: bool = False) -> Optional[np.ndarray]:  # noqa: ANN001
        """Audio-callback-safe: return the rendered buffer or None (never synthesises)."""
        return self._cache.get(self._key(clip, instrument, is_drum))

    def render(self, clip, instrument: int, is_drum: bool = False) -> np.ndarray:  # noqa: ANN001
        """Synthesise (UI thread) and cache. Returns silence if unavailable."""
        key = self._key(clip, instrument, is_drum)
        buf = self._cache.get(key)
        if buf is not None:
            return buf
        total = max(int(clip.duration * self.sr), 0)
        if not self._ensure():
            buf = np.zeros((total, 2), dtype=np.float32)
        else:
            buf = self._synth(clip, instrument, total, is_drum)
        self._cache[key] = buf
        return buf

    def warm(self, project) -> None:  # noqa: ANN001
        for track in project.tracks:
            for clip in track.clips:
                if clip.content_type == "midi":
                    self.render(clip, track.instrument, getattr(track, "is_drum", False))

    # ---- synthesis -------------------------------------------------------
    def _reset(self) -> None:
        try:
            self._fs.system_reset()
        except Exception:  # noqa: BLE001
            for p in range(128):
                self._fs.noteoff(0, p)

    def _synth(self, clip, instrument: int, total: int, is_drum: bool = False) -> np.ndarray:  # noqa: ANN001
        fs = self._fs
        sr = self.sr
        self._reset()
        bank = DRUM_BANK if is_drum else 0
        fs.program_select(0, self._sfid, bank, int(instrument))
        if total <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        # (frame, kind, pitch, vel); kind 0 = off, 1 = on. Offs sort before ons.
        events = []
        for note in clip.notes:
            s = max(0, int(note.start * sr))
            e = min(total, int((note.start + note.duration) * sr))
            if s >= total or e <= s:
                continue
            vel = int(max(1, min(127, note.velocity)))
            events.append((s, 1, int(note.pitch), vel))
            events.append((e, 0, int(note.pitch), 0))
        events.sort(key=lambda x: (x[0], x[1]))

        out = np.zeros((total, 2), dtype=np.float32)
        frame = 0
        i = 0
        while frame < total:
            next_f = min(events[i][0], total) if i < len(events) else total
            if next_f > frame:
                nf = next_f - frame
                seg = np.asarray(fs.get_samples(nf)).reshape(-1, 2).astype(np.float32)
                m = min(len(seg), total - frame)
                out[frame : frame + m] = seg[:m] / 32768.0
                frame = next_f
            while i < len(events) and events[i][0] <= frame:
                _, kind, pitch, vel = events[i]
                if kind == 1:
                    fs.noteon(0, pitch, vel)
                else:
                    fs.noteoff(0, pitch)
                i += 1
        return out
