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
import sys
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
    "/usr/share/sounds/sf2/*.sf2",       # Debian/Ubuntu (fluid-soundfont-*)
    "/usr/share/soundfonts/*.sf2",       # Arch / Fedora-style layout
]

# pyfluidsynth imports via ctypes.util.find_library (ldconfig / dyld only).
# Also search Homebrew, Debian, Fedora, and a user-local extract so MIDI
# works without a system package on the PATH.
_FLUID_LIB_GLOBS = [
    os.environ.get("FANTASIA_FLUIDSYNTH_LIB") or "",
    str(pathlib.Path.home() / ".local/lib/fantasia-deps/**/libfluidsynth.so*"),
    str(pathlib.Path.home() / ".local/lib/fantasia-deps/**/libfluidsynth*.dylib"),
    "/usr/lib64/libfluidsynth.so*",
    "/usr/lib/x86_64-linux-gnu/libfluidsynth.so*",
    "/usr/lib/libfluidsynth.so*",
    "/opt/homebrew/opt/fluid-synth/lib/libfluidsynth*.dylib",
    "/usr/local/opt/fluid-synth/lib/libfluidsynth*.dylib",
    "/opt/homebrew/lib/libfluidsynth*.dylib",
    "/usr/local/lib/libfluidsynth*.dylib",
]


def _native_elf(path: str) -> bool:
    """True if ``path`` is a dylib or an ELF matching this Python's bitness."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(5)
    except OSError:
        return False
    if head[:4] != b"\x7fELF":
        return True  # Mach-O / anything find_library already resolved
    want = 2 if sys.maxsize > 2**32 else 1
    return head[4] == want


def _find_fluidsynth_lib() -> Optional[str]:
    """Return a loadable libfluidsynth path, or None."""
    import ctypes.util

    for name in ("fluidsynth", "libfluidsynth", "fluidsynth-3", "libfluidsynth-3"):
        hit = ctypes.util.find_library(name)
        if hit and (not os.path.isfile(hit) or _native_elf(hit)):
            return hit
    found: list[str] = []
    for pattern in _FLUID_LIB_GLOBS:
        if not pattern:
            continue
        if os.path.isfile(pattern):
            found.append(pattern)
            continue
        found.extend(glob.glob(pattern, recursive=True))
    files = [p for p in found if os.path.isfile(p) and _native_elf(p)]
    # Prefer lib64 / x86_64, then the short soname symlink.
    files.sort(key=lambda p: (
        "lib64" not in p and "x86_64" not in p,
        len(pathlib.Path(p).name),
        p,
    ))
    return files[0] if files else None


def _prepare_fluidsynth_search() -> None:
    """Point pyfluidsynth at a known libfluidsynth if the loader cache misses."""
    import ctypes.util

    hit = _find_fluidsynth_lib()
    if not hit or ctypes.util.find_library("fluidsynth"):
        return
    orig = ctypes.util.find_library

    def find_library(name: str):  # noqa: ANN202
        found = orig(name)
        if found:
            return found
        if name and "fluid" in name.lower():
            return hit
        return None

    ctypes.util.find_library = find_library  # type: ignore[method-assign]


_prepare_fluidsynth_search()


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
        # A missing soundfont is a legitimate state — MIDI just renders silent
        # and the app still runs. Something that is not a path at all is a
        # caller bug, and silently degrading to silence hides it: the mix comes
        # out missing every MIDI track with no error anywhere.
        if soundfont is not None and not isinstance(soundfont, (str, os.PathLike)):
            raise TypeError(
                f"soundfont must be a path or None, got {type(soundfont).__name__} "
                f"({soundfont!r}) — the signature is MidiRenderer(soundfont, sample_rate)")
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
            _prepare_fluidsynth_search()
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
