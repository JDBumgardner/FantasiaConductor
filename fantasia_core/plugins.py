"""Audio plugins — hosting VST3/AU instruments and effects, and exposing them
to the agent.

The point is to host rather than reimplement. A synth like Vital ships tens of
thousands of lines of DSP *and* its whole interface inside the plugin; loading
it gives all of that, and every future update, for the cost of routing notes in
and audio out. It also keeps a GPL plugin's licence at arm's length from this
codebase, which reimplementing it against the plugin's source would not.

For the agent the interesting surface is the parameter list. A big synth exposes
hundreds of them, each with a human-readable name and value ("Filter 1 Cutoff",
"440 Hz"), and the plugin converts between text and its own normalized range —
so an instruction like "open the filter up" becomes a search plus a set, with no
per-plugin knowledge baked in here.

Headless: no Qt.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
from typing import Dict, List, Optional, Sequence

import numpy as np

# Where the OS keeps plugins. A user-level folder is searched too because that
# is where an installer without admin rights puts them.
_SEARCH = {
    "darwin": [
        "/Library/Audio/Plug-Ins/VST3", "~/Library/Audio/Plug-Ins/VST3",
        "/Library/Audio/Plug-Ins/Components", "~/Library/Audio/Plug-Ins/Components",
    ],
    "linux": ["/usr/lib/vst3", "/usr/local/lib/vst3", "~/.vst3"],
    "win32": [r"C:\Program Files\Common Files\VST3"],
}


def available() -> bool:
    try:
        import pedalboard  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


@dataclasses.dataclass
class PluginInfo:
    name: str
    path: str
    format: str            # "VST3" | "AU"
    is_instrument: bool
    manufacturer: str = ""

    @property
    def slug(self) -> str:
        return "".join(c for c in self.name.lower() if c.isalnum())


def search_paths() -> List[pathlib.Path]:
    import sys

    key = "darwin" if sys.platform == "darwin" else (
        "win32" if sys.platform.startswith("win") else "linux")
    out = []
    for p in _SEARCH.get(key, []):
        d = pathlib.Path(p).expanduser()
        if d.is_dir():
            out.append(d)
    extra = os.environ.get("FANTASIA_PLUGIN_PATH")
    if extra:
        out += [pathlib.Path(x).expanduser() for x in extra.split(os.pathsep)
                if pathlib.Path(x).expanduser().is_dir()]
    return out


# Above this, a "discrete" parameter is continuous in everything but the flag.
_MAX_CHOICE_STEPS = 256

_SCAN: Optional[List[PluginInfo]] = None


def scan(refresh: bool = False) -> List[PluginInfo]:
    """Installed plugins. Cached, because probing each file loads it."""
    global _SCAN
    if _SCAN is not None and not refresh:
        return _SCAN
    found: List[PluginInfo] = []
    if available():
        for d in search_paths():
            for ext, fmt in ((".vst3", "VST3"), (".component", "AU")):
                for f in sorted(d.glob(f"*{ext}")):
                    found.append(PluginInfo(f.stem, str(f), fmt, True))
    _SCAN = found
    return found


# A loaded plugin belongs to the thread that loaded it — pedalboard refuses to
# use one from anywhere else ("must be reloaded on the current thread"), and
# show_editor insists on the main thread regardless. So every call here has to
# come from the same thread; in the app that is the UI thread, and the agent's
# plugin tools marshal onto it rather than running on their worker.
#
# Instances are keyed by (plugin file, owner). One synth on several tracks needs
# a separate instance per track or they share one patch — a kick and a pad
# cannot be the same object. Vital costs 224MB for the first instance and very
# little after that (11 of them measured 465MB total), because the wavetables
# and code are shared, so this is affordable.
_LOADED: Dict[tuple, object] = {}


def resolve(path_or_name: str) -> str:
    """The canonical file for a plugin, from a path or an installed name.

    Paths are normalised, so "…/VST3/./Vital.vst3" and "…/VST3//Vital.vst3" are
    recognised as the same plugin rather than as several.
    """
    raw = str(path_or_name)
    # realpath rather than normpath: normpath keeps a leading "//", which POSIX
    # reserves, so "//Library/…" would not match "/Library/…".
    if os.path.exists(raw):
        return os.path.realpath(raw)
    hit = next((p for p in scan()
                if p.name.lower() == raw.lower() or p.slug == raw.lower()), None)
    if hit is None:
        names = ", ".join(p.name for p in scan()[:8]) or "none found"
        raise FileNotFoundError(f"no plugin named {raw!r}; installed: {names}")
    return os.path.realpath(hit.path)


def display_name(path_or_name: str) -> str:
    """A readable name for a plugin path, for showing on a track."""
    try:
        return pathlib.Path(resolve(path_or_name)).stem
    except Exception:  # noqa: BLE001
        return str(path_or_name)


def load(path_or_name: str, owner: Optional[str] = None):
    """Load a plugin. ``owner`` (a track id) gets its own instance and patch."""
    if not available():
        raise RuntimeError("plugin hosting needs pedalboard (pip install pedalboard)")
    import pedalboard

    path = resolve(path_or_name)
    key = (path, owner)
    if key not in _LOADED:
        _LOADED[key] = pedalboard.load_plugin(path)
    return _LOADED[key]


def unload(path_or_name: Optional[str] = None, owner: Optional[str] = None) -> None:
    """Drop instances. With no arguments, drop them all."""
    if path_or_name is None and owner is None:
        _LOADED.clear()
        return
    path = None
    if path_or_name is not None:
        try:
            path = resolve(path_or_name)
        except Exception:  # noqa: BLE001
            path = str(path_or_name)
    for k in [k for k in _LOADED
              if (path is None or k[0] == path) and (owner is None or k[1] == owner)]:
        del _LOADED[k]


# The one instance every track renders through, with its patch swapped in.
# Not a track id, so it never collides with one and prune() keeps it.
RENDER_OWNER = "\0render"


def render_slot(index: int = 0) -> str:
    """The slot a render worker uses. Workers may not share one instance —
    pedalboard's process call is not re-entrant — so each gets its own."""
    return RENDER_OWNER if index <= 0 else f"{RENDER_OWNER}-{index}"


def preload_slots(path_or_name: str, count: int) -> int:
    """Load the instances a worker pool will need. MUST run on the main thread.

    pedalboard refuses to construct a plugin from any other thread, so a worker
    that finds no instance cannot make one: it raises, the render is swallowed,
    and the track is simply silent with nothing to say why.
    """
    made = 0
    for i in range(max(1, count)):
        slot = render_slot(i)
        if (resolve(path_or_name), slot) not in _LOADED:
            load(path_or_name, owner=slot)
            made += 1
    return made


def instance_for(path_or_name: str, owner: Optional[str] = None,
                 slot: Optional[str] = None):
    """The instance a render should go through, and the slot it occupies.

    A track whose editor is open owns a dedicated instance the user may be
    turning knobs on; render through that one so what is heard matches what is
    shown. Everything else renders through one shared instance with the track's
    patch swapped in first.

    Swapping costs 6-38ms against a ~210ms render, and the render queue is
    grouped by track, so a song swaps once per track rather than once per clip.
    Holding an instance per track instead costs ~160MB each.
    """
    path = resolve(path_or_name)
    if owner is not None and (path, owner) in _LOADED:
        return _LOADED[(path, owner)], owner
    want = slot or RENDER_OWNER
    return load(path, owner=want), want


def is_resident(path_or_name: str) -> bool:
    """Whether an instance for this plugin is already loaded.

    Loading one costs seconds; rendering through a loaded one costs a couple of
    hundred milliseconds. Anything that must not block the user has to know the
    difference before it commits.
    """
    path = resolve(path_or_name)
    return any(k[0] == path for k in _LOADED)


def owners() -> set:
    """Track ids that currently hold an instance."""
    return {k[1] for k in _LOADED if k[1] is not None and k[1] != RENDER_OWNER}


def prune(keep: set) -> int:
    """Free instances whose owning track is gone. Returns how many were freed.

    Instances are held here rather than on the Track because the document model
    is plain serialisable data — it is written to JSON, snapshotted for undo and
    copied when a track is duplicated, none of which a live plugin handle
    survives. The lifetime still has to follow the track, so this reconciles the
    two: ownership by id, swept when the project changes.
    """
    dead = [k for k in _LOADED
            if k[1] is not None and k[1] != RENDER_OWNER and k[1] not in keep]
    for k in dead:
        del _LOADED[k]
    return len(dead)


# --- parameters ---------------------------------------------------------
def _params(plugin) -> Dict[str, object]:
    try:
        return dict(plugin.parameters)
    except Exception:  # noqa: BLE001
        return {}


def describe(plugin, query: str = "", limit: int = 40) -> List[dict]:
    """Parameters, optionally filtered by a substring of the name.

    A synth of Vital's size exposes hundreds; handing an agent all of them at
    once is neither useful nor cheap, so this searches and truncates.
    """
    terms = [t for t in query.lower().split() if t]
    out = []
    for key, p in _params(plugin).items():
        name = getattr(p, "name", key) or key
        hay = f"{key} {name}".lower()
        if terms and not all(t in hay for t in terms):
            continue
        row = {"key": key, "name": name,
               "value": getattr(p, "string_value", None),
               "raw": round(float(getattr(p, "raw_value", 0.0)), 4)}
        label = getattr(p, "label", "") or ""
        if label:
            row["unit"] = label
        if getattr(p, "is_boolean", False):
            row["type"] = "switch"
        elif getattr(p, "is_discrete", False):
            # Plugins are loose with this flag: Vital marks all 903 of its
            # parameters discrete, 557 of them with 2**31-1 steps, which plainly
            # means continuous. Only call it a choice when the count is one a
            # human could actually pick from.
            steps = int(getattr(p, "num_steps", 0) or 0)
            if 0 < steps <= _MAX_CHOICE_STEPS:
                row["type"] = "choice"
                row["steps"] = steps
        out.append(row)
        if len(out) >= limit:
            break
    return out


def set_param(plugin, name: str, value) -> dict:
    """Set one parameter, by human text ("880 Hz", "on") or a 0-1 raw value.

    Text is preferred where the plugin understands it: the mapping from a
    displayed value to the internal 0-1 range is the plugin's business, not
    something to guess at from outside.
    """
    params = _params(plugin)
    key = name if name in params else None
    if key is None:
        low = name.lower()
        key = next((k for k, p in params.items()
                    if k.lower() == low or (getattr(p, "name", "") or "").lower() == low), None)
    if key is None:
        key = next((k for k, p in params.items()
                    if low in f"{k} {getattr(p, 'name', '')}".lower()), None)
    if key is None:
        raise KeyError(f"no parameter matching {name!r}")
    p = params[key]

    if isinstance(value, str):
        try:
            raw = float(p.get_raw_value_for_text(value))
        except Exception:  # noqa: BLE001 — plugin could not parse it
            raw = None
        if raw is None:
            try:
                raw = float(value)
            except ValueError:
                raise ValueError(f"{key}: could not interpret {value!r}") from None
    else:
        raw = float(value)
    p.raw_value = max(0.0, min(1.0, raw)) if 0.0 <= raw <= 1.0 else raw
    return {"key": key, "name": getattr(p, "name", key),
            "value": getattr(p, "string_value", None),
            "raw": round(float(getattr(p, "raw_value", 0.0)), 4)}


# --- rendering ----------------------------------------------------------
def notes_to_midi(notes: Sequence, offset: float = 0.0, channel: int = 0):
    """``Note`` objects to the ``(bytes, seconds)`` pairs pedalboard wants."""
    msgs = []
    for n in sorted(notes, key=lambda x: x.start):
        pitch = int(max(0, min(127, n.pitch)))
        vel = int(max(1, min(127, getattr(n, "velocity", 100))))
        start = float(n.start) + offset
        msgs.append((bytes([0x90 | channel, pitch, vel]), start))
        msgs.append((bytes([0x80 | channel, pitch, 0]), start + float(n.duration)))
    return msgs


# How much silence to push through the plugin before a render when it cannot be
# reset. Measured on Vital's longest-release patch: 0.5s still leaked, 1.0s took
# the leading energy from 1.44 to 0.00001, and it costs ~34ms.
FLUSH_SECONDS = 1.5


def render_notes(plugin, notes: Sequence, duration: float, sr: int = 44100,
                 tail: float = 1.0, off_main_thread: bool = False) -> np.ndarray:
    """Render notes through an instrument plugin; returns ``(frames, channels)``.

    ``tail`` leaves room for the release to finish — cutting at the last
    note-off chops the ending off anything with a slow release.

    ``off_main_thread`` is required from a worker: pedalboard refuses to reset a
    plugin that was loaded on another thread, so the reset is skipped and the
    previous render's tail is flushed out with silence instead. Without that
    flush a loud clip leaks into the next one at 75x the new clip's own level.
    """
    if not notes:
        return np.zeros((0, 2), dtype=np.float32)
    if off_main_thread:
        plugin(notes_to_midi([]), duration=FLUSH_SECONDS,
               sample_rate=float(sr), reset=False)
        audio = plugin(notes_to_midi(notes), duration=float(duration) + float(tail),
                       sample_rate=float(sr), reset=False)
    else:
        audio = plugin(notes_to_midi(notes), duration=float(duration) + float(tail),
                       sample_rate=float(sr))
    a = np.asarray(audio, dtype=np.float32)
    return a.T if a.ndim == 2 and a.shape[0] <= 2 else a


def preset_bytes(plugin) -> Optional[bytes]:
    """The plugin's full state, for saving with the project."""
    try:
        return bytes(plugin.preset_data)
    except Exception:  # noqa: BLE001
        return None


def restore_preset(plugin, data: bytes) -> bool:
    try:
        plugin.preset_data = data
        return True
    except Exception:  # noqa: BLE001
        return False
