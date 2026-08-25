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


_LOADED: Dict[str, object] = {}


def load(path_or_name: str):
    """Load a plugin by path, or by (case-insensitive) name from the scan."""
    if not available():
        raise RuntimeError("plugin hosting needs pedalboard (pip install pedalboard)")
    import pedalboard

    key = str(path_or_name)
    if key in _LOADED:
        return _LOADED[key]
    path = key
    if not os.path.exists(path):
        hit = next((p for p in scan()
                    if p.name.lower() == key.lower() or p.slug == key.lower()), None)
        if hit is None:
            names = ", ".join(p.name for p in scan()[:8]) or "none found"
            raise FileNotFoundError(f"no plugin named {key!r}; installed: {names}")
        path = hit.path
    plugin = pedalboard.load_plugin(path)
    _LOADED[key] = plugin
    return plugin


def unload(path_or_name: Optional[str] = None) -> None:
    if path_or_name is None:
        _LOADED.clear()
    else:
        _LOADED.pop(str(path_or_name), None)


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
            row["type"] = "choice"
            row["steps"] = int(getattr(p, "num_steps", 0) or 0)
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


def render_notes(plugin, notes: Sequence, duration: float, sr: int = 44100,
                 tail: float = 1.0) -> np.ndarray:
    """Render notes through an instrument plugin; returns ``(frames, channels)``.

    ``tail`` leaves room for the release to finish — cutting at the last
    note-off chops the ending off anything with a slow release.
    """
    if not notes:
        return np.zeros((0, 2), dtype=np.float32)
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
