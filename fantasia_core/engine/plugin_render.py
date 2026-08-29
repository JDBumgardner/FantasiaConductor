"""Rendering MIDI clips through a hosted VST3/AU instrument.

Mirrors :mod:`fantasia_core.engine.midi_render`: clips are synthesized on the
UI thread and cached, and the audio callback only ever reads an already-rendered
buffer. That split is why the callback stays inside its deadline — it never
calls a plugin, which could take an unbounded amount of time and would hold the
GIL while doing it.

The cache key includes the plugin's state, so moving a knob re-renders the
clips that depend on it and leaves the rest alone.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Dict, Optional, Tuple

import numpy as np


class PluginRenderer:
    """Renders and caches MIDI clips played through a plugin instrument."""

    def __init__(self, sample_rate: int = 44100, tail: float = 1.0) -> None:
        self.sr = sample_rate
        self.tail = tail
        self._cache: Dict[Tuple, np.ndarray] = {}
        self._states: Dict[tuple, str] = {}

    # ---- keys ---------------------------------------------------------
    def _key(self, clip, plugin: str, state: str, owner: str = "") -> Tuple:  # noqa: ANN001
        notes = tuple((n.pitch, round(n.start, 4), round(n.duration, 4), n.velocity)
                      for n in clip.notes)
        # The state blob can be large; hash it so keys stay small.
        digest = hashlib.sha1((state or "").encode()).hexdigest()[:12]
        return (plugin, owner, digest, round(clip.duration, 4), notes)

    def cached(self, clip, plugin: str, state: str = "",
               owner: str = "") -> Optional[np.ndarray]:  # noqa: ANN001
        """Audio-callback-safe: the rendered buffer, or None. Never synthesizes."""
        return self._cache.get(self._key(clip, plugin, state, owner))

    # ---- rendering ----------------------------------------------------
    def render(self, clip, plugin: str, state: str = "",
               owner: str = "") -> np.ndarray:  # noqa: ANN001
        """Synthesize on a worker/UI thread and cache. Silence if unavailable.

        ``owner`` is the track id: one synth used on several tracks needs an
        instance each, or they all share whichever patch was applied last.
        """
        key = self._key(clip, plugin, state, owner)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        frames = max(int(clip.duration * self.sr), 0)
        buf = np.zeros((frames, 2), dtype=np.float32)
        try:
            from fantasia_core import plugins as plg

            inst, slot = plg.instance_for(plugin, owner or None)
            # Memoised against the slot, not the track: the shared instance
            # holds one patch at a time, so what matters is what is in it now.
            if state and self._states.get((plugin, slot)) != state:
                plg.restore_preset(inst, base64.b64decode(state))
                self._states[(plugin, slot)] = state
            audio = plg.render_notes(inst, clip.notes, clip.duration, self.sr,
                                     tail=self.tail)
            if len(audio):
                if audio.ndim == 1:
                    audio = np.stack([audio, audio], axis=1)
                take = min(len(audio), frames) if frames else len(audio)
                buf = np.zeros((max(frames, take), 2), dtype=np.float32)
                buf[:take] = audio[:take, :2]
                buf = buf[:frames] if frames else buf
        except Exception:  # noqa: BLE001 — a missing plugin must not kill playback
            pass
        self._cache[key] = buf
        return buf

    def pending(self, project) -> list:  # noqa: ANN001
        """Plugin clips with no rendered audio yet, as ``(clip, plugin, state)``.

        Rendering one clip through a plugin costs a few hundred milliseconds, so
        a project with a plugin on several tracks is many seconds of work. The
        caller spreads that over the event loop instead of blocking on it.
        """
        out = []
        for track in project.tracks:
            plugin = getattr(track, "plugin", "")
            if not plugin:
                continue
            state = getattr(track, "plugin_state", "")
            for clip in track.clips:
                if (clip.content_type == "midi"
                        and self.cached(clip, plugin, state, track.id) is None):
                    out.append((clip, plugin, state, track.id))
        return out

    def warm(self, project) -> None:  # noqa: ANN001
        """Render everything now. Blocks — prefer :meth:`pending` in the UI."""
        for clip, plugin, state, owner in self.pending(project):
            self.render(clip, plugin, state, owner)

    def forget_patches(self) -> None:
        """Forget which patch each instance holds, without dropping audio."""
        self._states.clear()

    def invalidate(self, plugin: Optional[str] = None,
                   owner: Optional[str] = None) -> None:
        """Drop cached audio. Narrow it with ``owner`` when one track changed.

        Without the owner this clears every track using the plugin, which on a
        project with a dozen tracks on one synth means re-rendering all of them
        to reflect a change that affected one.
        """
        if plugin is None and owner is None:
            self._cache.clear()
            self._states.clear()
            return
        for store in (self._cache, self._states):
            for k in [k for k in store
                      if (plugin is None or k[0] == plugin)
                      and (owner is None or k[1] == owner)]:
                del store[k]


def reset(renderer: "PluginRenderer") -> int:
    """Drop every track-owned instance. Call when a different song is loaded.

    Track ids restart at ``t1`` in every project, so song B's ``t3`` collides
    with song A's ``t3``. Reusing the instance mostly self-corrects, because a
    differing saved patch is restored before the render — but a track whose
    patch has not been saved yet restores nothing and would inherit the sound of
    an unrelated track from the previous song.
    """
    from fantasia_core import plugins as plg

    owned = list(plg.owners())
    for owner in owned:
        renderer.invalidate(owner=owner)
    # The shared instance survives, but whatever patch is in it belongs to the
    # old song — forget it so the next render loads the right one.
    renderer.forget_patches()
    return plg.prune(set())


def prune(project, renderer: "PluginRenderer") -> int:  # noqa: ANN001
    """Drop instances and cached audio for tracks that no longer exist."""
    from fantasia_core import plugins as plg

    keep = {t.id for t in getattr(project, "tracks", [])}
    for owner in list(plg.owners()):
        if owner not in keep:
            renderer.invalidate(owner=owner)
    return plg.prune(keep)


def capture_state(plugin_name: str, owner: Optional[str] = None) -> str:
    """A track's plugin state as base64, for saving with the project."""
    from fantasia_core import plugins as plg

    data = plg.preset_bytes(plg.load(plugin_name, owner=owner))
    return base64.b64encode(data).decode() if data else ""
