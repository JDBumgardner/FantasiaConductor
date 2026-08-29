"""Per-track effects via `pedalboard`.

An FX chain is stored on a track as a list of :class:`FxInsert` (id, type,
bypassed, params). :class:`FxHost` turns the live (non-bypassed) inserts into
a ``pedalboard.Pedalboard`` and processes audio, keeping one board per track
so effect state (e.g. reverb tails) carries across playback blocks.

The board is rebuilt only when the *graph* changes (plugin types, bypass, EQ
band types). Knob moves (EQ freq/gain/Q, enable) poke parameters on the live
plugins so dragging a band during playback does not allocate on the audio
thread.

pedalboard uses ``(num_channels, num_samples)`` arrays, so we transpose in/out.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from fantasia_core.engine.eq import band_as_fx, struct_sig

try:
    import pedalboard as pb
except Exception:  # noqa: BLE001
    pb = None


def _make(spec: dict):
    """Build one pedalboard plugin from a spec dict, or None."""
    if pb is None:
        return None
    kind = spec.get("type")
    p = spec.get("params", {})
    try:
        if kind == "reverb":
            return pb.Reverb(
                room_size=p.get("room_size", 0.6),
                wet_level=p.get("wet", 0.35),
                dry_level=p.get("dry", 0.7),
            )
        if kind == "delay":
            return pb.Delay(
                delay_seconds=p.get("time", 0.25),
                feedback=p.get("feedback", 0.3),
                mix=p.get("mix", 0.3),
            )
        if kind == "lowpass":
            return pb.LowpassFilter(cutoff_frequency_hz=p.get("cutoff", 1200.0))
        if kind == "highpass":
            return pb.HighpassFilter(cutoff_frequency_hz=p.get("cutoff", 250.0))
        if kind == "chorus":
            return pb.Chorus()
        if kind == "distortion":
            return pb.Distortion(drive_db=p.get("drive", 12.0))

        # ---- EQ bands -----------------------------------------------------
        # A "bell": boost/cut a band centred on freq. Q sets its width
        # (0.7 broad and musical, 4+ surgical).
        if kind == "eq_peak":
            return pb.PeakFilter(cutoff_frequency_hz=p.get("freq", 1000.0),
                                 gain_db=p.get("gain", 0.0), q=p.get("q", 1.0))
        # Shelves tilt everything above/below the corner instead of a band.
        if kind == "eq_low_shelf":
            return pb.LowShelfFilter(cutoff_frequency_hz=p.get("freq", 200.0),
                                     gain_db=p.get("gain", 0.0), q=p.get("q", 0.7))
        if kind == "eq_high_shelf":
            return pb.HighShelfFilter(cutoff_frequency_hz=p.get("freq", 6000.0),
                                      gain_db=p.get("gain", 0.0), q=p.get("q", 0.7))

        # ---- dynamics -----------------------------------------------------
        if kind == "compressor":
            return pb.Compressor(
                threshold_db=p.get("threshold", -16.0), ratio=p.get("ratio", 4.0),
                attack_ms=p.get("attack", 10.0), release_ms=p.get("release", 100.0),
            )
        if kind == "limiter":
            # NOT pedalboard's Limiter: that one applies makeup gain up to the
            # threshold (a maximizer), so it can push a quiet track to full
            # scale — the opposite of what you want on an insert. A compressor
            # with a near-infinite ratio and fast attack caps predictably.
            return pb.Compressor(
                threshold_db=p.get("threshold", -1.0), ratio=p.get("ratio", 20.0),
                attack_ms=p.get("attack", 1.0), release_ms=p.get("release", 100.0),
            )
        if kind == "gate":
            return pb.NoiseGate(threshold_db=p.get("threshold", -50.0),
                                ratio=p.get("ratio", 4.0),
                                attack_ms=p.get("attack", 1.0),
                                release_ms=p.get("release", 100.0))

        # ---- colour -------------------------------------------------------
        # Saturation = drive into a soft curve for harmonics, then pull the
        # level back so it warms rather than just gets louder.
        if kind == "saturator":
            drive = float(p.get("drive", 5.0))
            return pb.Pedalboard([
                pb.Distortion(drive_db=drive),
                pb.Gain(gain_db=p.get("output", -drive * 0.6)),
            ])
        if kind == "gain":
            return pb.Gain(gain_db=p.get("gain", 0.0))

        if kind == "vst":
            path = p.get("path") or p.get("name")
            if not path:
                return None
            return pb.load_plugin(str(path))

        # Stock 8-band EQ: always eight filters so toggling a band is a
        # parameter poke, not a graph rebuild (see FxHost + eq.struct_sig).
        if kind == "eq":
            from fantasia_core.engine.eq import default_bands

            bands = list(p.get("bands") or default_bands())
            plugs = []
            for band in bands[:8]:
                child = _make(band_as_fx(band))
                if child is not None:
                    plugs.append(child)
            return pb.Pedalboard(plugs) if plugs else None
    except Exception:  # noqa: BLE001 — bad params shouldn't crash audio
        return None
    return None


def _set_num(plugin, names: Tuple[str, ...], value: float) -> None:
    for name in names:
        if hasattr(plugin, name):
            try:
                setattr(plugin, name, float(value))
                return
            except Exception:  # noqa: BLE001
                continue


def _sync_one(plugin, spec: dict) -> None:
    """Poke live pedalboard parameters to match ``spec`` (audio thread)."""
    kind = spec.get("type")
    p = spec.get("params") or {}
    if kind in ("eq_peak", "eq_low_shelf", "eq_high_shelf"):
        _set_num(plugin, ("cutoff_frequency_hz", "cutoff_hz"), p.get("freq", 1000.0))
        _set_num(plugin, ("gain_db", "gain"), p.get("gain", 0.0))
        _set_num(plugin, ("q", "Q"), p.get("q", 1.0))
    elif kind in ("lowpass", "highpass"):
        _set_num(plugin, ("cutoff_frequency_hz", "cutoff_hz"), p.get("cutoff", 1000.0))
    elif kind == "gain":
        _set_num(plugin, ("gain_db", "gain"), p.get("gain", 0.0))
    elif kind == "reverb":
        _set_num(plugin, ("wet_level",), p.get("wet", 0.35))
        _set_num(plugin, ("dry_level",), p.get("dry", 0.7))
        _set_num(plugin, ("room_size",), p.get("room_size", 0.6))
    elif kind == "delay":
        _set_num(plugin, ("delay_seconds",), p.get("time", 0.25))
        _set_num(plugin, ("feedback",), p.get("feedback", 0.3))
        _set_num(plugin, ("mix",), p.get("mix", 0.3))
    elif kind == "distortion":
        _set_num(plugin, ("drive_db",), p.get("drive", 12.0))
    elif kind == "compressor":
        _set_num(plugin, ("threshold_db",), p.get("threshold", -16.0))
        _set_num(plugin, ("ratio",), p.get("ratio", 4.0))
        _set_num(plugin, ("attack_ms",), p.get("attack", 10.0))
        _set_num(plugin, ("release_ms",), p.get("release", 100.0))
    elif kind == "limiter":
        _set_num(plugin, ("threshold_db",), p.get("threshold", -1.0))
        _set_num(plugin, ("ratio",), p.get("ratio", 20.0))
        _set_num(plugin, ("attack_ms",), p.get("attack", 1.0))
        _set_num(plugin, ("release_ms",), p.get("release", 100.0))
    elif kind == "gate":
        _set_num(plugin, ("threshold_db",), p.get("threshold", -50.0))
        _set_num(plugin, ("ratio",), p.get("ratio", 4.0))
        _set_num(plugin, ("attack_ms",), p.get("attack", 1.0))
        _set_num(plugin, ("release_ms",), p.get("release", 100.0))
    elif kind == "eq":
        from fantasia_core.engine.eq import default_bands

        bands = list(p.get("bands") or default_bands())
        children = list(plugin) if plugin is not None else []
        for child, band in zip(children, bands):
            _sync_one(child, band_as_fx(band))


def _live_specs(specs: List) -> List[dict]:
    """Non-bypassed inserts as dicts the rest of this module understands."""
    from fantasia_core.document.fx_insert import as_dict, insert_bypassed

    out = []
    for s in specs or []:
        d = as_dict(s)
        if d.get("type") and not insert_bypassed(s):
            out.append(d)
    return out


def _sync_board(board, specs: List) -> None:
    children = list(board)
    live = _live_specs(specs)
    for child, spec in zip(children, live):
        _sync_one(child, spec)


def build_board(specs: List):
    if pb is None:
        return None
    plugins = [pl for pl in (_make(s) for s in _live_specs(specs)) if pl is not None]
    return pb.Pedalboard(plugins) if plugins else None


class FxHost:
    """Caches a Pedalboard per track and processes stereo blocks statefully."""

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[tuple, object]] = {}  # track_id -> (struct, board|dag)
        self._dag_plugins: Dict[str, dict] = {}  # track_id -> {insert_id: plugin}
        self._dag_bufs: Dict[str, dict] = {}

    def process(self, track, audio: np.ndarray, sr: int) -> np.ndarray:  # noqa: ANN001
        specs = getattr(track, "fx", None) or []
        if not specs or pb is None:
            return audio
        from fantasia_core.document.fx_insert import effective_wires, is_serial

        wires = getattr(track, "fx_wires", None) or []
        sig = struct_sig(specs, wires)
        if not is_serial(specs, wires):
            return self._process_dag(track, audio, sr, specs, wires, sig)

        entry = self._cache.get(track.id)
        if entry is None or entry[0] != sig:
            board = build_board(specs)
            self._cache[track.id] = (sig, board)
        else:
            board = entry[1]
            try:
                _sync_board(board, specs)
            except Exception:  # noqa: BLE001 — fall back to a rebuild
                board = build_board(specs)
                self._cache[track.id] = (sig, board)
        if board is None:
            return audio
        return self._run(board, audio, sr)

    def _run(self, board, audio: np.ndarray, sr: int) -> np.ndarray:  # noqa: ANN001
        try:
            out = board(audio.T.astype(np.float32), sr, reset=False).T
        except Exception:  # noqa: BLE001
            return audio
        n = len(audio)
        if len(out) < n:
            pad = np.zeros((n - len(out), out.shape[1]), dtype=np.float32)
            out = np.vstack([out, pad])
        elif len(out) > n:
            out = out[:n]
        return out

    def _process_dag(self, track, audio, sr, specs, wires, sig) -> np.ndarray:  # noqa: ANN001
        """Branch/merge graph: mix on join, copy on split. Off the serial board."""
        from fantasia_core.document.fx_insert import (
            OUT, SOURCE, as_dict, as_insert, effective_wires, insert_id,
            topo_order,
        )

        entry = self._cache.get(track.id)
        plugins = self._dag_plugins.get(track.id)
        if entry is None or entry[0] != sig or plugins is None:
            plugins = {}
            for spec in specs:
                d = as_dict(spec)
                if not d.get("type") or d.get("bypassed"):
                    continue
                nid = d.get("id") or d.get("type")
                plug = _make(d)
                if plug is not None:
                    plugins[nid] = plug
            self._dag_plugins[track.id] = plugins
            self._cache[track.id] = (sig, "dag")
        else:
            try:
                for spec in specs:
                    d = as_dict(spec)
                    nid = d.get("id") or d.get("type")
                    if nid in plugins and not d.get("bypassed"):
                        _sync_one(plugins[nid], d)
            except Exception:  # noqa: BLE001
                self._cache.pop(track.id, None)
                return self._process_dag(track, audio, sr, specs, wires, sig)

        graph = effective_wires(specs, wires)
        order = topo_order(specs, wires)
        bufs: dict = {SOURCE: audio}
        for spec in specs:
            nid = insert_id(spec) or as_dict(spec).get("type")
            incoming = [w.src for w in graph if w.dst == nid]
            mixed = self._mix_inputs(audio, bufs, incoming)
            d = as_dict(spec)
            plug = plugins.get(nid)
            if d.get("bypassed") or plug is None:
                bufs[nid] = mixed
                continue
            try:
                bufs[nid] = self._run(plug, mixed, sr)
            except Exception:  # noqa: BLE001
                bufs[nid] = mixed
        outgoing = [w.src for w in graph if w.dst == OUT]
        if not outgoing:
            return audio
        return self._mix_inputs(audio, bufs, outgoing)

    def _mix_inputs(self, audio: np.ndarray, bufs: dict, srcs: list) -> np.ndarray:
        if not srcs:
            return np.zeros_like(audio)
        acc = None
        for src in srcs:
            block = bufs.get(src)
            if block is None:
                continue
            if acc is None:
                acc = np.array(block, dtype=np.float32, copy=True)
            else:
                acc += block
        return acc if acc is not None else np.zeros_like(audio)
