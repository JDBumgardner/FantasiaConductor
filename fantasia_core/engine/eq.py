"""Stock parametric EQ — one FX insert with up to eight *bands*.

"Band" is the industry word (Ableton EQ Eight, Logic Channel EQ, FabFilter
Pro-Q). The numbered handles on the curve are those bands; each has a type
(bell, shelf, cut, notch), frequency, gain, Q, and an on/off flag.

The insert is stored on a track as an :class:`FxInsert`::

    FxInsert(id="fx12", type="eq", params={"bands": [ ... up to 8 ... ]})

Legacy one-filter FX types (``eq_peak``, ``highpass``, …) still process, and
the editor can promote them into this insert on first edit so older projects
and ``add_fx`` calls keep working.

Headless: no Qt. DSP still goes through pedalboard; this module owns the band
schema, the analytic frequency response used by the UI (cheap, off the audio
thread), and packing/unpacking the insert in a track's FX list.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np

MAX_BANDS = 8
F_MIN, F_MAX = 20.0, 20000.0
DB_MIN, DB_MAX = -24.0, 24.0

BAND_TYPES = (
    "bell",
    "low_shelf",
    "high_shelf",
    "low_cut",
    "high_cut",
    "notch",
)

# Filters whose frequency response is well-defined (plotted, editable as bands).
LINEAR_FX = frozenset({
    "eq", "eq_peak", "eq_low_shelf", "eq_high_shelf", "lowpass", "highpass",
})

_LEGACY_TO_BAND = {
    "eq_peak": "bell",
    "eq_low_shelf": "low_shelf",
    "eq_high_shelf": "high_shelf",
    "highpass": "low_cut",
    "lowpass": "high_cut",
}

_BAND_TO_LEGACY = {
    "bell": "eq_peak",
    "low_shelf": "eq_low_shelf",
    "high_shelf": "eq_high_shelf",
    "low_cut": "highpass",
    "high_cut": "lowpass",
    "notch": "eq_peak",
}

_DEFAULT_LAYOUT = (
    ("low_cut", 30.0, 0.0, 0.7, False),
    ("low_shelf", 120.0, 0.0, 0.7, True),
    ("bell", 250.0, 0.0, 1.0, True),
    ("bell", 800.0, 0.0, 1.0, True),
    ("bell", 2500.0, 0.0, 1.0, True),
    ("bell", 5000.0, 0.0, 1.0, True),
    ("high_shelf", 8000.0, 0.0, 0.7, True),
    ("high_cut", 18000.0, 0.0, 0.7, False),
)


def _clip_freq(f: float) -> float:
    return float(min(F_MAX, max(F_MIN, f)))


def _clip_gain(g: float) -> float:
    return float(min(DB_MAX, max(DB_MIN, g)))


def _clip_q(q: float) -> float:
    return float(min(18.0, max(0.1, q)))


CUT_TYPES = frozenset({"low_cut", "high_cut"})


def q_from_vertical_drag(q0: float, dy_norm: float) -> float:
    """Map a vertical drag to Q for high/low-cut bands.

    ``dy_norm`` is the drag as a fraction of the plot height, positive = up.
    Relative to the Q at press so grabbing a handle (which sits on the curve,
    not on a Q-encoded Y) never jumps the value.
    """
    return _clip_q(float(q0) * (10.0 ** (float(dy_norm) * 1.6)))


def default_bands() -> List[dict]:
    """Eight-band starting layout: HP / LS / four bells / HS / LP, cuts off."""
    return [
        normalize_band({
            "type": kind, "freq": freq, "gain": gain, "q": q, "enabled": on,
        })
        for kind, freq, gain, q, on in _DEFAULT_LAYOUT
    ]


def normalize_band(raw: Optional[dict] = None) -> dict:
    """Coerce one band dict to the canonical keys and ranges."""
    raw = raw or {}
    kind = str(raw.get("type", "bell")).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "peak": "bell", "eq_peak": "bell", "bell": "bell",
        "lowshelf": "low_shelf", "eq_low_shelf": "low_shelf",
        "highshelf": "high_shelf", "eq_high_shelf": "high_shelf",
        "highpass": "low_cut", "hpf": "low_cut", "low_cut": "low_cut",
        "lowpass": "high_cut", "lpf": "high_cut", "high_cut": "high_cut",
        "notch": "notch",
    }
    kind = aliases.get(kind, "bell")
    if kind not in BAND_TYPES:
        kind = "bell"
    freq = raw.get("freq", raw.get("cutoff", 1000.0))
    gain = raw.get("gain", -24.0 if kind == "notch" else 0.0)
    q = raw.get("q", 4.0 if kind == "notch" else (0.7 if "shelf" in kind or "cut" in kind else 1.0))
    enabled = raw.get("enabled", True)
    return {
        "type": kind,
        "freq": _clip_freq(float(freq)),
        "gain": _clip_gain(float(gain)),
        "q": _clip_q(float(q)),
        "enabled": bool(enabled),
    }


def _legacy_to_band(spec: dict) -> dict:
    kind = _LEGACY_TO_BAND.get(spec.get("type"), "bell")
    p = spec.get("params") or {}
    return normalize_band({"type": kind, **p, "enabled": True})


def bands_from_fx(fx: Optional[Sequence]) -> List[dict]:
    """Editor-facing 8 bands from a track FX list.

    Prefers a unified ``eq`` insert; otherwise lifts legacy linear filters into
    the first slots and pads with the default layout (remaining bands off).
    """
    from fantasia_core.document.fx_insert import as_dict

    fx = [as_dict(s) for s in (fx or [])]
    for spec in fx:
        if spec.get("type") == "eq":
            raw = list((spec.get("params") or {}).get("bands") or [])
            bands = [normalize_band(b) for b in raw[:MAX_BANDS]]
            pad = default_bands()[len(bands):]
            for b in pad:
                b["enabled"] = False
            return bands + pad

    lifted = [_legacy_to_band(s) for s in fx if s.get("type") in _LEGACY_TO_BAND]
    if not lifted:
        return default_bands()
    bands = lifted[:MAX_BANDS]
    pad = default_bands()[len(bands):]
    for b in pad:
        b["enabled"] = False
    return bands + pad


def fx_with_eq(fx: Optional[Sequence], bands: Sequence[dict]) -> list:
    """Write ``bands`` into ``fx`` as a single ``eq`` insert.

    Replaces an existing ``eq`` in place (keeping its id / bypass), or the
    first run of legacy linear filters, or appends. Non-EQ devices keep their
    position and identity.
    """
    from fantasia_core.document.fx_insert import FxInsert, as_insert

    bands = [normalize_band(b) for b in list(bands)[:MAX_BANDS]]
    params = {"bands": bands}
    chain = [as_insert(s) for s in (fx or [])]

    for i, spec in enumerate(chain):
        if spec.type == "eq":
            chain[i] = FxInsert(
                id=spec.id, type="eq", params=params, bypassed=spec.bypassed)
            return chain

    insert = FxInsert(id="", type="eq", params=params)
    first = next((i for i, s in enumerate(chain) if s.type in _LEGACY_TO_BAND), None)
    if first is None:
        chain.append(insert)
        return chain
    kept = [s for i, s in enumerate(chain)
            if not (i >= first and s.type in _LEGACY_TO_BAND)]
    kept.insert(first, insert)
    return kept


def band_as_fx(band: dict) -> dict:
    """One band as a legacy single-filter spec, with bypass baked into params.

    Disabled bells/shelves/notches sit at 0 dB; disabled cuts park at the
    audible edges so the plugin graph (always 8 filters) never has to rebuild
    when the user toggles a band.
    """
    b = normalize_band(band)
    kind, freq, gain, q = b["type"], b["freq"], b["gain"], b["q"]
    if not b["enabled"]:
        if kind in ("bell", "low_shelf", "high_shelf", "notch"):
            gain = 0.0
        elif kind == "low_cut":
            freq = F_MIN
        elif kind == "high_cut":
            freq = F_MAX
    if kind == "notch" and b["enabled"] and abs(gain) < 0.5:
        gain = -24.0
    legacy = _BAND_TO_LEGACY[kind]
    params = {"q": q}
    if kind in ("low_cut", "high_cut"):
        params["cutoff"] = freq
    else:
        params["freq"] = freq
        params["gain"] = gain
    return {"type": legacy, "params": params}


def expand_eq_specs(specs: Iterable) -> List[dict]:
    """Flatten ``eq`` inserts into the single-filter specs ``fx._make`` knows."""
    from fantasia_core.document.fx_insert import as_dict

    out: List[dict] = []
    for spec in specs or []:
        spec = as_dict(spec)
        if spec.get("type") == "eq":
            raw = list((spec.get("params") or {}).get("bands") or [])
            for band in raw[:MAX_BANDS]:
                out.append(band_as_fx(band))
        else:
            out.append(spec)
    return out


def struct_sig(specs: Sequence) -> tuple:
    """Identity of the *graph* (plugin types / bypass), not of the knob values.

    FxHost uses this to decide "rebuild the board" vs "poke parameters".
    Band type changes and bypass rebuild; dragging freq/gain/Q does not.
    """
    from fantasia_core.document.fx_insert import as_dict

    rows = []
    for spec in specs or []:
        spec = as_dict(spec)
        if spec.get("bypassed"):
            rows.append(("bypass", spec.get("id") or spec.get("type")))
            continue
        kind = spec.get("type")
        if kind == "eq":
            bands = list((spec.get("params") or {}).get("bands") or [])
            rows.append(("eq", tuple(
                normalize_band(b)["type"] for b in bands[:MAX_BANDS]
            )))
        else:
            rows.append((kind,))
    return tuple(rows)


# ---- analytic frequency response (UI thread; never the audio callback) ----
def _rbj(kind: str, freq: float, gain: float, q: float, sr: float):
    """RBJ cookbook biquad coefficients ``(b0, b1, b2, a0, a1, a2)``."""
    w0 = 2.0 * np.pi * (freq / sr)
    w0 = float(np.clip(w0, 1e-6, np.pi - 1e-6))
    cos, sin = np.cos(w0), np.sin(w0)
    q = max(0.05, float(q))
    alpha = sin / (2.0 * q)
    a = 10.0 ** (gain / 40.0)
    if kind == "bell":
        b0, b1, b2 = 1 + alpha * a, -2 * cos, 1 - alpha * a
        a0, a1, a2 = 1 + alpha / a, -2 * cos, 1 - alpha / a
    elif kind == "low_shelf":
        sa = 2.0 * np.sqrt(a) * alpha
        b0 = a * ((a + 1) - (a - 1) * cos + sa)
        b1 = 2 * a * ((a - 1) - (a + 1) * cos)
        b2 = a * ((a + 1) - (a - 1) * cos - sa)
        a0 = (a + 1) + (a - 1) * cos + sa
        a1 = -2 * ((a - 1) + (a + 1) * cos)
        a2 = (a + 1) + (a - 1) * cos - sa
    elif kind == "high_shelf":
        sa = 2.0 * np.sqrt(a) * alpha
        b0 = a * ((a + 1) + (a - 1) * cos + sa)
        b1 = -2 * a * ((a - 1) + (a + 1) * cos)
        b2 = a * ((a + 1) + (a - 1) * cos - sa)
        a0 = (a + 1) - (a - 1) * cos + sa
        a1 = 2 * ((a - 1) - (a + 1) * cos)
        a2 = (a + 1) - (a - 1) * cos - sa
    elif kind == "low_cut":  # high-pass
        b0, b1, b2 = (1 + cos) / 2, -(1 + cos), (1 + cos) / 2
        a0, a1, a2 = 1 + alpha, -2 * cos, 1 - alpha
    elif kind == "high_cut":  # low-pass
        b0, b1, b2 = (1 - cos) / 2, 1 - cos, (1 - cos) / 2
        a0, a1, a2 = 1 + alpha, -2 * cos, 1 - alpha
    elif kind == "notch":
        b0, b1, b2 = 1.0, -2 * cos, 1.0
        a0, a1, a2 = 1 + alpha, -2 * cos, 1 - alpha
    else:
        return 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
    return b0, b1, b2, a0, a1, a2


def _biquad_mag_db(coeffs, freqs: np.ndarray, sr: float) -> np.ndarray:
    b0, b1, b2, a0, a1, a2 = (float(c) for c in coeffs)
    w = 2.0 * np.pi * (freqs / sr)
    # H(e^{jw}) = (b0 + b1 z^{-1} + b2 z^{-2}) / (a0 + a1 z^{-1} + a2 z^{-2})
    z1 = np.cos(w) - 1j * np.sin(w)
    z2 = z1 * z1
    num = b0 + b1 * z1 + b2 * z2
    den = a0 + a1 * z1 + a2 * z2
    mag = np.abs(num) / np.maximum(np.abs(den), 1e-12)
    return 20.0 * np.log10(np.maximum(mag, 1e-12))


def response_db(bands: Sequence[dict], freqs: np.ndarray, sr: float = 44100.0) -> np.ndarray:
    """Cascade magnitude of enabled bands at ``freqs``, in dB. UI-thread safe."""
    db = np.zeros(len(freqs), dtype=np.float64)
    for raw in bands or []:
        b = normalize_band(raw)
        if not b["enabled"]:
            continue
        kind = b["type"]
        gain = b["gain"]
        if kind == "notch" and abs(gain) < 0.5:
            gain = -24.0
        coeffs = _rbj(kind, b["freq"], gain, b["q"], sr)
        db += _biquad_mag_db(coeffs, freqs, sr)
    return db


def log_freqs(n: int = 512, fmin: float = F_MIN, fmax: float = F_MAX) -> np.ndarray:
    return np.logspace(np.log10(fmin), np.log10(fmax), int(n))


def handle_gain(band: dict, freqs: Optional[np.ndarray] = None,
                db: Optional[np.ndarray] = None) -> float:
    """Y-position of a band handle: own gain, or the curve at its corner."""
    b = normalize_band(band)
    if b["type"] in ("bell", "low_shelf", "high_shelf"):
        return b["gain"] if b["enabled"] else 0.0
    if b["type"] == "notch":
        return b["gain"] if b["enabled"] else 0.0
    # Cuts sit on the measured/analytic curve at the corner frequency.
    if freqs is not None and db is not None and len(freqs) and len(db):
        idx = int(np.argmin(np.abs(freqs - b["freq"])))
        return float(db[idx])
    return 0.0


def format_freq(hz: float) -> str:
    if hz >= 1000.0:
        v = hz / 1000.0
        return f"{v:.1f} kHz" if v < 10 else f"{v:.0f} kHz"
    return f"{hz:.0f} Hz"
