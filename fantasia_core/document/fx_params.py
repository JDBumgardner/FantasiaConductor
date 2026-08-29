"""Editable parameter lists for stock instruments and FX.

Shared by the graph-node UI and (indirectly) agents: keys here are the same
dict keys stored on ``FxInsert.params`` / ``Track.synth``. Headless — no Qt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

# Keep in lockstep with fantasia_core.engine.synth.WAVEFORMS (document layer
# must not import the engine).
WAVEFORMS = ("sine", "saw", "square", "triangle")


@dataclass(frozen=True)
class ParamSpec:
    """One knob / menu on a stock device."""

    key: str
    label: str
    kind: str  # "float" | "choice" | "bool"
    default: Any
    minimum: float = 0.0
    maximum: float = 1.0
    decimals: int = 2
    suffix: str = ""
    choices: tuple = field(default_factory=tuple)


def _f(key: str, label: str, default: float, lo: float, hi: float,
       decimals: int = 2, suffix: str = "") -> ParamSpec:
    return ParamSpec(key, label, "float", default, lo, hi, decimals, suffix)


def _c(key: str, label: str, default: str, choices: Sequence[str]) -> ParamSpec:
    return ParamSpec(key, label, "choice", default, choices=tuple(choices))


def _b(key: str, label: str, default: bool = True) -> ParamSpec:
    return ParamSpec(key, label, "bool", default)


FX_PARAM_SPECS: dict[str, tuple[ParamSpec, ...]] = {
    "reverb": (
        _f("room_size", "Room", 0.6, 0.0, 1.0),
        _f("wet", "Wet", 0.35, 0.0, 1.0),
        _f("dry", "Dry", 0.7, 0.0, 1.0),
    ),
    "delay": (
        _f("time", "Time", 0.25, 0.01, 2.0, 3, " s"),
        _f("feedback", "Feedback", 0.3, 0.0, 0.95),
        _f("mix", "Mix", 0.3, 0.0, 1.0),
    ),
    "compressor": (
        _f("threshold", "Thresh", -16.0, -60.0, 0.0, 1, " dB"),
        _f("ratio", "Ratio", 4.0, 1.0, 20.0, 1),
        _f("attack", "Attack", 10.0, 0.1, 200.0, 1, " ms"),
        _f("release", "Release", 100.0, 10.0, 1000.0, 0, " ms"),
    ),
    "limiter": (
        _f("threshold", "Thresh", -1.0, -24.0, 0.0, 1, " dB"),
        _f("ratio", "Ratio", 20.0, 4.0, 40.0, 1),
        _f("attack", "Attack", 1.0, 0.1, 20.0, 1, " ms"),
        _f("release", "Release", 100.0, 10.0, 1000.0, 0, " ms"),
    ),
    "gate": (
        _f("threshold", "Thresh", -50.0, -80.0, 0.0, 1, " dB"),
        _f("ratio", "Ratio", 4.0, 1.0, 20.0, 1),
        _f("attack", "Attack", 1.0, 0.1, 50.0, 1, " ms"),
        _f("release", "Release", 100.0, 10.0, 1000.0, 0, " ms"),
    ),
    "saturator": (
        _f("drive", "Drive", 5.0, 0.0, 30.0, 1, " dB"),
        _f("output", "Output", -3.0, -24.0, 6.0, 1, " dB"),
    ),
    "distortion": (
        _f("drive", "Drive", 12.0, 0.0, 40.0, 1, " dB"),
    ),
    "lowpass": (
        _f("cutoff", "Cutoff", 1200.0, 20.0, 20000.0, 0, " Hz"),
    ),
    "highpass": (
        _f("cutoff", "Cutoff", 250.0, 20.0, 20000.0, 0, " Hz"),
    ),
    "gain": (
        _f("gain", "Gain", 0.0, -24.0, 24.0, 1, " dB"),
    ),
    "eq_peak": (
        _f("freq", "Freq", 1000.0, 20.0, 20000.0, 0, " Hz"),
        _f("gain", "Gain", 0.0, -24.0, 24.0, 1, " dB"),
        _f("q", "Q", 1.0, 0.1, 18.0, 2),
    ),
    "eq_low_shelf": (
        _f("freq", "Freq", 200.0, 20.0, 20000.0, 0, " Hz"),
        _f("gain", "Gain", 0.0, -24.0, 24.0, 1, " dB"),
        _f("q", "Q", 0.7, 0.1, 18.0, 2),
    ),
    "eq_high_shelf": (
        _f("freq", "Freq", 6000.0, 20.0, 20000.0, 0, " Hz"),
        _f("gain", "Gain", 0.0, -24.0, 24.0, 1, " dB"),
        _f("q", "Q", 0.7, 0.1, 18.0, 2),
    ),
}

SYNTH_PARAM_SPECS: tuple[ParamSpec, ...] = (
    _c("osc1", "Osc 1", "saw", WAVEFORMS),
    _c("osc2", "Osc 2", "saw", WAVEFORMS),
    _c("osc3", "Osc 3", "saw", WAVEFORMS),
    _f("mix", "Stack", 1.0, 0.0, 1.0),
    _f("detune", "Detune", 0.12, 0.0, 1.0, 2, " st"),
    _f("attack", "Attack", 0.01, 0.0, 2.0, 3, " s"),
    _f("decay", "Decay", 0.22, 0.0, 2.0, 3, " s"),
    _f("sustain", "Sustain", 0.68, 0.0, 1.0),
    _f("release", "Release", 0.20, 0.0, 2.0, 3, " s"),
    _f("cutoff", "Cutoff", 1600.0, 100.0, 12000.0, 0, " Hz"),
    _f("resonance", "Reso", 0.28, 0.0, 1.0),
    _f("env_amount", "Env", 800.0, 0.0, 8000.0, 0, " Hz"),
    _f("gain", "Gain", 0.48, 0.0, 1.0),
)


def eq_param_specs(n_bands: int = 8) -> tuple[ParamSpec, ...]:
    """Flattened 8-band EQ: on / freq / gain / Q per band."""
    rows: list[ParamSpec] = []
    for i in range(n_bands):
        n = i + 1
        rows.append(_b(f"b{i}.enabled", f"B{n} On", True))
        rows.append(_f(f"b{i}.freq", f"B{n} Freq", 1000.0, 20.0, 20000.0, 0, " Hz"))
        rows.append(_f(f"b{i}.gain", f"B{n} Gain", 0.0, -24.0, 24.0, 1, " dB"))
        rows.append(_f(f"b{i}.q", f"B{n} Q", 1.0, 0.1, 18.0, 2))
    return tuple(rows)


def specs_for(kind: str, params: Optional[dict] = None) -> tuple[ParamSpec, ...]:
    """Parameter rows for a stock FX type (empty for chorus / vst / unknown)."""
    if kind == "eq":
        bands = (params or {}).get("bands") or []
        n = max(8, len(bands)) if bands else 8
        return eq_param_specs(min(n, 8))
    return FX_PARAM_SPECS.get(kind, ())


def read_param(params: dict, spec: ParamSpec) -> Any:
    """Current value for ``spec`` from an insert's params (or synth patch)."""
    params = params or {}
    if spec.key.startswith("b") and "." in spec.key:
        idx_s, field = spec.key.split(".", 1)
        try:
            idx = int(idx_s[1:])
        except ValueError:
            return spec.default
        bands = params.get("bands") or []
        if 0 <= idx < len(bands) and isinstance(bands[idx], dict):
            val = bands[idx].get(field, spec.default)
            if spec.kind == "bool":
                return bool(val)
            return val
        return spec.default
    val = params.get(spec.key, spec.default)
    if spec.kind == "bool":
        return bool(val)
    return val


def apply_param(params: Optional[dict], key: str, value: Any) -> dict:
    """Return a new params dict with ``key`` set (supports ``b{i}.field``)."""
    out = dict(params or {})
    if key.startswith("b") and "." in key:
        idx_s, field = key.split(".", 1)
        try:
            idx = int(idx_s[1:])
        except ValueError:
            out[key] = value
            return out
        bands = [dict(b) if isinstance(b, dict) else {} for b in (out.get("bands") or [])]
        while len(bands) <= idx:
            bands.append({})
        band = dict(bands[idx])
        if field == "enabled":
            band[field] = bool(value)
        else:
            band[field] = value
        bands[idx] = band
        out["bands"] = bands
        return out
    out[key] = value
    return out
