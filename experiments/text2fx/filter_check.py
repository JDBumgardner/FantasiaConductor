"""Frame-domain filter vs the recursive SVF, both against real Vital. `python filter_check.py [static|sweep|both]`

The question the frame-domain filter cannot answer for itself: when the cutoff moves fast, is a per-sample recursion
closer to the plugin than a per-frame convolution? Two cases:

  static  cutoff parked (filter envelope amount 0) at several cutoffs and resonances — both models should be close,
          and this separates "the filter's shape is wrong" from "its motion is wrong".
  sweep   a pluck: short filter attack/decay with a large envelope amount, so the cutoff crosses the band in ~10 ms.
          This is where the frame filter's ~4 dB onset error was measured.

Reported per case: error against Vital in 24 log bands over the whole note, and over the first 30 ms alone.
"""
import math
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ARGS = [a for a in sys.argv[1:]]
sys.argv = [sys.argv[0]]                      # closed_loop_filter reads argv[1] as a voice count
import common as C  # noqa: E402
import closed_loop_filter as CL  # noqa: E402
import svf as SVF  # noqa: E402
from synth import WavetableSynth, raw_from_physical  # noqa: E402

SR = 48000
NOTE = [(57, 0.0, 1.2, 100)]          # one A2, long enough to see the tail
L = int(2.0 * SR)


def band_err(a, b, n_bands=24, f_lo=60.0, f_hi=16000.0, n_fft=2048, hop=512, first=None):
    """Per-band level difference in dB (a vs b), optionally over the first `first` seconds only."""
    if first is not None:
        k = int(first * SR)
        a, b = a[:k], b[:k]
        n_fft, hop = 512, 128
    win = torch.hann_window(n_fft)
    out = []
    for sig in (a, b):
        s = torch.stft(torch.as_tensor(sig, dtype=torch.float32), n_fft, hop, window=win, return_complex=True).abs() ** 2
        f = torch.fft.rfftfreq(n_fft, 1 / SR)
        edges = torch.logspace(math.log10(f_lo), math.log10(f_hi), n_bands + 1)
        out.append(torch.stack([s[(f >= edges[i]) & (f < edges[i + 1])].sum() for i in range(n_bands)]))
    return 10 * torch.log10((out[0] + 1e-12) / (out[1] + 1e-12))


def render_twin(phys, recursive):
    s = WavetableSynth(CL.TABLE, SR, voices=1, bounds=CL.B, interpolation=CL.TBL.get("interpolation", 1))
    s.random_phase = False
    s.recursive_filter = recursive
    raw = raw_from_physical(s, **phys)
    with torch.no_grad():
        return s.render({k: v.clone() for k, v in raw.items()}, NOTE, L)[0, 0].numpy(), s


def render_vital(phys):
    s = WavetableSynth(CL.TABLE, SR, voices=1, bounds=CL.B, interpolation=CL.TBL.get("interpolation", 1))
    vp = s.vital_params(raw_from_physical(s, **phys))
    vp["oscillator_1_level"] = phys["level"] ** 0.5
    vp["oscillator_1_phase_randomization"] = 0.0      # the twin renders at a fixed phase; let the plugin do the same
    old_midi = CL.midi
    CL.midi = [types.SimpleNamespace(pitch=p, start=st, duration=d, velocity=v) for p, st, d, v in NOTE]
    try:
        y = CL.vital_render(vp)
    finally:
        CL.midi = old_midi
    return np.pad(y[:L], (0, max(0, L - len(y[:L]))))


BASE = dict(frame=0.375 * (CL.F_T - 1), detune_semis=0.0, blend=0.5, level=0.5,
            attack=0.002, decay=0.8, sustain=0.8, release=0.3,
            cutoff_raw=0.5, resonance=0.3, fenv_amount=0.0,
            fattack=0.001, fdecay=0.3, fsustain=1.0, frelease=0.3)


def case(name, **over):
    phys = {**BASE, **over}
    vit = render_vital(phys)
    rows = []
    for label, rec in (("frame", False), ("svf", True)):
        y, _ = render_twin(phys, rec)
        n = min(len(y), len(vit))
        a, b = torch.tensor(y[:n]), torch.tensor(vit[:n])
        sc = float((b.pow(2).sum() / (a.pow(2).sum() + 1e-12)).sqrt())        # level-match before comparing shape
        e_all = band_err(a * sc, b)
        e_on = band_err(a * sc, b, first=0.03)
        rows.append((label, float(e_all.abs().mean()), float(e_all.abs().max()), float(e_on.abs().mean()), float(e_on.abs().max())))
    print(f"{name:34s}   whole note: mean|max dB        first 30 ms: mean|max dB")
    for label, m, mx, om, omx in rows:
        print(f"   {label:6s}  {m:5.2f} | {mx:5.2f}                  {om:5.2f} | {omx:5.2f}")
    return rows


if __name__ == "__main__":
    which = ARGS[0] if ARGS else "both"
    if which in ("static", "both"):
        print("— cutoff parked (filter envelope amount 0) —")
        case("cutoff 0.35, res 0.1", cutoff_raw=0.35, resonance=0.1)
        case("cutoff 0.5,  res 0.3", cutoff_raw=0.5, resonance=0.3)
        case("cutoff 0.65, res 0.7", cutoff_raw=0.65, resonance=0.7)
    if which in ("sweep", "both"):
        print("\n— a pluck: the envelope sweeps the cutoff across the band in ~10 ms —")
        case("fast sweep, res 0.3", cutoff_raw=0.25, resonance=0.3, fenv_amount=0.6, fattack=0.001, fdecay=0.05, fsustain=0.1)
        case("fast sweep, res 0.7", cutoff_raw=0.25, resonance=0.7, fenv_amount=0.6, fattack=0.001, fdecay=0.05, fsustain=0.1)
        case("slower sweep, res 0.3", cutoff_raw=0.3, resonance=0.3, fenv_amount=0.5, fattack=0.004, fdecay=0.25, fsustain=0.2)
