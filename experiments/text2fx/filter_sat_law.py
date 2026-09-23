"""Measure how Vital's filter resonance collapses with level. `python filter_sat_law.py [quick]`

`filter_level.py` showed the effect: at resonance 0.85 the plugin's resonance peak stays near 0 dB as the level rises
while our fixed-Q twin runs 26 dB past it. Vital's analog model saturates inside the filter loop, which lowers the
loop gain as the signal grows — a peak that flattens instead of screaming.

This measures the law our twin needs. For a grid of (resonance, oscillator level) it renders the plugin, then asks
which Q multiplier makes the twin's own render match: a per-cell search over a damping correction, scored on the
log-spectrum around the cutoff. The output is a table of (resonance, pre-filter rms) -> Q multiplier, written to
tables/filter_sat_law.json, plus the fit's residual so the next step knows what is left.
"""
import json
import math
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ARGS = list(sys.argv[1:])
sys.argv = [sys.argv[0]]
import closed_loop_filter as CL  # noqa: E402
from synth import WavetableSynth, raw_from_physical  # noqa: E402

SR = 48000
NOTE = [(45, 0.0, 1.5, 100)]
L = int(2.0 * SR)
BASE = dict(frame=0.0, detune_semis=0.0, blend=0.5, level=0.5,
            attack=0.002, decay=2.0, sustain=1.0, release=0.2,
            cutoff_raw=0.5, resonance=0.85, fenv_amount=0.0,
            fattack=0.001, fdecay=0.3, fsustain=1.0, frelease=0.3)
RES = [0.2, 0.4, 0.6, 0.75, 0.85, 0.95]
LEVELS = [0.08, 0.15, 0.3, 0.5, 0.75, 1.0]
QMULT = [0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0, 5.5, 8.0, 12.0, 18.0]


def _synth(qmult=1.0):
    s = WavetableSynth(CL.TABLE, SR, voices=1, bounds=CL.B, interpolation=CL.TBL.get("interpolation", 1))
    s.random_phase = False
    s.recursive_filter = True
    s.q_mult = qmult                      # damping correction under test (svf sees Q / q_mult)
    return s


def render_twin(phys, qmult):
    s = _synth(qmult)
    with torch.no_grad():
        return s.render(raw_from_physical(s, **phys), NOTE, L)[0, 0].numpy()


def render_vital(phys):
    s = _synth()
    vp = s.vital_params(raw_from_physical(s, **phys))
    vp["oscillator_1_level"] = phys["level"] ** 0.5
    old = CL.midi
    CL.midi = [types.SimpleNamespace(pitch=p, start=st, duration=d, velocity=v) for p, st, d, v in NOTE]
    try:
        y = CL.vital_render(vp)
    finally:
        CL.midi = old
    return np.pad(y[:L], (0, max(0, L - len(y[:L]))))


def logspec(y, n_fft=8192):
    seg = torch.as_tensor(y[int(0.25 * SR):int(1.2 * SR)], dtype=torch.float32)
    s = torch.stft(seg, n_fft, n_fft // 4, window=torch.hann_window(n_fft), return_complex=True).abs().mean(-1)
    return 20 * torch.log10(s + 1e-7), torch.fft.rfftfreq(n_fft, 1 / SR)


def score(a, b, fc):
    """Difference around the cutoff, level-matched — the resonance region is what the damping controls."""
    da, f = logspec(a)
    db, _ = logspec(b)
    m = (f > fc * 0.2) & (f < fc * 6)
    da, db = da[m], db[m]
    return float((da - da.mean() - (db - db.mean())).abs().mean())


def prefilter_rms(phys):
    """The signal level the filter actually sees in the twin (before its drive stage)."""
    s = _synth()
    raw = raw_from_physical(s, **phys)
    ph = s.physical(raw)
    return float(s.PRE_GAIN * (ph["level"] / 0.694))


if __name__ == "__main__":
    quick = "quick" in ARGS
    res_list = RES[::2] if quick else RES
    lv_list = LEVELS[::2] if quick else LEVELS
    fc = 261.6256 * 2 ** ((128.0 * BASE["cutoff_raw"] - 52.0) / 12)
    print(f"cutoff {fc:.0f} Hz, note A1. Per cell: the Q multiplier that best matches Vital, and the residual.\n")
    table = {}
    for r in res_list:
        row = {}
        for lv in lv_list:
            phys = {**BASE, "resonance": r, "level": lv}
            vit = render_vital(phys)
            best = min(((score(render_twin(phys, q), vit, fc), q) for q in QMULT), key=lambda t: t[0])
            amp = prefilter_rms(phys)
            row[f"{lv}"] = [best[1], round(best[0], 2), round(amp, 4)]
            print(f"  res {r:<5} level {lv:<5} (pre-filter rms {amp:.3f}): Q x{best[1]:<5} residual {best[0]:.2f} dB")
        table[f"{r}"] = row
    out = os.path.join(HERE, "tables", "filter_sat_law.json")
    json.dump(table, open(out, "w"), indent=1)
    print(f"\nwrote {out}")
