"""Fit the in-loop saturation law against Vital. `python filter_sat_law.py [quick]`

`filter_level.py` measured the gap: at resonance 0.85 the plugin's resonance peak goes -14.8 dB when quiet to +3.7 dB
at full level, where our fixed-Q filter runs to +30.2. Vital saturates inside the filter loop; `svf.svf_sat` models
that quasi-linearly as damping that rises with the resonance path's own level, k[n] = k (1 + (A[n]/a0)^p).

This fits a0 and p. For a grid of (resonance, oscillator level) it renders the plugin once, then scores each candidate
(a0, p) by the twin's band error against it — the same 7-band comparison used everywhere else, level-matched, so the
fit is about spectral shape rather than gain. Writes tables/filter_sat_law.json.
"""
import itertools
import json
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
EDGES = (60, 150, 350, 700, 1500, 3500, 8000, 16000)
BASE = dict(frame=0.0, detune_semis=0.0, blend=0.5, level=0.5,
            attack=0.002, decay=2.0, sustain=1.0, release=0.2,
            cutoff_raw=0.5, resonance=0.85, fenv_amount=0.0,
            fattack=0.001, fdecay=0.3, fsustain=1.0, frelease=0.3)
RES = [0.2, 0.5, 0.75, 0.95]
LEVELS = [0.1, 0.3, 0.6, 1.0]
A0 = [0.05, 0.1, 0.2, 0.35, 0.6, 1.0]
P = [1.0, 1.5, 2.0, 3.0]


def _synth(sat=None):
    s = WavetableSynth(CL.TABLE, SR, voices=1, bounds=CL.B, interpolation=CL.TBL.get("interpolation", 1))
    s.random_phase = False
    s.recursive_filter = True
    if sat is not None:
        s.sat_filter = True
        s.sat_a0, s.sat_p = sat
    return s


def twin(phys, sat=None):
    s = _synth(sat)
    with torch.no_grad():
        return s.render(raw_from_physical(s, **phys), NOTE, L)[0, 0].numpy()


def vital(phys):
    s = _synth()
    vp = s.vital_params(raw_from_physical(s, **phys))
    vp["oscillator_1_level"] = phys["level"] ** 0.5
    vp["oscillator_1_phase_randomization"] = 0.0
    old = CL.midi
    CL.midi = [types.SimpleNamespace(pitch=p, start=st, duration=d, velocity=v) for p, st, d, v in NOTE]
    try:
        y = CL.vital_render(vp)
    finally:
        CL.midi = old
    return np.pad(y[:L], (0, max(0, L - len(y[:L]))))


def bands(y):
    seg = torch.as_tensor(y[int(0.25 * SR):int(1.2 * SR)], dtype=torch.float32)
    s = torch.stft(seg, 8192, 2048, window=torch.hann_window(8192), return_complex=True).abs().pow(2).mean(-1)
    f = torch.fft.rfftfreq(8192, 1 / SR)
    return torch.tensor([10 * torch.log10(s[(f >= a) & (f < b)].sum() + 1e-12) for a, b in zip(EDGES, EDGES[1:])])


def err(a, b):
    sc = float(np.sqrt((b ** 2).sum() / ((a ** 2).sum() + 1e-12)))
    return float((bands(a * sc) - bands(b)).abs().mean())


if __name__ == "__main__":
    quick = "quick" in ARGS
    res_list, lv_list = (RES[::2], LEVELS[::2]) if quick else (RES, LEVELS)
    cells = [{**BASE, "resonance": r, "level": lv} for r in res_list for lv in lv_list]
    print(f"{len(cells)} cells (resonance x level); rendering the plugin…", flush=True)
    ref = [vital(c) for c in cells]
    base = np.mean([err(twin(c), v) for c, v in zip(cells, ref)])
    print(f"  no saturation model: mean band error {base:.2f} dB", flush=True)
    rows = []
    for a0, p in itertools.product(A0, P):
        e = np.mean([err(twin(c, (a0, p)), v) for c, v in zip(cells, ref)])
        rows.append((e, a0, p))
        print(f"  a0 {a0:<5} p {p:<4}: {e:.2f} dB", flush=True)
    rows.sort()
    best = rows[0]
    print(f"\nbest: a0 {best[1]}, p {best[2]} -> {best[0]:.2f} dB (from {base:.2f})")
    per_cell = [(c["resonance"], c["level"], round(err(twin(c, (best[1], best[2])), v), 2), round(err(twin(c), v), 2))
                for c, v in zip(cells, ref)]
    print("  per cell (resonance, level, with model, without):")
    for row in per_cell:
        print(f"    res {row[0]:<5} level {row[1]:<4}: {row[2]:5.2f} dB   (was {row[3]:5.2f})")
    json.dump({"a0": best[1], "p": best[2], "mean_db": round(best[0], 3), "baseline_db": round(float(base), 3),
               "per_cell": per_cell, "grid": [[round(e, 3), a, pp] for e, a, pp in rows]},
              open(os.path.join(HERE, "tables", "filter_sat_law.json"), "w"), indent=1)
    print("wrote tables/filter_sat_law.json")
