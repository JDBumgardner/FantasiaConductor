"""Re-measure the filter's resonance law against Vital. `python res_law_measure.py [cutoffs]`

`synth.RES_TAB` maps the resonance knob to (passband gain, Q) for the twin's 2-pole filter. It was measured on the
contaminated rig (renders depended on what preceded them), at one level, and on a coarse grid that stops at 0.9 — and
the twin's remaining error sits exactly at resonance 0.95, where Vital's own law (`2.15·sqrt(resonance)`, kMaxResonance
2.15) is climbing steeply toward self-oscillation.

Per resonance: render the plugin, then search Q for the best level-matched band error and take the passband gain from
the level ratio that is left. Writes tables/res_law.json and prints a table ready to paste into RES_TAB.
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
EDGES = (60, 120, 250, 400, 600, 900, 1400, 2200, 3500, 6000, 10000, 16000)
RES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 1.0]
QS = [0.5, 0.6, 0.75, 0.9, 1.1, 1.4, 1.7, 2.1, 2.6, 3.2, 4.0, 5.0, 6.5, 8.5, 11.0, 15.0, 20.0]
LEVEL = 0.3                      # quiet enough that the drive stage is near-linear and does not confound the fit
BASE = dict(frame=0.0, detune_semis=0.0, blend=0.5, level=LEVEL,
            attack=0.002, decay=2.0, sustain=1.0, release=0.2,
            cutoff_raw=0.5, resonance=0.5, fenv_amount=0.0,
            fattack=0.001, fdecay=0.3, fsustain=1.0, frelease=0.3)


def _synth(q=None):
    s = WavetableSynth(CL.TABLE, SR, voices=1, bounds=CL.B, interpolation=CL.TBL.get("interpolation", 1))
    s.random_phase = False
    s.recursive_filter = True
    if q is not None:
        s.force_q = q                  # bypass RES_TAB for the measurement
    return s


def twin(phys, q):
    s = _synth(q)
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


def rms(y):
    return float(np.sqrt(np.mean(y[int(0.25 * SR):int(1.2 * SR)] ** 2)))


if __name__ == "__main__":
    cutoffs = [float(a) for a in ARGS] or [0.5]
    out = {}
    for co in cutoffs:
        print(f"\ncutoff raw {co} ({261.6256 * 2 ** ((128.0 * co - 52.0) / 12):.0f} Hz), level {LEVEL}")
        print(f"  {'res':>5} {'Q':>6} {'gain dB':>8} {'err dB':>7}   (old table)")
        rows = []
        for r in RES:
            phys = {**BASE, "resonance": r, "cutoff_raw": co}
            v = vital(phys)
            scored = []
            for q in QS:
                a = twin(phys, q)
                sc = float(np.sqrt((v ** 2).sum() / ((a ** 2).sum() + 1e-12)))
                scored.append((float((bands(a * sc) - bands(v)).abs().mean()), q, rms(a)))
            e, q, a_rms = min(scored)
            gain_db = 20 * math.log10(max(rms(v), 1e-9) / max(a_rms, 1e-9))
            old = _synth().res_law(torch.tensor(r))
            rows.append([r, q, round(gain_db, 2), round(e, 2)])
            print(f"  {r:5.2f} {q:6.2f} {gain_db:8.2f} {e:7.2f}   (was Q {float(old[1]):.2f}, {20 * math.log10(float(old[0])):+.1f} dB)")
        out[str(co)] = rows
    json.dump(out, open(os.path.join(HERE, "tables", "res_law.json"), "w"), indent=1)
    print("\nwrote tables/res_law.json")
