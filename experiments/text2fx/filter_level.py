"""Is Vital's filter level-dependent? `python filter_level.py`

Our twin drives the signal through a tanh BEFORE a linear filter. Vital's analog model has the saturation inside the
filter loop, which behaves differently: the resonance peak compresses as the level into the filter rises, even where a
pre-filter waveshaper would still be in its linear region.

The test: park the cutoff, set a high resonance, and render the same note at several oscillator levels. Normalise each
output by its input level and compare spectra. A filter that is linear (or only pre-saturated, below the knee) gives
the same normalised spectrum at every level; an in-loop nonlinearity flattens the resonance peak as the level rises.

Reported per level: the height of the resonance peak over the level well below it (dB), and the normalised spectrum's
difference from the quietest render.
"""
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
NOTE = [(45, 0.0, 1.5, 100)]           # A1, low enough that many harmonics sit under the cutoff
L = int(2.0 * SR)
BASE = dict(frame=0.0, detune_semis=0.0, blend=0.5, level=0.5,
            attack=0.002, decay=2.0, sustain=1.0, release=0.2,
            cutoff_raw=0.5, resonance=0.85, fenv_amount=0.0,
            fattack=0.001, fdecay=0.3, fsustain=1.0, frelease=0.3)


def spectrum(y, n_fft=8192):
    """Average magnitude spectrum of the sustained part, in dB."""
    seg = torch.as_tensor(y[int(0.2 * SR):int(1.2 * SR)], dtype=torch.float32)
    s = torch.stft(seg, n_fft, n_fft // 4, window=torch.hann_window(n_fft), return_complex=True).abs().mean(-1)
    return 20 * torch.log10(s + 1e-9), torch.fft.rfftfreq(n_fft, 1 / SR)


def render(level, drive=None, twin=False, recursive=True):
    phys = {**BASE, "level": level}
    if drive is not None:
        phys["fdrive"] = drive
    s = WavetableSynth(CL.TABLE, SR, voices=1, bounds=CL.B, interpolation=CL.TBL.get("interpolation", 1))
    s.random_phase = False
    s.recursive_filter = recursive
    if twin:
        with torch.no_grad():
            return s.render(raw_from_physical(s, **phys), NOTE, L)[0, 0].numpy()
    vp = s.vital_params(raw_from_physical(s, **phys))
    vp["oscillator_1_level"] = level ** 0.5
    vp["oscillator_1_phase_randomization"] = 0.0
    old = CL.midi
    CL.midi = [types.SimpleNamespace(pitch=p, start=st, duration=d, velocity=v) for p, st, d, v in NOTE]
    try:
        y = CL.vital_render(vp)
    finally:
        CL.midi = old
    return np.pad(y[:L], (0, max(0, L - len(y[:L]))))


def peak_height(db, f, fc):
    """Resonance peak height: the maximum near the cutoff minus the mean of the band an octave below."""
    near = (f > fc * 0.7) & (f < fc * 1.4)
    below = (f > fc * 0.25) & (f < fc * 0.5)
    return float(db[near].max() - db[below].mean())


if __name__ == "__main__":
    cutoff_hz = 261.6256 * 2 ** ((128.0 * BASE["cutoff_raw"] - 52.0) / 12)
    print(f"cutoff {cutoff_hz:.0f} Hz, resonance {BASE['resonance']}, note A1 — same patch, rising oscillator level\n")
    levels = [0.1, 0.25, 0.5, 1.0]
    for who, twin in (("Vital", False), ("twin ", True)):
        ref = None
        print(f"  {who}:")
        for lv in levels:
            y = render(lv, twin=twin)
            db, f = spectrum(y)
            rms = float(np.sqrt(np.mean(y[int(0.2 * SR):int(1.2 * SR)] ** 2)))
            norm = db - 20 * math.log10(max(rms, 1e-9))
            if ref is None:
                ref = norm
            m = (f > 60) & (f < 12000)
            print(f"    level {lv:<5}: out rms {rms:.4f} | resonance peak {peak_height(db, f, cutoff_hz):5.1f} dB | "
                  f"normalised spectrum vs the quietest: mean {float((norm - ref)[m].abs().mean()):4.2f} dB, max {float((norm - ref)[m].abs().max()):5.2f} dB")
