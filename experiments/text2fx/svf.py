"""Recursive time-varying state-variable filter — the twin's filter, one sample at a time instead of one STFT frame
at a time.

The frame-domain filter in `synth.tv_lowpass` freezes the cutoff over a whole analysis frame, which is exact when the
cutoff sits still and wrong where it moves fastest: a pluck's envelope opens the filter over ~10 ms, and that burst is
most of the note's high-band energy (measured error ~4 dB at the onset, which is why two frame sizes are crossfaded).
A per-sample recursion has no such window. It also puts the saturation *inside* the loop, where Vital's analog model
has it.

The topology is the TPT (topology-preserving transform) SVF — Vital's analog filter and the bilinear transform of the
same prototype the frame-domain version uses:

    g = tan(pi fc / sr),  k = 1/Q
    D(z) = (1 + kg + g^2) + (2g^2 - 2) z^-1 + (1 - kg + g^2) z^-2
    LP = g^2 (1 + z^-1)^2 / D      BP = g (1 - z^-2) / D      HP = (1 - z^-1)^2 / D

`torchlpc.sample_wise_lpc` runs the all-pole part with per-sample coefficients and a custom backward (numba: CUDA where
there is one, a parallel CPU scan otherwise — on this machine CPU, ~3 ms forward / 4 ms backward at 480k samples, which
is cheaper than the frame-domain filter it replaces). The numerators are three-tap FIRs, so they are plain shifts.
"""
import math

import torch

try:
    from torchlpc import sample_wise_lpc
except ImportError:  # pragma: no cover - torchlpc is a hard dependency of this module
    sample_wise_lpc = None

SR = 48000


def _shift(x, n):
    """x delayed by n samples, zero-filled."""
    if n == 0:
        return x
    return torch.cat([x.new_zeros(x.shape[0], n), x[:, :-n]], dim=-1)


def svf(x, fc, Q, sr=SR, blend=None, G=1.0, zi=None):
    """Time-varying 2-pole SVF. x, fc: (B, T); Q: scalar or (B, T). Returns (B, T).

    blend: None -> lowpass. Otherwise the twin's measured morph, 0 = LP, 1 = BP, 2 = HP:
        d <= 1: (1-d) LP + sqrt(1-(1-d)^2) BP,   d > 1: (d-1) HP + sqrt(1-(d-1)^2) BP
    G scales the output (the measured passband gain that comes with resonance).
    """
    if sample_wise_lpc is None:
        raise RuntimeError("torchlpc is required for the recursive filter (pip install torchlpc==0.6)")
    B, T = x.shape
    g = torch.tan(math.pi * fc.clamp(10.0, 0.49 * sr) / sr)
    k = (1.0 / torch.as_tensor(Q, dtype=x.dtype, device=x.device)).expand_as(g)
    gg = g * g
    d0 = 1.0 + k * g + gg
    a1 = (2.0 * gg - 2.0) / d0
    a2 = (1.0 - k * g + gg) / d0
    # numerators, per sample, already divided by d0
    nlp, nbp, nhp = gg / d0, g / d0, 1.0 / d0
    x1, x2 = _shift(x, 1), _shift(x, 2)
    lp_in = nlp * (x + 2 * x1 + x2)
    bp_in = nbp * (x - x2)
    hp_in = nhp * (x - 2 * x1 + x2)
    if blend is None:
        num = lp_in
    else:
        d = torch.as_tensor(blend, dtype=x.dtype, device=x.device)
        lo = torch.clamp(1 - d, 0, 1)
        hi = torch.clamp(d - 1, 0, 1)
        mid = torch.sqrt(torch.clamp(1 - torch.where(d <= 1, lo, hi) ** 2, min=1e-9))
        num = torch.where(d <= 1, lo * lp_in + mid * bp_in, hi * hp_in + mid * bp_in)
    a = torch.stack([a1, a2], dim=-1)
    # torchlpc runs on the CPU here (no CUDA on Apple silicon); its backward is exact either way
    dev = num.device
    y = sample_wise_lpc(num.cpu(), a.cpu(), zi=None if zi is None else zi.cpu()).to(dev)
    return G * y


def response(fc, Q, sr=SR, n=2048, blend=None):
    """The digital filter's magnitude response at one fixed cutoff, on the rfft grid — for checking against the
    analog prototype the frame-domain filter uses."""
    z = torch.exp(-2j * math.pi * torch.fft.rfftfreq(n, 1.0))
    g = math.tan(math.pi * min(max(fc, 10.0), 0.49 * sr) / sr)
    k = 1.0 / Q
    d = (1 + k * g + g * g) + (2 * g * g - 2) * z + (1 - k * g + g * g) * z ** 2
    lp = g * g * (1 + z) ** 2 / d
    bp = g * (1 - z ** 2) / d
    hp = (1 - z) ** 2 / d
    if blend is None:
        return lp.abs()
    lo, hi = max(0.0, min(1.0, 1 - blend)), max(0.0, min(1.0, blend - 1))
    mid = math.sqrt(max(1e-9, 1 - (lo if blend <= 1 else hi) ** 2))
    return ((lo * lp + mid * bp) if blend <= 1 else (hi * hp + mid * bp)).abs()


def analog_response(fc, Q, sr=SR, n=2048, blend=None):
    """The prototype `synth.tv_lowpass` applies per frame: H = wc^2 / (s^2 + s wc/Q + wc^2)."""
    w = 2 * math.pi * torch.fft.rfftfreq(n, 1 / sr)
    wc = 2 * math.pi * fc
    s = 1j * w
    den = s ** 2 + s * wc / Q + wc ** 2
    lp, bp, hp = wc ** 2 / den, (s * wc / Q) / den, s ** 2 / den
    if blend is None:
        return lp.abs()
    lo, hi = max(0.0, min(1.0, 1 - blend)), max(0.0, min(1.0, blend - 1))
    mid = math.sqrt(max(1e-9, 1 - (lo if blend <= 1 else hi) ** 2))
    return ((lo * lp + mid * bp) if blend <= 1 else (hi * hp + mid * bp)).abs()
