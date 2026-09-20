"""FX node registry: build / init / prior / describe per node type, and a chain builder.

This is the seed of the graph compiler: an app FX DAG becomes a list of node types here,
each with a differentiable twin, a musically-bounded init, a prior at its knees, and a
plain-language description of what its parameters mean.

Node types: eq, comp, drive (level-referenced tanh), dist (GRAFX tanh at absolute level), pwtanh, chorus, transient, gate, delay, reverb.
"""
import os, math, torch, torch.utils.checkpoint, numpy as np
from grafx.data import GRAFX, NodeConfigs, convert_to_tensor
from grafx.render import render_grafx, reorder_for_fast_render
from grafx.render.prepare import prepare_render
from grafx import processors as P

SR = 48000
DB_PER_LG, DB_PER_LPG, DB_PER_LNE = 40 / math.log(10), 20 / math.log(10), 10 / math.log(10)   # eq gain, pre-gain, ln-energy

def _r(g, *shape, sd=0.3): return sd * torch.randn(*shape, generator=g)
def _logit(x): return torch.logit(torch.tensor(float(x)).clamp(1e-4, 1 - 1e-4))

import grafx.processors.core.convolution as _conv
def _convolve(x, h, mode="zerophase", pad_mode="pow2"):
    """GRAFX's convolve calls irfft without n=, so an odd x+h-1 (ours: 480000 + 4000 - 1) comes back one sample
    short and the whole output is time-stretched by that sample — dry + wet then combs at the top of the band.
    Power-of-two padding is also 3-10x faster than the exact minimum length."""
    n = x.shape[-1] + h.shape[-1] - 1; n = int(2 ** math.ceil(math.log2(n)))
    Y = torch.fft.rfft(torch.nn.functional.pad(x, (0, n - x.shape[-1]))) * torch.fft.rfft(torch.nn.functional.pad(h, (0, n - h.shape[-1])))
    y = torch.fft.irfft(Y, n=n)
    return y[..., h.shape[-1] // 2: h.shape[-1] // 2 + x.shape[-1]] if mode == "zerophase" else y[..., : x.shape[-1]]
_conv.convolve = _convolve

class DryWet(P.DryWet):
    """GRAFX's DryWet documents w = sigmoid(z) but its forward uses z raw, so an init of -1 mixed -1 x wet + 2 x dry
    and 'mix 0.5' meant z = 0 = fully dry. Every run before 2026-09-15 had this. This subclass applies the sigmoid."""
    def forward(self, input_signals, drywet_weight, **kw): return super().forward(input_signals, torch.sigmoid(drywet_weight), **kw)

# Initialisation policy. "neutral": every effect starts at (near) bypass -- flat EQ, ratio 1.1, drive -24 dB, mixes
# 0.015, shaper 0, gate range 1 dB -- so a search begins AT the instrument and the locality weight decides how much
# each node switches on. "prior": the older musical mid-settings (compressor 3:1, mixes 0.2-0.3...), which put every
# result a fixed hop from the instrument before the prompt had said anything (e-piano: distance 0.46 for +0.03).
NEUTRAL = os.environ.get("T2_FX_INIT", "neutral") == "neutral"
def _mix0(g, sd=0.15): return torch.logit(torch.tensor(0.015)) + _r(g, 1, 1, sd=sd)     # a mix of ~1.5 %: a live gradient, an inaudible effect

# ---------------------------------------------------------------------------------------------
NODES = {}
def node(name):
    def deco(cls): NODES[name] = cls; return cls
    return deco

@node("eq")
class EQ:
    @staticmethod
    def scale(p, a): return {**p, "log_gain": p["log_gain"] * a}
    @staticmethod
    def make(N): return P.GainStagingRegularization(P.ParametricEqualizer(num_filters=6, processor_channel="mono"))
    @staticmethod
    def init(g):
        f0 = torch.tensor(np.geomspace(80, 12000, 6), dtype=torch.float32)
        return {"w0": torch.logit(2 * f0 / SR)[None, None] + _r(g, 1, 1, 6, sd=0.1), "q_inv": _r(g, 1, 1, 6), "log_gain": _r(g, 1, 1, 6, sd=0.02 if NEUTRAL else 0.2)}
    @staticmethod
    def prior(p):
        hz = torch.sigmoid(p["w0"][0, 0]) * SR / 2; db = DB_PER_LG * p["log_gain"][0, 0]; q = torch.exp(-p["q_inv"][0, 0])
        pen = (torch.relu(db.abs() - 6.0) ** 2).sum() / 36 + (torch.relu(q - 2.0) ** 2).sum() / 4
        lf = torch.log10(hz); d = (lf[:, None] - lf[None, :]).abs() + torch.eye(6, device=lf.device) * 9
        return pen + 0.35 * (torch.relu(0.2 - d) * db.abs()[:, None] * db.abs()[None, :]).sum() / 6
    @staticmethod
    def describe(p):
        hz = torch.sigmoid(p["w0"][0, 0]) * SR / 2; db = DB_PER_LG * p["log_gain"][0, 0]
        return {"eq": [(int(h), round(float(g), 1)) for h, g in sorted(zip(hz, db))]}

from torchcomp.core import CompressorFunction as _CompressorFunction
class _Ballistics(_CompressorFunction):
    """torchcomp's attack/release recursion, minus its `ctx.save_for_forward(..., y, ...)`: saving the output for
    forward-mode AD makes a grad_fn -> y -> grad_fn cycle the GC cannot see, so every forward whose graph is not
    backpropagated (an evaluation, an aborted line search) leaks the whole upstream graph (~230 MB here)."""
    @staticmethod
    def setup_context(ctx, inputs, output):
        x, zi, at, rt = inputs; y, at_mask = output
        ctx.mark_non_differentiable(at_mask); ctx.save_for_backward(x, y, zi, at, rt, at_mask)

class Compressor(torch.nn.Module):
    """Feed-forward RMS compressor with an amplitude-domain gain law (dB in, dB out).

    GRAFX's Compressor applies its ln-energy gain straight to the waveform (a squared gain: ratio 2 is
    already a limiter, ratio > 2 inverts) and shifts the threshold by -26 dB, so we keep only its FFT
    one-pole smoother and write the law ourselves.  Parameters: thr_db, log_ratio (ratio = 1 + e^x),
    log_knee (knee width dB = e^x), z_alpha (one-pole coefficient sigmoid(z); tau = -1/(sr ln alpha)).
    With ballistics=True the smoother is torchcomp's exact attack/release recursion (z_alpha has 2
    entries), run on the CPU because it is 200x slower on MPS; the energy round trip costs ~1 ms."""
    def __init__(self, N, iir_len=16384, ballistics=False):
        super().__init__()
        from grafx.processors.core.convolution import FIRConvolution
        self.register_buffer("arange", torch.arange(iir_len)[None, :].float())
        self.conv = FIRConvolution(mode="causal", flashfftconv=False, max_input_len=N)
        self.ballistics = ballistics
    def smooth(self, e, z_alpha):
        if self.ballistics:
            # torchcomp's compressor_core smooths a GAIN: its coefficient is the update fraction 1 - alpha (ms2coef =
            # 1 - exp(-2200/ms/sr)), and "attack" is the coefficient for a FALLING input. Fed a level, both conventions
            # were wrong here until 2026-09-20: alpha (~0.999) as the fraction made the ballistics near-instantaneous
            # (a sample-by-sample waveshaper with a hard knee), and unswapped roles hug the troughs (a 0.5 sine read as
            # 0.16; JUCE reads 0.44). Found by the round trip against pedalboard's compressor.
            ts = (1 - torch.sigmoid(z_alpha)).cpu(); ec = e.cpu()
            y = _Ballistics.apply(ec, torch.ones(ec.shape[0]), ts[..., 1], ts[..., 0])[0]                 # (release, attack): rising level -> attack coefficient
            return y.to(e.device)
        alpha = torch.sigmoid(z_alpha).clamp(max=1 - 1e-5)
        h = torch.exp(self.arange * torch.log(alpha)); h = h / h.sum(-1, keepdim=True)   # unit-DC truncated one-pole
        return torch.relu(self.conv(e, h))
    detector = "rms"
    def forward(self, x, thr_db, log_ratio, log_knee, z_alpha):
        if self.detector == "peak":                                           # JUCE-style: ballistics on |x|, level = 20 log10
            lvl = 20 * torch.log10(self.smooth(x.abs().mean(-2), z_alpha) + 1e-4)
        else:
            e = self.smooth(x.square().mean(-2), z_alpha)                   # (B, L) smoothed energy
            lvl = 10 * torch.log10(e + 1e-8)                                  # RMS dB, floor -80
        s, W = 1 / (1 + torch.exp(log_ratio)), torch.exp(log_knee)            # slope 1/ratio, knee width dB
        over = lvl - thr_db
        mid = (s - 1) * (over + W / 2).clamp(min=0) ** 2 / (2 * W)
        gain_db = torch.where(over > W / 2, (s - 1) * over, mid)
        return x * torch.pow(10.0, gain_db / 20)[:, None, :]

@node("comp")
class Comp:
    """Not gain-staged: lowering the peaks is its job; the loudness normaliser downstream restores level."""
    ballistics = True
    @staticmethod
    def scale(p, a): return {**p, "log_ratio": torch.log((a * torch.exp(p["log_ratio"])).clamp(min=1e-6))}
    @classmethod
    def make(cls, N): return Compressor(N, ballistics=cls.ballistics)
    @classmethod
    def init(cls, g):
        za = torch.tensor([[7.3, 9.2]]) if cls.ballistics else torch.tensor([[7.3]])                # ~30 ms attack, ~200 ms release
        return {"thr_db": torch.tensor([[-6.0 if NEUTRAL else -18.0]]) + _r(g, 1, 1, sd=3.0), "log_ratio": torch.tensor([[math.log(0.05 if NEUTRAL else 2.0)]]) + _r(g, 1, 1, sd=0.3 if NEUTRAL else 1.0),   # neutral: 1.05:1 above -6 dB touches only peaks
                "log_knee": torch.tensor([[math.log(6.0)]]) + _r(g, 1, 1, sd=0.2), "z_alpha": za + _r(g, *za.shape, sd=0.5)}
    @staticmethod
    def prior(p):
        thr = p["thr_db"][0, 0]; ratio = 1 + torch.exp(p["log_ratio"][0, 0]); z = p["z_alpha"][0]
        knee = torch.exp(p["log_knee"][0, 0])
        return (torch.relu(-40 - thr) ** 2 + torch.relu(thr) ** 2) / 100 + torch.relu(ratio - 8.0) ** 2 / 16 + torch.relu(knee - 24.0) ** 2 / 100 \
               + (torch.relu(3.9 - z) ** 2 + torch.relu(z - 9.9) ** 2).sum()                                  # 1 ms .. 400 ms
    @staticmethod
    def describe(p):
        tau = [round(-1000 / (SR * math.log(min(1 / (1 + math.exp(-float(z))), 0.999999))), 1) for z in p["z_alpha"][0]]
        return {"comp": dict(threshold_db=round(float(p["thr_db"][0, 0]), 1), ratio=round(float(1 + torch.exp(p["log_ratio"][0, 0])), 2),
                             knee_db=round(float(torch.exp(p["log_knee"][0, 0])), 1), **({"attack_ms": tau[0], "release_ms": tau[1]} if len(tau) == 2 else {"smoothing_ms": tau[0]}))}

@node("dist")
class Dist:
    @staticmethod
    def scale(p, a): return {**p, "log_pre_gain": p["log_pre_gain"] * a}
    @staticmethod
    def make(N): return P.GainStagingRegularization(P.TanhDistortion())
    @staticmethod
    def init(g): return {"log_pre_gain": _r(g, 1, 1)}         # absolute-level tanh: 0 dB is already near-linear on our signals
    @staticmethod
    def prior(p):
        drive = DB_PER_LPG * p["log_pre_gain"][0, 0]; return torch.relu(drive - 10.0) ** 2 / 100 + torch.relu(-drive) ** 2 / 4
    @staticmethod
    def describe(p): return {"drive_db": round(float(DB_PER_LPG * p["log_pre_gain"][0, 0]), 1)}

class Drive(torch.nn.Module):
    """Level-referenced soft clipper: the input is normalised to unit RMS, scaled by 0.3 * 10^(drive/20), pushed
    through tanh, and brought back to the input RMS. So 'drive' means saturation, not level: 0 dB is a gentle bend
    (0.3 rms into tanh), +10 dB is heavy, +20 dB is a near-hard clip — on any input level. Gain-neutral by
    construction, so no gain-staging wrapper. (The GRAFX TanhDistortion 'dist' node was a tanh at absolute level:
    on a -20 dBFS signal it did nothing until its +10 dB prior cap, and the grid never used it.)"""
    def forward(self, x, drive_db):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6
        g = 0.3 * torch.pow(10.0, drive_db / 20)[:, None, :] if drive_db.ndim == 2 else 0.3 * torch.pow(10.0, drive_db / 20)
        y = torch.tanh(g * x / rms)
        return y * rms / (y.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)

@node("drive")
class DriveNode:
    @staticmethod
    def make(N): return Drive()
    @staticmethod
    def init(g): return {"drive_db": torch.tensor([[-24.0 if NEUTRAL else 0.0]]) + _r(g, 1, 1, sd=2.0)}
    @staticmethod
    def prior(p): d = p["drive_db"][0, 0]; return torch.relu(d - 20.0) ** 2 / 100 + torch.relu(-6.0 - d) ** 2 / 36
    @staticmethod
    def describe(p): return {"drive_db": round(float(p["drive_db"][0, 0]), 1)}
    @staticmethod
    def scale(p, a): return {**p, "drive_db": p["drive_db"] * a - 40.0 * (1 - a)}     # amount 0 -> -40 dB: tanh linear, a bypass

class LevelRef(torch.nn.Module):
    """Run a saturator at a fixed reference level: normalise the input to 0.3 RMS x pre-gain, process, restore the RMS."""
    def __init__(self, proc): super().__init__(); self.proc = proc
    def forward(self, x, **kw):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6
        y = self.proc(0.3 * x / rms, **kw); y = y[0] if isinstance(y, tuple) else y
        return y * rms / (y.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)

@node("pwtanh")
class PwTanh:
    """Piecewise tanh: asymmetric hardness (e^x) and thresholds (sigmoid) — a different saturation flavour, level-referenced."""
    @staticmethod
    def scale(p, a): return {**p, "log_pre_gain": p["log_pre_gain"] * a - (1 - a) * 40 / DB_PER_LPG}
    @staticmethod
    def make(N): return LevelRef(P.PiecewiseTanhDistortion())
    @staticmethod
    def init(g): return {"log_hardness": _r(g, 1, 2, sd=0.3), "z_threshold": _r(g, 1, 2, sd=0.5), "log_pre_gain": torch.tensor([[-24.0 / DB_PER_LPG if NEUTRAL else 0.0]]) + _r(g, 1, 1)}
    @staticmethod
    def prior(p):
        drive = DB_PER_LPG * p["log_pre_gain"][0, 0]
        return torch.relu(drive - 20.0) ** 2 / 100 + torch.relu(-6.0 - drive) ** 2 / 36 + (torch.relu(p["log_hardness"].abs() - 1.1) ** 2).sum()
    @staticmethod
    def describe(p):
        return {"pwtanh": dict(drive_db=round(float(DB_PER_LPG * p["log_pre_gain"][0, 0]), 1), hardness=[round(float(v), 2) for v in torch.exp(p["log_hardness"][0])],
                               threshold=[round(float(v), 2) for v in torch.sigmoid(p["z_threshold"][0])])}

class Chorus(torch.nn.Module):
    """LFO-modulated fractional delay, linear interpolation, dry + wet mix. Delay d(n) = d0 + depth*sin(2*pi*rate*n/sr).
    Flanger is the same node at 1-5 ms; no feedback path (that needs a recursion we don't have yet)."""
    def __init__(self, N, sr=SR):
        super().__init__(); self.register_buffer("n", torch.arange(N).float()); self.sr = sr
    def forward(self, x, z_delay_ms, z_depth, z_rate, drywet_weight):
        d0 = 1 + 29 * torch.sigmoid(z_delay_ms)                 # 1..30 ms centre
        depth = torch.sigmoid(z_depth) * d0 * 0.9              # up to 90 % of the centre, never negative delay
        rate = 0.05 * torch.exp(torch.sigmoid(z_rate) * math.log(10 / 0.05))     # 0.05..10 Hz, log-spaced
        d = (d0 + depth * torch.sin(2 * math.pi * rate * self.n / self.sr)) * self.sr / 1000     # samples, (B, L)
        pos = (self.n - d).clamp(min=0); i0 = pos.floor(); frac = pos - i0; i0 = i0.long()
        i1 = (i0 + 1).clamp(max=x.shape[-1] - 1)
        xb = x[:, 0]                                             # mono (B, L)
        wet = torch.gather(xb, 1, i0) * (1 - frac) + torch.gather(xb, 1, i1) * frac
        m = torch.sigmoid(drywet_weight)
        return ((1 - m) * xb + m * wet)[:, None, :]

@node("chorus")
class ChorusNode:
    @staticmethod
    def scale(p, a): return {**p, "drywet_weight": torch.logit((a * torch.sigmoid(p["drywet_weight"])).clamp(1e-6, 1 - 1e-6))}
    @staticmethod
    def make(N): return P.GainStagingRegularization(Chorus(N))
    @staticmethod
    def init(g): return {"z_delay_ms": torch.full((1, 1), -0.5) + _r(g, 1, 1), "z_depth": torch.full((1, 1), -1.0) + _r(g, 1, 1),
                         "z_rate": torch.full((1, 1), -0.5) + _r(g, 1, 1), "drywet_weight": _mix0(g) if NEUTRAL else torch.full((1, 1), -1.0) + _r(g, 1, 1)}
    @staticmethod
    def prior(p): return torch.relu(torch.sigmoid(p["drywet_weight"][0, 0]) - 0.5) ** 2 / 0.25
    @staticmethod
    def describe(p):
        d0 = 1 + 29 * float(torch.sigmoid(p["z_delay_ms"][0, 0])); depth = float(torch.sigmoid(p["z_depth"][0, 0])) * d0 * 0.9
        rate = 0.05 * math.exp(float(torch.sigmoid(p["z_rate"][0, 0])) * math.log(10 / 0.05))
        return {"chorus": dict(delay_ms=round(d0, 1), depth_ms=round(depth, 1), rate_hz=round(rate, 2), mix=round(float(torch.sigmoid(p["drywet_weight"][0, 0])), 2))}

class EnvFollowers(torch.nn.Module):
    """Shared helper: RMS level in dB through one or more unit-DC truncated one-pole smoothers (FFT conv)."""
    def __init__(self, N, iir_len=16384):
        super().__init__()
        from grafx.processors.core.convolution import FIRConvolution
        self.register_buffer("arange", torch.arange(iir_len)[None, :].float())
        self.conv = FIRConvolution(mode="causal", flashfftconv=False, max_input_len=N)
    def level_db(self, x, z_alpha):
        alpha = torch.sigmoid(z_alpha).clamp(max=1 - 1e-5)                              # (B, K)
        h = torch.exp(self.arange[None] * torch.log(alpha)[..., None]); h = h / h.sum(-1, keepdim=True)   # (B, K, iir_len)
        e = x.square().mean(-2, keepdim=True).expand(-1, h.shape[1], -1)                # (B, K, L)
        e = torch.relu(self.conv(e, h))
        return 10 * torch.log10(e + 1e-8)

class TransientShaper(EnvFollowers):
    """Fast (1 ms) vs slow (z_slow, ~20-100 ms) follower. Where fast > slow the sound is attacking: gain_db = a*(fast-slow);
    elsewhere sustaining: gain_db = s*(slow-fast). a, s in [-1, 1] (dB per dB), so +-12 dB of difference maps to +-12 dB."""
    def forward(self, x, attack, sustain, z_slow):
        z = torch.cat([torch.full_like(z_slow, 3.9), z_slow], -1)                    # 1 ms fast, learnable slow
        lv = self.level_db(x, z); diff = lv[:, 0] - lv[:, 1]                          # (B, L) fast - slow
        a, s = torch.tanh(attack), torch.tanh(sustain)
        gain_db = a * torch.relu(diff) - s * torch.relu(-diff)
        return x * torch.pow(10.0, gain_db.clamp(-24, 24) / 20)[:, None, :]

@node("transient")
class TransientNode:
    @staticmethod
    def scale(p, a): return {**p, "attack": torch.atanh(a * torch.tanh(p["attack"])), "sustain": torch.atanh(a * torch.tanh(p["sustain"]))}
    @staticmethod
    def make(N): return TransientShaper(N)
    @staticmethod
    def init(g): return {"attack": _r(g, 1, 1, sd=0.02 if NEUTRAL else 0.3), "sustain": _r(g, 1, 1, sd=0.02 if NEUTRAL else 0.3), "z_slow": torch.tensor([[8.4]]) + _r(g, 1, 1, sd=0.3)}   # ~90 ms
    @staticmethod
    def prior(p): z = p["z_slow"][0, 0]; return torch.relu(6.9 - z) ** 2 + torch.relu(z - 9.2) ** 2      # 20..200 ms
    @staticmethod
    def describe(p):
        tau = -1000 / (SR * math.log(min(1 / (1 + math.exp(-float(p["z_slow"][0, 0]))), 0.999999)))
        return {"transient": dict(attack=round(float(torch.tanh(p["attack"][0, 0])), 2), sustain=round(float(torch.tanh(p["sustain"][0, 0])), 2), window_ms=round(tau, 1))}

class Gate(EnvFollowers):
    """Soft expander: gain_db = -range * (1 - sigmoid((level - thr) / width)), level from a smoothed RMS follower."""
    def forward(self, x, thr_db, log_width, z_alpha, log_range):
        lv = self.level_db(x, z_alpha)[:, 0]
        openness = torch.sigmoid((lv - thr_db) / torch.exp(log_width))
        gain_db = -torch.exp(log_range) * (1 - openness)
        return x * torch.pow(10.0, gain_db / 20)[:, None, :]

@node("gate")
class GateNode:
    @staticmethod
    def scale(p, a): return {**p, "log_range": p["log_range"] + math.log(max(a, 1e-6))}
    @staticmethod
    def make(N): return Gate(N)
    @staticmethod
    def init(g): return {"thr_db": torch.tensor([[-40.0]]) + _r(g, 1, 1, sd=3), "log_width": torch.tensor([[math.log(6.0)]]) + _r(g, 1, 1, sd=0.2),
                         "z_alpha": torch.tensor([[7.3]]) + _r(g, 1, 1, sd=0.3), "log_range": torch.tensor([[math.log(1.0 if NEUTRAL else 24.0)]]) + _r(g, 1, 1, sd=0.2)}
    @staticmethod
    def prior(p):
        thr = p["thr_db"][0, 0]; z = p["z_alpha"][0, 0]
        return (torch.relu(-70 - thr) ** 2 + torch.relu(thr + 10) ** 2) / 100 + torch.relu(3.9 - z) ** 2 + torch.relu(z - 9.2) ** 2 + torch.relu(torch.exp(p["log_range"][0, 0]) - 25) ** 2 / 100
    @staticmethod
    def describe(p):
        tau = -1000 / (SR * math.log(min(1 / (1 + math.exp(-float(p["z_alpha"][0, 0]))), 0.999999)))
        return {"gate": dict(threshold_db=round(float(p["thr_db"][0, 0]), 1), width_db=round(float(torch.exp(p["log_width"][0, 0])), 1),
                             range_db=round(float(torch.exp(p["log_range"][0, 0])), 1), smoothing_ms=round(tau, 1))}

@node("delay")
class Delay:
    """8 taps up to ~0.5 s, learnable times (surrogate delay lines), no per-tap colour, explicit dry/wet.
    Returns a radii_reg intermediate that must be added to the loss (keeps the surrogate a true delay)."""
    @staticmethod
    def scale(p, a): return {**p, "drywet_weight": torch.logit((a * torch.sigmoid(p["drywet_weight"])).clamp(1e-6, 1 - 1e-6))}
    @staticmethod
    def make(N):
        return DryWet(P.MultitapDelay(processor_channel="mono", num_segments=8, zp_filter_per_tap=False, flashfftconv=False, max_input_len=N), external_param=False)
    @staticmethod
    def init(g): return {"delay_z": _r(g, 1, 8, 2, sd=0.5), "drywet_weight": _mix0(g) if NEUTRAL else torch.full((1, 1), -1.5) + _r(g, 1, 1)}
    @staticmethod
    def prior(p): return torch.relu(torch.sigmoid(p["drywet_weight"][0, 0]) - 0.4) ** 2 / 0.16
    @staticmethod
    def describe(p): return {"delay_mix": round(float(torch.sigmoid(p["drywet_weight"][0, 0])), 2)}

@node("reverb")
class Reverb:
    """12-band filtered-noise reverb. GRAFX draws the noise bank from the *unseeded* numpy RNG at construction
    and re-draws a random offset every forward by default, so we seed it and pin the offset: the render you hear
    is the render that was optimised, and CPU-vs-MPS gradient checks are meaningful."""
    @staticmethod
    def scale(p, a): return {**p, "drywet_weight": torch.logit((a * torch.sigmoid(p["drywet_weight"])).clamp(1e-6, 1 - 1e-6))}
    @staticmethod
    def make(N):
        np.random.seed(0)
        return P.GainStagingRegularization(DryWet(P.FilteredNoiseShapingReverb(sr=SR, processor_channel="mono", zerophase=False, noise_randomness="fixed", flashfftconv=False, max_input_len=N), external_param=False))
    @staticmethod
    def init(g): return {"log_decay": _r(g, 1, 1, 12, sd=0.5), "log_gain": 1.0 + _r(g, 1, 1, 12, sd=0.1), "drywet_weight": _mix0(g) if NEUTRAL else torch.full((1, 1), -1.0) + _r(g, 1, 1)}
    @staticmethod
    def prior(p): return (torch.relu(-p["log_gain"]) ** 2).sum() + torch.relu(torch.sigmoid(p["drywet_weight"][0, 0]) - 0.5) ** 2 / 0.25
    @staticmethod
    def describe(p):
        rt = [int(50 + 1950 * float(torch.sigmoid(p["log_decay"][0, 0, i]))) for i in (1, 6, 10)]
        return {"reverb": dict(mix=round(float(torch.sigmoid(p["drywet_weight"][0, 0])), 2), rt60_lo_mid_hi_ms=rt)}

# ---------------------------------------------------------------------------------------------
class Chain:
    """A serial chain of node types, as one GRAFX graph."""
    def __init__(self, types, N, device):
        self.types, self.N, self.device = list(types), N, device
        G = GRAFX(config=NodeConfigs(self.types)); G.add_serial_chain(["in", *self.types, "out"])
        self.rd = prepare_render(reorder_for_fast_render(convert_to_tensor(G), method="beam"))
        self.procs = torch.nn.ModuleDict({t: NODES[t].make(N) for t in self.types}).to(device)
    def init(self, seed):
        g = torch.Generator().manual_seed(seed)
        return {t: {k: v.to(self.device).requires_grad_(True) for k, v in NODES[t].init(g).items()} for t in self.types}
    def render(self, x, params, input_grad=True, checkpoint=True):
        """x: (1,1,L) -> (audio (L,), regulariser sum: gain-staging + delay radii).

        Serial chains are rendered node by node here rather than through render_grafx so each node can be
        activation-checkpointed: on the 8 GB machine the saved FFT spectra of eight FIR-convolution nodes
        plus CLAP's activations do not fit under the MPS cap, and recomputing a node's forward in backward
        costs less than a fifth of a step."""
        reg = torch.zeros((), device=x.device)
        for t in self.types:
            proc, p = self.procs[t], params[t]
            def f(x, *vals, proc=proc, keys=tuple(p.keys())):
                out = proc(x, **dict(zip(keys, vals)))
                if isinstance(out, tuple):
                    out, inter = out
                    r = sum(inter[k].sum() for k in ("gain_reg", "radii_reg") if k in inter) if isinstance(inter, dict) else inter.sum()
                else: r = torch.zeros((), device=x.device)
                return out, r
            if checkpoint and getattr(NODES[t], "checkpoint", True) and torch.is_grad_enabled():
                x, r = torch.utils.checkpoint.checkpoint(f, x, *p.values(), use_reentrant=False)
            else: x, r = f(x, *p.values())
            reg = reg + r
        return x[0, 0], reg
    def prior(self, params): return sum(NODES[t].prior(params[t]) for t in self.types)
    def describe(self, params):
        d = {}
        for t in self.types: d.update(NODES[t].describe({k: v.detach() for k, v in params[t].items()}))
        return d
    def scale(self, params, a):
        """Every node a fraction `a` of the way from bypass to its setting — the 'amount' knob of the writeup."""
        return {t: NODES[t].scale(params[t], a) for t in self.types}
    def params_flat(self, params): return [v for d in params.values() for v in d.values()]

def selftest(N=480000, seed=0):
    """Per node: (1) amount 0 returns the input (bypass), (2) CPU-vs-MPS gradient cosine at full length.
    Test (1) is what caught GRAFX's raw dry/wet weight and the one-sample time stretch; run it after any node change."""
    torch.manual_seed(seed); x = torch.randn(1, 1, N) * 0.1
    devs = ("cpu", "mps") if torch.backends.mps.is_available() else ("cpu",)
    for t in NODES:
        grads, res = {}, None
        for dev in devs:
            ch = Chain([t], N, dev); p = ch.init(seed)
            with torch.no_grad(): y0, _ = ch.render(x.to(dev), ch.scale(p, 0.0), checkpoint=False)
            res = float((y0.cpu() - x[0, 0]).pow(2).mean().sqrt() / 0.1) if dev == "cpu" else res
            out, reg = ch.render(x.to(dev), p); ((out ** 2).mean() + 0.01 * reg).backward()
            grads[dev] = torch.cat([v.grad.flatten().cpu() for v in ch.params_flat(p)])
        cos = float(torch.nn.functional.cosine_similarity(grads["cpu"], grads[devs[-1]], dim=0)) if len(devs) > 1 else float("nan")
        note = "" if t not in ("dist", "pwtanh") else "  (tanh at 0 dB drive is not an exact identity; 5-10 % expected)"
        print(f"  {t:10s} bypass residual {res:.1e}   grad cos {cos:+.4f}{note}", flush=True)

if __name__ == "__main__":
    if torch.backends.mps.is_available(): torch.mps.set_per_process_memory_fraction(0.5)
    selftest()

def z_from_ms(ms, sr=SR):
    """One-pole coefficient logit for a time constant in ms (alpha = exp(-1/(tau*sr)), z = logit(alpha))."""
    alpha = math.exp(-1.0 / (max(float(ms), 0.05) / 1000 * sr)); return torch.tensor(math.log(alpha / (1 - alpha)))
def ms_from_z(z, sr=SR):
    alpha = 1 / (1 + math.exp(-float(z))); return round(-1000 / (sr * math.log(min(alpha, 0.999999))), 2)
