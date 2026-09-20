"""Differentiable twins of the app's stock FX inserts, parameterised in the APP'S OWN UNITS (Hz, dB, ms, ratio,
0-1 mixes) so a search starts at the user's current settings and exports back as plain parameter edits.

Each twin: make(N) -> module; from_app(params) -> raw tensors; to_app(raw) -> app params; prior(raw); describe(raw);
scale(raw, a) (a=0 is the current setting, a=1 the result -- the amount knob is relative to where the user was).
Raw tensors are unconstrained: frequencies as log2(Hz), times as log(ms|s), dB as dB, ratios as log(ratio-1),
mixes as logits. The DSP is fxgraph's / GRAFX's; only the parameterisation is new. Which twins are EXACT copies of
pedalboard (the app's engine) and which are approximations is stated per node and checked by roundtrip.py."""
import math, torch
from grafx import processors as P
import fxgraph as FG
SR = 48000
sig, logit = torch.sigmoid, lambda x: torch.logit(torch.as_tensor(float(x)).clamp(1e-4, 1 - 1e-4))
def _t(x): return torch.as_tensor(float(x), dtype=torch.float32)
def _f(x): return float(x.detach().cpu()) if torch.is_tensor(x) else float(x)

class Gain(torch.nn.Module):
    def forward(self, x, gain_db): return x * torch.pow(10.0, gain_db / 20)

class Saturation(torch.nn.Module):
    """pedalboard Distortion + Gain, exactly: tanh(x * 10^(drive/20)) * 10^(out/20)."""
    def forward(self, x, drive_db, out_db): return torch.tanh(x * torch.pow(10.0, drive_db / 20)) * torch.pow(10.0, out_db / 20)

class Biquads(torch.nn.Module):
    """A fixed list of RBJ biquad types (peak / lowshelf / highshelf / lowpass / highpass), one set of (freq, gain, q)
    per band, run through GRAFX's frequency-sampled IIR. pedalboard's filters are JUCE's RBJ designs, so this is exact
    up to the FSM truncation (4000 taps)."""
    def __init__(self, kinds, N):
        super().__init__(); self.kinds = list(kinds)
        self.iir = P.core.iir.IIRFilter(order=2, backend="fsm", flashfftconv=False, fsm_max_input_len=N)
    def coeffs(self, kind, hz, gain_db, q):
        w0 = 2 * math.pi * hz.clamp(20.0, SR * 0.45) / SR; cos_w0, sin_w0 = torch.cos(w0), torch.sin(w0)
        A = torch.pow(10.0, gain_db / 40); alpha = sin_w0 / (2 * q.clamp(min=0.05))
        if kind == "eq_peak":
            b = torch.stack([1 + alpha * A, -2 * cos_w0, 1 - alpha * A]); a = torch.stack([1 + alpha / A, -2 * cos_w0, 1 - alpha / A])
        elif kind in ("eq_low_shelf", "eq_high_shelf"):
            s = 2 * torch.sqrt(A) * alpha; sgn = 1.0 if kind == "eq_low_shelf" else -1.0
            b = torch.stack([A * ((A + 1) - sgn * (A - 1) * cos_w0 + s), sgn * 2 * A * ((A - 1) - sgn * (A + 1) * cos_w0), A * ((A + 1) - sgn * (A - 1) * cos_w0 - s)])
            a = torch.stack([(A + 1) + sgn * (A - 1) * cos_w0 + s, -sgn * 2 * ((A - 1) + sgn * (A + 1) * cos_w0), (A + 1) + sgn * (A - 1) * cos_w0 - s])
        elif kind in ("lowpass", "highpass"):                      # JUCE first-order designs (pedalboard's Low/HighpassFilter): 6 dB/oct
            n = torch.tan(w0 / 2); z = torch.zeros_like(n)
            if kind == "lowpass": b = torch.stack([n / (1 + n), n / (1 + n), z]); a = torch.stack([torch.ones_like(n), (n - 1) / (1 + n), z])
            else: b = torch.stack([1 / (1 + n), -1 / (1 + n), z]); a = torch.stack([torch.ones_like(n), (n - 1) / (1 + n), z])
        else: raise ValueError(kind)
        return b / a[0], a / a[0]
    def forward(self, x, log2_hz, gain_db, q_log):
        Bs, As = [], []
        for i, kind in enumerate(self.kinds):
            b, a = self.coeffs(kind, torch.pow(2.0, log2_hz[0, i]), gain_db[0, i], torch.exp(q_log[0, i])); Bs.append(b); As.append(a)
        Bs = torch.stack(Bs)[None, None]; As = torch.stack(As)[None, None]           # (B=1, C=1, K, 3)
        return self.iir(x, Bs, As)

class FeedbackDelay(torch.nn.Module):
    """pedalboard Delay (time, feedback, mix) as a frequency-domain FIR of K echoes: H = sum_k fb^(k-1) e^{-j w k tau};
    differentiable in tau through the phase. Exact for a feedback delay up to K echoes (fb^K < 1e-3 at fb 0.7, K 24)."""
    def __init__(self, N, K=24):
        super().__init__(); self.K, self.N = K, N; self.n = int(2 ** math.ceil(math.log2(N + 2 * SR * 2)))
        self.register_buffer("f", torch.fft.rfftfreq(self.n, 1 / SR))
    def forward(self, x, log_time, fb_logit, mix_logit):
        tau = torch.exp(log_time).reshape(()); fb = (sig(fb_logit) * 0.95).reshape(()); k = torch.arange(1, self.K + 1, device=x.device, dtype=torch.float32)
        phase = torch.remainder(self.f[None, :] * (k[:, None] * tau), 1.0)                   # (K, F) turns, wrapped before the exponential: 1e6 rad is not float32-safe
        H = (torch.pow(fb, k - 1)[:, None] * torch.exp(-2j * math.pi * phase)).sum(0)         # (F,); pedalboard: first echo at full mix, then x fb per echo (measured on an impulse)
        X = torch.fft.rfft(torch.nn.functional.pad(x, (0, self.n - x.shape[-1])))
        wet = torch.fft.irfft(X * H, n=self.n)[..., :x.shape[-1]]; m = sig(mix_logit).reshape(())
        return (1 - m) * x + m * wet

# ------------------------------------------------------------------------------------------------------------
class AppNode:
    """Base: subclasses fill kinds/keys; from_app/to_app do the unit conversion."""
    exact = True
    def make(self, N): raise NotImplementedError
    def from_app(self, p): raise NotImplementedError
    def to_app(self, raw): raise NotImplementedError
    def prior(self, raw): return torch.zeros(())
    def describe(self, raw): return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.to_app(raw).items()}
    def scale(self, raw, raw0, a): return {k: raw0[k] + a * (raw[k] - raw0[k]) for k in raw}

class GainNode(AppNode):
    def make(self, N): return Gain()
    def from_app(self, p): return {"gain_db": _t(p.get("gain", 0.0))[None, None]}
    def to_app(self, raw): return {"gain": _f(raw["gain_db"][0, 0])}
    def prior(self, raw): return torch.relu(raw["gain_db"].abs() - 24).pow(2).sum() / 100

class SaturationNode(AppNode):
    def __init__(self, kind): self.kind = kind; self.d0 = 5.0 if kind == "saturator" else 12.0; self.ratio = 0.6 if kind == "saturator" else 0.35
    def make(self, N): return Saturation()
    def from_app(self, p): d = float(p.get("drive", self.d0)); return {"drive_db": _t(d)[None, None], "out_db": _t(p.get("output", -d * self.ratio))[None, None]}
    def to_app(self, raw): return {"drive": _f(raw["drive_db"][0, 0]), "output": _f(raw["out_db"][0, 0])}
    def prior(self, raw): return (torch.relu(-raw["drive_db"]).pow(2).sum() + torch.relu(raw["drive_db"] - 40).pow(2).sum() + torch.relu(raw["out_db"].abs() - 24).pow(2).sum()) / 100

class EQNode(AppNode):
    """The stock 8-band EQ: bands are dicts {type, freq, gain, q, on?}; disabled bands are kept at 0 dB and frozen."""
    def __init__(self, bands): self.bands = [dict(b) for b in bands]; self.kinds = [b.get("type", "eq_peak") for b in self.bands]
    def make(self, N): return Biquads(self.kinds, N)
    def from_app(self, p):
        bands = p.get("bands") or self.bands
        return {"log2_hz": torch.tensor([[math.log2(float(b.get("freq", 1000.0))) for b in bands]]), "gain_db": torch.tensor([[float(b.get("gain", 0.0)) if b.get("on", True) else 0.0 for b in bands]]),
                "q_log": torch.tensor([[math.log(float(b.get("q", 1.0))) for b in bands]])}
    def to_app(self, raw):
        return {"bands": [{**b, "freq": round(2.0 ** _f(raw["log2_hz"][0, i]), 1), "gain": round(_f(raw["gain_db"][0, i]), 2), "q": round(math.exp(_f(raw["q_log"][0, i])), 3)} for i, b in enumerate(self.bands)]}
    def prior(self, raw):
        return torch.relu(raw["gain_db"].abs() - 6).pow(2).sum() / 36 + torch.relu(raw["q_log"] - math.log(4.0)).pow(2).sum() + torch.relu(math.log(0.3) - raw["q_log"]).pow(2).sum() \
               + torch.relu(raw["log2_hz"] - math.log2(16000)).pow(2).sum() + torch.relu(math.log2(30) - raw["log2_hz"]).pow(2).sum()

class FilterNode(AppNode):
    """lowpass / highpass inserts: JUCE first-order designs, exact."""
    exact = True
    def __init__(self, kind): self.kind = kind
    def make(self, N): return Biquads([self.kind], N)
    def from_app(self, p): return {"log2_hz": torch.tensor([[math.log2(float(p.get("cutoff", 1200.0 if self.kind == "lowpass" else 250.0)))]]), "gain_db": torch.zeros(1, 1), "q_log": torch.full((1, 1), math.log(0.707))}
    def to_app(self, raw): return {"cutoff": round(2.0 ** _f(raw["log2_hz"][0, 0]), 1)}
    def prior(self, raw): return torch.relu(raw["log2_hz"] - math.log2(20000)).pow(2).sum() + torch.relu(math.log2(20) - raw["log2_hz"]).pow(2).sum()

class CompressorWithMakeup(torch.nn.Module):
    def __init__(self, N): super().__init__(); self.comp = FG.Compressor(N, ballistics=True); self.comp.detector = "peak"      # JUCE's compressor follows |x|, not x^2
    def forward(self, x, thr_db, log_ratio, log_knee, z_alpha, makeup_db): return self.comp(x, thr_db, log_ratio, log_knee, z_alpha) * torch.pow(10.0, makeup_db / 20)

class CompressorNode(AppNode):
    """Our feed-forward RMS compressor with ballistics + makeup. pedalboard's (JUCE) is peak-based with a hard knee --
    calibrate the mapping by roundtrip; approximate until then."""
    exact = False
    def make(self, N): return CompressorWithMakeup(N)
    def from_app(self, p):
        return {"thr_db": _t(p.get("threshold", -16.0))[None, None], "log_ratio": torch.log(_t(max(float(p.get("ratio", 4.0)) - 1, 1e-3)))[None, None], "log_knee": torch.log(_t(1.0))[None, None],
                "z_alpha": torch.stack([FG.z_from_ms(p.get("attack", 10.0)), FG.z_from_ms(p.get("release", 100.0))])[None], "makeup_db": _t(p.get("makeup", 0.0))[None, None]}
    def to_app(self, raw):
        return {"threshold": _f(raw["thr_db"][0, 0]), "ratio": 1 + math.exp(_f(raw["log_ratio"][0, 0])), "attack": FG.ms_from_z(raw["z_alpha"][0, 0]), "release": FG.ms_from_z(raw["z_alpha"][0, 1]), "makeup": _f(raw["makeup_db"][0, 0])}
    def prior(self, raw):
        thr = raw["thr_db"][0, 0]; z = raw["z_alpha"][0]
        return (torch.relu(-60 - thr) ** 2 + torch.relu(thr) ** 2) / 100 + torch.relu(raw["log_ratio"] - math.log(19.0)).pow(2).sum() + (torch.relu(3.9 - z) ** 2 + torch.relu(z - 9.9) ** 2).sum() + torch.relu(raw["makeup_db"].abs() - 24).pow(2).sum() / 100

class LimiterNode(CompressorNode):
    def from_app(self, p): return super().from_app({"threshold": p.get("threshold", -1.0), "ratio": 20.0, "attack": 1.0, "release": p.get("release", 100.0), "makeup": 0.0})
    def to_app(self, raw): a = super().to_app(raw); return {"threshold": a["threshold"], "release": a["release"]}

class ChorusNode(AppNode):
    """Our LFO delay; pedalboard/JUCE Chorus differs in its depth scaling and LFO shape -- calibrate by roundtrip."""
    exact = False
    def make(self, N): return FG.Chorus(N)
    def from_app(self, p):
        rate, depth, cd, mix = float(p.get("rate", 1.0)), float(p.get("depth", 0.25)), float(p.get("centre_delay", 7.0)), float(p.get("mix", 0.5))
        return {"z_delay_ms": logit((cd - 1) / 29)[None, None], "z_depth": logit(min(depth, 0.999))[None, None], "z_rate": logit(math.log(rate / 0.05) / math.log(10 / 0.05))[None, None], "drywet_weight": logit(mix)[None, None]}
    def to_app(self, raw):
        return {"rate": round(0.05 * math.exp(_f(sig(raw["z_rate"][0, 0])) * math.log(10 / 0.05)), 3), "depth": round(_f(sig(raw["z_depth"][0, 0])), 3), "centre_delay": round(1 + 29 * _f(sig(raw["z_delay_ms"][0, 0])), 2), "mix": round(_f(sig(raw["drywet_weight"][0, 0])), 3)}

class DelayNode(AppNode):
    def make(self, N): return FeedbackDelay(N)
    def from_app(self, p): return {"log_time": torch.log(_t(p.get("time", 0.25)))[None, None], "fb_logit": logit(float(p.get("feedback", 0.3)) / 0.95)[None, None], "mix_logit": logit(p.get("mix", 0.3))[None, None]}
    def to_app(self, raw): return {"time": round(math.exp(_f(raw["log_time"][0, 0])), 4), "feedback": round(0.95 * _f(sig(raw["fb_logit"][0, 0])), 3), "mix": round(_f(sig(raw["mix_logit"][0, 0])), 3)}
    def prior(self, raw): return torch.relu(raw["log_time"] - math.log(2.0)).pow(2).sum() + torch.relu(math.log(0.01) - raw["log_time"]).pow(2).sum()

class Freeverb(torch.nn.Module):
    """JUCE's Reverb (pedalboard.Reverb) is Freeverb: input gain 0.015 -> 8 parallel damped feedback combs (feedback =
    0.28 room + 0.7, damp = 0.4 damping) -> 4 series 'allpasses' (feedback 0.5) -> wet1/wet2 by width, + dry x 2.
    It is LTI for fixed parameters, so it is evaluated EXACTLY in the frequency domain: each comb is
    z^-D / (1 - fb (1-d)/(1 - d z^-1) z^-D), each 'allpass' (1.5 z^-D - 1)/(1 - 0.5 z^-D); the right channel uses the same
    lines 23 samples longer; the mono output is the channel mean. Differentiable in room, damping, wet, dry, width."""
    COMBS, ALLPASSES, SPREAD, gain = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617), (556, 441, 341, 225), 23, 0.015
    def __init__(self, N, sr=SR):
        super().__init__(); self.n = int(2 ** math.ceil(math.log2(N + 4 * sr))); s = sr / 44100.0
        self.combs = [round(c * s) for c in self.COMBS]; self.aps = [round(a * s) for a in self.ALLPASSES]; self.spread = round(self.SPREAD * s)
        self.register_buffer("z1", torch.exp(-2j * math.pi * torch.fft.rfftfreq(self.n, 1.0)))     # z^-1 on the FFT grid
    def wet_response(self, fb, d, offset):
        z1 = self.z1; H = 0
        for D in self.combs:
            zD = z1 ** (D + offset); H = H + zD / (1 - fb * (1 - d) / (1 - d * z1) * zD)
        for D in self.aps:
            zD = z1 ** (D + offset); H = H * (1.5 * zD - 1) / (1 - 0.5 * zD)          # JUCE: temp = x + 0.5 b[n-D]; y = b[n-D] - x
        return H
    def forward(self, x, room_logit, damp_logit, wet_logit, dry_logit, width_logit):
        room, damping, wet, dry, width = (sig(v).reshape(()) for v in (room_logit, damp_logit, wet_logit, dry_logit, width_logit))
        fb, d = room * 0.28 + 0.7, damping * 0.4; wet_g = wet * 3.0; wet1, wet2 = wet_g * (width * 0.5 + 0.5), wet_g * (1 - width) * 0.5
        HL, HR = self.wet_response(fb, d, 0), self.wet_response(fb, d, self.spread)
        Hmono = self.gain * 2 * ((wet1 + wet2) * (HL + HR) / 2) + dry * 2.0           # input = (L + R) * gain with L = R = x; mean of L = wet1 HL + wet2 HR + dry x, R = wet1 HR + wet2 HL + dry x
        X = torch.fft.rfft(torch.nn.functional.pad(x, (0, self.n - x.shape[-1])))
        return torch.fft.irfft(X * Hmono, n=self.n)[..., :x.shape[-1]]

class ReverbNode(AppNode):
    """pedalboard Reverb == Freeverb, evaluated exactly (see Freeverb)."""
    exact = True
    def make(self, N): return Freeverb(N)
    def from_app(self, p):
        return {k: logit(p.get(a, dflt))[None, None] for k, a, dflt in (("room_logit", "room_size", 0.5), ("damp_logit", "damping", 0.5), ("wet_logit", "wet", 0.3), ("dry_logit", "dry", 0.7), ("width_logit", "width", 1.0))}
    def to_app(self, raw): return {a: round(_f(sig(raw[k][0, 0])), 3) for k, a in (("room_logit", "room_size"), ("damp_logit", "damping"), ("wet_logit", "wet"), ("dry_logit", "dry"), ("width_logit", "width"))}
    def prior(self, raw): return torch.relu(sig(raw["wet_logit"]) - 0.6).pow(2).sum() * 4      # Freeverb's wet is x3 internally: past ~0.6 it swamps the dry

class GateNode(AppNode):
    """pedalboard NoiseGate is a ratio-based downward expander; ours is a soft expander with a range. Approximate."""
    exact = False
    def make(self, N): return FG.Gate(N)
    def from_app(self, p):
        ratio = float(p.get("ratio", 4.0)); return {"thr_db": _t(p.get("threshold", -50.0))[None, None], "log_width": torch.log(_t(6.0))[None, None], "z_alpha": FG.z_from_ms(p.get("release", 100.0))[None, None], "log_range": torch.log(_t(min(60.0, 6.0 * (ratio - 1) + 1e-3)))[None, None]}
    def to_app(self, raw): return {"threshold": _f(raw["thr_db"][0, 0]), "ratio": round(1 + math.exp(_f(raw["log_range"][0, 0])) / 6.0, 2), "release": FG.ms_from_z(raw["z_alpha"][0, 0]), "attack": 1.0}
    def prior(self, raw): return torch.relu(torch.exp(raw["log_range"]) - 25).pow(2).sum() / 100

class Identity(torch.nn.Module):
    def forward(self, x): return x
class FrozenNode(AppNode):
    """An insert we have no twin for (a VST): identity during the search, flagged. The exported graph keeps it as is."""
    exact = False
    def __init__(self, kind): self.kind = kind
    def make(self, N): return Identity()
    def from_app(self, p): return {}
    def to_app(self, raw): return {}

def twin_for(spec):
    kind, p = spec.get("type", ""), spec.get("params") or {}
    if kind == "eq": return EQNode(p.get("bands") or [])
    if kind == "gain": return GainNode()
    if kind in ("saturator", "distortion"): return SaturationNode(kind)
    if kind in ("lowpass", "highpass"): return FilterNode(kind)
    if kind == "compressor": return CompressorNode()
    if kind == "limiter": return LimiterNode()
    if kind == "gate": return GateNode()
    if kind == "chorus": return ChorusNode()
    if kind == "delay": return DelayNode()
    if kind == "reverb": return ReverbNode()
    return FrozenNode(kind)
