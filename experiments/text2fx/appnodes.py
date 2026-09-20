"""Differentiable twins of the app's stock FX inserts, parameterised in the APP'S OWN UNITS (Hz, dB, ms, ratio,
0-1 mixes) so a search starts at the user's current settings and exports back as plain parameter edits.

Each twin: make(N) -> module; from_app(params) -> raw tensors; to_app(raw) -> app params; prior(raw); describe(raw);
scale(raw, a) (a=0 is the current setting, a=1 the result -- the amount knob is relative to where the user was).
Raw tensors are unconstrained: frequencies as log2(Hz), times as log(ms|s), dB as dB, ratios as log(ratio-1),
mixes as logits. The DSP is fxgraph's / GRAFX's; only the parameterisation is new. Which twins are EXACT copies of
pedalboard (the app's engine) and which are approximations is stated per node and checked by roundtrip.py."""
import math, torch, numpy as np
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

def host_delay_samples(time_s, sr=SR): return math.floor(float(np.float32(time_s)) * sr)

class FeedbackDelay(torch.nn.Module):
    """pedalboard Delay (time, feedback, mix) as a frequency-domain FIR of K echoes: H = sum_k fb^(k-1) e^{-j w k tau};
    differentiable in tau through the phase (tau quantised to the host's whole-sample truncation, straight-through).
    Exact for a feedback delay up to K echoes (fb^K < 1e-3 at fb 0.7, K 24)."""
    def __init__(self, N, K=24):
        super().__init__(); self.K, self.N = K, N; self.n = int(2 ** math.ceil(math.log2(N + 2 * SR * 2)))
        self.register_buffer("f", torch.fft.rfftfreq(self.n, 1 / SR))
    def forward(self, x, time, fb_logit, mix_logit):
        tau = time.reshape(()).clamp(min=1 / SR); fb = (sig(fb_logit) * 0.95).reshape(()); k = torch.arange(1, self.K + 1, device=x.device, dtype=torch.float32)
        # pedalboard truncates float32(time) * sr to whole samples, no interpolation (0.35 s -> 16799, measured on an impulse).
        # Quantise the same way, straight-through so the gradient still flows through time; to_app exports the sample centre.
        tau = tau + (host_delay_samples(float(tau)) / SR - tau).detach()
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

def z_juce(ms, sr=SR):
    """One-pole logit for a JUCE BallisticsFilter time: JUCE uses alpha = exp(-2 pi * 1000 / (sr * ms)), i.e. a time constant
    of ms / 2 pi (measured on pedalboard gate + compressor level steps, 2026-09-20: fitted tau ratio 6.36 ~ 2 pi)."""
    return FG.z_from_ms(max(float(ms), 0.05) / (2 * math.pi), sr)
def ms_juce(z, sr=SR): return round(FG.ms_from_z(z, sr) * 2 * math.pi, 2)

class CompressorNode(AppNode):
    """pedalboard/JUCE compressor: peak detector (ballistics on |x|, JUCE time convention, state from 0), hard knee,
    then makeup. Round-trips at ~100 dB SNR (2026-09-20)."""
    exact = True
    def make(self, N): return CompressorWithMakeup(N)
    def from_app(self, p):
        return {"thr_db": _t(p.get("threshold", -16.0))[None, None], "log_ratio": torch.log(_t(max(float(p.get("ratio", 4.0)) - 1, 1e-3)))[None, None], "log_knee": torch.log(_t(0.01))[None, None],
                "z_alpha": torch.stack([z_juce(p.get("attack", 10.0)), z_juce(p.get("release", 100.0))])[None], "makeup_db": _t(p.get("makeup", 0.0))[None, None]}
    def to_app(self, raw):
        return {"threshold": _f(raw["thr_db"][0, 0]), "ratio": 1 + math.exp(_f(raw["log_ratio"][0, 0])), "attack": ms_juce(raw["z_alpha"][0, 0]), "release": ms_juce(raw["z_alpha"][0, 1]), "makeup": _f(raw["makeup_db"][0, 0])}
    def prior(self, raw):
        thr = raw["thr_db"][0, 0]; z = raw["z_alpha"][0]
        return (torch.relu(-60 - thr) ** 2 + torch.relu(thr) ** 2) / 100 + torch.relu(raw["log_ratio"] - math.log(19.0)).pow(2).sum() + (torch.relu(3.9 - z) ** 2 + torch.relu(z - 9.9) ** 2).sum() + torch.relu(raw["makeup_db"].abs() - 24).pow(2).sum() / 100

class LimiterNode(CompressorNode):
    def from_app(self, p): return super().from_app({"threshold": p.get("threshold", -1.0), "ratio": 20.0, "attack": 1.0, "release": p.get("release", 100.0), "makeup": 0.0})
    def to_app(self, raw): a = super().to_app(raw); return {"threshold": a["threshold"], "release": a["release"]}

class DelayNode(AppNode):
    def make(self, N): return FeedbackDelay(N)
    # time is a LINEAR raw parameter: the host truncates float32(time) * sr, and only the very same float32 value lands on the
    # same sample at round numbers (0.35 s is 16799.9997 samples; exp(log(0.35)) in float32 came out above 16800).
    def from_app(self, p): return {"time": _t(p.get("time", 0.25))[None, None], "fb_logit": logit(float(p.get("feedback", 0.3)) / 0.95)[None, None], "mix_logit": logit(p.get("mix", 0.3))[None, None]}
    def to_app(self, raw): return {"time": round((host_delay_samples(_f(raw["time"][0, 0])) + 0.5) / SR, 7), "feedback": round(0.95 * _f(sig(raw["fb_logit"][0, 0])), 3), "mix": round(_f(sig(raw["mix_logit"][0, 0])), 3)}   # mid-sample: truncates to the same sample the twin used
    def prior(self, raw): return (torch.relu(raw["time"] - 2.0).pow(2).sum() + torch.relu(0.01 - raw["time"]).pow(2).sum()) * 100

class Freeverb(torch.nn.Module):
    """JUCE's Reverb (pedalboard.Reverb) is Freeverb: input gain 0.015 -> 8 parallel damped feedback combs (feedback =
    0.28 room + 0.7, damp = 0.4 damping) -> 4 series 'allpasses' (feedback 0.5) -> wet1/wet2 by width, + dry x 2.
    It is LTI for fixed parameters, so it is evaluated EXACTLY in the frequency domain: each comb is
    z^-D / (1 - fb (1-d)/(1 - d z^-1) z^-D), each 'allpass' (1.5 z^-D - 1)/(1 - 0.5 z^-D); the right channel uses the same
    lines 23 samples longer; the mono output is the channel mean. Differentiable in room, damping, wet, dry, width."""
    COMBS, ALLPASSES, SPREAD, gain = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617), (556, 441, 341, 225), 23, 0.015
    def __init__(self, N, sr=SR):
        super().__init__(); self.n = int(2 ** math.ceil(math.log2(N + 4 * sr))); isr = int(sr)
        # JUCE sizes every line with integer division of (sr * tuning) / 44100 -- the right channel's +23 is added BEFORE the
        # division, not after. round() here (and spread added after) put half the lines a sample out: 13 dB round-trip SNR.
        self.lines = [([isr * c // 44100 for c in self.COMBS], [isr * a // 44100 for a in self.ALLPASSES]),
                      ([isr * (c + self.SPREAD) // 44100 for c in self.COMBS], [isr * (a + self.SPREAD) // 44100 for a in self.ALLPASSES])]
        self.register_buffer("z1", torch.exp(-2j * math.pi * torch.fft.rfftfreq(self.n, 1.0)))     # z^-1 on the FFT grid
    def wet_response(self, fb, d, channel):
        z1 = self.z1; H = 0; combs, aps = self.lines[channel]
        for D in combs:
            zD = z1 ** D; H = H + zD / (1 - fb * (1 - d) / (1 - d * z1) * zD)
        for D in aps:
            zD = z1 ** D; H = H * (1.5 * zD - 1) / (1 - 0.5 * zD)                     # JUCE: temp = x + 0.5 b[n-D]; y = b[n-D] - x
        return H
    def forward(self, x, room_logit, damp_logit, wet_logit, dry_logit, width_logit):
        room, damping, wet, dry, width = (sig(v).reshape(()) for v in (room_logit, damp_logit, wet_logit, dry_logit, width_logit))
        fb, d = room * 0.28 + 0.7, damping * 0.4; wet_g = wet * 3.0; wet1, wet2 = wet_g * (width * 0.5 + 0.5), wet_g * (1 - width) * 0.5
        HL, HR = self.wet_response(fb, d, 0), self.wet_response(fb, d, 1)
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


def juce_lfo_rate(rate_hz, sr=SR):
    """The rate a JUCE dsp::Oscillator actually runs at. Its phase is a float32 accumulator wrapped at 2 pi; within each
    binade of the phase the increment rounds to a whole number of ulps, so the effective period is a closed-form sum.
    Off by up to ~0.1 %% (-0.12 %% at 0.3 Hz, +0.04 %% at 1.5 Hz): inaudible, but it is what the round trip hears."""
    inc = float(np.float32(rate_hz * 2 * math.pi / sr)); steps = 2.0 ** -20 / inc
    for k in range(-20, 3):
        a, b = 2.0 ** k, min(2.0 ** (k + 1), 2 * math.pi)
        if a >= 2 * math.pi: break
        ulp = 2.0 ** (k - 23); steps += (b - a) / (max(round(inc / ulp), 1) * ulp)
    return sr / steps

class AppChorus(torch.nn.Module):
    """pedalboard/JUCE Chorus, measured on a click train: delay(t) = clamp(centre - 10 ms * depth * sin(2 pi rate t), 1 ms),
    identical on both channels, linear dry/wet mix. Feedback (rare; default 0 in the app) has no differentiable form
    here and is ignored -- reported by the node."""
    def __init__(self, N, sr=SR): super().__init__(); self.sr = sr; self.register_buffer("t", torch.arange(N).float() / sr)
    def forward(self, x, log_rate, depth_logit, log_centre, mix_logit):
        rate, depth, centre, mix = torch.exp(log_rate).reshape(()), sig(depth_logit).reshape(()), torch.exp(log_centre).reshape(()), sig(mix_logit).reshape(())
        rate = rate * (juce_lfo_rate(float(rate), self.sr) / float(rate))          # host's float32 LFO drift, detached
        d_ms = (centre - 10.0 * depth * torch.sin(2 * math.pi * rate * self.t)).clamp(min=1.0)
        pos = (torch.arange(x.shape[-1], device=x.device).float() - d_ms * self.sr / 1000).clamp(min=0); i0 = pos.floor().long(); frac = pos - i0
        xb = x[:, 0]; wet = torch.gather(xb, 1, i0[None].expand(xb.shape[0], -1)) * (1 - frac) + torch.gather(xb, 1, (i0 + 1).clamp(max=x.shape[-1] - 1)[None].expand(xb.shape[0], -1)) * frac
        return ((1 - mix) * xb + mix * wet)[:, None, :]

class ChorusNode(AppNode):
    """Exact to the measured JUCE law except feedback (ignored)."""
    exact = True
    def make(self, N): return AppChorus(N)
    def from_app(self, p):
        self.feedback = float(p.get("feedback", 0.0))
        return {"log_rate": torch.log(_t(p.get("rate", 1.0)))[None, None], "depth_logit": logit(min(float(p.get("depth", 0.25)), 0.999))[None, None], "log_centre": torch.log(_t(p.get("centre_delay", 7.0)))[None, None], "mix_logit": logit(p.get("mix", 0.5))[None, None]}
    def to_app(self, raw): return {"rate": round(math.exp(_f(raw["log_rate"][0, 0])), 3), "depth": round(_f(sig(raw["depth_logit"][0, 0])), 3), "centre_delay": round(math.exp(_f(raw["log_centre"][0, 0])), 2), "mix": round(_f(sig(raw["mix_logit"][0, 0])), 3), "feedback": getattr(self, "feedback", 0.0)}
    def prior(self, raw): return torch.relu(raw["log_rate"] - math.log(8.0)).pow(2).sum() + torch.relu(math.log(0.1) - raw["log_rate"]).pow(2).sum() + torch.relu(raw["log_centre"] - math.log(30.0)).pow(2).sum() + torch.relu(math.log(1.0) - raw["log_centre"]).pow(2).sum()

class AppGate(torch.nn.Module):
    """pedalboard/JUCE NoiseGate: a 1 ms RMS level, attack/release ballistics on that level (peak type: attack when it
    rises), then static gain (ratio - 1) * (level_dB - thr_dB) below the threshold -- a downward expander. Measured on tone
    bursts 2026-09-20 (the rms/peak 3 dB is a sine's, so JUCE compares rms to the threshold directly). torchcomp's core
    smooths a gain and its 'attack' coefficient applies to a falling input, so the roles are swapped here."""
    def __init__(self, N, sr=SR):
        super().__init__(); from grafx.processors.core.convolution import FIRConvolution
        self.sr = sr; self.register_buffer("arange", torch.arange(4096)[None, :].float()); self.conv = FIRConvolution(mode="causal", flashfftconv=False, max_input_len=N)
    def forward(self, x, thr_db, log_ratio, z_attack, z_release):
        alpha = math.exp(-2 * math.pi / (0.001 * self.sr)); h = torch.exp(self.arange * math.log(alpha)); h = h / h.sum()   # JUCE's 1 ms RMS pre-filter
        rms = torch.sqrt(torch.relu(self.conv(x.square().mean(-2), h)) + 1e-12)
        ts = (1 - torch.sigmoid(torch.cat([z_attack, z_release], -1))).cpu(); rc = rms.cpu()
        env = FG._Ballistics.apply(rc, torch.zeros(rc.shape[0]), ts[..., 1], ts[..., 0])[0].to(x.device)   # (falling=release, rising=attack)
        lvl = 20 * torch.log10(env + 1e-6); ratio = 1 + torch.exp(log_ratio)
        gain_db = (ratio - 1) * torch.clamp(lvl - (thr_db - 3.0), max=0.0)
        return x * torch.pow(10.0, gain_db / 20)[:, None, :]

class GateNode(AppNode):
    exact = True
    def make(self, N): return AppGate(N)
    def from_app(self, p):
        return {"thr_db": _t(p.get("threshold", -50.0))[None, None], "log_ratio": torch.log(_t(max(float(p.get("ratio", 4.0)) - 1, 1e-3)))[None, None], "z_attack": z_juce(p.get("attack", 1.0))[None, None], "z_release": z_juce(p.get("release", 100.0))[None, None]}
    def to_app(self, raw): return {"threshold": round(_f(raw["thr_db"][0, 0]), 2), "ratio": round(1 + math.exp(_f(raw["log_ratio"][0, 0])), 2), "attack": ms_juce(raw["z_attack"][0, 0]), "release": ms_juce(raw["z_release"][0, 0])}
    def prior(self, raw): return (torch.relu(-80 - raw["thr_db"]) ** 2 + torch.relu(raw["thr_db"]) ** 2).sum() / 100 + torch.relu(raw["log_ratio"] - math.log(19.0)).pow(2).sum()

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
