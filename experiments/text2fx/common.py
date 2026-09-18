"""Shared harness for the text-guided FX experiments (Text2FX, arXiv 2409.18847).

Differentiable CLAP scoring: a torchaudio mel front end that reproduces the HF
ClapProcessor features exactly (embedding cosine 1.0000), so gradients reach
the waveform. Plus K-weighted loudness matching, a multi-resolution STFT
distance, and the dasp_pytorch ParametricEQ the paper used.

Nothing here is product code; it lives in the repo only because the scratchpad
kept getting wiped between sessions.
"""
from __future__ import annotations

import os, sys, types, warnings
import numpy as np
import soundfile as sf
import math, torch, torchaudio

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
SR = 48000
CLAP = "laion/clap-htsat-unfused"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ---- test audio -------------------------------------------------------------
def kalimba(path=os.path.join(HERE, "G_kalimba_gm.wav")) -> np.ndarray:
    """10 s GM kalimba figure at -3 dBFS, rendered once and cached beside us."""
    if not os.path.exists(path):
        from fantasia_core.engine.midi_render import MidiRenderer
        N = lambda p, s, d, v=100: types.SimpleNamespace(pitch=p, start=s, duration=d, velocity=v)
        pat = [67, 70, 74, 75, 74, 70, 67, 74]; notes = []; t = 0.0
        for r in range(10):
            for i, p in enumerate(pat):
                notes.append(N(p + (12 if r % 3 == 2 else 0), t, 0.45, 88 + (10 if i == 0 else 0)))
                t += 0.125
        clip = types.SimpleNamespace(notes=notes, duration=10.0, content_type="midi", id="c1")
        r = MidiRenderer(os.path.join(ROOT, "assets/soundfonts/GeneralUser-GS-v1.471.sf2"), SR)
        sf.write(path, r.render(clip, 108), SR)         # GM 108 = Kalimba
    y, _ = sf.read(path, dtype="float32")
    if y.ndim > 1: y = y.mean(axis=1)
    return (y / np.abs(y).max() * 0.7).astype(np.float32)

# ---- MPS workaround ---------------------------------------------------------
# torch's `reflect` pad has a broken backward on MPS for 1-D signals longer than
# ~65k samples (verified: cos(grad_cpu, grad_mps) ~0.2 at 100k+, 1.0 at 48k).
# torch.stft(center=True) and torchaudio's MelSpectrogram both reflect-pad, so
# every STFT on 10 s of audio trained on wrong gradients. Reflect via flip+cat
# (plain index ops) and run the transforms with center=False: bit-exact forward.
def reflect_pad(x, p):
    return torch.cat([x[..., 1:p + 1].flip(-1), x, x[..., -p - 1:-1].flip(-1)], dim=-1)

# ---- CLAP, differentiable ---------------------------------------------------
_model = _proc = _mel = _todb = None
def _load():
    global _model, _proc, _mel, _todb
    if _model is None:
        from transformers import ClapModel, ClapProcessor
        _model = ClapModel.from_pretrained(CLAP).eval().to(DEVICE)
        _proc = ClapProcessor.from_pretrained(CLAP)
        for p in _model.parameters(): p.requires_grad_(False)
        fe = _proc.feature_extractor
        _mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=fe.fft_window_size, hop_length=fe.hop_length,
            win_length=fe.fft_window_size, n_mels=fe.feature_size, f_min=fe.frequency_min,
            f_max=fe.frequency_max, power=2.0, center=False,       # padded by hand, see reflect_pad
            norm="slaney", mel_scale="slaney").to(DEVICE)
        _todb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=None).to(DEVICE)
    return _model, _proc

_un = lambda o: o.pooler_output if hasattr(o, "pooler_output") else o
_nrm = lambda e: e / e.norm(dim=-1, keepdim=True)

def embed(w: torch.Tensor) -> torch.Tensor:
    """(samples,) or (B, samples) waveform -> (B, 512) unit embedding, with gradient."""
    m, p = _load(); n = p.feature_extractor.nb_max_samples
    w = w.to(DEVICE)
    if w.ndim == 1: w = w[None, :]
    w = torch.nn.functional.pad(w, (0, max(0, n - w.shape[-1])))[:, :n]
    w = reflect_pad(w, p.feature_extractor.fft_window_size // 2)
    f = _todb(_mel(w)).transpose(1, 2)[:, None, :1001, :]          # (B,1,frames,mels)
    lon = torch.zeros(w.shape[0], 1, dtype=torch.bool, device=DEVICE)
    return _nrm(_un(m.get_audio_features(input_features=f, is_longer=lon)))

def text_emb(s: str) -> torch.Tensor:
    m, p = _load()
    with torch.no_grad():
        t = p(text=[s], return_tensors="pt", padding=True).to(DEVICE)
        return _nrm(_un(m.get_text_features(**t))).detach()

# ---- loudness / distance ----------------------------------------------------
rms = lambda w: torch.sqrt((w ** 2).mean(-1, keepdim=True) + 1e-12)
_KB1 = torch.tensor([1.53512485958697, -2.69169618940638, 1.19839281085285])   # BS.1770 @48k
_KA1 = torch.tensor([1.0, -1.69065929318241, 0.73248077421585])
_KB2 = torch.tensor([1.0, -2.0, 1.0])
_KA2 = torch.tensor([1.0, -1.99004745483398, 0.99007225036621])
def _biquad_H(b, a, n, dev):
    z = torch.exp(-1j * torch.linspace(0, np.pi, n // 2 + 1, device=dev))
    return (b[0] + b[1] * z + b[2] * z ** 2) / (a[0] + a[1] * z + a[2] * z ** 2)
def k_weight(w):
    """BS.1770 K-weighting applied in the frequency domain (MPS-friendly)."""
    n, d = w.shape[-1], w.device
    H = _biquad_H(_KB1.to(d), _KA1.to(d), n, d) * _biquad_H(_KB2.to(d), _KA2.to(d), n, d)
    return torch.fft.irfft(torch.fft.rfft(w) * H, n=n)
def loudness(w): return torch.sqrt((k_weight(w) ** 2).mean(-1, keepdim=True) + 1e-12)
def level_match(w, ref):
    """Match K-weighted loudness to ref; the floor caps the boost at ~60 dB so
    silence stays silent instead of being amplified into a phantom score."""
    lr = loudness(ref)
    return w * (lr / (loudness(w) + 1e-3 * lr))

def mrstft(a, b, ffts=(128, 512, 1024, 2048)):
    """Spectral convergence + log-magnitude L1, averaged over resolutions."""
    tot = 0.0
    for n in ffts:
        win = torch.hann_window(n, device=a.device)
        A = torch.stft(reflect_pad(a, n // 2), n, n // 4, window=win, center=False, return_complex=True).abs()
        B = torch.stft(reflect_pad(b, n // 2), n, n // 4, window=win, center=False, return_complex=True).abs()
        sc = (A - B).flatten(-2).norm(dim=-1) / (A.flatten(-2).norm(dim=-1) + 1e-8)
        lm = (torch.log(A + 1e-5) - torch.log(B + 1e-5)).abs().flatten(-2).mean(-1)
        tot = tot + sc + lm
    return tot / len(ffts)                                   # (B,) or scalar

def score(w, target: torch.Tensor, src: torch.Tensor) -> float:
    """The neutral yardstick: plain cosine on loudness-matched audio."""
    w, src, target = w.to(DEVICE), src.to(DEVICE), target.to(DEVICE)
    with torch.no_grad():
        return float((embed(level_match(w, src)) @ target.T).squeeze())

# ---- the paper's EQ ---------------------------------------------------------
def dasp_eq():
    from dasp_pytorch import ParametricEQ
    return ParametricEQ(sample_rate=SR)

def eq_apply(eq, raw, w):
    """raw: unbounded (NP,) tensor -> sigmoid -> (0,1) -> dasp ranges."""
    return eq.process_normalized(w[None, None, :], torch.sigmoid(raw)[None, :])[0, 0]

def eq_denorm(eq, p01):
    names = list(eq.param_ranges)
    lo = torch.tensor([eq.param_ranges[n][0] for n in names], device=p01.device)
    hi = torch.tensor([eq.param_ranges[n][1] for n in names], device=p01.device)
    return lo + p01 * (hi - lo)

def hand_shelf(y: np.ndarray, hz=2000, db=-12) -> torch.Tensor:
    from pedalboard import Pedalboard, HighShelfFilter
    return torch.tensor(Pedalboard([HighShelfFilter(cutoff_frequency_hz=hz, gain_db=db)])(y.copy(), SR))

# ---- reverb that fits in memory ---------------------------------------------
def fast_reverb(sample_rate=SR):
    """dasp NoiseShapedReverb, but the 65,536-tap IR is applied by FFT convolution.

    dasp uses a direct time-domain conv1d for the IR, whose backward on a 10 s
    signal needs ~100 GB — it OOM-killed this machine twice. Same maths, same
    parameters, same IR construction; only the convolution changes.
    """
    from functools import partial
    import dasp_pytorch
    from dasp_pytorch.modules import NoiseShapedReverb
    import dasp_pytorch.functional as DF

    def nsr_fft(x, sample_rate, *, mix, num_samples=65536, num_bandpass_taps=1023, **bands):
        bs, chs, seq_len = x.size()
        if chs == 1: x = x.repeat(1, 2, 1)
        gains = torch.stack([bands[f"band{i}_gain"] for i in range(12)], 1).view(bs, 1, 12, 1)
        decays = torch.stack([bands[f"band{i}_decay"] for i in range(12)], 1).view(bs, 1, 12, 1)
        mix = mix.view(bs, 1, 1)
        with torch.no_grad():        # the coloured noise depends on no parameter: no backward needed
            filters = dasp_pytorch.signal.octave_band_filterbank(num_bandpass_taps, sample_rate).type_as(x)
            wn = torch.randn(bs * 2, 12, num_samples + num_bandpass_taps - 1).type_as(x)
            wn = torch.nn.functional.conv1d(wn, filters, groups=12).view(bs, 2, 12, num_samples)
        t = torch.linspace(0, 1, steps=num_samples).type_as(x).view(1, 1, 1, -1)
        ir = (wn * torch.exp(-(decays * 10.0 + 1.0) * t) * gains).mean(2)        # (bs, 2, num_samples)
        n = 1 << (seq_len + num_samples - 1).bit_length()
        y = torch.fft.irfft(torch.fft.rfft(x, n) * torch.fft.rfft(ir, n), n)[..., :seq_len]
        return (1 - mix) * x + mix * y

    rv = NoiseShapedReverb(sample_rate)
    rv.process_fn = nsr_fft
    return rv

# ---- losses for real-audio targets --------------------------------------------
def env_db(w, win=240):
    """Frame RMS in dB with a -60 dB floor. (..., T) -> (..., frames)."""
    fr = w[..., : w.shape[-1] // win * win].unfold(-1, win, win)
    return 20 * torch.log10(torch.sqrt((fr ** 2).mean(-1) + 1e-12) + 1e-3)

def env_loss(a, b, win=240):
    """L1 between RMS envelopes in dB — identifies ADSR shape directly, where the
    log-STFT term rewards being quiet wherever the fine structure can't be matched."""
    return (env_db(a, win) - env_db(b, win)).abs().mean() / 20.0

def mrstft_lin(a, b, ffts=(128, 512, 1024, 2048), log_weight=0.1):
    """MR-STFT with the log-magnitude term down-weighted so loud parts dominate."""
    tot = 0.0
    for n in ffts:
        win = torch.hann_window(n, device=a.device)
        A = torch.stft(reflect_pad(a, n // 2), n, n // 4, window=win, center=False, return_complex=True).abs()
        B = torch.stft(reflect_pad(b, n // 2), n, n // 4, window=win, center=False, return_complex=True).abs()
        sc = (A - B).flatten(-2).norm(dim=-1) / (A.flatten(-2).norm(dim=-1) + 1e-8)
        lm = (torch.log(A + 1e-5) - torch.log(B + 1e-5)).abs().flatten(-2).mean(-1)
        tot = tot + sc + log_weight * lm
    return tot / len(ffts)

def beat_depth(w, notes, win=240, hold=(0.4, 1.4)):
    """Ripple of the dB envelope INSIDE each note's sustained span (after decay, before
    release) — the only modulation there is unison beating. Note on/off edges are 60 dB
    steps and swamp any whole-signal measure. Returns mean ripple RMS in dB (scalar)."""
    e = env_db(w, win); r = SR / win; out = []
    for _, start, dur, _ in notes:
        h0, h1 = min(hold[0], 0.3 * dur), min(hold[1], dur - 0.03)          # windows relative to short notes
        i0, i1 = int((start + h0) * r), int((start + h1) * r)
        if i1 - i0 < 8: continue
        seg = e[..., i0:i1]; out.append(torch.sqrt(((seg - seg.mean(-1, keepdim=True)) ** 2).mean(-1) + 1e-6))
    if not out: return torch.zeros((), device=w.device)                      # no note long enough to hold
    return torch.stack(out).mean()

def mod_depth_loss(a, b, notes):
    return (beat_depth(a, notes) - beat_depth(b, notes)).abs() / 3.0

def harmonic_profile(w, notes, K=64, hold=(0.3, 1.3), max_detune_semis=0.6):
    """Per-note log-power at each harmonic of the known pitch, summed over a window
    wide enough to include detuned unison sidebands. Linear in the wavetable rows, so
    as a function of frame position it has no ridges. Returns (notes, K) in dB."""
    out = []
    for pitch, start, dur, _ in notes:
        f0 = 440.0 * 2 ** ((pitch - 69) / 12)
        h0, h1 = min(hold[0], 0.3 * dur), min(hold[1], dur - 0.02)          # windows relative to short notes
        i0, i1 = int((start + h0) * SR), int((start + h1) * SR)
        if i1 - i0 < SR // 40: continue
        seg = w[..., i0:i1]; n = seg.shape[-1]
        X = torch.fft.rfft(seg * torch.hann_window(n, device=w.device)).abs() ** 2
        binhz = SR / n; spread = 2 ** (max_detune_semis / 12) - 1
        rows = []
        for k in range(1, K + 1):
            fc = k * f0
            if fc >= SR / 2: rows.append(torch.full_like(X[..., 0], -80.0)); continue
            half = min(fc * spread + 2 * binhz, f0 * 0.45)              # sidebands, capped at half the spacing
            lo, hi = int((fc - half) / binhz), int((fc + half) / binhz) + 1
            rows.append(10 * torch.log10(X[..., lo:hi].sum(-1) + 1e-9))
        out.append(torch.stack(rows, -1))
    if not out: return torch.full((1, K), -80.0, device=w.device)
    return torch.stack(out)                                            # (notes, K)

def harmonic_loss(a, b, notes, K=64):
    ha, hb = harmonic_profile(a, notes, K), harmonic_profile(b, notes, K)
    ha = ha - ha.max(-1, keepdim=True).values; hb = hb - hb.max(-1, keepdim=True).values   # shape, not level
    ha, hb = ha.clamp(min=-60.0), hb.clamp(min=-60.0)          # -60 dB floor: don't compare noise floors
    return (ha - hb).abs().mean() / 10.0


_BANDS = {}
def band_energy_loss(a, b, n_fft=2048, hop=480, n_bands=32, f_lo=60.0, smooth=8, sr=48000):
    """Log band energies over time (32 log-spaced bands of at least 4 bins, 43 Hz frames, power averaged over 8
    frames = 80 ms), L1. Noise-realisation-invariant: two independent white noises of the same level agree here
    to ~0.03 (log10) where the fine STFT terms see a Rayleigh-fluctuation floor of 0.73 that barely moves with
    level -- which made the twin prefer *less* noise than the target and compensate with cutoff/level.
    Complements mrstft_lin; does not replace it (it has no phase/onset resolution)."""
    key = (n_fft, n_bands, str(a.device))
    if key not in _BANDS:
        f = torch.fft.rfftfreq(n_fft, 1 / sr); edges = torch.logspace(math.log10(f_lo), math.log10(sr / 2), n_bands + 1)
        rows, lo = [], 0
        for i in range(n_bands):
            hi_idx = int((f < edges[i + 1]).sum()); hi_idx = max(hi_idx, lo + 4)          # at least 4 bins per band
            if hi_idx > f.numel(): break
            r = torch.zeros(f.numel()); r[lo:hi_idx] = 1.0 / (hi_idx - lo); rows.append(r); lo = hi_idx
        _BANDS[key] = torch.stack(rows).to(a.device)
    M = _BANDS[key]; win = torch.hann_window(n_fft, device=a.device)
    def P(x):
        S = M @ torch.stft(reflect_pad(x, n_fft // 2), n_fft, hop, window=win, center=False, return_complex=True).abs() ** 2
        return torch.nn.functional.avg_pool1d(S[None], smooth, stride=smooth // 2)[0]
    return (torch.log10(P(a) + 1e-9) - torch.log10(P(b) + 1e-9)).abs().mean()
