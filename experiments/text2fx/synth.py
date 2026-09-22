"""Differentiable wavetable oscillator + ADSR, shaped to map 1:1 onto Vital.

Oscillator: harmonic-domain wavetable. A table is F frames of K harmonic
amplitudes; playback of a continuous frame position p is a linear blend of
adjacent frames, summed as sin(k*phase). Harmonics above Nyquist are masked,
so it is band-limited per note for free — equivalent to Vital playing a
static frame (its phase/spectral-morph modes are not modelled).

Unison: V voices (config, not optimised — Vital's is a stepped switch), detune
spread in semitones, blend between centre voice and the detuned ones.

Envelope: Vital's ADSR with time = 32 * raw**4 (measured, see vital.json).
Attack linear; decay and release exponential toward sustain / zero.

All optimisable parameters are unbounded raw tensors mapped through sigmoid
into Vital's own 0-1 raw space, so `vital_params()` is a direct read-out.
"""
from __future__ import annotations
import math, os, sys
import numpy as np, torch, torch.nn as nn, torch.utils.checkpoint
TAIL_MAX = 2.0        # longest release tail rendered after note-off (memory bound; longer releases fade out here)


# ---- tables -------------------------------------------------------------------
def basic_shapes(K=64):
    """4 key frames of harmonic amplitudes: sine -> triangle -> saw -> square."""
    k = torch.arange(1, K + 1, dtype=torch.float32)
    sine = torch.zeros(K); sine[0] = 1.0
    tri = torch.where(k % 2 == 1, (8 / math.pi ** 2) / k ** 2 * torch.where(((k - 1) // 2) % 2 == 0, 1.0, -1.0), 0.0)
    saw = (2 / math.pi) / k * torch.where(k % 2 == 1, 1.0, -1.0)
    sqr = torch.where(k % 2 == 1, (4 / math.pi) / k, 0.0)
    return torch.stack([sine, tri, saw, sqr])          # (F=4, K), signed amplitudes

def bounds_from_notes(notes, release_cap=4.0):
    """Envelope-time ceilings implied by the clip: attack no longer than the shortest
    note, decay no longer than two of the longest, release capped."""
    durs = [d for _, _, d, _ in notes]
    return dict(attack=min(durs), decay=2.0 * max(durs), release=release_cap)

def sustained_notes():
    """Five 1.5 s notes with 0.5 s gaps: release tails and unison beating are observable
    here, where the kalimba's 8 notes/s bury both under the note rhythm."""
    return [(p, 2.0 * i, 1.5, 96) for i, p in enumerate((62, 65, 69, 72, 69))]

def kalimba_notes():
    pat = [67, 70, 74, 75, 74, 70, 67, 74]; notes = []; t = 0.0
    for r in range(10):
        for i, p in enumerate(pat):
            notes.append((p + (12 if r % 3 == 2 else 0), t, 0.45, 88 + (10 if i == 0 else 0))); t += 0.125
    return notes

# ---- the synth ------------------------------------------------------------------
class WavetableSynth(nn.Module):
    RAW = ("frame", "detune", "blend", "level", "attack", "decay", "sustain", "release")
    FILTER_RAW = ("cutoff", "resonance", "fenv_amount", "fattack", "fdecay", "fsustain", "frelease")
    OPTIONAL_RAW = ("apow", "dpow", "rpow", "fapow", "fdpow", "frpow", "keytrack", "frame_spread", "fblend", "fdrive", "noise")   # present only if given
    # Vital's sample oscillator, measured 2026-09-17 on the plugin: the default sample is white noise (band energies
    # within 0.4 dB of white), it takes the voice's amplitude envelope like the oscillators, its output rms is
    # 0.207 * sample_level_raw^2 (sustain 1, velocity 100, filter off), keytrack barely tilts it (skip), and
    # destination FILTER 1 sends it through the filter with the oscillators. The twin's `noise` raw is the linear
    # amplitude fraction a (gradient-friendly); Vital's sample_level raw = sqrt(a). NOISE_GAIN calibrated below.
    NOISE_GAIN = 0.28                                  # twin unit-variance noise -> Vital filter-input amplitude (measured, see below)
    # Level chain, measured: osc level is applied BEFORE the filter (display level 0.5 -> 0.224 peak for a
    # sine at the filter input), the filter input always passes tanh(2 g x)/(2 sqrt g) with g = 10^(drive_raw)
    # (drive display = 20 raw dB; THD -33 dB at level 0.5 even at 0 dB drive), then the amp env. PRE_GAIN
    # puts the twin's unit waveform at Vital's filter-input amplitude; OUT_GAIN keeps the end-to-end
    # calibration (Vital level = torch level / 0.694) unchanged.
    PRE_GAIN, OUT_GAIN = 0.52, 0.694 / 0.52 / 1.334     # /1.334: -2.5 dB measured end-to-end vs Vital (filter off, 1 and 3 voices, two levels)
    # Vital filter 1, Analog 12 dB LP, measured at 0% keytrack: f_c = 261.6 * 2^((128 raw - 52) / 12) Hz
    # ("0 semitones" is middle C; keytrack raw 0.5 = 0%, raw 0.0 = -100%). A modulation amount of
    # 1.0 adds the full 128 semitones; env 2 applies LINEARLY (amp env is squared). Resonance raw ->
    # (passband dB, Q) tabulated from the resonant-peak sweep at cutoff raw 0.55.
    # Re-measured 2026-09-13 at fc 250-2000 Hz: cutoff-independent above ~250 Hz (rows agree to 0.1 dB).
    RES_TAB = torch.tensor([[0.0, -0.9, 0.6], [0.2, -3.1, 0.75], [0.4, -5.1, 1.4], [0.5, -6.0, 1.65], [0.6, -6.8, 2.1],
                            [0.7, -7.6, 2.95], [0.8, -8.4, 5.4], [0.9, -9.2, 25.0], [1.0, -10.0, 60.0]])

    def __init__(self, table, sr=48000, voices=1, detune_range_semitones=2.0, seed=0, bounds=None,
                 velocity_track=0.0, interpolation=1, bipolar_fenv=False):
        """bounds: dict(attack=, decay=, release=) max seconds, from `bounds_from_notes`.
        Envelope times are sigmoid(raw) * max, so an attack longer than the shortest
        note — where decay and sustain have exactly zero gradient — is not representable."""
        super().__init__()
        self.bounds = bounds or dict(attack=32.0, decay=32.0, release=32.0)   # Vital's own ceiling
        # velocity: shape = 1 - vt (1 - vel/127), squared by the amp path (measured to 0.1 dB, vt in [-1, 1])
        self.velocity_track = velocity_track
        self.interpolation = interpolation      # 0 = stepped table (Basic Shapes): snap frame to the keyframe below
        # Filter env amount: unipolar [0, 1] by default. Vital allows [-1, 1], but "negative amount +
        # filter attack" and "positive amount + fast decay" are the same trajectory below the filter's
        # frame resolution, and the search wanders between them; the conventional pluck is positive.
        self.bipolar_fenv = bipolar_fenv
        self.tmin = dict(attack=0.001, decay=0.01, release=0.01)               # log-time floors
        # Vital randomises every note's start phase (phase_randomization 100%). A fixed-phase
        # twin then scores WORST at the true detune: its beats land exactly out of step with
        # Vital's. Drawing fresh per-note phases makes the loss an expectation over phases,
        # minimised at the truth. Set False only to compare against a deterministic Vital.
        self.random_phase = True
        self.register_buffer("table", table)             # (F, K)
        self.sr, self.voices, self.detune_range = sr, voices, detune_range_semitones
        F, K = table.shape
        self.F, self.K = F, K
        self._seed = seed; g = torch.Generator().manual_seed(seed)
        self.register_buffer("voice_phase", 2 * math.pi * torch.rand(voices, generator=g))   # Vital: random phase
        # Measured on Vital (detune_power default): 5 voices sit at +/-1.0 and +/-0.35 of
        # the detune, i.e. offset = sign(i) * (|i|/K)^1.5, centre voice at 0 when odd.
        half = (voices - 1) // 2                     # NB: not `K` — that is the harmonic count above
        idx = torch.arange(voices, dtype=torch.float32) - (voices - 1) / 2
        off = torch.sign(idx) * (idx.abs() / max(half, 0.5)) ** 1.5 if voices > 1 else torch.zeros(1)
        self.register_buffer("voice_offset", off)
        self.register_buffer("k", torch.arange(1, K + 1, dtype=torch.float32))

    # raw (Vital 0-1) -> physical
    @staticmethod
    def vital_time_raw(seconds): return (seconds / 32.0) ** 0.25          # Vital: time = 32 * raw**4
    def _t(self, key, s01):
        """Log-time: t = tmin * (tmax/tmin)^s. Vital's quartic has ~zero gradient at
        short times (d attack/d raw ~ 0.002 at 0.5 ms) and traps the search there."""
        lo, hi = self.tmin[key], self.bounds[key]
        return lo * (hi / lo) ** s01
    def _t_inv(self, key, seconds):
        lo, hi = self.tmin[key], self.bounds[key]
        return math.log(max(seconds, lo) / lo) / math.log(hi / lo)
    def physical(self, p):
        """p: dict of unbounded tensors -> dict of physical values (all differentiable)."""
        s = {k: torch.sigmoid(v) for k, v in p.items()}                # 0-1
        out = {}
        pw = lambda k, default: 40 * (s[k] - 0.5) if k in s else torch.tensor(default)
        out.update(attack_power=pw("apow", 0.0), decay_power=pw("dpow", -2.0), release_power=pw("rpow", -2.0))
        if "keytrack" in s: out["keytrack"] = 2 * s["keytrack"] - 1
        if "fblend" in s: out["fblend"] = 2 * s["fblend"]                                   # display 0 LP .. 1 BP .. 2 HP
        if "fdrive" in s: out["fdrive"] = s["fdrive"]                                       # raw: 0..1 = 0..20 dB
        if "frame_spread" in s: out["frame_spread"] = 256 * (s["frame_spread"] - 0.5)     # DISPLAY frames (raw 0.5 = 0); rows in render()
        if "noise" in s: out["noise"] = s["noise"]                                          # white-noise amplitude fraction at the filter input
        if "cutoff" in s:                                              # filter section present
            out.update(cutoff_raw=s["cutoff"], resonance=s["resonance"],
                       fenv_amount=(2 * s["fenv_amount"] - 1) if self.bipolar_fenv else s["fenv_amount"],
                       fattack=self._t("attack", s["fattack"]), fdecay=self._t("decay", s["fdecay"]),
                       fsustain=s["fsustain"], frelease=self._t("release", s["frelease"]),
                       fattack_power=pw("fapow", 0.0), fdecay_power=pw("fdpow", -2.0), frelease_power=pw("frpow", -2.0))
        out.update(
            frame=s["frame"] * (self.F - 1),
            detune_semis=(s["detune"] ** 2) * self.detune_range,      # percent params display raw^2
            blend=s["blend"],                                       # linear percent (read-back: 0.7746 -> "77.46%")
            level=s["level"] ** 2,
            attack=self._t("attack", s["attack"]), decay=self._t("decay", s["decay"]),
            sustain=s["sustain"], release=self._t("release", s["release"]),
        )
        return out
    def vital_params(self, p):
        s = {k: float(torch.sigmoid(v)) for k, v in p.items()}; ph = self.physical(p)
        out = {"velocity_track": (self.velocity_track + 1) / 2}
        for k, name in (("apow", "envelope_1_attack_power"), ("dpow", "envelope_1_decay_power"), ("rpow", "envelope_1_release_power"),
                        ("fapow", "envelope_2_attack_power"), ("fdpow", "envelope_2_decay_power"), ("frpow", "envelope_2_release_power"),
                        ("keytrack", "filter_1_key_track"), ("frame_spread", "oscillator_1_unison_frame_spread"), ("fblend", "filter_1_blend"), ("fdrive", "filter_1_drive")):
            if k in s: out[name] = s[k]
        if "noise" in s:
            out.update({"sample_switch": 1.0 if s["noise"] > 1e-3 else 0.0, "sample_level": s["noise"] ** 0.5, "sample_destination": 0.0,   # FILTER 1
                        "sample_keytrack": 0.0, "sample_loop": 1.0, "sample_random_phase": 1.0})
        if "cutoff" in s:
            out.update({"filter_1_cutoff": s["cutoff"], "filter_1_resonance": s["resonance"],
                        "modulation_1_amount": (float(ph["fenv_amount"]) + 1) / 2,     # raw = (amount + 1) / 2
                        "envelope_2_attack": self.vital_time_raw(float(ph["fattack"])), "envelope_2_decay": self.vital_time_raw(float(ph["fdecay"])),
                        "envelope_2_sustain": s["fsustain"], "envelope_2_release": self.vital_time_raw(float(ph["frelease"]))})
        return out | {"oscillator_1_wave_frame": s["frame"], "oscillator_1_unison_detune": s["detune"],
                "oscillator_1_blend": s["blend"], "oscillator_1_level": s["level"],
                "oscillator_1_unison_voices": (self.voices - 1) / 15,
                "envelope_1_attack": self.vital_time_raw(float(ph["attack"])),
                "envelope_1_decay": self.vital_time_raw(float(ph["decay"])),
                "envelope_1_sustain": s["sustain"],
                "envelope_1_release": self.vital_time_raw(float(ph["release"]))}

    def harmonics(self, frame):
        """Continuous frame position -> (K,) signed harmonic amplitudes."""
        lo, w = self.frame_parts(frame)                  # stepped tables (interpolation 0): no gradient to frame, the sweep finds it
        return (1 - w) * self.table[lo] + w * self.table[lo + 1]

    def frame_parts(self, frame):
        """Continuous frame -> (integer frame below, morph weight). The table is a constant; the weight carries the gradient."""
        if self.interpolation == 0: frame = torch.floor(frame.detach())
        lo = torch.clamp(torch.floor(frame), 0, self.F - 2); return lo.long(), frame - lo

    @staticmethod
    def power_curve(x, p):
        """Vital's segment curve, measured: (1 - e^{p x}) / (1 - e^p); linear at p = 0.
        Matches the plugin at p = -2, -12, 0 to 3 dp (display power = 40 (raw - 0.5))."""
        p = torch.as_tensor(p, dtype=x.dtype, device=x.device)
        lin = x
        ex = (1 - torch.exp(p * x)) / (1 - torch.exp(p) + 1e-9)
        return torch.where(p.abs() < 1e-3, lin, ex)

    def envelope_shape(self, t, off, ph):
        """Vital's ADSR as measured on the real plugin. Attack rises along the power
        curve; decay reaches sustain at exactly t=d, release reaches zero at exactly t=r,
        each along 1 - curve(x, power). Default powers 0 / -2 / -2. The amplitude path
        applies the shape SQUARED (sustain 0.3 -> 0.09 of peak)."""
        a, d, s, r = ph["attack"], ph["decay"], ph["sustain"], ph["release"]
        pa, pd, pr = ph.get("attack_power", 0.0), ph.get("decay_power", -2.0), ph.get("release_power", -2.0)
        def on(tt):
            atk = self.power_curve(torch.clamp(tt / a, 0.0, 1.0), pa)
            xd = torch.clamp((tt - a) / d, 0.0, 1.0)
            dec = s + (1 - s) * (1 - self.power_curve(xd, pd))
            return torch.where(tt < a, atk, dec)
        v_off = on(torch.tensor(off, device=t.device))
        xr = torch.clamp((t - off) / r, 0.0, 1.0)
        rel = v_off * (1 - self.power_curve(xr, pr))
        return torch.where(t < off, on(t), rel)

    def envelope(self, t, off, ph): return self.envelope_shape(t, off, ph) ** 2      # amplitude path is squared
    def res_law(self, r):
        """resonance raw -> (passband gain linear, Q), piecewise-linear in the measured table."""
        tab = self.RES_TAB.to(r.device); x = tab[:, 0]
        i = torch.clamp(torch.searchsorted(x, r.reshape(1)) - 1, 0, len(x) - 2)[0]
        w = (r - x[i]) / (x[i + 1] - x[i])
        pb = tab[i, 1] + w * (tab[i + 1, 1] - tab[i, 1]); q = tab[i, 2] + w * (tab[i + 1, 2] - tab[i, 2])
        return 10 ** (pb / 20), q

    NF, HOP = 4096, 1024      # 4096: -29 dB error at fc 275 Hz / Q 5 (1024 was -11 dB); 21 ms time resolution
    ONSET_S, ONSET_NF = 0.10, 1024    # swept 256..2048 vs Vital: 1024 is the best all-round (worst band error 4.4 dB)
    def tv_lowpass(self, x, fc, G, Q, blend=None):
        """Time-varying 2-pole lowpass, two frame sizes crossfaded. Long frames (NF) are
        accurate when the cutoff is LOW (long ring) but smear the onset; a pluck's envelope
        opens the filter for ~10 ms and that burst is most of the note's high-band energy.
        Short frames (ONSET_NF) resolve it and are accurate there because the cutoff is high.
        Crossfade short -> long over the first ONSET_S of the signal (each chunk is one note)."""
        long = self._tv_lowpass(x, fc, G, Q, self.NF, self.NF // 4, blend)
        # the short-frame pass only matters through the crossfade; run it on that head alone (halves the filter cost)
        n_on = min(x.shape[-1], int((self.ONSET_S + 0.06) * self.sr))
        short = self._tv_lowpass(x[:, :n_on], fc[:, :n_on], G, Q, self.ONSET_NF, self.ONSET_NF // 4, blend)
        t = torch.arange(n_on, device=x.device) / self.sr
        w_long = torch.sigmoid((t - self.ONSET_S) / 0.01)[None, :]          # 0.9975 at n_on; beyond it: long only
        head = w_long * long[:, :n_on] + (1 - w_long) * short
        return torch.cat([head, long[:, n_on:]], -1)

    def _tv_lowpass(self, x, fc, G, Q, n, h, blend=None):
        """One frame size. x: (B, T); fc: (B, T) Hz. Per STFT frame (Hann, 75% overlap) the
        analog prototype at the frame's mean cutoff: H = G wc^2 / (s^2 + s wc/Q + wc^2)."""
        B, T = x.shape
        win = torch.hann_window(n, device=x.device)
        pad = n - h
        xp = torch.nn.functional.pad(x, (pad, pad)); fp = torch.nn.functional.pad(fc, (pad, pad), mode="replicate")
        frames = xp.unfold(-1, n, h) * win                                   # (B, nfr, n)
        fcf = fp.unfold(-1, n, h).mean(-1)                                   # (B, nfr)
        X = torch.fft.rfft(frames)                                           # (B, nfr, n/2+1)
        w = 2 * math.pi * torch.fft.rfftfreq(n, 1 / self.sr).to(x.device)    # (n/2+1,)
        wc = 2 * math.pi * fcf.clamp(20.0, 0.45 * self.sr)[..., None]        # (B, nfr, 1)
        sj = 1j * w
        den = sj ** 2 + sj * wc / Q + wc ** 2
        lp, bp, hp = wc ** 2 / den, (sj * wc / Q) / den, sj ** 2 / den
        if blend is None:
            H = G * lp
        else:   # measured: d<=1: (1-d) LP + sqrt(1-(1-d)^2) BP ; d>1: (d-1) HP + sqrt(1-(d-1)^2) BP  (max err 0.9 dB)
            d = blend
            a = torch.clamp(1 - d, 0, 1); b = torch.clamp(d - 1, 0, 1)
            H = G * torch.where(d <= 1, a * lp + torch.sqrt(1 - a * a + 1e-9) * bp, b * hp + torch.sqrt(1 - b * b + 1e-9) * bp)
        y = torch.fft.irfft(X * H, n=n) * win                                # (B, nfr, n)
        # overlap-add; Hann^2 at 75% overlap sums to a constant 1.5
        out = torch.nn.functional.fold(y.transpose(1, 2), (1, xp.shape[-1]), (1, n), stride=(1, h))[:, 0, 0, :]
        return out[:, pad:pad + T] / 1.5

    W = 2048                                   # table length, like Vital

    def frame_tables(self, amps, cuts):
        """(K,) signed harmonic amplitudes -> {cutoff: (W,) waveform}, one per distinct
        cutoff. Exact band-limiting per pitch (cut = floor(Nyquist / f0)): octave mips
        cost a 392 Hz note half its harmonics (12.5 kHz instead of 24 kHz).
        x[n] = sum_k a_k sin(2 pi k n / W)  ==  irfft of  -i * a_k * W/2.  Differentiable in amps."""
        W, K = self.W, self.K
        spec = torch.zeros(W // 2 + 1, dtype=torch.complex64, device=amps.device)
        out = {}
        for cut in cuts:
            m = (self.k <= cut).float()
            sp = spec.clone(); sp[1:K + 1] = torch.complex(torch.zeros_like(amps), -(amps * m) * (W / 2))
            out[cut] = torch.fft.irfft(sp, n=W)
        return out

    def lookup(self, table, turns):
        """Linear-interpolated wavetable read. turns: (..., T) in [0,1). Gradient flows to
        the table (-> frame) and through the interpolation slope to turns (-> detune)."""
        pos = turns * self.W
        i0 = torch.floor(pos).long() % self.W; frac = pos - torch.floor(pos)
        return table[i0] * (1 - frac) + table[(i0 + 1) % self.W] * frac

    def _chunk(self, t, f0, dur, vel, tables, ws, detune, blend, level, attack, decay, sustain, release, filt=None, extra=None, ph0=None, white=None):
        """B notes sharing one duration and one mip table. t: (T,), f0/vel: (B,). -> (B, T).
        filt: dict(cutoff_raw, resonance, fenv_amount, fattack, fdecay, fsustain, frelease) or None."""
        ph = dict(attack=attack, decay=decay, sustain=sustain, release=release, attack_power=extra["attack_power"],
                  decay_power=extra["decay_power"], release_power=extra["release_power"])
        vshape = 1 - self.velocity_track * (1 - vel / 127.0)                                   # (B,)
        env = (self.envelope_shape(t, dur, ph)[None, :] * vshape[:, None]) ** 2                 # amp path: squared
        sig = torch.zeros_like(env); wsq = 0.0
        B = f0.shape[0]
        if ph0 is None: ph0 = self.draw_phases(B, t.device)
        for v in range(self.voices):
            fv = f0 * 2 ** (self.voice_offset[v] * detune / 12)
            # phase in turns, wrapped: float32 at 1e6 rad loses ~0.1 rad and devices differ
            turns = torch.remainder(fv[:, None] * t[None, :] + ph0[v][:, None], 1.0)
            # per-voice table (frame spread), read as a morph of the two CONSTANT neighbouring frames' waveforms: the gathers
            # touch no learnable tensor, so the backward has no 10M-way scatter (nondeterministic on MPS and CPU alike, and
            # 2e-3 off in float32) -- the frame's gradient arrives through the dense morph weight instead (2026-09-21)
            wave = (1 - ws[v]) * self.lookup(tables[v][0], turns) + ws[v] * self.lookup(tables[v][1], turns)
            centre = self.voices == 1 or abs(float(self.voice_offset[v])) < 1e-6
            # Measured on Vital (mono sum): EVERY outer voice has the same weight relative to
            # the centre, r(b) = 0.503 b + 0.202 b^2 (fits 0/0.25/0.5/0.75/1 within 0.01),
            # independent of voice count. Outer voices are hard-panned; the mono sum halves them.
            w = 1.0 if centre else 0.503 * blend + 0.202 * blend ** 2
            sig = sig + w * wave; wsq = wsq + w ** 2
        # Vital keeps detuned unison at constant power for any voices/blend (measured:
        # rms 1.000 +/- 0.01 for 2-8 voices, blend 0-1); blend changes character, not level
        sig = sig / torch.sqrt(torch.as_tensor(wsq, device=sig.device) + 1e-8)
        sig = self.PRE_GAIN * (level / 0.694) * sig                        # Vital's filter-input amplitude (level is torch units)
        if "noise" in extra and white is not None:                         # sample oscillator (white noise) summed at the filter input
            sig = sig + self.NOISE_GAIN * extra["noise"] * white
        if filt is not None:                                               # osc -> drive -> filter -> amp, as in Vital
            g = 10 ** extra.get("fdrive", torch.tensor(0.0, device=sig.device))
            sig = torch.tanh(2 * g * sig) / (2 * torch.sqrt(g))
            e2 = self.envelope_shape(t, dur, dict(attack=filt["fattack"], decay=filt["fdecay"], sustain=filt["fsustain"], release=filt["frelease"],
                                                  attack_power=filt["fattack_power"], decay_power=filt["fdecay_power"], release_power=filt["frelease_power"]))
            semis = 128.0 * (filt["cutoff_raw"] + filt["fenv_amount"] * e2)[None, :]           # (1, T)
            if "keytrack" in extra:                                                              # + kt (note - 60), measured
                note = 69 + 12 * torch.log2(f0 / 440.0)
                semis = semis + (extra["keytrack"] * (note - 60))[:, None]
            fc = 261.6256 * 2 ** ((semis - 52.0) / 12)
            G, Q = self.res_law(filt["resonance"])
            sig = self.tv_lowpass(sig, fc.expand(sig.shape[0], -1), G, Q, extra.get("fblend", None))
        return self.OUT_GAIN * env * sig

    def draw_phases(self, B, device):
        """Per-note start phases in turns: Vital draws a random phase per note; deterministic mode uses the voice phases."""
        return torch.rand(self.voices, B, device=device) if self.random_phase else (self.voice_phase / (2 * math.pi))[:, None].expand(self.voices, B)

    def draw_noise(self, B, T, device):
        """Unit-variance white noise per note; a fixed realisation in deterministic mode so renders reproduce."""
        if self.random_phase: return torch.randn(B, T, device=device)
        g = torch.Generator().manual_seed(self._seed); return torch.randn(B, T, generator=g).to(device)

    checkpoint = False        # recompute each note group's forward in backward: ~400 MB -> ~30 MB saved activations at 20 notes

    def render(self, p, notes, L, chunk=80):
        dev = self.table.device; ph = self.physical(p)
        out = torch.zeros(L, device=dev); nyq = self.sr / 2
        args = (ph["detune_semis"], ph["blend"], ph["level"], ph["attack"], ph["decay"], ph["sustain"], ph["release"])
        filt = {k: ph[k] for k in ("cutoff_raw", "resonance", "fenv_amount", "fattack", "fdecay", "fsustain", "frelease",
                                   "fattack_power", "fdecay_power", "frelease_power")} if "cutoff_raw" in ph else None
        extra = {k: ph[k] for k in ("attack_power", "decay_power", "release_power", "keytrack", "fblend", "fdrive", "noise") if k in ph}
        spread = ph["frame_spread"] * (self.F - 1) / 256 if "frame_spread" in ph else None    # display frames -> rows
        groups = {}
        for pitch, start, dur, vel in notes:
            f0 = 440.0 * 2 ** ((pitch - 69) / 12)
            cut = min(self.K, int(nyq / (f0 * 2 ** (self.detune_range / 12))))   # safe for the sharpest voice
            groups.setdefault((round(dur, 4), cut), []).append((f0, start, vel))
        cuts = {c for _, c in groups}
        # one table set per voice: with unison frame spread, outer voices play frame + offset * spread
        frames = [ph["frame"] + (self.voice_offset[v] * spread if spread is not None else 0.0) for v in range(self.voices)]
        frames = [torch.clamp(f, 0, self.F - 1) for f in frames]
        parts = [self.frame_parts(f) for f in frames]                                          # [voice] -> (lo, w)
        with torch.no_grad(): tables = [tuple(self.frame_tables(self.table[lo + d], cuts) for d in (0, 1)) for lo, _ in parts]   # [voice] -> (below, above)[cut], constants
        ws = [w for _, w in parts]
        # tail after note-off: Vital's release segment reaches zero at exactly `release` seconds, so render
        # that long (plus a margin) and no longer — a fixed 1 s tail both truncated long releases mid-curve
        # (a click) and wasted most of the render on silence for short ones. The length is a shape, so it is
        # detached; releases past TAIL_MAX get a 20 ms raised-cosine fade at the end instead of a cut.
        want = float(ph["release"].detach()) + 0.05
        tail = min(math.ceil(want / 0.25) * 0.25, TAIL_MAX)                # quantised so the MPS allocator can reuse blocks
        fade = None
        if want > TAIL_MAX:
            nf = int(0.02 * self.sr); fade = 0.5 * (1 + torch.cos(math.pi * torch.arange(nf, device=dev) / nf))
        for (dur, cut), group in groups.items():
            T = int((dur + tail) * self.sr)
            t = torch.arange(T, device=dev, dtype=torch.float32) / self.sr
            for i in range(0, len(group), chunk):
                g = group[i:i + chunk]
                f0 = torch.tensor([x[0] for x in g], device=dev); vel = torch.tensor([float(x[2]) for x in g], device=dev)
                ph0 = self.draw_phases(len(g), dev)                       # drawn outside the checkpoint so the recompute sees the same phases
                white = self.draw_noise(len(g), T, dev) if "noise" in extra else None
                if self.checkpoint and torch.is_grad_enabled():
                    fk, fv_ = (list(filt.keys()), list(filt.values())) if filt is not None else ([], []); ek, ev = list(extra.keys()), list(extra.values())
                    nw = len(ws)
                    def fn(t, f0, vel, ph0, white, *flat, tables=[(tb[0][cut], tb[1][cut]) for tb in tables], nf=len(fv_), fk=fk, ek=ek):
                        w_ = flat[:nw]; rest = flat[nw:]
                        a = rest[:len(args)]; fl = dict(zip(fk, rest[len(args):len(args) + nf])) if fk else None; ex = dict(zip(ek, rest[len(args) + nf:]))
                        return self._chunk(t, f0, dur, vel, tables, w_, *a, filt=fl, extra=ex, ph0=ph0, white=white)
                    sig = torch.utils.checkpoint.checkpoint(fn, t, f0, vel, ph0, white, *ws, *args, *fv_, *ev, use_reentrant=False, preserve_rng_state=False)
                else:
                    sig = self._chunk(t, f0, dur, vel, [(tb[0][cut], tb[1][cut]) for tb in tables], ws, *args, filt=filt, extra=extra, ph0=ph0, white=white)
                if fade is not None: sig = torch.cat([sig[:, :-len(fade)], sig[:, -len(fade):] * fade], -1)
                for row, (_, start, _) in zip(sig, g):
                    n0 = int(start * self.sr); n1 = min(L, n0 + T)
                    if n1 > n0: out[n0:n1] = out[n0:n1] + row[: n1 - n0]
        return out[None, None, :]

# Vital's init patch, in its own raw 0-1 space (vital_params.json): the natural starting
# point for any search. Random inits put the attack past the note length, where decay
# and sustain have exactly zero gradient and the search stalls.
VITAL_INIT = dict(frame=0.0, detune=0.4472, blend=0.8, level=0.7071, sustain=1.0,
                  attack_s=0.0005, decay_s=1.0, release_s=0.09)

# Envelope families to restart from. Vital's init is a pad; a pluck target starts
# far from it, and one init + noise is not diversity.
INIT_FAMILIES = [
    dict(name="vital-init", attack_s=0.0005, decay_s=1.0,  sustain=1.0, release_s=0.09),
    dict(name="pluck",      attack_s=0.002,  decay_s=0.15, sustain=0.2, release_s=0.2),
    dict(name="medium",     attack_s=0.02,   decay_s=0.4,  sustain=0.5, release_s=0.5),
]

def init_raw(synth, noise=0.2, seed=0):
    g = torch.Generator().manual_seed(seed)
    logit = lambda x: torch.logit(torch.tensor(float(x)).clamp(1e-3, 1 - 1e-3))
    v = VITAL_INIT; e = INIT_FAMILIES[seed % len(INIT_FAMILIES)]
    base = dict(frame=v["frame"], detune=v["detune"], blend=v["blend"], level=v["level"], sustain=e["sustain"],
                attack=synth._t_inv("attack", e["attack_s"]), decay=synth._t_inv("decay", e["decay_s"]),
                release=synth._t_inv("release", e["release_s"]))
    return {k: logit(x) + noise * torch.randn((), generator=g) for k, x in base.items()}

def raw_from_physical(synth, **phys):
    """Inverse map for building test targets: physical -> unbounded raw."""
    inv = lambda x: torch.logit(torch.tensor(float(x)).clamp(1e-4, 1 - 1e-4))
    t_raw = lambda sec, key: inv(synth._t_inv(key, sec))
    return {"frame": inv(phys["frame"] / (synth.F - 1)),
            "detune": inv((phys["detune_semis"] / synth.detune_range) ** 0.5),
            "blend": inv(phys["blend"]), "level": inv(phys["level"] ** 0.5),
            "attack": t_raw(phys["attack"], "attack"), "decay": t_raw(phys["decay"], "decay"),
            "sustain": inv(phys["sustain"]), "release": t_raw(phys["release"], "release"),
            **{k: inv(phys[n] / 40 + 0.5) for k, n in (("apow", "attack_power"), ("dpow", "decay_power"), ("rpow", "release_power"),
                                                       ("fapow", "fattack_power"), ("fdpow", "fdecay_power"), ("frpow", "frelease_power")) if n in phys},
            **({"keytrack": inv((phys["keytrack"] + 1) / 2)} if "keytrack" in phys else {}),
            **({"fblend": inv(phys["fblend"] / 2)} if "fblend" in phys else {}),
            **({"fdrive": inv(phys["fdrive"])} if "fdrive" in phys else {}),
            **({"frame_spread": inv(phys["frame_spread"] / 256 + 0.5)} if "frame_spread" in phys else {}),
            **({"cutoff": inv(phys["cutoff_raw"]), "resonance": inv(phys["resonance"]),
                "fenv_amount": inv((phys["fenv_amount"] + 1) / 2 if synth.bipolar_fenv else phys["fenv_amount"]),
                "fattack": t_raw(phys["fattack"], "attack"), "fdecay": t_raw(phys["fdecay"], "decay"),
                "fsustain": inv(phys["fsustain"]), "frelease": t_raw(phys["frelease"], "release")} if "cutoff_raw" in phys else {})}

if __name__ == "__main__":
    import time, warnings; warnings.filterwarnings("ignore")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import common as C
    SR = 48000; L = 10 * SR; notes = kalimba_notes()
    if torch.backends.mps.is_available(): torch.mps.set_per_process_memory_fraction(0.5)
    TRUE = dict(frame=2.3, detune_semis=0.3, blend=0.6, level=0.5, attack=0.01, decay=0.3, sustain=0.4, release=0.5)

    # 1) render the target on CPU, time it, listen
    B = bounds_from_notes(notes); print(f"  bounds from clip: {B}")
    synth = WavetableSynth(basic_shapes(), SR, voices=3, bounds=B)
    p_true = {k: v.clone() for k, v in raw_from_physical(synth, **TRUE).items()}
    t0 = time.time(); target = synth.render(p_true, notes, L); t_cpu = time.time() - t0
    print(f"  render (cpu, {len(notes)} notes, 3 voices, 10 s): {t_cpu*1000:.0f} ms  peak={float(target.abs().max()):.3f}")
    import soundfile as sf
    sf.write(os.path.join(C.HERE, "synth_target.wav"), target[0, 0].numpy(), SR)
    print("  vital_params:", {k: round(v, 4) for k, v in synth.vital_params(p_true).items()})

    # 2) CPU-vs-MPS gradient at full length
    def grad_on(dev):
        s = WavetableSynth(basic_shapes(), SR, voices=3, bounds=B).to(dev)
        p = {k: (v.clone() + 0.3).to(dev).requires_grad_(True) for k, v in p_true.items()}
        out = s.render(p, notes, L); tgt = target.to(dev)
        loss = C.mrstft(out[0, 0], tgt[0, 0]); loss.backward()
        if dev == "mps": torch.mps.synchronize()
        return float(loss), torch.stack([p[k].grad.cpu() for k in synth.RAW])
    lc, gc = grad_on("cpu"); t0 = time.time(); lm, gm = grad_on("mps"); t_mps = time.time() - t0
    print(f"  grad check: loss cpu={lc:.4f} mps={lm:.4f}  cos={float(torch.nn.functional.cosine_similarity(gc, gm, dim=0)):+.4f}  (mps fwd+bwd {t_mps*1000:.0f} ms)")

    # 3) recovery: from a random init, recover TRUE from the audio alone
    def recover(voices, steps=400, lr=0.03, restarts=3):
        dev = C.DEVICE; s_ = WavetableSynth(basic_shapes(), SR, voices=voices, bounds=B).to(dev)
        tr = {k: v.clone() for k, v in raw_from_physical(s_, **TRUE).items()}
        with torch.no_grad(): tgt = s_.render({k: v.to(dev) for k, v in tr.items()}, notes, L)[0, 0]
        best = None
        for restart in range(restarts):
            r = _recover_once(s_, tgt, dev, steps, lr, restart)
            if best is None or r[0] < best[0]: best = r
        loss, p, ms = best
        got = s_.physical(p)
        print(f"  recovery voices={voices}: best of {restarts} restarts loss={loss:.4f}  {ms:.0f} ms/step")
        print(f"  {'param':14s} {'true':>8s} {'recovered':>10s}")
        for k in TRUE:
            if voices == 1 and k in ("detune_semis", "blend"): continue
            print(f"  {k:14s} {TRUE[k]:8.3f} {float(got[k]):10.3f}")

    def _recover_once(s_, tgt, dev, steps, lr, restart):
        p = {k: v.to(dev).requires_grad_(True) for k, v in init_raw(s_, seed=restart).items()}
        def sweep_frame():
            # frame is 1-D with ridges; sweep it in frame units, other params as they are now
            with torch.no_grad():
                cands = torch.logit(torch.linspace(0.02, 0.98, 20)); losses = []
                for c in cands:
                    p["frame"].fill_(float(c)); losses.append(float(C.mrstft(s_.render(p, notes, L)[0, 0], tgt)))
                p["frame"].fill_(float(cands[int(np.argmin(losses))]))
            return min(losses)
        opt = torch.optim.Adam(p.values(), lr=lr); t0 = time.time()
        for i in range(steps):
            if i in (0, steps // 3):                       # sweep at start, and again once the envelope has settled
                sw = sweep_frame(); print(f"    sweep@{i}: frame={float(s_.physical(p)['frame']):.2f} loss={sw:.3f}", flush=True)
            opt.zero_grad(); loss = C.mrstft(s_.render(p, notes, L)[0, 0], tgt); loss.backward(); opt.step()
        print(f"    restart {restart}: loss={float(loss):.4f}", flush=True)
        return float(loss), {k: v.detach() for k, v in p.items()}, (time.time() - t0) / steps * 1000
    recover(1)
    recover(3)
