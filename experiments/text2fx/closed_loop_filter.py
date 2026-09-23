"""Closed loop on real Vital audio WITH the filter section.

Vital: Moog Pluck preset with only its env_2 -> filter_1_cutoff routing kept, osc 1 +
filter 1 (Analog 12 dB LP) on, everything else off. Secret = oscillator + amp env +
cutoff/resonance/env amount + filter env. Twin recovers all 15 from the audio, exports
back into Vital, and the two Vital renders are compared.
"""
import os, sys, glob, time, types, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, soundfile as sf
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from fantasia_core import plugins as P, presets as PR
from synth import WavetableSynth, sustained_notes, bounds_from_notes, init_raw, raw_from_physical

SR = 48000; L = 10 * SR; notes = sustained_notes(); B = bounds_from_notes(notes); DEV = C.DEVICE
if DEV == "mps": torch.mps.set_per_process_memory_fraction(0.5)
VOICES = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LEVEL_CAL = 0.694; N = types.SimpleNamespace
midi = [N(pitch=p, start=s, duration=d, velocity=v) for p, s, d, v in notes]
TBL = torch.load(os.path.join(C.HERE, "tables", "basic_shapes.pt")); TABLE = TBL["table"]; F_T = TABLE.shape[0]

SECRET_PHYS = dict(frame=0.375 * (F_T - 1), detune_semis=0.4, blend=0.5, level=0.5, attack=0.005, decay=0.4, sustain=0.6, release=0.3,
                   cutoff_raw=0.35, resonance=0.4, fenv_amount=0.5, fattack=0.002, fdecay=0.25, fsustain=0.15, frelease=0.3,
                   fdecay_power=-5.54)          # the preset's own curve, left as shipped: the twin must recover it

def load_vital():
    patch = PR.read_vital_file(glob.glob(os.path.expanduser("~/Music/Vital/*/Presets/Moog Pluck.vital"))[0])
    mods = patch["settings"]["modulations"]
    patch["settings"]["modulations"] = [mods[0]] + [{"source": "", "destination": ""} for _ in mods[1:]]
    vital = P.load("Vital"); assert P.restore_preset(vital, PR.splice_vital(P.preset_bytes(vital), patch))
    for k, v in {"oscillator_2_switch": 0, "oscillator_3_switch": 0, "sample_switch": 0, "chorus_switch": 0, "reverb_switch": 0, "delay_switch": 0,
                 "distortion_switch": 0, "compressor_switch": 0, "eq_switch": 0, "flanger_switch": 0, "phaser_switch": 0, "filter_fx_switch": 0, "filter_2_switch": 0,
                 "filter_1_switch": 1, "filter_1_model": 0.0, "filter_1_style": 0.0, "filter_1_blend": 0.0, "filter_1_key_track": 0.5, "filter_1_drive": 0.0, "filter_1_mix": 1.0,          # key_track raw 0.5 = 0% (0.0 is -100%)
                 "envelope_2_attack_power": 0.5, "envelope_2_release_power": 0.45,     # decay power is part of the secret now
                 "oscillator_1_unison_voices": (VOICES - 1) / 15, "oscillator_1_phase_randomization": 1.0, "oscillator_1_frequency_morph_type": 0.0,
                 "oscillator_1_distortion_type": 0.0, "oscillator_1_unison_frame_spread": 0.5, "oscillator_1_detune_power": 0.65, "oscillator_1_transpose": 0.5,
                 "oscillator_1_tune": 0.5, "volume": 0.7397, "portamento_force": 0.0}.items():
        P.set_param(vital, k, v)
    for k, want in (("filter_1_switch", "On"), ("oscillator_1_frequency_morph_type", "None"), ("filter_1_key_track", "0%")):
        got = P._params(vital)[k].string_value
        if want: assert got == want, f"{k} is {got!r}"
    return vital

def vital_render(params, settle=True):
    """Render `params` in Vital. The first pass is rendered and THROWN AWAY: the plugin instance is shared and Vital
    smooths parameter changes, so a render that follows a different patch starts with the old values still gliding --
    measured 2026-09-22 as up to 3.8 dB in a band, i.e. a render's result depended on what was rendered before it.
    One discarded pass costs 0.4 s and makes it order-independent (a full unload+reload costs 3.6 s and agrees)."""
    vital = load_vital()
    for k, v in params.items(): P.set_param(vital, k, v)
    if settle: P.render_notes(vital, midi, 10.0, sr=SR, tail=0.0)
    y = P.render_notes(vital, midi, 10.0, sr=SR, tail=0.0).mean(axis=1)[:L]
    return np.pad(y, (0, max(0, L - len(y)))).astype(np.float32)

def onset_profile_loss(a, b, notes, K=64):
    """Harmonic profile in the first 70 ms of each note, where the filter envelope has it
    open: the waveform (frame) is observable here and NOT in the hold span, where a darker
    frame and a lower cutoff look the same."""
    ha, hb = C.harmonic_profile(a, notes, K, hold=(0.005, 0.075)), C.harmonic_profile(b, notes, K, hold=(0.005, 0.075))
    ha = (ha - ha.max(-1, keepdim=True).values).clamp(min=-60.0); hb = (hb - hb.max(-1, keepdim=True).values).clamp(min=-60.0)
    return (ha - hb).abs().mean() / 10.0

BAND_W = float(os.environ.get("T2_BAND_W", "3.0"))            # smoothed-spectrum term: the one that can see noise level (T2_BAND_W=0 for the old loss)
def loss_fn(a, b):
    return (C.mrstft_lin(a, b, ffts=(128, 512, 2048, 8192, 16384)) + 2.0 * C.env_loss(a, b, win=7200)
            + 1.0 * C.mod_depth_loss(a, b, notes) + 1.0 * C.harmonic_loss(a, b, notes) + 1.0 * onset_profile_loss(a, b, notes)
            + (BAND_W * C.band_energy_loss(a, b) if BAND_W else 0.0))

NOISE = os.environ.get("T2_NOISE", "1") == "1"                 # fit the noise source too (T2_NOISE=0 for the pre-2026-09-17 behaviour)

def init_all(s_, seed):
    p = init_raw(s_, seed=seed)                                     # oscillator + amp env families
    g = torch.Generator().manual_seed(100 + seed)
    logit = lambda x: torch.logit(torch.tensor(float(x)).clamp(1e-3, 1 - 1e-3))
    fam = [dict(cut=0.5, res=0.2, amt=0.5, a=0.002, d=0.2, su=0.3, r=0.3), dict(cut=0.4, res=0.5, amt=0.75, a=0.005, d=0.5, su=0.5, r=0.5),
           dict(cut=0.6, res=0.1, amt=0.5, a=0.01, d=0.1, su=0.1, r=0.2)][seed % 3]
    p.update(cutoff=logit(fam["cut"]), resonance=logit(fam["res"]), fenv_amount=logit(fam["amt"]), fdpow=logit(0.45),   # start at the default -2
             fattack=logit(s_._t_inv("attack", fam["a"])), fdecay=logit(s_._t_inv("decay", fam["d"])), fsustain=logit(fam["su"]), frelease=logit(s_._t_inv("release", fam["r"])))
    if NOISE: p["noise"] = logit(0.05)                             # Vital's sample oscillator (white noise into the filter); starts quiet, not off
    return {k: v + 0.2 * torch.randn((), generator=g) for k, v in p.items()}

def recover(tgt, steps=300, lr=0.03, restarts=None):
    """Frame on a stepped table is a discrete choice: one restart starts at each keyframe (or at
    8 spread positions for a smooth table); envelope family and filter init rotate with it."""
    s_ = WavetableSynth(TABLE, SR, voices=VOICES, bounds=B, interpolation=TBL.get("interpolation", 1)).to(DEV); tgt = torch.tensor(tgt, device=DEV); best = None
    kf = TBL.get("keyframes") or []
    starts = [k / 256 * (s_.F - 1) for k in kf] if (TBL.get("interpolation", 1) == 0 and kf) else list(np.linspace(0, s_.F - 1, 8))
    restarts = restarts or len(starts)
    for r in range(restarts):
        p = {k: v.to(DEV).requires_grad_(True) for k, v in init_all(s_, r).items()}
        with torch.no_grad(): p["frame"].fill_(float(torch.logit(torch.tensor(min(max(starts[r % len(starts)] / (s_.F - 1), 1e-3), 1 - 1e-3)))))
        with torch.no_grad():                                          # level init: match the target's loudness
            y0 = s_.render(p, notes, L)[0, 0]; ratio = float(C.rms(tgt) / (C.rms(y0) + 1e-9))
            lv = float(torch.sigmoid(p["level"])) ** 2 * ratio          # physical level = sigmoid^2
            p["level"].fill_(float(torch.logit(torch.tensor(min(max(lv, 1e-4), 0.9999) ** 0.5))))
        def sweep():
            # score frame candidates on the pitch-aware profiles only: shape-normalised and taken
            # where the filter is open, so the choice does not depend on the (still wrong) envelope/filter
            with torch.no_grad():
                cands = torch.logit(torch.linspace(0.01, 0.99, 40)); ls = []
                for c in cands: p["frame"].fill_(float(c)); ls.append(float(loss_fn(s_.render(p, notes, L)[0, 0], tgt)))
                p["frame"].fill_(float(cands[int(np.argmin(ls))]))
        opt = torch.optim.Adam(p.values(), lr=lr); t0 = time.time()
        for i in range(steps):
            if i in (steps // 3, 2 * steps // 3): sweep()          # re-check the frame once the rest has adapted
            opt.zero_grad(); loss = loss_fn(s_.render(p, notes, L)[0, 0], tgt); loss.backward(); opt.step()
        print(f"    restart {r} (start frame {starts[r % len(starts)]:5.1f}): loss={float(loss):.4f}  frame={float(s_.physical(p)['frame']):5.2f}  ({(time.time()-t0)/steps*1000:.0f} ms/step)", flush=True)
        if best is None or float(loss) < best[0]: best = (float(loss), {k: v.detach() for k, v in p.items()})
    return s_, best[1]

def env_corr(a, b, w=int(0.005 * SR)):
    e = lambda x: np.sqrt((x[:len(x)//w*w].reshape(-1, w) ** 2).mean(1)); return float(np.corrcoef(e(a), e(b))[0, 1])
def band_diff(a, b):
    def bands(x):
        X = np.abs(np.fft.rfft(x)) ** 2; f = np.fft.rfftfreq(len(x), 1 / SR); t = X.sum()
        return np.array([10 * np.log10(X[(f >= lo) & (f < hi)].sum() / t + 1e-12) for lo, hi in ((300, 1000), (1000, 3000), (3000, 8000), (8000, 16000))])
    return np.round(bands(a) - bands(b), 1).tolist()

if __name__ == "__main__":
    print(f"  voices={VOICES}\n  1) Vital renders the secret patch (with filter + filter env)")
    s0 = WavetableSynth(TABLE, SR, voices=VOICES, bounds=B); sp = s0.vital_params(raw_from_physical(s0, **SECRET_PHYS))
    sp["oscillator_1_level"] = SECRET_PHYS["level"] ** 0.5
    ref = vital_render(sp); sf.write(os.path.join(C.HERE, f"loopf_ref_{VOICES}v.wav"), ref, SR)
    print("  2) recover on the twin"); s_, p = recover(ref); got = s_.physical(p)
    print(f"\n  {'param':14s} {'secret':>8s} {'recovered':>10s}")
    for k in SECRET_PHYS:
        if VOICES == 1 and k in ("detune_semis", "blend"): continue
        v = float(got[k]) / (LEVEL_CAL if k == "level" else 1.0)
        print(f"  {k:14s} {SECRET_PHYS[k]:8.3f} {v:10.3f}")
    vp_check = s_.vital_params(p); print(f"  export envelope_2_decay_power raw {vp_check['envelope_2_decay_power']:.3f} -> power {40*(vp_check['envelope_2_decay_power']-0.5):+.2f}  (secret -5.54)")
    print("  3) export to Vital and render")
    vp = s_.vital_params(p); vp["oscillator_1_level"] = min(1.0, (float(got["level"]) / LEVEL_CAL) ** 0.5)
    out = vital_render(vp); sf.write(os.path.join(C.HERE, f"loopf_out_{VOICES}v.wav"), out, SR)
    print(f"  Vital(secret) vs Vital(recovered): band energy diff dB {band_diff(out, ref)}   envelope corr {env_corr(out, ref):.3f}")
    with torch.no_grad(): tw = s_.render({k: v.to(DEV) for k, v in p.items()}, notes, L)[0, 0].cpu().numpy()
    print(f"  torch twin vs Vital(secret):        band energy diff dB {band_diff(tw, ref)}   envelope corr {env_corr(tw, ref):.3f}")
