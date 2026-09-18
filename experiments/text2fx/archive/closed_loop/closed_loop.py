"""Closed loop on REAL Vital audio.

  1. Set Vital to a secret patch, render the kalimba MIDI  -> reference
  2. Recover the patch on the torch twin from that audio alone (frame left free:
     it should land on the saw by itself)
  3. Export the recovered parameters back into Vital, render, compare
"""
import os, sys, types, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, soundfile as sf
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from fantasia_core import plugins as P
from synth import WavetableSynth, basic_shapes, kalimba_notes, sustained_notes, bounds_from_notes, init_raw

CLIP = sys.argv[4] if len(sys.argv) > 4 else "kalimba"
TABLE = sys.argv[5] if len(sys.argv) > 5 else None            # extracted-table .pt -> load its preset into Vital
SR = 48000; L = 10 * SR; notes = sustained_notes() if CLIP == "sustained" else kalimba_notes(); B = bounds_from_notes(notes); DEV = C.DEVICE
if DEV == "mps": torch.mps.set_per_process_memory_fraction(0.5)
LEVEL_CAL = 0.694                    # measured with the squared envelope: Vital level = torch level / 0.694
N = types.SimpleNamespace
midi = [N(pitch=p, start=s, duration=d, velocity=v) for p, s, d, v in notes]

VOICES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
LOSS = sys.argv[2] if len(sys.argv) > 2 else "mrstft"
DETERMINISTIC = len(sys.argv) > 3 and sys.argv[3] == "det"
def LOSSFN(a, b):
    if LOSS == "mrstft":  return C.mrstft(a, b)
    if LOSS == "lin":     return C.mrstft_lin(a, b)
    if LOSS == "env":     return C.mrstft(a, b) + 2.0 * C.env_loss(a, b)
    if LOSS == "lin+env": return C.mrstft_lin(a, b) + 2.0 * C.env_loss(a, b)
    if LOSS == "lin+cenv": return C.mrstft_lin(a, b) + 2.0 * C.env_loss(a, b, win=7200)   # 150 ms > a beat period
    if LOSS == "multi":    # short windows -> attack; long windows -> detune sidebands (2.9 Hz); coarse env -> ADSR
        return C.mrstft_lin(a, b, ffts=(128, 512, 2048, 8192, 16384)) + 2.0 * C.env_loss(a, b, win=7200)
    if LOSS == "full":       # multi+mod + per-note harmonic profile (pitch-aware) -> frame without ridges
        return (C.mrstft_lin(a, b, ffts=(128, 512, 2048, 8192, 16384)) + 2.0 * C.env_loss(a, b, win=7200)
                + 1.0 * C.mod_depth_loss(a, b, notes) + 1.0 * C.harmonic_loss(a, b, notes))
    if LOSS == "multi+mod":  # + beating depth (phase-independent) -> blend
        return C.mrstft_lin(a, b, ffts=(128, 512, 2048, 8192, 16384)) + 2.0 * C.env_loss(a, b, win=7200) + 1.0 * C.mod_depth_loss(a, b, notes)
SECRET = {"oscillator_1_unison_voices": (VOICES - 1) / 15, "oscillator_1_unison_detune": (0.4 / 2.0) ** 0.5,   # 0.4 semis
          "oscillator_1_blend": 0.5, "oscillator_1_level": 0.5 ** 0.5,
          "envelope_1_attack": (0.02 / 32) ** 0.25, "envelope_1_decay": (0.2 / 32) ** 0.25,
          "envelope_1_sustain": 0.3, "envelope_1_release": (0.6 / 32) ** 0.25}
SECRET_PHYS = dict(frame=2.0, detune_semis=0.4, blend=0.5, level=0.5, attack=0.02, decay=0.2, sustain=0.3, release=0.6)
if TABLE:
    tbl = torch.load(TABLE); TABLE_T = tbl["table"]; F_T = TABLE_T.shape[0]
    SECRET_FRAME_RAW = 0.30                                   # between keyframes on purpose
    SECRET["oscillator_1_wave_frame"] = SECRET_FRAME_RAW
    SECRET_PHYS["frame"] = SECRET_FRAME_RAW * (F_T - 1)
    from extract_table import load_clean
    def make_synth(voices): return WavetableSynth(TABLE_T, SR, voices=voices, bounds=B)
else:
    def make_synth(voices): return WavetableSynth(basic_shapes(), SR, voices=voices, bounds=B)

def vital_render(params):
    if TABLE: vital, _ = load_clean(tbl["preset"])            # preset table, everything else off
    else: vital = P.load("Vital")
    if DETERMINISTIC: P.set_param(vital, "oscillator_1_phase_randomization", 0.0)
    elif TABLE: P.set_param(vital, "oscillator_1_phase_randomization", 1.0)
    for k, v in params.items(): P.set_param(vital, k, v)
    y = P.render_notes(vital, midi, 10.0, sr=SR, tail=0.0).mean(axis=1)[:L]
    return np.pad(y, (0, max(0, L - len(y)))).astype(np.float32)

def recover(tgt, voices=3, steps=400, lr=0.03, restarts=3):
    s_ = make_synth(voices).to(DEV); tgt = torch.tensor(tgt, device=DEV)
    if DETERMINISTIC: s_.voice_phase.zero_(); s_.random_phase = False
    best = None
    for r in range(restarts):
        p = {k: v.to(DEV).requires_grad_(True) for k, v in init_raw(s_, seed=r).items()}
        def sweep():
            with torch.no_grad():
                cands = torch.logit(torch.linspace(0.01, 0.99, 40)); ls = []
                for c in cands: p["frame"].fill_(float(c)); ls.append(float(LOSSFN(s_.render(p, notes, L)[0, 0], tgt)))
                p["frame"].fill_(float(cands[int(np.argmin(ls))]))
        opt = torch.optim.Adam(p.values(), lr=lr); t0 = time.time()
        for i in range(steps):
            if i in (0, steps // 3): sweep()
            opt.zero_grad(); loss = LOSSFN(s_.render(p, notes, L)[0, 0], tgt); loss.backward(); opt.step()
        print(f"    restart {r}: loss={float(loss):.4f}  frame={float(s_.physical(p)['frame']):.2f}  ({(time.time()-t0)/steps*1000:.0f} ms/step)", flush=True)
        if best is None or float(loss) < best[0]: best = (float(loss), {k: v.detach() for k, v in p.items()})
    return s_, best[1], best[0]

def ltas_bands(a, b):
    def ltas(x, n=4096):
        X = np.abs(np.fft.rfft(x[:len(x)//n*n].reshape(-1, n) * np.hanning(n), axis=1)).mean(0); return 20*np.log10(X+1e-9), np.fft.rfftfreq(n, 1/SR)
    La, f = ltas(a); Lb, _ = ltas(b); off = 20*np.log10(np.sqrt((a**2).mean())/np.sqrt((b**2).mean()))
    return {f"{lo}-{hi}": round(float((La[m]-Lb[m]).mean() - off), 1) for lo, hi in [(300,1000),(1000,3000),(3000,8000),(8000,16000)] for m in [(f>=lo)&(f<hi)]}
def env_corr(a, b, w=int(0.005*SR)):
    e = lambda x: np.sqrt((x[:len(x)//w*w].reshape(-1, w)**2).mean(1)); return float(np.corrcoef(e(a), e(b))[0, 1])

if __name__ == "__main__":
    print(f"  loss={LOSS}  voices={VOICES}  deterministic={DETERMINISTIC}  clip={CLIP}\n  1) Vital renders the secret patch"); ref = vital_render(SECRET)
    sf.write(os.path.join(C.HERE, f"loop_ref_vital_{VOICES}v_{CLIP}.wav"), ref, SR)
    print("  2) recover on the torch twin from that audio (frame free)")
    s_, p, loss = recover(ref, voices=VOICES)
    got = s_.physical(p)
    print(f"\n  {'param':14s} {'secret':>8s} {'recovered':>10s}")
    for k in SECRET_PHYS:
        if VOICES == 1 and k in ("detune_semis", "blend"): continue
        v = float(got[k]) / (LEVEL_CAL if k == "level" else 1.0)
        if k == "frame" and TABLE: print(f"  {'wave_frame raw':14s} {SECRET_FRAME_RAW:8.3f} {v / (F_T - 1):10.3f}")          # torch level -> Vital level
        print(f"  {k:14s} {SECRET_PHYS[k]:8.3f} {v:10.3f}")
    print("  3) export to Vital and render")
    vp = s_.vital_params(p); vp["oscillator_1_level"] = min(1.0, (float(got["level"]) / LEVEL_CAL) ** 0.5)
    out = vital_render(vp)
    sf.write(os.path.join(C.HERE, f"loop_out_vital_{VOICES}v_{CLIP}.wav"), out, SR)
    print(f"  Vital(secret) vs Vital(recovered):  LTAS diff dB {ltas_bands(out, ref)}   envelope corr {env_corr(out, ref):.3f}")
    with torch.no_grad(): tw = s_.render({k: v.to(DEV) for k, v in p.items()}, notes, L)[0, 0].cpu().numpy()
    print(f"  torch twin vs Vital(secret):        LTAS diff dB {ltas_bands(tw, ref)}   envelope corr {env_corr(tw, ref):.3f}")
    # what a mismatch would look like: Vital at its init patch vs the secret
    init = vital_render({k: 0.0 if "voices" in k else v for k, v in {**({"oscillator_1_wave_frame": 0.0} if TABLE else {}), "oscillator_1_unison_voices": 0.0, "oscillator_1_unison_detune": 0.4472, "oscillator_1_blend": 0.8, "oscillator_1_level": 0.7071, "envelope_1_attack": 0.0629, "envelope_1_decay": 0.4204, "envelope_1_sustain": 1.0, "envelope_1_release": 0.2302}.items()})
    print(f"  (baseline) Vital(init) vs secret:  LTAS diff dB {ltas_bands(init, ref)}   envelope corr {env_corr(init, ref):.3f}")
