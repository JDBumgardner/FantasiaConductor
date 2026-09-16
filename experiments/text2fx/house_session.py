"""Instrument first, then prompts on top of it — on a house hook.

1. Render a GM soundfont instrument (cello / brass) playing the hook -> reference.
2. Twin recovers a Vital patch for it (the "synth cello" / "synth brass").
3. For each prompt: start FROM that patch, optimise synth + FX toward the prompt with a
   locality term to the instrument's own dry render, so the result is the instrument
   under the words rather than a new sound.
"""
import os, sys, json, time, math, types, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, soundfile as sf
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], "1"]
import common as C, closed_loop_filter as CL, text2synth as T2
from fantasia_core.engine.midi_render import MidiRenderer
from synth import WavetableSynth, bounds_from_notes, raw_from_physical
from grafx.render import render_grafx

SR = 48000; L = 10 * SR; DEV = C.DEVICE; OUT = os.path.join(C.HERE, "house")
N = lambda p, s, d, v: types.SimpleNamespace(pitch=p, start=s, duration=d, velocity=v)

def house_hook(bpm=124, loops=2):
    """Two-bar minor-key lead hook with off-beat stabs, looped. (pitch, beat, beats, vel)"""
    F3, Ab3, Bb3, C4, Eb4, G3 = 53, 56, 58, 60, 63, 55
    bar1 = [(F3, 0, .75, 110), (Ab3, 1.5, .5, 96), (C4, 2, .5, 104), (Bb3, 2.75, .25, 90), (Ab3, 3.5, .5, 100)]
    bar2 = [(F3, 0, .5, 110), (Eb4, 1, .75, 104), (C4, 2.5, .5, 96), (Ab3, 3, .5, 100), (G3, 3.5, .5, 92)]
    beat = 60 / bpm; out = []
    for k in range(loops):
        for bar, ev in ((0, bar1), (1, bar2)):
            for p, b, d, v in ev: out.append((p, (k * 8 + bar * 4 + b) * beat, d * beat * 0.9, v))
    return out
NOTES = house_hook(); MIDI = [N(*n) for n in NOTES]; B = bounds_from_notes(NOTES)
for m in (CL, T2): m.notes, m.B, m.midi = NOTES, B, MIDI            # experiment modules keep these as globals
CL.midi = MIDI

def reference(program):
    clip = types.SimpleNamespace(notes=MIDI, duration=10.0, content_type="midi", id="ref")
    y = MidiRenderer("assets/soundfonts/GeneralUser-GS-v1.471.sf2", SR).render(clip, program).mean(axis=1)[:L]
    y = np.pad(y, (0, max(0, L - len(y)))); return (y / (np.abs(y).max() + 1e-9) * 0.5).astype(np.float32)

def make_instrument(name, program):
    t0 = time.time(); ref = reference(program); sf.write(os.path.join(OUT, f"{name}_ref.wav"), ref, SR)
    s_, p = CL.recover(ref, steps=250, restarts=6); got = s_.physical(p)
    vp = s_.vital_params(p); vp["oscillator_1_level"] = min(1.0, (float(got["level"]) / CL.LEVEL_CAL) ** 0.5)
    out = CL.vital_render(vp); sf.write(os.path.join(OUT, f"{name}_vital.wav"), out / (np.abs(out).max() + 1e-9) * 0.5, SR)
    with torch.no_grad(): dry = s_.render({k: v.to(DEV) for k, v in p.items()}, NOTES, L)[0, 0].cpu().numpy()
    sf.write(os.path.join(OUT, f"{name}_clone.wav"), dry / (np.abs(dry).max() + 1e-9) * 0.5, SR)
    json.dump({"vital_params": vp, "physical": {k: round(float(v), 4) for k, v in got.items()}, "raw": {k: float(v) for k, v in p.items()}},
              open(os.path.join(OUT, f"{name}_patch.json"), "w"), indent=1)
    print(f"  [{name}] {time.time()-t0:.0f}s  Vital-vs-ref corr {CL.env_corr(out, ref):.2f} bands {CL.band_diff(out, ref)} | "
          f"frame {int(got['frame']*256/(s_.F-1))} cutoff {float(got['cutoff_raw']):.2f} res {float(got['resonance']):.2f} envamt {float(got['fenv_amount']):.2f} "
          f"amp A{float(got['attack'])*1000:.0f}ms D{float(got['decay']):.2f} S{float(got['sustain']):.2f} R{float(got['release']):.2f}", flush=True)
    return p

def prompt_on(name, p_inst, prompt, lam_loc=0.05, steps=300, restarts=3):
    slug = "".join(ch if ch.isalnum() else "_" for ch in prompt)[:28].strip("_")
    T = C.text_emb("this sound is " + prompt); procs, rd = T2.build_fx(); best = None
    s_ = WavetableSynth(CL.TABLE, SR, voices=1, bounds=B, interpolation=CL.TBL.get("interpolation", 1)).to(DEV)
    with torch.no_grad(): anchor = s_.render({k: v.to(DEV) for k, v in p_inst.items()}, NOTES, L)[0, 0]
    for r in range(restarts):
        g = torch.Generator().manual_seed(r)
        ps = {k: (v.clone().to(DEV) + (0.15 * torch.randn((), generator=g).to(DEV) if r else 0)).requires_grad_(k != "level") for k, v in p_inst.items()}
        pf = T2.init_fx(r)
        opt = torch.optim.Adam([v for v in ps.values() if v.requires_grad] + [v for d in pf.values() for v in d.values()], lr=0.02); t0 = time.time()
        for i in range(steps):
            opt.zero_grad()
            y = s_.render(ps, NOTES, L); out, inter, _ = render_grafx(procs, y, pf, rd, input_signal_grad=True)
            gs = sum(v["gain_reg"].sum() for v in inter if isinstance(v, dict) and "gain_reg" in v)
            w = T2.loudness_norm(out[0, 0])
            loss = (-(C.embed(w) @ T.T).squeeze() + T2.LAM_PRI * T2.fx_prior(pf) + T2.LAM_GS * gs
                    + lam_loc * C.mrstft(T2.loudness_norm(y[0, 0]), T2.loudness_norm(anchor)))     # stay the instrument
            loss.backward(); opt.step()
        with torch.no_grad():
            y = s_.render(ps, NOTES, L); out, _, _ = render_grafx(procs, y, pf, rd); w = T2.loudness_norm(out[0, 0])
            score = float((C.embed(w) @ T.T).squeeze()); dist = float(C.mrstft(T2.loudness_norm(y[0, 0]), T2.loudness_norm(anchor)))
        print(f"    restart {r}: score={score:+.4f} dist-to-instrument={dist:.2f} ({(time.time()-t0)/steps*1000:.0f} ms/step)", flush=True)
        if best is None or score > best[0]: best = (score, dist, {k: v.detach() for k, v in ps.items()}, {t: {k: v.detach() for k, v in d.items()} for t, d in pf.items()}, y[0, 0].detach(), out[0, 0].detach())
    score, dist, ps, pf, dry, wet = best; d = T2.describe(s_, ps, pf); vp = s_.vital_params(ps)
    sf.write(os.path.join(OUT, f"{name}__{slug}_dry.wav"), (dry / (dry.abs().max() + 1e-9) * 0.5).cpu().numpy(), SR)
    sf.write(os.path.join(OUT, f"{name}__{slug}_fx.wav"), (wet / (wet.abs().max() + 1e-9) * 0.5).cpu().numpy(), SR)
    json.dump({"prompt": prompt, "score": score, "dist": dist, "vital_params": vp, "describe": d}, open(os.path.join(OUT, f"{name}__{slug}.json"), "w"), indent=1)
    sy = d["synth"]
    print(f"  [{name} + '{prompt}'] score {score:+.3f} dist {dist:.2f} | cutoff {sy['cutoff_raw']:.2f} res {sy['resonance']:.2f} envamt {sy['fenv_amount']:.2f} "
          f"amp A{sy['attack']*1000:.0f}ms D{sy['decay']:.2f} S{sy['sustain']:.2f} R{sy['release']:.2f} filt D{sy['fdecay']:.2f} S{sy['fsustain']:.2f} | eq {d['eq']} drive {d['drive_db']} reverb {d['reverb_mix']}", flush=True)

if __name__ == "__main__":
    PROMPTS = ["airy, soft and light, whistling through the trees", "under water, in the deep depths, burbling"]
    for name, prog in (("cello", 42), ("brass", 61)):
        p = make_instrument(name, prog)
        for pr in PROMPTS: prompt_on(name, p, pr)
    print("ALL DONE")
