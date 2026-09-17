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

SR = 48000; L = 10 * SR; DEV = C.DEVICE; OUT = os.path.join(C.HERE, "house")
OPT = os.environ.get("T2_OPT", "sh")                            # "sh": successive halving over 8 inits (default); "adam": 3 x 300 restarts
ROUNDS = tuple(tuple(int(x) for x in r.split(":")) for r in os.environ.get("T2_ROUNDS", "30:4,60:2,220:1").split(","))
TAG = ("" if T2.CHAIN_TYPES == ["eq", "dist", "reverb"] else "_" + "-".join(T2.CHAIN_TYPES)) + ("" if OPT == "sh" else "_" + OPT)
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

OBJ = os.environ.get("T2_OBJ", "sim")                          # "sim": CLAP cosine to the prompt; "contrast": cosine(prompt) - cosine(anchor text), e.g. "a dark cello" minus "a cello"

def prompt_on(name, p_inst, prompt, lam_loc=0.05, steps=300, restarts=3, anchor_text=None):
    slug = "".join(ch if ch.isalnum() else "_" for ch in prompt)[:28].strip("_")
    T_named = C.text_emb("this sound is " + prompt); best = None
    T_self = C.text_emb("this sound is " + anchor_text) if anchor_text else None
    T = T_named - T_self if (OBJ == "contrast" and T_self is not None) else T_named        # the optimised direction
    s_ = WavetableSynth(CL.TABLE, SR, voices=1, bounds=B, interpolation=CL.TBL.get("interpolation", 1)).to(DEV)
    with torch.no_grad(): anchor = s_.render({k: v.to(DEV) for k, v in p_inst.items()}, NOTES, L)[0, 0]; anchor_n = T2.loudness_norm(anchor)
    def forward(ps, pf):
        y = s_.render(ps, NOTES, L); out, gs = T2.render_fx(y, pf); return y, out, gs, T2.loudness_norm(out)
    def loss_of(ps, pf):
        y, out, gs, w = forward(ps, pf)
        return (-(C.embed(w) @ T.T).squeeze() + T2.LAM_PRI * T2.fx_prior(pf) + T2.LAM_GS * gs + lam_loc * C.mrstft(T2.loudness_norm(y[0, 0]), anchor_n))
    def evaluate(ps, pf):
        y, out, gs, w = forward(ps, pf); e = C.embed(w); score = float((e @ T.T).squeeze())
        robust = float(sum((C.embed(torch.roll(w, int(sh * SR))) @ T.T).squeeze() for sh in (0.0, 0.37, 0.71, 1.9)) / 4)
        ev = dict(score=score, robust=robust, dist=float(C.mrstft(T2.loudness_norm(y[0, 0]), anchor_n)))
        if T_self is not None: ev.update(named=float((e @ T_named.T).squeeze()), self=float((e @ T_self.T).squeeze()))
        return ev, y[0, 0].detach(), out.detach()
    def make_candidate(r, noise):
        g = torch.Generator().manual_seed(r)
        ps = {k: (v.clone().to(DEV) + (noise * torch.randn((), generator=g).to(DEV) if noise else 0)).requires_grad_(k != "level") for k, v in p_inst.items()}
        return ps, T2.init_fx(r)
    t0 = time.time()
    if OPT == "sh":                                                     # successive halving + cosine + L-BFGS polish
        from optim import Candidate, successive_halving, free_cache
        import contextlib
        @contextlib.contextmanager
        def fixed_phases():
            s_.random_phase = False
            try: yield
            finally: s_.random_phase = True
        for attempt in (0, 1):
            cands = [Candidate(*make_candidate(r, [0, 0.15, 0.15, 0.3, 0.3, 0.5, 0.5, 0.8][r % 8]), label=f"c{r}") for r in range(restarts)]
            try:
                frontier, total = successive_halving(cands, lambda c: loss_of(c.ps, c.pf), lambda c: evaluate(c.ps, c.pf)[0], rounds=ROUNDS, lr=0.02, polish_ctx=fixed_phases); break
            except RuntimeError as e:                                    # out of memory: retry once with the synth's note groups checkpointed (-240 MB peak, +30 % time)
                if "out of memory" not in str(e) or attempt: raise
                print(f"    OOM — retrying with synth checkpointing", flush=True); del cands; free_cache(); s_.checkpoint = True
        ev, c = frontier[0]; ps, pf = c.ps, c.pf
        with torch.no_grad(): ev, dry, wet = evaluate(ps, pf)
        score, dist = ev["score"], ev["dist"]; ps = {k: v.detach() for k, v in ps.items()}; pf = {t: {k: v.detach() for k, v in d.items()} for t, d in pf.items()}
        extra = dict(robust=ev["robust"], frontier=[dict(label=c.label, **e) for e, c in frontier], steps=total, opt="sh", objective=OBJ, anchor_text=anchor_text, **({"named": ev["named"], "self": ev["self"]} if T_self is not None else {}))
        print(f"    frontier: " + "  ".join(f"{c.label} {e['score']:+.3f} (robust {e['robust']:+.3f}, dist {e['dist']:.2f})" for e, c in frontier) + f"   {total} steps in {time.time()-t0:.0f}s", flush=True)
    else:
        for r in range(restarts):
            ps, pf = make_candidate(r, 0.15 if r else 0)
            opt = torch.optim.Adam([v for v in ps.values() if v.requires_grad] + [v for d in pf.values() for v in d.values()], lr=0.02); t1 = time.time()
            for i in range(steps):
                opt.zero_grad(); loss = loss_of(ps, pf); loss.backward(); opt.step()
            with torch.no_grad(): ev, dry_r, wet_r = evaluate(ps, pf)
            print(f"    restart {r}: score={ev['score']:+.4f} robust={ev['robust']:+.4f} dist-to-instrument={ev['dist']:.2f} ({(time.time()-t1)/steps*1000:.0f} ms/step)", flush=True)
            if best is None or ev["score"] > best[0]: best = (ev["score"], ev["dist"], {k: v.detach() for k, v in ps.items()}, {t: {k: v.detach() for k, v in d.items()} for t, d in pf.items()}, dry_r, wet_r, ev["robust"])
        score, dist, ps, pf, dry, wet, robust = best; extra = dict(robust=robust, steps=steps * restarts, opt="adam")
    d = T2.describe(s_, ps, pf); vp = s_.vital_params(ps)
    # amount knob: half-way from the instrument to the result (synth in raw space, every FX node half-way from bypass)
    with torch.no_grad():
        ps_half = {k: (p_inst[k].to(DEV) + 0.5 * (v - p_inst[k].to(DEV))) for k, v in ps.items()}
        ev_half, _, wet_half = evaluate(ps_half, T2.CHAIN.scale(pf, 0.5))
    extra["amount50"] = ev_half
    sf.write(os.path.join(OUT, f"{name}__{slug}{TAG}_fx_amt50.wav"), (wet_half / (wet_half.abs().max() + 1e-9) * 0.5).cpu().numpy(), SR)
    sf.write(os.path.join(OUT, f"{name}__{slug}{TAG}_dry.wav"), (dry / (dry.abs().max() + 1e-9) * 0.5).cpu().numpy(), SR)
    sf.write(os.path.join(OUT, f"{name}__{slug}{TAG}_fx.wav"), (wet / (wet.abs().max() + 1e-9) * 0.5).cpu().numpy(), SR)
    json.dump({"prompt": prompt, "score": score, "dist": dist, "chain": T2.CHAIN_TYPES, "vital_params": vp, "describe": d, "raw": {k: float(v) for k, v in ps.items()},
               "fx_raw": {t: {k: v.cpu().tolist() for k, v in dd.items()} for t, dd in pf.items()}, **extra}, open(os.path.join(OUT, f"{name}__{slug}{TAG}.json"), "w"), indent=1)
    if os.environ.get("T2_VERIFY", "1") == "1":                      # re-render the saved parameters with a fresh synth + chain: the result must reproduce its score
        with torch.no_grad():
            import fxgraph as FG
            s2 = WavetableSynth(CL.TABLE, SR, voices=1, bounds=B, interpolation=CL.TBL.get("interpolation", 1)).to(DEV); s2.random_phase = False
            ch2 = FG.Chain(T2.CHAIN_TYPES, L, DEV)
            y2 = s2.render({k: torch.tensor(float(v), device=DEV) for k, v in ps.items()}, NOTES, L); out2, _ = ch2.render(y2, {t: {k: v.clone() for k, v in dd.items()} for t, dd in pf.items()}, checkpoint=False)
            s_fresh = float((C.embed(T2.loudness_norm(out2)) @ T.T).squeeze())
            s_.random_phase = False; y3 = s_.render(ps, NOTES, L); out3, _ = T2.render_fx(y3, pf); s_.random_phase = True
            s_same = float((C.embed(T2.loudness_norm(out3)) @ T.T).squeeze())
            def bands(x):
                X = torch.fft.rfft(x / x.abs().max()).abs() ** 2; f = torch.fft.rfftfreq(len(x), 1 / SR, device=x.device); tot = X.sum()
                return [round(float(10 * torch.log10(X[(f >= lo) & (f < hi)].sum() / tot + 1e-12))) for lo, hi in ((0, 300), (300, 1000), (1000, 3000), (3000, 8000), (8000, 20000))]
            extra["verify"] = dict(fresh=s_fresh, same_objects_fixed_phase=s_same)
            flag = "" if abs(s_fresh - score) < 0.03 else "   <<< DOES NOT REPRODUCE"
            print(f"    verify: eval {score:+.3f} | same synth+chain, fixed phases {s_same:+.3f} | fresh synth+chain {s_fresh:+.3f}{flag} | dry bands eval {bands(dry)} fresh {bands(y2[0, 0])}", flush=True)
    sy = d["synth"]
    print(f"  [{name} + '{prompt}'] score {score:+.3f} (robust {extra['robust']:+.3f}, at 50% amount {ev_half['score']:+.3f}" + (f", prompt {extra['named']:+.3f} vs anchor '{anchor_text}' {extra['self']:+.3f}" if T_self is not None else "") + f") dist {dist:.2f} | cutoff {sy['cutoff_raw']:.2f} res {sy['resonance']:.2f} envamt {sy['fenv_amount']:.2f} "
          f"amp A{sy['attack']*1000:.0f}ms D{sy['decay']:.2f} S{sy['sustain']:.2f} R{sy['release']:.2f} filt D{sy['fdecay']:.2f} S{sy['fsustain']:.2f} | {', '.join(f'{k}={v}' for k, v in d.items() if k != 'synth')}", flush=True)

if __name__ == "__main__":
    PROMPTS = ["airy, soft and light, whistling through the trees", "under water, in the deep depths, burbling"]
    for name, prog in (("cello", 42), ("brass", 61)):
        p = make_instrument(name, prog)
        for pr in PROMPTS: prompt_on(name, p, pr)
    print("ALL DONE")
