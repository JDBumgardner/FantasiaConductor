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

LOC_DRY = float(os.environ.get("T2_LOC_DRY", "0.05"))          # locality to the instrument on the dry synth output (the patch stays the instrument)
LOC_WET = float(os.environ.get("T2_LOC_WET", "0.0"))           # ... and on the heard output (so FX moves are costed like synth moves)
LR_EQ = os.environ.get("T2_LR_EQ", "0") == "1"                 # per-parameter lr from measured embedding sensitivity
EXCERPT_S = float(os.environ.get("T2_EXCERPT", "0"))
WARM_K = int(os.environ.get("T2_WARM", "2"))                    # this many candidates come from earlier results on the same instrument (nearest prompts); 0 = cold
WARM_DIR = os.environ.get("T2_WARM_DIR", "")                    # where to look for them (default: the output folder and everything under it)
ROUNDS_WARM = tuple(tuple(int(x) for x in r.split(":")) for r in os.environ.get("T2_ROUNDS_WARM", "15:3,30:2,110:1").split(","))   # used when warm candidates were found: 370 steps beat 700 cold on 6/6 cells (mean 0.341 vs 0.322)           # > 0: early halving stages optimise on the first EXCERPT_S seconds of the hook (fewer notes to render)
OBJ = os.environ.get("T2_OBJ", "sim")                          # "sim": CLAP cosine to the prompt; "contrast": cosine(prompt) - cosine(anchor text), e.g. "a dark cello" minus "a cello"

def prompt_on(name, p_inst, prompt, lam_loc=None, steps=300, restarts=3, anchor_text=None):
    lam_loc = LOC_DRY if lam_loc is None else lam_loc
    slug = "".join(ch if ch.isalnum() else "_" for ch in prompt)[:28].strip("_")
    T_named = C.text_emb("this sound is " + prompt); best = None
    T_self = C.text_emb("this sound is " + anchor_text) if anchor_text else None
    T = T_named - T_self if (OBJ == "contrast" and T_self is not None) else T_named        # the optimised direction
    s_ = WavetableSynth(CL.TABLE, SR, voices=1, bounds=B, interpolation=CL.TBL.get("interpolation", 1)).to(DEV)
    p_dev = {k: v.to(DEV) for k, v in p_inst.items()}
    with torch.no_grad(): anchor = s_.render(p_dev, NOTES, L)[0, 0]; anchor_n = C.SpectralAnchor(T2.loudness_norm(anchor), ffts=(512, 2048))   # two resolutions: half the cost of four, same job
    cur = dict(notes=NOTES, anchor=anchor_n)                                           # what the loss currently renders (coarse-to-fine swaps this)
    short = [n for n in NOTES if n[1] < EXCERPT_S] if EXCERPT_S else None
    if short:
        with torch.no_grad(): a_s = C.SpectralAnchor(T2.loudness_norm(s_.render(p_dev, short, L)[0, 0]), ffts=(512, 2048))
    def stage_hook(si, last):
        if short: cur.update(notes=NOTES if last else short, anchor=anchor_n if last else a_s)
    def forward(ps, pf):
        y = s_.render(ps, cur["notes"], L); out, gs = T2.render_fx(y, pf); return y, out, gs, T2.loudness_norm(out)
    def loss_of(ps, pf):
        y, out, gs, w = forward(ps, pf); a = cur["anchor"]
        loc = lam_loc * C.mrstft(T2.loudness_norm(y[0, 0]), a) + (LOC_WET * C.mrstft(w, a) if LOC_WET else 0.0)
        return (-(C.embed(w) @ T.T).squeeze() + T2.LAM_PRI * T2.fx_prior(pf) + T2.LAM_GS * gs + loc)
    def evaluate(ps, pf):
        saved = dict(cur); cur.update(notes=NOTES, anchor=anchor_n)                         # always judge on the whole hook
        y, out, gs, w = forward(ps, pf); e = C.embed(w); score = float((e @ T.T).squeeze()); cur.update(saved)
        robust = float(sum((C.embed(torch.roll(w, int(sh * SR))) @ T.T).squeeze() for sh in (0.0, 0.37, 0.71, 1.9)) / 4)
        ev = dict(score=score, robust=robust, dist=float(C.mrstft(T2.loudness_norm(y[0, 0]), anchor_n)))
        cur.update(saved)
        if T_self is not None: ev.update(named=float((e @ T_named.T).squeeze()), self=float((e @ T_self.T).squeeze()))
        return ev, y[0, 0].detach(), out.detach()
    def warm_candidates(k):
        """Earlier results for this instrument whose prompts CLAP finds closest to this one: their synth parameters, and the
        FX parameters of the node types the current chain shares with theirs; the rest of the FX from a fresh init."""
        import glob
        pool = []
        for f in glob.glob(os.path.join(WARM_DIR or OUT, "**", f"{name}__*.json"), recursive=True):
            try: d = json.load(open(f))
            except Exception: continue
            if d.get("prompt") == prompt or "raw" not in d or "fx_raw" not in d: continue
            if any(q["prompt"] == d["prompt"] and q["score"] >= d["score"] for q in pool): continue        # one result per prompt: the best
            pool = [q for q in pool if q["prompt"] != d["prompt"]] + [d]
        if not pool: return []
        te = torch.cat([C.text_emb("this sound is " + d["prompt"]) for d in pool]); sims = (te @ T_named.T).squeeze(-1)
        picks = [pool[i] for i in sims.argsort(descending=True)[:k].tolist()]
        out = []
        for i, d in enumerate(picks):
            ps = {kk: torch.tensor(float(v), device=DEV).requires_grad_(kk != "level") for kk, v in d["raw"].items()}
            if CL.NOISE and "noise" not in ps: ps["noise"] = torch.logit(torch.tensor(0.05, device=DEV)).requires_grad_(True)
            pf = T2.init_fx(100 + i)
            for t, dd in d["fx_raw"].items():
                if t in pf and all(kk in pf[t] and list(torch.tensor(v).shape) == list(pf[t][kk].shape) for kk, v in dd.items()):
                    pf[t] = {kk: torch.tensor(v, device=DEV).requires_grad_(True) for kk, v in dd.items()}
            out.append((ps, pf, f"w{i}:{d['prompt'][:12]}"))
        print("    warm start from: " + ", ".join(f"'{d['prompt']}' ({float(s):+.2f})" for d, s in zip(picks, sims.sort(descending=True).values[:k].tolist())), flush=True)
        return out

    def make_candidate(r, noise):
        g = torch.Generator().manual_seed(r)
        ps = {k: (v.clone().to(DEV) + (noise * torch.randn((), generator=g).to(DEV) if noise else 0)).requires_grad_(k != "level") for k, v in p_inst.items()}
        if CL.NOISE and "noise" not in ps:                                            # instruments recovered before the noise source existed start it quiet
            ps["noise"] = torch.logit(torch.tensor(0.05, device=DEV)).requires_grad_(True)
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
        lr_mult = {}
        if LR_EQ:
            from optim import sensitivity_lr
            def embed_fn(ps_, pf_):
                y_ = s_.render(ps_, NOTES, L); o_, _ = T2.CHAIN.render(y_, pf_, checkpoint=False); return C.embed(T2.loudness_norm(o_))[0]
            s_.random_phase = False; lr_mult, sens = sensitivity_lr(Candidate(*make_candidate(0, 0)), embed_fn); s_.random_phase = True
            print("    lr multipliers: " + "  ".join(f"{k.split('/')[-1]}×{m:.2g}" for k, m in sorted(lr_mult.items(), key=lambda kv: -abs(math.log(kv[1]))) if abs(m - 1) > 0.05), flush=True)
        warm = warm_candidates(WARM_K) if WARM_K else []
        rounds = ROUNDS_WARM if warm else ROUNDS; n_cands = (len(warm) + 2) if warm else restarts       # warm: the retrieved ones + the instrument + one perturbed
        for attempt in (0, 1):
            cands = [Candidate(ps_, pf_, label=lab, lr_mult=lr_mult) for ps_, pf_, lab in warm]
            cands += [Candidate(*make_candidate(r, [0, 0.3, 0.15, 0.5, 0.3, 0.5, 0.15, 0.8][r % 8]), label=f"c{r}", lr_mult=lr_mult) for r in range(n_cands - len(cands))]
            try:
                frontier, total = successive_halving(cands, lambda c: loss_of(c.ps, c.pf), lambda c: evaluate(c.ps, c.pf)[0], rounds=rounds, lr=0.02, polish_ctx=fixed_phases, stage_hook=stage_hook); break
            except RuntimeError as e:                                    # out of memory: retry once with the synth's note groups checkpointed (-240 MB peak, +30 % time)
                if "out of memory" not in str(e) or attempt: raise
                print(f"    OOM — retrying with synth checkpointing", flush=True); del cands; free_cache(); s_.checkpoint = True
        ev, c = frontier[0]; ps, pf = c.ps, c.pf
        with torch.no_grad(): ev, dry, wet = evaluate(ps, pf)
        score, dist = ev["score"], ev["dist"]; ps = {k: v.detach() for k, v in ps.items()}; pf = {t: {k: v.detach() for k, v in d.items()} for t, d in pf.items()}
        extra = dict(robust=ev["robust"], frontier=[dict(label=c.label, **e) for e, c in frontier], steps=total, opt="sh", objective=OBJ, anchor_text=anchor_text, warm=[lab for _, _, lab in warm], loc_dry=lam_loc, loc_wet=LOC_WET, lr_eq=LR_EQ, lr_mult=lr_mult, **({"named": ev["named"], "self": ev["self"]} if T_self is not None else {}))
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
        p0 = lambda k: p_inst[k].to(DEV) if k in p_inst else torch.logit(torch.tensor(0.05, device=DEV))      # noise: the instrument patch may predate it
        ps_half = {k: (p0(k) + 0.5 * (v - p0(k))) for k, v in ps.items()}
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
