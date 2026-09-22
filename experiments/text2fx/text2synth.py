"""Text -> synth + FX, end to end differentiable.

clone (wavetable osc, unison off, amp env, filter + filter env)  ->  GRAFX chain (fxgraph.Chain; default
EQ -> comp -> drive -> pwtanh -> chorus -> transient -> gate -> delay -> reverb; override with T2_CHAIN="eq,dist,reverb")
->  loudness-normalise  ->  CLAP cosine to the prompt.  Every parameter gets a gradient.

Unlike the FX-only runs, the generator can only produce synth sounds, so there is no
off-manifold cheat; what remains is that a prompt gives a direction, and parameters
run to their (musical) rails. Level is fixed so silence is not an option.
"""
import os, sys, json, time, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, soundfile as sf
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from synth import WavetableSynth, sustained_notes, bounds_from_notes, init_raw, raw_from_physical
from fxgraph import Chain

SR = 48000; L = 10 * SR; DEV = C.DEVICE
if DEV == "mps": torch.mps.set_per_process_memory_fraction(0.5)
STEPS, RESTARTS, LAM_PRI, LAM_GS = 300, 3, 0.05, 0.02
OUT = os.path.join(C.HERE, "text2synth"); notes = sustained_notes(); B = bounds_from_notes(notes)
TBL = torch.load(os.path.join(C.HERE, "tables", "classic_fade.pt")); TABLE = TBL["table"]; F_T = TABLE.shape[0]
FIXED_LEVEL = 0.35 * 0.694                      # not optimised: silence must not be reachable

CHAIN_TYPES = os.environ.get("T2_CHAIN", "eq,comp,drive,pwtanh,chorus,transient,gate,delay,reverb").split(",")   # every node in the race: +0.02 mean over the 5-node chain, +10 % time
CHAIN = Chain(CHAIN_TYPES, L, DEV)

def build_fx(): return CHAIN.procs, CHAIN.rd
def init_fx(seed): return CHAIN.init(seed)
def fx_prior(p): return CHAIN.prior(p)
def render_fx(y, pf):
    """synth (1,1,L) -> (fx audio (L,), regulariser: gain-staging + delay radii)."""
    return CHAIN.render(y, pf)

def init_synth(s_, seed):
    p = init_raw(s_, seed=seed)                                     # osc + amp env family
    logit = lambda x: torch.logit(torch.tensor(float(x)).clamp(1e-3, 1 - 1e-3))
    p["frame"] = logit([0.2, 0.5, 0.8][seed % 3])
    fam = [dict(cut=0.5, res=0.2, amt=0.4, a=0.002, d=0.2, su=0.3, r=0.3), dict(cut=0.4, res=0.5, amt=0.6, a=0.005, d=0.5, su=0.5, r=0.5), dict(cut=0.7, res=0.1, amt=0.2, a=0.05, d=0.3, su=0.8, r=0.5)][seed % 3]
    p.update(cutoff=logit(fam["cut"]), resonance=logit(fam["res"]), fenv_amount=logit(fam["amt"]), fattack=logit(s_._t_inv("attack", fam["a"])),
             fdecay=logit(s_._t_inv("decay", fam["d"])), fsustain=logit(fam["su"]), frelease=logit(s_._t_inv("release", fam["r"])))
    p["level"] = raw_from_physical(s_, frame=0, detune_semis=0, blend=0, level=FIXED_LEVEL, attack=0.01, decay=0.1, sustain=1, release=0.1)["level"]
    if os.environ.get("T2_NOISE", "1") == "1": p["noise"] = logit(0.05)             # Vital's sample oscillator, quiet
    return {k: v.to(DEV).requires_grad_(k != "level") for k, v in p.items()}

def loudness_norm(w, target_rms=0.12):
    lw = C.loudness(w); return w * (target_rms / (lw + 1e-3 * target_rms))

def describe(s_, ps, pf):
    ph = s_.physical(ps)
    return dict(synth={k: round(float(v), 3) for k, v in ph.items() if k not in ("attack_power", "decay_power", "release_power", "fattack_power", "fdecay_power", "frelease_power")},
                **CHAIN.describe(pf))

OPT = os.environ.get("T2_OPT", "sh")            # "sh": successive halving over 8 inits (default); "adam": 3 x 300 restarts
ROUNDS = tuple(tuple(int(x) for x in r.split(":")) for r in os.environ.get("T2_ROUNDS", "30:4,60:2,220:1").split(","))

def run(prompt):
    """-> (score, ps, pf, synth, dry (L,), wet (L,), extra). The synth is shared by all candidates (it holds no state
    but the table and phases), so a candidate is just its parameter dicts."""
    torch.manual_seed(0)                                   # seeded phase/noise draws: a run repeats bit for bit (see ladder.run)
    T = C.text_emb(prompt); s_ = WavetableSynth(TABLE, SR, voices=1, bounds=B, interpolation=1).to(DEV)
    def loss_of(ps, pf):
        y = s_.render(ps, notes, L); out, gs = render_fx(y, pf); w = loudness_norm(out)
        return -(C.embed(w) @ T.T).squeeze() + LAM_PRI * fx_prior(pf) + LAM_GS * gs
    def evaluate(ps, pf):
        y = s_.render(ps, notes, L); out, _ = render_fx(y, pf); w = loudness_norm(out); score = float((C.embed(w) @ T.T).squeeze())
        robust = float(sum((C.embed(torch.roll(w, int(sh * SR))) @ T.T).squeeze() for sh in (0.0, 0.37, 0.71, 1.9)) / 4)
        return dict(score=score, robust=robust), y[0, 0].detach(), out.detach()
    t0 = time.time()
    if OPT == "sh":
        from optim import Candidate, successive_halving
        import contextlib
        @contextlib.contextmanager
        def fixed_phases():
            s_.random_phase = False
            try: yield
            finally: s_.random_phase = True
        cands = [Candidate(init_synth(s_, r), init_fx(r), label=f"c{r}") for r in range(8)]
        frontier, total = successive_halving(cands, lambda c: loss_of(c.ps, c.pf), lambda c: evaluate(c.ps, c.pf)[0], rounds=ROUNDS, polish_ctx=fixed_phases)
        ev, c = frontier[0]
        with torch.no_grad(): ev, dry, wet = evaluate(c.ps, c.pf)
        print(f"    frontier: " + "  ".join(f"{c.label} {e['score']:+.3f} (robust {e['robust']:+.3f})" for e, c in frontier) + f"   {total} steps in {time.time()-t0:.0f}s", flush=True)
        ps = {k: v.detach() for k, v in c.ps.items()}; pf = {t: {k: v.detach() for k, v in d.items()} for t, d in c.pf.items()}
        return ev["score"], ps, pf, s_, dry, wet, dict(robust=ev["robust"], frontier=[dict(label=c.label, **e) for e, c in frontier], steps=total, opt="sh")
    best = None
    for r in range(RESTARTS):
        ps, pf = init_synth(s_, r), init_fx(r)
        opt = torch.optim.Adam([v for v in ps.values() if v.requires_grad] + [v for d in pf.values() for v in d.values()], lr=0.02); t1 = time.time()
        for i in range(STEPS):
            opt.zero_grad(); loss = loss_of(ps, pf); loss.backward(); opt.step()
        with torch.no_grad(): ev, dry, wet = evaluate(ps, pf)
        print(f"    restart {r}: score={ev['score']:+.4f} robust={ev['robust']:+.4f}  ({(time.time()-t1)/STEPS*1000:.0f} ms/step)", flush=True)
        if best is None or ev["score"] > best[0]: best = (ev["score"], {k: v.detach() for k, v in ps.items()}, {t: {k: v.detach() for k, v in d.items()} for t, d in pf.items()}, s_, dry, wet, dict(robust=ev["robust"], steps=STEPS * RESTARTS, opt="adam"))
    return best

if __name__ == "__main__":
    prompts = sys.argv[1:] or ["this sound is underwater", "this sound is woody", "this sound is church bells ringing", "this sound is an african dance on fire", "this sound is folkish and melodic"]
    for prompt in prompts:
        slug = "".join(ch if ch.isalnum() else "_" for ch in prompt.replace("this sound is ", ""))[:32].strip("_")
        print(f"  '{prompt}'", flush=True)
        score, ps, pf, s_, dry, wet, extra = run(prompt)
        d = describe(s_, ps, pf); vp = s_.vital_params(ps)
        sf.write(os.path.join(OUT, f"{slug}_synth_dry.wav"), (dry / (dry.abs().max() + 1e-9) * 0.5).cpu().numpy(), SR)
        sf.write(os.path.join(OUT, f"{slug}_synth_fx.wav"), (wet / (wet.abs().max() + 1e-9) * 0.5).cpu().numpy(), SR)
        json.dump({"prompt": prompt, "score": score, "chain": CHAIN_TYPES, "vital_params": vp, "describe": d, "raw": {k: float(v) for k, v in ps.items()}, "fx_raw": {t: {k: v.cpu().tolist() for k, v in dd.items()} for t, dd in pf.items()}, **extra}, open(os.path.join(OUT, f"{slug}.json"), "w"), indent=1)
        sy = d["synth"]
        print(f"  BEST {score:+.4f} | frame {sy['frame']:.1f}/{F_T-1} cutoff {sy['cutoff_raw']:.2f} res {sy['resonance']:.2f} envamt {sy['fenv_amount']:.2f} "
              f"amp A{sy['attack']*1000:.0f}ms D{sy['decay']:.2f} S{sy['sustain']:.2f} R{sy['release']:.2f} filt D{sy['fdecay']:.2f} S{sy['fsustain']:.2f} "
              f"| {', '.join(f'{k}={v}' for k, v in d.items() if k != 'synth')}", flush=True)
    print("ALL DONE")
