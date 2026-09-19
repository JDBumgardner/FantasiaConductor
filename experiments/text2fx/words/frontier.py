"""Score-vs-distance frontier by continuation: optimise at a tight locality weight first (a small move), then
warm-start each looser weight from the previous optimum. Every point is the best sound *at that distance*, not a
scaled-down extreme. Both routes: the twin (synth + FX) and the FX chain on the soundfont recording.
Records score on the word, distance from the source, identity (CLAP "a <instrument>") and saves each stop.
Distance is the phase-invariant band-energy loss of the HEARD output against the untouched instrument (same params,
other note phases: 0.01; a big cutoff move: 1.2) -- the same definition for both routes; measuring the twin's distance on
its dry synth alone made its FX free and its frontier degenerate. Candidates are ranked by the objective (score - lam * dist), not by score alone,
otherwise the locality has no say in which start wins."""
import os, sys, json, time, contextlib
CELLS = sys.argv[1:] or ["cello:dark", "cello:punchy", "epiano:dark"]
LAMS = [3.0, 1.0, 0.3, 0.1, 0.03, 0.01]                         # locality weights on the band-energy distance of the HEARD output, tight -> loose
os.environ["T2_WARM"] = "0"; sys.argv = [sys.argv[0]]
sys.path.insert(0, "."); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch, soundfile as sf
import common as C, text2synth as T2, house_session as H
from synth import WavetableSynth
from optim import Candidate, successive_halving
from word_grid import DISPLAY, art
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "frontier"); os.makedirs(OUT, exist_ok=True)
SR = 48000; L = 10 * SR; DEV = C.DEVICE
ROUNDS_FIRST, ROUNDS_NEXT = ((20, 4), (40, 2), (120, 1)), ((80, 1),)      # first stop: a small search; later stops: continue the optimum (30 steps under-reached the loose end)

def fx_sweep(name, word):
    src, _ = sf.read(os.path.join(HERE, "instruments", f"{name}_ref.wav")); x = torch.nn.functional.pad(torch.tensor(src[:L], dtype=torch.float32, device=DEV), (0, max(0, L - min(len(src), L))))[None, None]
    T = C.text_emb("this sound is " + word); T_self = C.text_emb("this sound is " + art(DISPLAY[name])); x_n = T2.loudness_norm(x[0, 0])
    lam_now = [LAMS[0]]
    def make_loss(lam):
        lam_now[0] = lam
        def loss_of(c):
            out, gs = T2.render_fx(x, c.pf); w = T2.loudness_norm(out)
            return -(C.embed(w) @ T.T).squeeze() + T2.LAM_PRI * T2.fx_prior(c.pf) + T2.LAM_GS * gs + lam * C.band_energy_loss(w, x_n)
        return loss_of
    def evaluate(c):
        with torch.no_grad():
            out, _ = T2.render_fx(x, c.pf); w = T2.loudness_norm(out); e = C.embed(w); clap = float((e @ T.T).squeeze()); dist = float(C.band_energy_loss(w, x_n))
            return dict(score=clap - lam_now[0] * dist, clap=clap, self=float((e @ T_self.T).squeeze()), dist=dist), out
    with torch.no_grad(): e0 = C.embed(x_n); base = dict(clap=float((e0 @ T.T).squeeze()), self=float((e0 @ T_self.T).squeeze()), dist=0.0)
    stops, cands = [base], [Candidate({}, T2.init_fx(r), label=f"c{r}") for r in range(8)]
    for i, lam in enumerate(LAMS):
        frontier, _ = successive_halving(cands, make_loss(lam), lambda c: evaluate(c)[0], rounds=ROUNDS_FIRST if i == 0 else ROUNDS_NEXT, polish=False, log=lambda *a: None)
        ev, c = frontier[0]; ev, out = evaluate(c); ev["lam"] = lam; stops.append(ev)
        sf.write(os.path.join(OUT, f"{name}__{word}__fx_lam{lam}.wav"), (out / (out.abs().max() + 1e-9) * 0.5).cpu().numpy(), SR)
        cands = [Candidate({}, {t: {k: v.detach().clone().requires_grad_(True) for k, v in dd.items()} for t, dd in c.pf.items()}, label=f"lam{lam}")]
    return stops

def twin_sweep(name, word):
    raw = json.load(open(os.path.join(HERE, "instruments", f"{name}_patch.json")))["raw"]; p_inst = {k: torch.tensor(float(v), device=DEV) for k, v in raw.items()}
    s_ = WavetableSynth(H.CL.TABLE, SR, voices=1, bounds=H.B, interpolation=H.CL.TBL.get("interpolation", 1)).to(DEV)
    T = C.text_emb("this sound is " + word); T_self = C.text_emb("this sound is " + art(DISPLAY[name]))
    with torch.no_grad(): y0 = s_.render(p_inst, H.NOTES, L)[0, 0]; y0_n = T2.loudness_norm(y0)
    lam_now = [LAMS[0]]
    def make_loss(lam):
        lam_now[0] = lam
        def loss_of(c):
            y = s_.render(c.ps, H.NOTES, L); out, gs = T2.render_fx(y, c.pf); w = T2.loudness_norm(out)
            return -(C.embed(w) @ T.T).squeeze() + T2.LAM_PRI * T2.fx_prior(c.pf) + T2.LAM_GS * gs + lam * C.band_energy_loss(w, y0_n)      # distance of what you hear, same as the FX route
        return loss_of
    def evaluate(c):
        with torch.no_grad():
            s_.random_phase = False; y = s_.render(c.ps, H.NOTES, L); out, _ = T2.render_fx(y, c.pf); w = T2.loudness_norm(out); e = C.embed(w); s_.random_phase = True
            clap = float((e @ T.T).squeeze()); dist = float(C.band_energy_loss(w, y0_n))
            return dict(score=clap - lam_now[0] * dist, clap=clap, self=float((e @ T_self.T).squeeze()), dist=dist), out
    with torch.no_grad(): e0 = C.embed(y0_n); base = dict(clap=float((e0 @ T.T).squeeze()), self=float((e0 @ T_self.T).squeeze()), dist=0.0)
    def cand(r, noise):
        g = torch.Generator().manual_seed(r)
        ps = {k: (v.clone() + (noise * torch.randn((), generator=g).to(DEV) if noise else 0)).requires_grad_(k != "level") for k, v in p_inst.items()}
        ps["noise"] = torch.logit(torch.tensor(0.005, device=DEV)).requires_grad_(True); return Candidate(ps, T2.init_fx(r), label=f"c{r}")   # near-silent noise start (0.05 was an audible hiss: e-piano distance 0.21 before any move)
    stops, cands = [base], [cand(r, [0, 0.3, 0.15, 0.5, 0.3, 0.5, 0.15, 0.8][r]) for r in range(8)]
    for i, lam in enumerate(LAMS):
        frontier, _ = successive_halving(cands, make_loss(lam), lambda c: evaluate(c)[0], rounds=ROUNDS_FIRST if i == 0 else ROUNDS_NEXT, polish=False, log=lambda *a: None)
        ev, c = frontier[0]; ev, out = evaluate(c); ev["lam"] = lam; stops.append(ev)
        sf.write(os.path.join(OUT, f"{name}__{word}__twin_lam{lam}.wav"), (out / (out.abs().max() + 1e-9) * 0.5).cpu().numpy(), SR)
        cands = [Candidate({k: v.detach().clone().requires_grad_(v.requires_grad) for k, v in c.ps.items()}, {t: {k: v.detach().clone().requires_grad_(True) for k, v in dd.items()} for t, dd in c.pf.items()}, label=f"lam{lam}")]
    return stops

res = {}
for cell in CELLS:
    name, word = cell.split(":")
    for route, fn in (("fx", fx_sweep), ("twin", twin_sweep)):
        t0 = time.time(); stops = fn(name, word); res[f"{name}/{word}/{route}"] = stops
        print(f"  [{name} {word} {route}] " + "  ".join(f"{'λ'+str(s['lam']) if 'lam' in s else 'start'}: clap {s['clap']:+.2f} dist {s['dist']:.2f} self {s['self']:+.2f}" for s in stops) + f"   ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(os.path.join(OUT, "frontier.json"), "w"), indent=1); print("FRONTIER DONE")
