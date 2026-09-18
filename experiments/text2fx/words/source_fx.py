"""Text -> FX on the soundfont recording itself (no twin): the wide chain on the fixed audio, same words, same
optimiser, so the twin's contribution can be read off against the same cells. Results in words/source/."""
import os, sys, json, time, contextlib
CELLS = sys.argv[1:] or ["cello:dark", "cello:soft", "cello:punchy", "cello:metallic", "epiano:dark", "epiano:punchy"]
sys.argv = [sys.argv[0]]; sys.path.insert(0, "."); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch, soundfile as sf
import common as C, text2synth as T2
from optim import Candidate, successive_halving
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "source"); os.makedirs(OUT, exist_ok=True)
SR = 48000; L = 10 * SR; DEV = C.DEVICE; ROUNDS = ((30, 4), (60, 2), (220, 1)); LOC = 0.05
def run(name, word):
    src, _ = sf.read(os.path.join(HERE, "instruments", f"{name}_ref.wav")); x = torch.tensor(src[:L], dtype=torch.float32, device=DEV); x = torch.nn.functional.pad(x, (0, L - x.numel()))[None, None]
    T = C.text_emb("this sound is " + word); anchor = C.SpectralAnchor(T2.loudness_norm(x[0, 0]), ffts=(512, 2048))
    def loss_of(c):
        out, gs = T2.render_fx(x, c.pf); w = T2.loudness_norm(out)
        return -(C.embed(w) @ T.T).squeeze() + T2.LAM_PRI * T2.fx_prior(c.pf) + T2.LAM_GS * gs + LOC * C.mrstft(w, anchor)
    def evaluate(c):
        with torch.no_grad():
            out, _ = T2.render_fx(x, c.pf); w = T2.loudness_norm(out); e = C.embed(w)
            return dict(score=float((e @ T.T).squeeze()), robust=float(sum((C.embed(torch.roll(w, int(s * SR))) @ T.T).squeeze() for s in (0.0, 0.37, 0.71, 1.9)) / 4), dist=float(C.mrstft(w, anchor))), out
    with torch.no_grad(): base = float((C.embed(T2.loudness_norm(x[0, 0])) @ T.T).squeeze())
    cands = [Candidate({}, T2.init_fx(r), label=f"c{r}") for r in range(8)]
    t0 = time.time(); frontier, total = successive_halving(cands, loss_of, lambda c: evaluate(c)[0], rounds=ROUNDS)
    ev, c = frontier[0]; ev, out = evaluate(c)
    with torch.no_grad(): ev_half, out_half = evaluate(Candidate({}, T2.CHAIN.scale(c.pf, 0.5)))
    d = T2.CHAIN.describe(c.pf)
    sf.write(os.path.join(OUT, f"{name}__{word}_fx.wav"), (out / (out.abs().max() + 1e-9) * 0.5).cpu().numpy(), SR)
    sf.write(os.path.join(OUT, f"{name}__{word}_fx_amt50.wav"), (out_half / (out_half.abs().max() + 1e-9) * 0.5).cpu().numpy(), SR)
    json.dump({"prompt": word, "source": f"{name}_ref.wav", "score": ev["score"], "robust": ev["robust"], "baseline": base, "amount50": ev_half, "dist": ev["dist"], "chain": T2.CHAIN_TYPES, "describe": d,
               "fx_raw": {t: {k: v.detach().cpu().tolist() for k, v in dd.items()} for t, dd in c.pf.items()}, "steps": total}, open(os.path.join(OUT, f"{name}__{word}.json"), "w"), indent=1)
    print(f"  [{name} soundfont + '{word}'] score {ev['score']:+.3f} (untouched {base:+.2f}, robust {ev['robust']:+.3f}, half amount {ev_half['score']:+.3f}) | " + ", ".join(f"{k}={v}" for k, v in d.items()) + f"   {total} steps in {time.time()-t0:.0f}s", flush=True)
for cell in CELLS:
    name, w = cell.split(":")
    if not os.path.exists(os.path.join(OUT, f"{name}__{w}.json")): run(name, w)
print("SOURCE DONE")
