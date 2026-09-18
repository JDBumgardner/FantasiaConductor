"""Perceptual step size per parameter: how far the CLAP embedding moves for one Adam step (lr 0.02) of each raw
parameter, at the instrument's starting point with the default FX init. Adam moves every parameter ~lr per step
regardless of gradient size, so this is the real 'weight' each component has in the search."""
import os, sys, json, torch, warnings; warnings.filterwarnings("ignore")
ARGS = sys.argv[1:]; sys.argv = [sys.argv[0]]
sys.path.insert(0, "."); sys.path.insert(0, "experiments/text2fx"); sys.path.insert(0, "experiments/text2fx/words")
import common as C
torch.mps.set_per_process_memory_fraction(0.5)
import synth as S, house_session as H, text2synth as T2
from word_grid import INSTRUMENTS
SR = 48000; L = 10 * SR; DEV = C.DEVICE; W = "experiments/text2fx/words"; STEP = 0.02
s_ = S.WavetableSynth(H.CL.TABLE, SR, voices=1, bounds=H.B, interpolation=H.CL.TBL.get("interpolation", 1)).to(DEV); s_.random_phase = False
def embed(ps, pf):
    with torch.no_grad(): y = s_.render(ps, H.NOTES, L); out, _ = T2.CHAIN.render(y, pf, checkpoint=False); return C.embed(T2.loudness_norm(out))[0]
res = {}
for name in ARGS or [n for n, _ in INSTRUMENTS]:
    raw = json.load(open(os.path.join(W, "instruments", f"{name}_patch.json")))["raw"]
    ps = {k: torch.tensor(float(v), device=DEV) for k, v in raw.items()}; pf = T2.init_fx(0); pf = {t: {k: v.detach() for k, v in d.items()} for t, d in pf.items()}
    e0 = embed(ps, pf); rows = []
    for k in ps:
        if k == "level": continue
        d = 0
        for sgn in (1, -1):
            p2 = dict(ps); p2[k] = ps[k] + sgn * STEP; d += float((embed(p2, pf) - e0).norm()) / 2
        rows.append(("synth", k, d))
    for t in pf:
        for k, v in pf[t].items():
            flat = v.flatten(); ds = []
            for i in range(flat.numel()):
                d = 0
                for sgn in (1, -1):
                    v2 = v.clone().flatten(); v2[i] += sgn * STEP; p2 = {tt: dict(pf[tt]) for tt in pf}; p2[t][k] = v2.view_as(v); d += float((embed(ps, p2) - e0).norm()) / 2
                ds.append(d)
            rows.append((t, k, sum(ds) / len(ds), max(ds)))
    res[name] = rows
    print(f"\n{name}: embedding displacement per Adam step (lr {STEP}); [component] parameter  mean (max over elements)")
    for r in sorted(rows, key=lambda r: -r[2]):
        print(f"  {r[0]:7s} {r[1]:14s} {r[2]:.4f}" + (f"  (max {r[3]:.4f})" if len(r) > 3 else ""), flush=True)
json.dump({n: [list(r) for r in rows] for n, rows in res.items()}, open(os.path.join(W, "sensitivity.json"), "w"), indent=1)
