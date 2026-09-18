"""For every plain-grid cell: how much of the gain came from the synth move vs the FX, and from which FX node.
score(untouched) -> score(dry result, no fx) -> score(result with one node bypassed each) -> score(result)."""
import os, sys, json, glob, torch, warnings; warnings.filterwarnings("ignore")
sys.argv = [sys.argv[0]]; sys.path.insert(0, "."); sys.path.insert(0, "experiments/text2fx"); sys.path.insert(0, "experiments/text2fx/words")
import common as C
torch.mps.set_per_process_memory_fraction(0.5)
import synth as S, house_session as H, text2synth as T2, fxgraph as FG
from word_grid import INSTRUMENTS, WORDS
SR = 48000; L = 10 * SR; DEV = C.DEVICE; W = "experiments/text2fx/words"
s_ = S.WavetableSynth(H.CL.TABLE, SR, voices=1, bounds=H.B, interpolation=H.CL.TBL.get("interpolation", 1)).to(DEV); s_.random_phase = False
ch = T2.CHAIN; base = json.load(open(os.path.join(W, "baseline.json")))
def find(name, prompt):
    for f in glob.glob(os.path.join(W, "plain", f"{name}__*_eq-comp-dist-delay-reverb.json")):
        d = json.load(open(f))
        if d["prompt"] == prompt: return d
out = {}; agg = {"synth": [], "fx": [], **{t: [] for t in ch.types}}
print(f"  {'cell':18s} {'base':>6s} {'dry':>6s} {'fx':>6s} | synth  fx  | " + "  ".join(f"-{t:>6s}" for t in ch.types) + "   (score drop when that node is bypassed)")
for name, _ in INSTRUMENTS:
    for w in WORDS:
        d = find(name, w); T = C.text_emb("this sound is " + w)
        ps = {k: torch.tensor(float(v), device=DEV) for k, v in d["raw"].items()}; pf = {t: {k: torch.tensor(v, device=DEV) for k, v in dd.items()} for t, dd in d["fx_raw"].items()}
        with torch.no_grad():
            y = s_.render(ps, H.NOTES, L); sc = lambda a: float((C.embed(T2.loudness_norm(a)) @ T.T).squeeze())
            s_dry = sc(y[0, 0]); full, _ = ch.render(y, pf, checkpoint=False); s_fx = sc(full)
            drops = {}
            for t in ch.types:
                p2 = {tt: (FG.NODES[tt].scale(pf[tt], 0.0) if tt == t else pf[tt]) for tt in ch.types}
                o, _ = ch.render(y, p2, checkpoint=False); drops[t] = s_fx - sc(o)
        b = base[name][w]; row = dict(base=b, dry=s_dry, fx=s_fx, synth_gain=s_dry - b, fx_gain=s_fx - s_dry, drops=drops); out[f"{name}/{w}"] = row
        agg["synth"].append(s_dry - b); agg["fx"].append(s_fx - s_dry)
        for t in ch.types: agg[t].append(drops[t])
        print(f"  {name+'/'+w:18s} {b:+.2f} {s_dry:+.2f} {s_fx:+.2f} | {s_dry-b:+.2f} {s_fx-s_dry:+.2f} | " + "  ".join(f"{drops[t]:+.3f}" for t in ch.types), flush=True)
json.dump(out, open(os.path.join(W, "who_did_the_work.json"), "w"), indent=1)
m = lambda v: sum(v) / len(v)
print(f"\n  mean gain from the synth move {m(agg['synth']):+.3f}, from the fx {m(agg['fx']):+.3f}; mean score drop when bypassing: " + ", ".join(f"{t} {m(agg[t]):+.3f}" for t in ch.types))
