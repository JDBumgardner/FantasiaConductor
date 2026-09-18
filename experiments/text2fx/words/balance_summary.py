"""Table for the rebalancing A/B: score and which components moved, per variant."""
import os, sys, json, glob
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
base = json.load(open(os.path.join(D, "baseline.json")))
inst = {n: json.load(open(os.path.join(D, f"{n}_patch.json")))["physical"] for n in ("cello", "brass", "epiano", "flute")}
def row(d, n):
    s, de = d["describe"]["synth"], d["describe"]; i = inst[n]
    eqmax = max(abs(g) for _, g in de["eq"])
    return dict(score=d["score"], half=d["amount50"]["score"], dcut=s["cutoff_raw"] - i["cutoff_raw"], dres=s["resonance"] - i["resonance"], datt=(s["attack"] - i["attack"]) * 1000,
                drive=de["drive_db"], ratio=de["comp"]["ratio"], delay=de["delay_mix"], reverb=de["reverb"]["mix"], eqmax=eqmax, dist=d["dist"])
variants = sorted(os.listdir(os.path.join(D, "balance"))) if os.path.isdir(os.path.join(D, "balance")) else []
cells = sorted({os.path.basename(f).split("_eq-comp")[0] for v in variants for f in glob.glob(os.path.join(D, "balance", v, "*.json"))})
print(f"{'cell':18s} {'var':4s} {'score':>6s} {'grid':>6s} {'half':>6s} | {'Δcut':>6s} {'Δres':>6s} {'Δatt':>6s} {'drive':>6s} {'comp':>5s} {'delay':>5s} {'rev':>5s} {'eq|max|':>7s} {'dist':>5s}")
for c in cells:
    n, w = c.split("__")
    g = glob.glob(os.path.join(D, f"{c}_eq-comp-dist-delay-reverb.json")); grid = json.load(open(g[0]))["score"] if g else float("nan")
    for v in variants:
        f = glob.glob(os.path.join(D, "balance", v, f"{c}_eq-comp-*.json"))
        if not f: continue
        r = row(json.load(open(f[0])), n)
        print(f"{c:18s} {v:4s} {r['score']:+.3f} {grid:+.3f} {r['half']:+.3f} | {r['dcut']:+.2f} {r['dres']:+.2f} {r['datt']:+5.0f} {r['drive']:+6.1f} {r['ratio']:5.1f} {r['delay']:5.2f} {r['reverb']:5.2f} {r['eqmax']:7.1f} {r['dist']:5.2f}")
