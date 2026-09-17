"""Score every result on the prompts it did NOT optimise for: the bare word, the named prompt, and the instrument name.
Runs on the CPU so a grid can keep the GPU."""
import os, sys, json, glob, torch, warnings; warnings.filterwarnings("ignore")
ARGS = sys.argv[1:]
sys.path.insert(0, "."); sys.path.insert(0, "experiments/text2fx"); sys.path.insert(0, "experiments/text2fx/words")
import common as C; C.DEVICE = "cpu"
import synth as S, house_session as H, text2synth as T2
from word_grid import INSTRUMENTS, WORDS, DISPLAY, art
SR = 48000; L = 10 * SR; W = "experiments/text2fx/words"
s_ = S.WavetableSynth(H.CL.TABLE, SR, voices=1, bounds=H.B, interpolation=H.CL.TBL.get("interpolation", 1)); s_.random_phase = False
base = json.load(open(os.path.join(W, "baseline.json")))
def render(d):
    ps = {k: torch.tensor(float(v)) for k, v in d["raw"].items()}; pf = {t: {k: torch.tensor(v) for k, v in dd.items()} for t, dd in d["fx_raw"].items()}
    with torch.no_grad(): torch.manual_seed(0); y = s_.render(ps, H.NOTES, L); out, _ = T2.CHAIN.render(y, pf, checkpoint=False); return C.embed(T2.loudness_norm(out))
def find(folder, name, prompt):
    for f in glob.glob(os.path.join(folder, f"{name}__*_eq-comp-dist-delay-reverb.json")):
        d = json.load(open(f))
        if d["prompt"] == prompt: return d
out = json.load(open(os.path.join(W, "cross_scores.json"))) if os.path.exists(os.path.join(W, "cross_scores.json")) else {}
for name in ARGS or [n for n, _ in INSTRUMENTS]:
    T = {w: {"word": C.text_emb(f"this sound is {w}"), "named": C.text_emb("this sound is " + art(f"{w} {DISPLAY[name]}"))} for w in WORDS}; T_self = C.text_emb("this sound is " + art(DISPLAY[name]))
    print(f"\n{name}: result of the PLAIN prompt | result of the NAMED prompt — each scored on: bare word / named / 'a {DISPLAY[name]}'   (untouched: self {base[name]['_self']:+.2f})")
    print(f"  {'word':9s} {'plain→word':>11s} {'plain→named':>12s} {'plain→self':>11s}   {'named→word':>11s} {'named→named':>12s} {'named→self':>11s}   {'contr→word':>11s} {'contr→named':>12s} {'contr→self':>11s}")
    for w in WORDS:
        dp, dn = find(W, name, w), (find(os.path.join(W, "named"), name, f"a {w} {DISPLAY[name]}") or find(os.path.join(W, "named"), name, art(f"{w} {DISPLAY[name]}")))
        dc = find(os.path.join(W, "contrast"), name, art(f"{w} {DISPLAY[name]}")) if os.path.isdir(os.path.join(W, "contrast")) else None
        row = {}
        for tag, d in (("plain", dp), ("named", dn), ("contrast", dc)):
            if d is None: row[tag] = None; continue
            e = render(d); row[tag] = dict(word=float(e @ T[w]["word"].T), named=float(e @ T[w]["named"].T), self=float(e @ T_self.T))
        f = lambda r, k: f"{r[k]:+.2f}" if r else "   ·"
        print(f"  {w:9s} {f(row['plain'],'word'):>11s} {f(row['plain'],'named'):>12s} {f(row['plain'],'self'):>11s}   {f(row['named'],'word'):>11s} {f(row['named'],'named'):>12s} {f(row['named'],'self'):>11s}   {f(row['contrast'],'word'):>11s} {f(row['contrast'],'named'):>12s} {f(row['contrast'],'self'):>11s}", flush=True)
        out[f"{name}/{w}"] = row
json.dump(out, open(os.path.join(W, "cross_scores.json"), "w"), indent=1)
