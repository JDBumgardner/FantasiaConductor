"""Results of word_grid.py as tables: score / robust / half-amount per (instrument, word), and what moved."""
import os, sys, json, glob
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D); from word_grid import INSTRUMENTS, WORDS, DISPLAY
NAMED = len(sys.argv) > 1 and sys.argv[1] == "named"; R = os.path.join(D, "named") if NAMED else D
rows = {}
for f in glob.glob(os.path.join(R, "*__*_eq-comp-dist-delay-reverb.json")):
    d = json.load(open(f)); name = os.path.basename(f).split("__", 1)[0]
    word = d["prompt"].split(" ")[1] if NAMED else d["prompt"]
    rows[(name, word)] = d
names = [n for n, _ in INSTRUMENTS if any(k[0] == n for k in rows)]
base = json.load(open(os.path.join(D, "baseline.json"))) if os.path.exists(os.path.join(D, "baseline.json")) else {}
def b0(n, w): return (base[n]["_named"][w] if NAMED else base[n][w]) if n in base else None
def cell(n, w):
    d = rows.get((n, w))
    if not d: return "·"
    b = b0(n, w); return f"{d['score']:+.2f}" + (f" ({b:+.2f} → {d['score']-b:+.2f})" if b is not None else "") + f" · ½ {d['amount50']['score']:+.2f}"
lines = [f"# Word grid — CLAP cosine to “this sound is {'a <word> <instrument>' if NAMED else '<word>'}”", "",
         "Per cell: score of the result (score of the untouched instrument → gain) · score at half amount.", "",
         "| word | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
for w in WORDS:
    lines.append(f"| {w} | " + " | ".join(cell(n, w) for n in names) + " |")
if base:
    lines += ["", "Untouched instrument vs “this sound is a ⟨instrument⟩”: " + ", ".join(f"{n} {base[n]['_self']:+.2f}" for n in names if n in base) + " — CLAP recognises the brass and e-piano twins as their instruments, not the cello or flute twins."]
lines += ["", "## What moved (result minus instrument; physical units)", ""]
for n in names:
    inst = json.load(open(os.path.join(D, f"{n}_patch.json")))["physical"]
    lines += [f"### {n}", "", "| word | score | dist | cutoff | res | env amt | attack ms | release s | drive dB | comp ratio | delay | reverb | eq (Hz: dB) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for w in WORDS:
        d = rows.get((n, w))
        if not d: continue
        s = d["describe"]["synth"]; de = d["describe"]
        dl = lambda k, sc=1, f="{:+.2f}": f.format((s[k] - inst[k]) * sc)
        eq = ", ".join(f"{h}:{g:+.0f}" for h, g in de["eq"] if abs(g) >= 3)
        lines.append(f"| {w} | {d['score']:+.3f} | {d['dist']:.2f} | {dl('cutoff_raw')} | {dl('resonance')} | {dl('fenv_amount')} | {dl('attack', 1000, '{:+.0f}')} | {dl('release')} | {de['drive_db']:+.1f} | {de['comp']['ratio']:.1f} | {de['delay_mix']:.2f} | {de['reverb']['mix']:.2f} | {eq} |")
    lines.append("")
out = "\n".join(lines); open(os.path.join(R, "RESULTS.md"), "w").write(out); print(out)
