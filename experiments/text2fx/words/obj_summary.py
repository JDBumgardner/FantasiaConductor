"""Directional objective vs cosine: the same cells, per stop, and at matched distance. `python obj_summary.py`.
cos arms come from ladder_det/a (deterministic runs, 2026-09-21) or ladder_obj/cos; dir arms from ladder_obj/dir.
Stops from older runs lack the 'dir' field; it is recomputed from the stop wav against the untouched source."""
import os, sys, json, glob, numpy as np
ARM = sys.argv[1] if len(sys.argv) > 1 else "direq"                       # read before house_session rewrites sys.argv; dir (one-sided stops) or direq (two-sided)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
CELLS = [("epiano", "dark", "twin"), ("cello", "punchy", "fx"), ("cello", "soft", "fx"), ("cello", "soft", "twin")]

def load(arm_dir, name, word, route):
    f = os.path.join(HERE, arm_dir, f"{name}_{route}__{word}__stops.json"); return json.load(open(f)) if os.path.exists(f) else None

def with_dir(stops, arm_dir, name, word, route):
    if all("dir" in s for s in stops): return stops
    import torch, soundfile as sf, common as C, text2synth as T2, house_session as H
    from synth import WavetableSynth
    from word_grid import DISPLAY
    SR = 48000; L = 10 * SR
    if route == "fx":
        a, _ = sf.read(os.path.join(HERE, "instruments", f"{name}_ref.wav")); src = torch.tensor(a[:L].astype(np.float32), device=C.DEVICE)
    else:
        raw = json.load(open(os.path.join(HERE, "instruments", f"{name}_patch.json")))["raw"]; p = {k: torch.tensor(float(v), device=C.DEVICE) for k, v in raw.items()}
        s_ = WavetableSynth(H.CL.TABLE, SR, voices=1, bounds=H.B, interpolation=H.CL.TBL.get("interpolation", 1)).to(C.DEVICE); s_.random_phase = False
        with torch.no_grad(): src = s_.render(p, H.NOTES, L)[0, 0]
    T_dir = C.text_emb("this sound is " + word) - C.text_emb("this sound is not " + word); T_dir = T_dir / T_dir.norm()
    with torch.no_grad():
        e0 = C.embed(T2.loudness_norm(src))
        for i, s in enumerate(stops):
            if "dir" in s: continue
            if i == 0: s["dir"] = 0.0; continue
            f = os.path.join(HERE, arm_dir, f"{name}_{route}__{word}__stop{i}_d{s['target'] if s['target'] is not None else 'inf'}.wav")
            y, _ = sf.read(f); e = C.embed(T2.loudness_norm(torch.tensor(y[:L].astype(np.float32), device=C.DEVICE))); d = e - e0
            s["dir"] = float((d @ T_dir.T).squeeze() / (d.norm() + 1e-3))
    return stops

def curve(stops, key):
    d = np.array([s["dist"] for s in stops]); v = np.array([s[key] for s in stops]); o = np.argsort(d); return d[o], v[o]

for name, word, route in CELLS:
    cos = load("ladder_obj/cos", name, word, route) or load("ladder_det/a", name, word, route); dr = load(f"ladder_obj/{ARM}", name, word, route)
    if not (cos and dr): print(f"{name} {word} {route}: missing an arm"); continue
    cos = with_dir(cos, "ladder_obj/cos" if os.path.exists(os.path.join(HERE, "ladder_obj/cos", f"{name}_{route}__{word}__stops.json")) else "ladder_det/a", name, word, route)
    print(f"\n{name} {word} ({route}) — per stop: word score · direction cosine · distance · identity · flags")
    print(f"{'target':>7} | {'cos: word':>9} {'dir':>6} {'dist':>5} {'id':>5} {'flags':>8} | {'dir: word':>9} {'dir':>6} {'dist':>5} {'id':>5} {'flags':>8}")
    for a, b in zip(cos, dr):
        fa, fb = ",".join(x[:6] for x in a.get("flags", [])) or "-", ",".join(x[:6] for x in b.get("flags", [])) or "-"
        print(f"{str(a.get('target')):>7} | {a['clap']:+9.3f} {a['dir']:+6.2f} {a['dist']:5.2f} {a['self']:5.2f} {fa:>8} | {b['clap']:+9.3f} {b['dir']:+6.2f} {b['dist']:5.2f} {b['self']:5.2f} {fb:>8}")
    lo = max(min(curve(cos, "clap")[0][1:]), min(curve(dr, "clap")[0][1:])); hi = min(max(curve(cos, "clap")[0]), max(curve(dr, "clap")[0]))
    print(f"   at matched distance:", end="")
    for D in (0.5, 0.8, 1.2, 1.6):
        if lo <= D <= hi:
            wc, wd = np.interp(D, *curve(cos, "clap")), np.interp(D, *curve(dr, "clap")); ic, idd = np.interp(D, *curve(cos, "self")), np.interp(D, *curve(dr, "self"))
            print(f"  d={D}: word {wc:+.3f}→{wd:+.3f} ({wd - wc:+.3f}), identity {ic:.2f}→{idd:.2f}", end="")
    print()
