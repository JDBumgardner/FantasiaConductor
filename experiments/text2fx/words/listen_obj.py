"""Listening set for the objective A/B: at matched distance (~0.6 and ~1.0), the cosine stop and the two-sided
directional stop side by side, per cell. Writes words/listen_obj/<cell>__d<D>__{cos,dir}.wav (full 10 s, already
loudness-matched to the source by the ladder) and <cell>__d<D>__AB.wav: original, cosine, directional -- the first bar
of each, 2.4 s, so the difference is heard back to back. README.txt carries the numbers."""
import os, sys, json, numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "listen_obj"); os.makedirs(OUT, exist_ok=True); SR = 48000; BAR = int(2.4 * SR)
CELLS = [("epiano", "dark", "twin"), ("cello", "punchy", "fx"), ("cello", "soft", "fx"), ("cello", "soft", "twin")]
ARMS = {"cos": ("ladder_obj/cos", "ladder_det/a"), "dir": ("ladder_obj/direq",)}
lines = ["Objective A/B at matched distance: cosine ('this sound is X') vs two-sided directional (change aligned with X minus not-X, the stop sets the amount).",
         "Per cell and distance: word = CLAP cosine to 'this sound is X'; dir = cosine of the change with the word's direction; id = 'a <instrument>'. AB clips: original, cosine, directional (2.4 s each).", ""]
def stops_of(arm, name, word, route):
    for d in ARMS[arm]:
        f = os.path.join(HERE, d, f"{name}_{route}__{word}__stops.json")
        if os.path.exists(f): return d, json.load(open(f))
    return None, None
def nearest(stops, D):
    i = min(range(1, len(stops)), key=lambda i: abs(stops[i]["dist"] - D) if stops[i]["target"] is not None else 9); return i
def bar(y):
    y = y[:BAR]; ramp = np.minimum(1, np.arange(len(y)) / (0.01 * SR)); return y * ramp * ramp[::-1]
for name, word, route in CELLS:
    src, _ = sf.read(os.path.join(HERE, "instruments", f"{name}_ref.wav")) if route == "fx" else (None, None)
    arms = {a: stops_of(a, name, word, route) for a in ARMS}
    if any(v[1] is None for v in arms.values()): print("skip", name, word, route); continue
    lines.append(f"{name} {word} ({route})")
    for D in (0.6, 1.0):
        clips = {}; row = []
        for a, (d, stops) in arms.items():
            i = nearest(stops, D); st = stops[i]
            f = os.path.join(HERE, d, f"{name}_{route}__{word}__stop{i}_d{st['target'] if st['target'] is not None else 'inf'}.wav"); y, _ = sf.read(f); clips[a] = y
            sf.write(os.path.join(OUT, f"{name}_{word}_{route}__d{D}__{a}.wav"), y, SR)
            row.append(f"{a}: stop {i} dist {st['dist']:.2f} word {st['clap']:+.3f} dir {st.get('dir', float('nan')):+.2f} id {st['self']:.2f}")
        if src is None:                                                             # twin route: the untouched sound is the first segment of the ladder clip
            lad, _ = sf.read(os.path.join(HERE, arms["cos"][0], f"{name}_{route}__{word}__LADDER.wav")); orig = lad[:BAR]
        else: orig = src[:BAR] / (np.abs(src[:BAR]).max() + 1e-9) * np.abs(clips["cos"][:BAR]).max()
        sf.write(os.path.join(OUT, f"{name}_{word}_{route}__d{D}__AB.wav"), np.concatenate([bar(orig), np.zeros(int(0.3 * SR)), bar(clips["cos"]), np.zeros(int(0.3 * SR)), bar(clips["dir"])]), SR)
        lines.append(f"  d~{D}:  " + "   |   ".join(row))
    lines.append("")
open(os.path.join(OUT, "README.txt"), "w").write("\n".join(lines)); print("\n".join(lines))
