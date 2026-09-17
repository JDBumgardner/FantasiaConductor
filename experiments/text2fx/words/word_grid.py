"""Single-word prompts x a handful of instruments, on the house hook — the prompt -> parameter map.

Four GM instruments are recovered as twin patches (cello, brass, electric piano, flute), then each is pushed
toward ten one-word prompts spanning the space: spectrum (bright / dark / warm), texture (harsh / metallic /
wooden), dynamics (punchy / soft), space (airy / distant). Same objective as house_session.prompt_on: CLAP
cosine to "this sound is <word>" + priors + a locality term to the instrument, successive halving over 8 inits.
Results land next to this file; summarize.py turns them into a table.
"""
import os, sys, json, time
ARGS = sys.argv[1:]                                           # read BEFORE importing house_session, which rewrites sys.argv
os.environ.setdefault("T2_ROUNDS", "20:4,40:2,140:1")        # 460 steps + polish per run, ~2.5 min: 40 runs in under two hours
sys.path.insert(0, "."); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import house_session as H
HERE = os.path.dirname(os.path.abspath(__file__))
NAMED = os.environ.get("T2_NAMED", "0") == "1"                # "this sound is a dark cello" instead of "this sound is dark"
CONTRAST = H.OBJ == "contrast"                                # optimise cosine("a dark cello") - cosine("a cello"): the adjective's direction, not the instrument's
H.OUT = os.path.join(HERE, "contrast" if CONTRAST else "named") if NAMED else HERE
os.makedirs(H.OUT, exist_ok=True)

INSTRUMENTS = [("cello", 42), ("brass", 61), ("epiano", 4), ("flute", 73)]
DISPLAY = {"cello": "cello", "brass": "brass section", "epiano": "electric piano", "flute": "flute"}
WORDS = ["bright", "dark", "warm", "harsh", "soft", "punchy", "metallic", "wooden", "airy", "distant"]
def art(s): return ("an " if s[0] in "aeiou" else "a ") + s
def prompt_for(name, w): return art(f"{w} {DISPLAY[name]}") if NAMED else w

if __name__ == "__main__":
    recover_only = ARGS[:1] == ["recover"]; names = [a for a in ARGS if a != "recover"] or [n for n, _ in INSTRUMENTS]
    for name, prog in INSTRUMENTS:
        if name not in names: continue
        patch = os.path.join(H.OUT, f"{name}_patch.json")
        if not os.path.exists(patch) and os.path.exists(os.path.join(HERE, f"{name}_patch.json")): patch = os.path.join(HERE, f"{name}_patch.json")   # named grid reuses the plain grid's instruments
        if os.path.exists(patch):
            p = {k: torch.tensor(float(v)) for k, v in json.load(open(patch))["raw"].items()}; print(f"  [{name}] patch loaded", flush=True)
        else:
            p = H.make_instrument(name, prog)
        if recover_only: continue                                   # run_grid.sh does the words in a fresh process
        for w in WORDS:
            pr = prompt_for(name, w); slug = "".join(ch if ch.isalnum() else "_" for ch in pr)[:28].strip("_")
            if os.path.exists(os.path.join(H.OUT, f"{name}__{slug}{H.TAG}.json")): continue
            t0 = time.time(); H.prompt_on(name, p, pr, restarts=8, anchor_text=art(DISPLAY[name]) if NAMED else None); print(f"      ({time.time()-t0:.0f}s)", flush=True)
    print("INSTRUMENT DONE")
