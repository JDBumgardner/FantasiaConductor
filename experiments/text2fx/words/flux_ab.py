"""A/B of the artefact flux penalty's gradient (fixed 2026-09-20 -- it was zero on MPS): the same ladder cell twice, identical
seeds, arm 'nograd' detaching the flux term (T2_FLUX_NOGRAD=1) and arm 'fixed' not. One process per arm; results in
words/ladder_flux/<arm>/. `python flux_ab.py summary` tabulates both arms per stop."""
import os, sys, json, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "ladder_flux"); CELL = os.environ.get("T2_AB_CELL", "cello:punchy"); ROUTE = os.environ.get("T2_AB_ROUTE", "fx")
PY_ = os.path.join(ROOT, ".venv", "bin", "python")

def run_arm(arm):
    out = os.path.join(OUT, arm); os.makedirs(out, exist_ok=True)
    name, word = CELL.split(":")
    if os.path.exists(os.path.join(out, f"{name}_{ROUTE}__{word}__stops.json")): print(f"[{arm}] done already", flush=True); return
    env = dict(os.environ, T2_FLUX_NOGRAD="1" if arm == "nograd" else "0", T2_LADDER_OUT=out)
    t0 = time.time(); print(f"[{arm}] {CELL} {ROUTE} ->", out, flush=True)
    subprocess.run([PY_, os.path.join(HERE, "ladder_test.py"), CELL, ROUTE], env=env, cwd=ROOT, check=False)
    print(f"[{arm}] {time.time() - t0:.0f}s", flush=True)

def summary():
    name, word = CELL.split(":"); rows = {}
    for arm in ("nograd", "nograd_rep", "nograd2", "fixed", "fixed_rep"):
        f = os.path.join(OUT, arm, f"{name}_{ROUTE}__{word}__stops.json")
        if os.path.exists(f): rows[arm] = json.load(open(f))
    if not rows: print("nothing yet"); return
    keys = ("clap", "dist", "d_flux", "d_jump", "d_trem", "d_crest", "d_stutter", "flags")
    print(f"{CELL} {ROUTE}: per stop, nograd (old MPS behaviour) vs fixed")
    print(f"{'stop':>6} {'arm':>10} {'clap':>7} {'dist':>6} {'flux':>6} {'jump':>6} {'trem':>6} {'crest':>6} {'stut':>5}  flags")
    n = max(len(v) for v in rows.values())
    for i in range(n):
        for arm, stops in rows.items():
            if i >= len(stops): continue
            s = stops[i]; a = s.get("art", {}); tgt = s.get("target")
            print(f"{('d' + str(tgt)) if tgt is not None else ('dry' if i == 0 else 'free'):>6} {arm:>10} {s.get('clap', 0):7.3f} {s.get('dist', 0):6.2f} {a.get('d_flux', 0):6.2f} {a.get('d_jump', 0):6.2f} {a.get('d_trem', 0):6.2f} {a.get('d_crest', 0):6.2f} {a.get('d_stutter', 0):5} {','.join(s.get('flags', [])) or '-'}")
        print()

if __name__ == "__main__":
    if "summary" in sys.argv: summary()
    else:
        for arm in ("nograd", "fixed"): run_arm(arm)
        summary(); print("FLUX AB DONE", flush=True)
