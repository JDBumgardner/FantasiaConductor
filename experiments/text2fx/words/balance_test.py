"""A/B for the rebalancing: same cells, three settings.
  A  new chain (level-referenced drive), locality on the dry synth only (as the grid)
  B  A + locality split between dry and heard output (0.025 + 0.025)
  C  B + per-parameter learning rates from measured sensitivity
  D  B + the wide chain: every node type (eq, comp, drive, pwtanh, chorus, transient, gate, delay, reverb)
Results in words/balance/<variant>/."""
import os, sys, json, time
VARIANT = sys.argv[1]; CELLS = sys.argv[2:] or ["cello:dark", "cello:soft", "cello:punchy", "cello:metallic", "epiano:dark", "epiano:punchy"]
os.environ.setdefault("T2_ROUNDS", "20:4,40:2,140:1")
if VARIANT in ("B", "C", "E", "F"): os.environ["T2_LOC_DRY"] = "0.025"; os.environ["T2_LOC_WET"] = "0.025"
if VARIANT == "C": os.environ["T2_LR_EQ"] = "1"
if VARIANT in ("D", "E", "F"): os.environ["T2_CHAIN"] = "eq,comp,drive,pwtanh,chorus,transient,gate,delay,reverb"     # every node type in the race
if VARIANT in ("E", "F"): os.environ["T2_LR_EQ"] = "1"          # with the outlier-only rule (slow the hyper-sensitive, never speed the weak)
if VARIANT == "F": os.environ["T2_EXCERPT"] = "2.0"             # early stages on the first bar (5 notes), last stage on the whole hook
sys.argv = [sys.argv[0]]
sys.path.insert(0, "."); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch, house_session as H
HERE = os.path.dirname(os.path.abspath(__file__)); H.OUT = os.path.join(HERE, "balance", VARIANT); os.makedirs(H.OUT, exist_ok=True)
for cell in CELLS:
    name, w = cell.split(":")
    p = {k: torch.tensor(float(v)) for k, v in json.load(open(os.path.join(HERE, "instruments", f"{name}_patch.json")))["raw"].items()}
    if os.path.exists(os.path.join(H.OUT, f"{name}__{w}{H.TAG}.json")): continue
    t0 = time.time(); H.prompt_on(name, p, w, restarts=8); print(f"      ({time.time()-t0:.0f}s)", flush=True)
print("VARIANT DONE")
