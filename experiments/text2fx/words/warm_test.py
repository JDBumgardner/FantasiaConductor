"""Warm start vs cold: hold one word out, start from the nearest other words' results on the same instrument, half the budget."""
import os, sys, json, time
CELLS = sys.argv[1:] or ["cello:dark", "cello:soft", "cello:punchy", "cello:metallic", "epiano:dark", "epiano:punchy"]
os.environ["T2_WARM"] = "2"; os.environ["T2_WARM_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)))     # search the whole words tree (plain grid + balance variants)
os.environ["T2_ROUNDS"] = "15:3,30:2,110:1"                                                                            # 4 candidates: 60 + 90 + 220 = 370 steps (cold D: 700)
sys.argv = [sys.argv[0]]; sys.path.insert(0, "."); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch, house_session as H
HERE = os.path.dirname(os.path.abspath(__file__)); H.OUT = os.path.join(HERE, "balance", "W"); os.makedirs(H.OUT, exist_ok=True)
for cell in CELLS:
    name, w = cell.split(":"); p = {k: torch.tensor(float(v)) for k, v in json.load(open(os.path.join(HERE, f"{name}_patch.json")))["raw"].items()}
    if os.path.exists(os.path.join(H.OUT, f"{name}__{w}{H.TAG}.json")): continue
    t0 = time.time(); H.prompt_on(name, p, w, restarts=4); print(f"      ({time.time()-t0:.0f}s)", flush=True)
print("WARM DONE")
