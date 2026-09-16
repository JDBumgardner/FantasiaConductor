"""Extra prompts on the saved synth instruments (run after house_session.py)."""
import os, sys, json, torch
PROMPTS = sys.argv[1:]                       # read BEFORE importing house_session, which rewrites sys.argv
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import house_session as H
PROMPTS = PROMPTS or ["catholic orchestral, reaching to heaven in chorus", "hollow pipes, like an over sized wind chime"]
for name in ("cello", "brass"):
    raw = json.load(open(os.path.join(H.OUT, f"{name}_patch.json")))["raw"]
    p = {k: torch.tensor(float(v)) for k, v in raw.items()}
    for pr in PROMPTS: H.prompt_on(name, p, pr)
print("ALL DONE")
