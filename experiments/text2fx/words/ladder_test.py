"""Ladders for a few cells, both routes (twin instrument, soundfont recording). Results in words/ladder/."""
import os, sys, json, time
CELLS = [a for a in sys.argv[1:] if ":" in a] or ["cello:dark", "cello:punchy", "cello:soft", "epiano:dark"]; ROUTES = [a for a in sys.argv[1:] if a in ("fx", "twin")] or ["fx", "twin"]
sys.argv = [sys.argv[0]]; sys.path.insert(0, "."); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch, soundfile as sf
import common as C, house_session as H, text2synth as T2, ladder
from synth import WavetableSynth
from word_grid import DISPLAY, art
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.environ.get("T2_LADDER_OUT") or os.path.join(HERE, "ladder"); SR = 48000; L = 10 * SR; DEV = C.DEVICE
for cell in CELLS:
    name, word = cell.split(":")
    src, _ = sf.read(os.path.join(HERE, "instruments", f"{name}_ref.wav")); x = torch.tensor(src[:L], dtype=torch.float32, device=DEV); x = torch.nn.functional.pad(x, (0, L - x.numel()))
    if "fx" in ROUTES and not os.path.exists(os.path.join(OUT, f"{name}_fx__{word}__stops.json")):
        print(f"  [{name} {word} — FX on the recording]", flush=True); t0 = time.time()
        ladder.run(word, x, H.NOTES, OUT, f"{name}_fx", art(DISPLAY[name]), log=lambda s: print(s, flush=True)); print(f"    ({time.time()-t0:.0f}s)", flush=True)
    if "twin" not in ROUTES or os.path.exists(os.path.join(OUT, f"{name}_twin__{word}__stops.json")): continue
    raw = json.load(open(os.path.join(HERE, "instruments", f"{name}_patch.json")))["raw"]; p_inst = {k: torch.tensor(float(v), device=DEV) for k, v in raw.items()}
    s_ = WavetableSynth(H.CL.TABLE, SR, voices=1, bounds=H.B, interpolation=H.CL.TBL.get("interpolation", 1)).to(DEV)
    with torch.no_grad(): s_.random_phase = False; y0 = s_.render(p_inst, H.NOTES, L)[0, 0]; s_.random_phase = True
    print(f"  [{name} {word} — twin]", flush=True); t0 = time.time()
    ladder.run(word, y0, H.NOTES, OUT, f"{name}_twin", art(DISPLAY[name]), synth=s_, p_inst=p_inst, log=lambda s: print(s, flush=True)); print(f"    ({time.time()-t0:.0f}s)", flush=True)
print("LADDER DONE")
