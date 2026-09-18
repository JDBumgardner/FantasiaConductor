"""Re-recover the instruments with the noise source on and the band-energy term in the loss; compare with the
noiseless patches on CLAP's own terms ("this sound is a <instrument>") and save the audio for listening."""
import os, sys, json, time
ARGS = sys.argv[1:]; sys.argv = [sys.argv[0]]
sys.path.insert(0, "."); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch, soundfile as sf
import common as C, house_session as H, text2synth as T2
from synth import WavetableSynth
from word_grid import INSTRUMENTS, DISPLAY, art
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "noise"); H.OUT = OUT
SR = 48000; L = 10 * SR; DEV = C.DEVICE
s_ = WavetableSynth(H.CL.TABLE, SR, voices=1, bounds=H.B, interpolation=H.CL.TBL.get("interpolation", 1)).to(DEV); s_.random_phase = False
def clap_self(y, name):
    with torch.no_grad(): return float((C.embed(T2.loudness_norm(torch.as_tensor(y, device=DEV))) @ C.text_emb("this sound is " + art(DISPLAY[name])).T).squeeze())
for name, prog in INSTRUMENTS:
    if ARGS and name not in ARGS: continue
    old = json.load(open(os.path.join(HERE, "instruments", f"{name}_patch.json")))["raw"]
    with torch.no_grad(): y_old = s_.render({k: torch.tensor(float(v), device=DEV) for k, v in old.items()}, H.NOTES, L)[0, 0].cpu().numpy()
    ref, _ = sf.read(os.path.join(HERE, "instruments", f"{name}_ref.wav"))
    t0 = time.time(); p = H.make_instrument(name, prog)                     # writes <name>_{ref,clone,vital}.wav and _patch.json into OUT
    new = json.load(open(os.path.join(OUT, f"{name}_patch.json")))
    with torch.no_grad(): y_new = s_.render({k: torch.tensor(float(v), device=DEV) for k, v in new["raw"].items()}, H.NOTES, L)[0, 0].cpu().numpy()
    vit, _ = sf.read(os.path.join(OUT, f"{name}_vital.wav"))
    print(f"  [{name}] CLAP “{art(DISPLAY[name])}”: soundfont {clap_self(ref.astype(np.float32), name):+.2f} | old twin {clap_self(y_old, name):+.2f} | new twin (noise a={new['physical'].get('noise', 0):.3f} -> sample_level {new['vital_params'].get('sample_level', 0):.2f}) {clap_self(y_new, name):+.2f} | Vital render of the new patch {clap_self(vit.astype(np.float32), name):+.2f}   ({time.time()-t0:.0f}s)", flush=True)
print("RECOVER DONE")
