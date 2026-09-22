"""A Vital track through the compiler: the e-piano twin patch as the source, app inserts after it, the word "dark", the
fixed-distance ladder -- then the closed loop: the exported Vital parameters played by REAL Vital (pedalboard) through
the app's FxHost with the exported insert parameters, scored against what the twin promised. Results in words/vitalgraph/."""
import os, sys, json, time, types, numpy as np, torch, soundfile as sf
ARGS = sys.argv[1:]                                           # before importing house_session, which rewrites sys.argv
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
import common as C, house_session as H, ladder, compile as CP, roundtrip as R, text2synth as T2, closed_loop_filter as CL
from synth import WavetableSynth, raw_from_vital, vital_coverage
from word_grid import DISPLAY, art
from fantasia_core.document.fx_insert import FxInsert
from fantasia_core.engine.fx import FxHost
NAME, WORD = (ARGS[0] if ARGS else "epiano"), (ARGS[1] if len(ARGS) > 1 else "dark")
OUT = os.path.join(HERE, "vitalgraph"); os.makedirs(OUT, exist_ok=True); SR = 48000; L = 10 * SR; DEV = C.DEVICE
if torch.backends.mps.is_available(): torch.mps.set_per_process_memory_fraction(0.5)
# the instrument, as the app would hand it over: Vital raw parameters (here from the twin's recovered patch)
raw = json.load(open(os.path.join(HERE, "instruments", f"{NAME}_patch.json")))["raw"]
s_ = WavetableSynth(H.CL.TABLE, SR, voices=1, bounds=H.B, interpolation=H.CL.TBL.get("interpolation", 1)).to(DEV)
vp0 = s_.vital_params({k: torch.tensor(float(v)) for k, v in raw.items()}); print("coverage:", vital_coverage(vp0) or "the twin models this patch")
p_inst = {k: v.to(DEV) for k, v in raw_from_vital(s_, vp0, keys=raw.keys()).items()}
inserts = [FxInsert(id="fx1", type="eq", params=dict(bands=[dict(type="eq_low_shelf", freq=150.0, gain=0.0, q=0.7), dict(type="eq_peak", freq=900.0, gain=0.0, q=1.0), dict(type="eq_peak", freq=3000.0, gain=0.0, q=1.0), dict(type="eq_high_shelf", freq=8000.0, gain=0.0, q=0.7)])),
           FxInsert(id="fx2", type="compressor", params=dict(threshold=-18.0, ratio=1.5, attack=20.0, release=150.0, makeup=0.0)),
           FxInsert(id="fx3", type="reverb", params=dict(room_size=0.5, damping=0.5, wet=0.15, dry=0.85))]
g = CP.compile_track(inserts, None, synth=s_, notes=H.NOTES, N=L, device=DEV)
with torch.no_grad(): s_.random_phase = False; y0 = s_.render(p_inst, H.NOTES, L)[0, 0]; s_.random_phase = True
t0 = time.time(); stops = ladder.run(WORD, y0, H.NOTES, OUT, f"{NAME}_vitalgraph", art(DISPLAY[NAME]), synth=s_, p_inst=p_inst, graph=g, log=lambda s: print(s, flush=True)); print(f"    ({time.time()-t0:.0f}s)")
# the closed loop: Vital plays the exported patch, FxHost runs the exported inserts, CLAP scores the result
T = C.text_emb("this sound is " + WORD); host = FxHost()
def score(y): 
    with torch.no_grad(): return float((C.embed(T2.loudness_norm(torch.tensor(y, dtype=torch.float32, device=DEV))) @ T.T).squeeze())
vit0 = CL.vital_render(vp0); tw0 = y0.cpu().numpy()
print(f"  start: twin dry {score(tw0):+.3f}   Vital playing the same patch {score(vit0):+.3f}   (the twin's own fidelity; env corr {R.env_corr(vit0, tw0):.3f})")
print("  stop | twin score | Vital + FxHost with the exported params | delta")
rows = []
for i in range(1, len(stops)):
    tgt = stops[i]["target"]; d = json.load(open(os.path.join(OUT, f"{NAME}_vitalgraph__{WORD}__stop{i}_d{tgt if tgt is not None else 'inf'}.json"))); ex = d["export"]
    vit = CL.vital_render(ex["vital"]); fx = [FxInsert(id=ins.id, type=ins.type, params=ex.get(ins.id, ins.params)) for ins in inserts]
    out = host.process(types.SimpleNamespace(id="t", fx=fx, fx_wires=[]), np.stack([vit, vit]), SR).mean(axis=0)
    sc = score(out); rows.append(dict(stop=i, target=tgt, twin=stops[i]["clap"], app=sc)); sf.write(os.path.join(OUT, f"{NAME}_vitalgraph__{WORD}__stop{i}_app.wav"), out / (np.abs(out).max() + 1e-9) * 0.5, SR)
    print(f"  {i}    | {stops[i]['clap']:+.3f}     | {sc:+.3f}                                   | {sc - stops[i]['clap']:+.3f}   {json.dumps({k: v for k, v in ex['vital'].items() if k in ('filter_1_cutoff', 'filter_1_resonance', 'oscillator_1_wave_frame', 'envelope_1_release')})}")
json.dump(dict(start=dict(twin=score(tw0), vital=score(vit0)), stops=rows), open(os.path.join(OUT, f"{NAME}_vitalgraph__{WORD}__closed_loop.json"), "w"), indent=1)
print("VITALGRAPH DONE")
