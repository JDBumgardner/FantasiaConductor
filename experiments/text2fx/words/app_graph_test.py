"""End to end on an APP graph: the cello recording through a user-style chain (eq > compressor > delay > reverb), the
ladder for one word, export each stop as app parameters, and render the EXPORTED parameters through the app's own
engine (FxHost) -- the round trip that makes the result usable in the app."""
import os, sys, json, time, types, numpy as np, torch, soundfile as sf
WORD = sys.argv[1] if len(sys.argv) > 1 else "dark"; sys.argv = [sys.argv[0]]
sys.path.insert(0, "."); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common as C, house_session as H, compile as CP, ladder, roundtrip as R
from fantasia_core.document.fx_insert import FxInsert
from fantasia_core.engine.fx import FxHost
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "appgraph"); SR = 48000; L = 10 * SR; DEV = C.DEVICE
src, _ = sf.read(os.path.join(HERE, "instruments", "cello_ref.wav")); x = torch.nn.functional.pad(torch.tensor(src[:L], dtype=torch.float32), (0, max(0, L - min(len(src), L)))).to(DEV)
inserts = [FxInsert(id="fx1", type="eq", params=dict(bands=[dict(type="eq_low_shelf", freq=150.0, gain=0.0, q=0.7), dict(type="eq_peak", freq=600.0, gain=0.0, q=1.0), dict(type="eq_peak", freq=2500.0, gain=0.0, q=1.0), dict(type="eq_high_shelf", freq=8000.0, gain=0.0, q=0.7)])),
           FxInsert(id="fx2", type="compressor", params=dict(threshold=-20.0, ratio=2.0, attack=10.0, release=100.0, makeup=0.0)),
           FxInsert(id="fx3", type="delay", params=dict(time=0.3, feedback=0.3, mix=0.1)),
           FxInsert(id="fx4", type="reverb", params=dict(room_size=0.5, damping=0.5, wet=0.15, dry=0.85))]
g = CP.compile_track(inserts, None, source_audio=x, N=L, device=DEV)
t0 = time.time(); stops = ladder.run(WORD, x, H.NOTES, OUT, "cello_appgraph", "a cello", graph=g, log=lambda s: print(s, flush=True)); print(f"    ({time.time()-t0:.0f}s)")
# the round trip: exported params -> FxHost -> CLAP score, vs the twin's own render
T = C.text_emb("this sound is " + WORD); host = FxHost()
print("  stop | twin score | exported params through the app engine: score, level vs twin, env corr")
for i in range(1, len(stops)):
    d = json.load(open(os.path.join(OUT, f"cello_appgraph__{WORD}__stop{i}_d{stops[i]['target'] if stops[i]['target'] is not None else 'inf'}.json")))
    ex = d["export"]; fx = [FxInsert(id=ins.id, type=ins.type, params=ex.get(ins.id, ins.params)) for ins in inserts]
    track = types.SimpleNamespace(id="t", fx=fx, fx_wires=[]); a = x.cpu().numpy(); ref = host.process(track, np.stack([a, a]), SR).mean(axis=0)
    twin = sf.read(os.path.join(OUT, f"cello_appgraph__{WORD}__stop{i}_d{stops[i]['target'] if stops[i]['target'] is not None else 'inf'}.wav"))[0]
    with torch.no_grad(): sc = float((C.embed(__import__('text2synth').loudness_norm(torch.tensor(ref, dtype=torch.float32, device=DEV))) @ T.T).squeeze())
    lvl = 20 * np.log10(np.std(ref) / (np.std(twin) + 1e-9)); print(f"  {i}    | {stops[i]['clap']:+.3f}     | {sc:+.3f}  {lvl:+.2f} dB  {R.env_corr(ref, twin):.3f}   | {json.dumps({k: {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'bands'} for k, v in ex.items()})[:160]}")
print("APPGRAPH DONE")
