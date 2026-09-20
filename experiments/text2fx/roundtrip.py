"""Does the compiled twin of an app graph sound like the app's engine? Render the same graph at the SAME parameters
through pedalboard (fantasia_core.engine.fx.FxHost) and through SearchGraph, compare per node and end to end."""
import os, sys, types, math, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compile as CP
from fantasia_core.engine.fx import FxHost, _make
from fantasia_core.document.fx_insert import FxInsert, FxWire
SR = 48000

def bands(x, sr=SR):
    X = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64))) ** 2; f = np.fft.rfftfreq(len(x), 1 / sr); t = X.sum() + 1e-12
    return np.array([10 * np.log10(X[(f >= lo) & (f < hi)].sum() / t + 1e-12) for lo, hi in ((20, 200), (200, 1000), (1000, 4000), (4000, 10000), (10000, 20000))])
def env_corr(a, b, w=2400):
    e = lambda x: np.sqrt((np.asarray(x)[:len(x) // w * w].reshape(-1, w) ** 2).mean(1)); return float(np.corrcoef(e(a), e(b))[0, 1])
def compare(a, b, label):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32); n = min(len(a), len(b)); a, b = a[:n], b[:n]
    lvl = 20 * math.log10((np.std(a) + 1e-9) / (np.std(b) + 1e-9)); bd = bands(a) - bands(b); ec = env_corr(a, b)
    snr = 10 * math.log10(np.sum(b ** 2) / (np.sum((a - b) ** 2) + 1e-12))
    print(f"  {label:34s} level {lvl:+5.2f} dB  bands {np.round(bd, 1).tolist()}  env corr {ec:.3f}  sample snr {snr:5.1f} dB")
    return dict(level_db=lvl, bands=bd.tolist(), env_corr=ec, snr=snr)

def node_roundtrip(spec, audio, N=None):
    """One insert: pedalboard vs its twin at the app's parameters."""
    N = N or len(audio); plug = _make(spec)
    a = np.asarray(audio, dtype=np.float32); ref = plug(np.stack([a, a]), SR).mean(axis=0) if plug is not None else a      # the app runs stereo tracks; Freeverb differs between its mono and stereo paths
    g = CP.compile_track([FxInsert(id="fx1", type=spec["type"], params=spec.get("params") or {})], None, source_audio=torch.as_tensor(audio, dtype=torch.float32), N=N)
    with torch.no_grad(): out, _ = g.render(g.init())
    return compare(out.numpy(), ref, f"{spec['type']:11s} {'exact' if g.twins['fx1'].exact else 'approx'}"), out.numpy(), ref

if __name__ == "__main__":
    import soundfile as sf
    audio, _ = sf.read("experiments/text2fx/words/instruments/cello_ref.wav"); audio = audio[:5 * SR].astype(np.float32)
    print("per node, pedalboard vs twin at the same app parameters (cello soundfont, 5 s):")
    specs = [dict(type="gain", params=dict(gain=-6.0)), dict(type="saturator", params=dict(drive=12.0, output=-6.0)), dict(type="distortion", params=dict(drive=20.0)),
             dict(type="eq", params=dict(bands=[dict(type="eq_low_shelf", freq=150.0, gain=4.0, q=0.7), dict(type="eq_peak", freq=800.0, gain=-6.0, q=2.0), dict(type="eq_peak", freq=3000.0, gain=5.0, q=1.0), dict(type="eq_high_shelf", freq=8000.0, gain=-5.0, q=0.7)])),
             dict(type="lowpass", params=dict(cutoff=1500.0)), dict(type="highpass", params=dict(cutoff=300.0)),
             dict(type="delay", params=dict(time=0.3, feedback=0.5, mix=0.5)), dict(type="chorus", params=dict(rate=1.5, depth=0.5, centre_delay=7.0, mix=0.5)),
             dict(type="compressor", params=dict(threshold=-24.0, ratio=4.0, attack=10.0, release=100.0, makeup=0.0)), dict(type="limiter", params=dict(threshold=-12.0)),
             dict(type="gate", params=dict(threshold=-30.0, ratio=4.0, attack=1.0, release=100.0)), dict(type="reverb", params=dict(room_size=0.7, damping=0.5, wet=0.4, dry=0.6))]
    for s in specs: node_roundtrip(s, audio)


def graph_roundtrip(inserts, wires, audio, label, N=None):
    """A whole track graph through the app's FxHost vs the compiled twin, at the app's current parameters."""
    N = N or len(audio)
    track = types.SimpleNamespace(id="t1", fx=inserts, fx_wires=wires or [])
    a = np.asarray(audio, dtype=np.float32); ref = FxHost().process(track, np.stack([a, a]), SR)      # the host wants (channels, samples)
    ref = ref.mean(axis=0) if ref.ndim == 2 else ref
    g = CP.compile_track(inserts, wires, source_audio=torch.as_tensor(a), N=N)
    with torch.no_grad(): out, _ = g.render(g.init())
    r = compare(out.numpy(), ref, label); r["frozen"] = g.frozen; return r, g

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "graphs":
    import soundfile as sf
    audio, _ = sf.read("experiments/text2fx/words/instruments/cello_ref.wav"); audio = audio[:5 * SR].astype(np.float32)
    print("whole graphs, FxHost vs compiled twin:")
    serial = [FxInsert(id="fx1", type="eq", params=dict(bands=[dict(type="eq_low_shelf", freq=150.0, gain=3.0, q=0.7), dict(type="eq_peak", freq=900.0, gain=-4.0, q=1.5), dict(type="eq_high_shelf", freq=7000.0, gain=2.0, q=0.7)])),
              FxInsert(id="fx2", type="compressor", params=dict(threshold=-22.0, ratio=3.0, attack=15.0, release=120.0, makeup=2.0)),
              FxInsert(id="fx3", type="saturator", params=dict(drive=8.0, output=-4.0)),
              FxInsert(id="fx4", type="delay", params=dict(time=0.35, feedback=0.4, mix=0.3))]
    graph_roundtrip(serial, None, audio, "serial eq>comp>sat>delay")
    # a branch: source -> eq -> out and source -> highpass -> saturator -> out (merge at out)
    par = [FxInsert(id="a", type="eq", params=dict(bands=[dict(type="eq_peak", freq=500.0, gain=-6.0, q=1.0)])), FxInsert(id="b", type="highpass", params=dict(cutoff=2000.0)), FxInsert(id="c", type="saturator", params=dict(drive=15.0, output=-9.0))]
    wires = [FxWire("in", "a"), FxWire("a", "out"), FxWire("in", "b"), FxWire("b", "c"), FxWire("c", "out")]
    graph_roundtrip(par, wires, audio, "parallel: eq || highpass>sat")
    # a frozen VST in the middle: identity in the twin; the real one changes the sound -> the mismatch is the flag's meaning
    vst = [FxInsert(id="fx1", type="gain", params=dict(gain=-3.0)), FxInsert(id="fx2", type="vst", params=dict(name="Nonexistent"))]
    r, g = graph_roundtrip(vst, None, audio, "gain > unknown vst (frozen)"); print("   frozen:", g.frozen)
