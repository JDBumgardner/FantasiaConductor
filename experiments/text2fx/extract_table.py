"""Extract a Vital wavetable as harmonic amplitudes by playing it.

Loads a .vital preset with routings stripped and everything but osc 1 off,
sweeps osc_1_wave_frame over F positions, renders a low sustained note, and
reads the magnitude at each harmonic. Result: (F, K) table the twin can play.
Phases are discarded (all losses are magnitude-based), so the waveform differs
from Vital's while every harmonic magnitude matches.
"""
import os, sys, glob, json, types, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fantasia_core import plugins as P, presets as PR

SR = 48000; N = types.SimpleNamespace
HERE = os.path.dirname(os.path.abspath(__file__))

def load_clean(preset_name):
    path = glob.glob(os.path.expanduser(f"~/Music/Vital/*/Presets/{preset_name}.vital"))[0]
    patch = PR.read_vital_file(path)
    patch["settings"]["modulations"] = []                       # routings are not parameters
    vital = P.load("Vital")
    assert P.restore_preset(vital, PR.splice_vital(P.preset_bytes(vital), patch))
    for k, v in {"oscillator_2_switch": 0, "oscillator_3_switch": 0, "sample_switch": 0,
                 "filter_1_switch": 0, "filter_2_switch": 0, "filter_fx_switch": 0,
                 "chorus_switch": 0, "reverb_switch": 0, "delay_switch": 0, "distortion_switch": 0,
                 "compressor_switch": 0, "eq_switch": 0, "flanger_switch": 0, "phaser_switch": 0,
                 "oscillator_1_unison_voices": 0.0, "oscillator_1_phase_randomization": 0.0,
                 "oscillator_1_level": 0.7071, "oscillator_1_transpose": 0.5, "oscillator_1_tune": 0.5,
                 "oscillator_1_frequency_morph_type": 0.0, "oscillator_1_distortion_type": 0.0,   # key != json name
                 "oscillator_1_unison_frame_spread": 0.5, "oscillator_1_detune_power": 0.65,       # outers on the same frame; init spread law
                 "envelope_1_attack": 0.0, "envelope_1_decay": 0.0, "envelope_1_sustain": 1.0, "envelope_1_release": 0.0,
                 "portamento_force": 0.0, "volume": 0.7397}.items():             # init-patch volume (-6.02 dB): keeps LEVEL_CAL valid
        r = P.set_param(vital, k, v)                                 # no silent misses: KeyError propagates
    for k, want in (("oscillator_1_frequency_morph_type", "None"), ("oscillator_1_unison_frame_spread", "0"), ("filter_1_switch", "Off")):
        got = P._params(vital)[k].string_value; assert got == want, f"{k} is {got!r}, wanted {want!r}"
    return vital, patch["settings"]["wavetables"][0]["name"]

def table_meta(preset_name):
    path = glob.glob(os.path.expanduser(f"~/Music/Vital/*/Presets/{preset_name}.vital"))[0]
    w = PR.read_vital_file(path)["settings"]["wavetables"][0]
    comp = w["groups"][0]["components"][0]
    return {"interpolation": comp.get("interpolation_style", comp.get("interpolation")),
            "keyframes": [k.get("position") for k in comp.get("keyframes", [])]}

def extract(preset_name, frames=33, K=128, pitch=36):
    vital, tname = load_clean(preset_name)
    f0 = 440 * 2 ** ((pitch - 69) / 12)
    table = np.zeros((frames, K), dtype=np.float32)
    for i, raw in enumerate(np.linspace(0, 1, frames)):
        P.set_param(vital, "oscillator_1_wave_frame", float(raw))
        y = P.render_notes(vital, [N(pitch=pitch, start=0.0, duration=1.5, velocity=100)], 1.5, sr=SR, tail=0.0).mean(axis=1)
        m = y[int(0.4 * SR):int(1.4 * SR)]; n = len(m)                       # 1 s -> 1 Hz bins
        X = np.abs(np.fft.rfft(m * np.hanning(n))); fr = np.fft.rfftfreq(n, 1 / SR)
        for k in range(1, K + 1):
            j = int(round(k * f0 / (SR / n))); table[i, k - 1] = X[max(j - 2, 0):j + 3].max()
    table /= table.max()
    # Absolute scale: the twin must be as loud as Vital at the same nominal level, or the
    # exported level is wrong. Render both at frame raw 0.5 / level 0.5 and match RMS,
    # folding in the saw-measured LEVEL_CAL (Vital level = torch level / 0.694).
    from synth import WavetableSynth, raw_from_physical, bounds_from_notes
    note = [(pitch, 0.0, 1.5, 100)]
    P.set_param(vital, "oscillator_1_wave_frame", 0.5); P.set_param(vital, "oscillator_1_level", 0.5 ** 0.5)
    yv = P.render_notes(vital, [N(pitch=pitch, start=0.0, duration=1.5, velocity=100)], 1.5, sr=SR, tail=0.0).mean(axis=1)[int(0.4*SR):int(1.4*SR)]
    tw = WavetableSynth(torch.tensor(table), SR, voices=1, bounds=bounds_from_notes(note))
    p = raw_from_physical(tw, frame=(frames - 1) / 2, detune_semis=0.0, blend=0.0, level=0.5 * 0.694, attack=0.001, decay=0.5, sustain=1.0, release=0.1)
    with torch.no_grad(): yt = tw.render(p, note, int(1.5 * SR))[0, 0].numpy()[int(0.4*SR):int(1.4*SR)]
    amp_scale = float(np.sqrt((yv ** 2).mean()) / np.sqrt((yt ** 2).mean()))
    table *= amp_scale
    print(f"  amp_scale={amp_scale:.3f}")
    out = os.path.join(HERE, "tables", f"{tname.lower().replace(' ', '_')}.pt")
    torch.save({"table": torch.tensor(table), "name": tname, "preset": preset_name, "frames": frames, "K": K, **table_meta(preset_name)}, out)
    return table, tname, out

if __name__ == "__main__":
    for preset in sys.argv[1:] or ["Moog Pluck"]:
        t, name, out = extract(preset)
        print(f"  '{name}' from {preset}: {t.shape} -> {out}")
        print("  first 8 harmonics (rel h1) at 9 frame positions:")
        for i in np.linspace(0, len(t) - 1, 9).astype(int):
            row = t[i, :8] / (t[i, 0] + 1e-9)
            print(f"    frame {i/(len(t)-1)*256:5.0f}: " + " ".join(f"{v:5.2f}" for v in row) + f"   h1={t[i,0]:.2f}")
