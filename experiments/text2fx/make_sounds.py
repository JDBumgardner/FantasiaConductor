"""Make Vital patches that imitate other instruments.

For each GM soundfont instrument: render the sustained phrase as the reference, let
the twin find a Vital patch for it (oscillator, unison off, filter + both envelopes),
export the patch into real Vital and render it. Outputs pairs in sounds/.
"""
import os, sys, json, time, types, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, soundfile as sf
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], "1"]                                   # closed_loop_filter reads VOICES from argv
import common as C, closed_loop_filter as CL
from fantasia_core.engine.midi_render import MidiRenderer
from synth import sustained_notes

SR = 48000; L = 10 * SR; notes = sustained_notes(); OUT = os.path.join(C.HERE, "sounds")
TARGETS = {"marimba": 12, "kalimba": 108, "flute": 73, "epiano": 4, "pizzicato": 45, "brass": 61}
N = lambda p, s, d, v: types.SimpleNamespace(pitch=p, start=s, duration=d, velocity=v)

def reference(program):
    clip = types.SimpleNamespace(notes=[N(*n) for n in notes], duration=10.0, content_type="midi", id="ref")
    y = MidiRenderer("assets/soundfonts/GeneralUser-GS-v1.471.sf2", SR).render(clip, program).mean(axis=1)[:L]
    y = np.pad(y, (0, max(0, L - len(y)))); return (y / (np.abs(y).max() + 1e-9) * 0.5).astype(np.float32)

if __name__ == "__main__":
    for name, prog in TARGETS.items():
        t0 = time.time(); ref = reference(prog); sf.write(os.path.join(OUT, f"{name}_ref.wav"), ref, SR)
        s_, p = CL.recover(ref, steps=250, restarts=6)
        got = s_.physical(p); vp = s_.vital_params(p); vp["oscillator_1_level"] = min(1.0, (float(got["level"]) / CL.LEVEL_CAL) ** 0.5)
        out = CL.vital_render(vp); sf.write(os.path.join(OUT, f"{name}_vital.wav"), out / (np.abs(out).max() + 1e-9) * 0.5, SR)
        phys = {k: round(float(v), 4) for k, v in got.items()}
        json.dump({"vital_params": vp, "physical": phys}, open(os.path.join(OUT, f"{name}_patch.json"), "w"), indent=1)
        kf = int(phys["frame"] * 256 / (s_.F - 1))
        print(f"  {name:10s} {time.time()-t0:4.0f}s  frame {kf:3d}  cutoff {phys['cutoff_raw']:.2f} res {phys['resonance']:.2f} envamt {phys['fenv_amount']:.2f} "
              f"| amp A{phys['attack']*1000:.0f}ms D{phys['decay']:.2f} S{phys['sustain']:.2f} R{phys['release']:.2f} "
              f"| filt A{phys['fattack']*1000:.0f}ms D{phys['fdecay']:.2f} S{phys['fsustain']:.2f} R{phys['frelease']:.2f} "
              f"| Vital-vs-ref corr {CL.env_corr(out, ref):.2f} bands {CL.band_diff(out, ref)}", flush=True)
    print("ALL DONE")
