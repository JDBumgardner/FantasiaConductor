"""Text -> synth + FX, end to end differentiable.

clone (wavetable osc, unison off, amp env, filter + filter env)  ->  GRAFX (EQ -> drive -> reverb)
->  loudness-normalise  ->  CLAP cosine to the prompt.  Every parameter gets a gradient.

Unlike the FX-only runs, the generator can only produce synth sounds, so there is no
off-manifold cheat; what remains is that a prompt gives a direction, and parameters
run to their (musical) rails. Level is fixed so silence is not an option.
"""
import os, sys, json, time, math, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, soundfile as sf
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from synth import WavetableSynth, sustained_notes, bounds_from_notes, init_raw, raw_from_physical
from grafx.data import GRAFX, NodeConfigs, convert_to_tensor
from grafx.render import render_grafx, reorder_for_fast_render
from grafx.render.prepare import prepare_render
from grafx import processors as P

SR = 48000; L = 10 * SR; DEV = C.DEVICE
if DEV == "mps": torch.mps.set_per_process_memory_fraction(0.5)
STEPS, RESTARTS, LAM_PRI, LAM_GS = 300, 3, 0.05, 0.02
OUT = os.path.join(C.HERE, "text2synth"); notes = sustained_notes(); B = bounds_from_notes(notes)
TBL = torch.load(os.path.join(C.HERE, "tables", "classic_fade.pt")); TABLE = TBL["table"]; F_T = TABLE.shape[0]
DB_PER_LG, DB_PER_LPG = 40 / math.log(10), 20 / math.log(10)
FIXED_LEVEL = 0.35 * 0.694                      # not optimised: silence must not be reachable

def build_fx():
    G = GRAFX(config=NodeConfigs(["eq", "dist", "reverb"])); G.add_serial_chain(["in", "eq", "dist", "reverb", "out"])
    rd = prepare_render(reorder_for_fast_render(convert_to_tensor(G), method="beam"))
    rev = P.DryWet(P.FilteredNoiseShapingReverb(sr=SR, processor_channel="mono", zerophase=False, flashfftconv=False, max_input_len=L), external_param=False)
    procs = torch.nn.ModuleDict({"eq": P.GainStagingRegularization(P.ParametricEqualizer(num_filters=6, processor_channel="mono")),
                                 "dist": P.GainStagingRegularization(P.TanhDistortion()), "reverb": P.GainStagingRegularization(rev)}).to(DEV)
    return procs, rd

def init_fx(seed):
    g = torch.Generator().manual_seed(seed); r = lambda *s, sd=0.3: sd * torch.randn(*s, generator=g)
    f0 = torch.tensor(np.geomspace(80, 12000, 6), dtype=torch.float32)
    p = {"eq": {"w0": torch.logit(2 * f0 / SR)[None, None] + r(1, 1, 6, sd=0.1), "q_inv": r(1, 1, 6), "log_gain": r(1, 1, 6, sd=0.2)},
         "dist": {"log_pre_gain": r(1, 1)},
         "reverb": {"log_decay": r(1, 1, 12, sd=0.5), "log_gain": 1.0 + r(1, 1, 12, sd=0.1), "drywet_weight": torch.full((1, 1), -1.0) + r(1, 1)}}
    return {t: {k: v.to(DEV).requires_grad_(True) for k, v in d.items()} for t, d in p.items()}

def fx_prior(p):
    hz = torch.sigmoid(p["eq"]["w0"][0, 0]) * SR / 2; db = DB_PER_LG * p["eq"]["log_gain"][0, 0]; q = torch.exp(-p["eq"]["q_inv"][0, 0])
    pen = (torch.relu(db.abs() - 6.0) ** 2).sum() / 36 + (torch.relu(q - 2.0) ** 2).sum() / 4
    lf = torch.log10(hz); d = (lf[:, None] - lf[None, :]).abs() + torch.eye(6, device=lf.device) * 9
    pen = pen + 0.35 * (torch.relu(0.2 - d) * db.abs()[:, None] * db.abs()[None, :]).sum() / 6
    drive = DB_PER_LPG * p["dist"]["log_pre_gain"][0, 0]; pen = pen + torch.relu(drive - 10.0) ** 2 / 100 + torch.relu(-drive) ** 2 / 4
    pen = pen + (torch.relu(-p["reverb"]["log_gain"]) ** 2).sum() + torch.relu(torch.sigmoid(p["reverb"]["drywet_weight"][0, 0]) - 0.5) ** 2 / 0.25
    return pen

def init_synth(s_, seed):
    p = init_raw(s_, seed=seed)                                     # osc + amp env family
    logit = lambda x: torch.logit(torch.tensor(float(x)).clamp(1e-3, 1 - 1e-3))
    p["frame"] = logit([0.2, 0.5, 0.8][seed % 3])
    fam = [dict(cut=0.5, res=0.2, amt=0.4, a=0.002, d=0.2, su=0.3, r=0.3), dict(cut=0.4, res=0.5, amt=0.6, a=0.005, d=0.5, su=0.5, r=0.5), dict(cut=0.7, res=0.1, amt=0.2, a=0.05, d=0.3, su=0.8, r=0.5)][seed % 3]
    p.update(cutoff=logit(fam["cut"]), resonance=logit(fam["res"]), fenv_amount=logit(fam["amt"]), fattack=logit(s_._t_inv("attack", fam["a"])),
             fdecay=logit(s_._t_inv("decay", fam["d"])), fsustain=logit(fam["su"]), frelease=logit(s_._t_inv("release", fam["r"])))
    p["level"] = raw_from_physical(s_, frame=0, detune_semis=0, blend=0, level=FIXED_LEVEL, attack=0.01, decay=0.1, sustain=1, release=0.1)["level"]
    return {k: v.to(DEV).requires_grad_(k != "level") for k, v in p.items()}

def loudness_norm(w, target_rms=0.12):
    lw = C.loudness(w); return w * (target_rms / (lw + 1e-3 * target_rms))

def describe(s_, ps, pf):
    ph = s_.physical(ps); hz = torch.sigmoid(pf["eq"]["w0"][0, 0]) * SR / 2; db = DB_PER_LG * pf["eq"]["log_gain"][0, 0]
    return dict(synth={k: round(float(v), 3) for k, v in ph.items() if k not in ("attack_power", "decay_power", "release_power", "fattack_power", "fdecay_power", "frelease_power")},
                eq=[(int(h), round(float(g), 1)) for h, g in sorted(zip(hz, db))], drive_db=round(float(DB_PER_LPG * pf["dist"]["log_pre_gain"][0, 0]), 1),
                reverb_mix=round(float(torch.sigmoid(pf["reverb"]["drywet_weight"][0, 0])), 2))

def run(prompt):
    T = C.text_emb(prompt); procs, rd = build_fx(); best = None
    for r in range(RESTARTS):
        s_ = WavetableSynth(TABLE, SR, voices=1, bounds=B, interpolation=1).to(DEV)
        ps, pf = init_synth(s_, r), init_fx(r)
        opt = torch.optim.Adam([v for v in ps.values() if v.requires_grad] + [v for d in pf.values() for v in d.values()], lr=0.02)
        t0 = time.time()
        for i in range(STEPS):
            opt.zero_grad()
            y = s_.render(ps, notes, L)                                     # (1,1,L)
            out, inter, _ = render_grafx(procs, y, pf, rd, input_signal_grad=True)   # default False detaches the synth!
            gs = sum(v["gain_reg"].sum() for v in inter if isinstance(v, dict) and "gain_reg" in v)
            w = loudness_norm(out[0, 0])
            loss = -(C.embed(w) @ T.T).squeeze() + LAM_PRI * fx_prior(pf) + LAM_GS * gs
            loss.backward(); opt.step()
        with torch.no_grad():
            y = s_.render(ps, notes, L); out, _, _ = render_grafx(procs, y, pf, rd); w = loudness_norm(out[0, 0])
            score = float((C.embed(w) @ T.T).squeeze())
        print(f"    restart {r}: score={score:+.4f}  ({(time.time()-t0)/STEPS*1000:.0f} ms/step)", flush=True)
        if best is None or score > best[0]: best = (score, {k: v.detach() for k, v in ps.items()}, {t: {k: v.detach() for k, v in d.items()} for t, d in pf.items()}, s_, y[0, 0].detach(), out[0, 0].detach())
    return best

if __name__ == "__main__":
    prompts = sys.argv[1:] or ["this sound is underwater", "this sound is woody", "this sound is church bells ringing", "this sound is an african dance on fire", "this sound is folkish and melodic"]
    for prompt in prompts:
        slug = "".join(ch if ch.isalnum() else "_" for ch in prompt.replace("this sound is ", ""))[:32].strip("_")
        print(f"  '{prompt}'", flush=True)
        score, ps, pf, s_, dry, wet = run(prompt)
        d = describe(s_, ps, pf); vp = s_.vital_params(ps)
        sf.write(os.path.join(OUT, f"{slug}_synth_dry.wav"), (dry / (dry.abs().max() + 1e-9) * 0.5).cpu().numpy(), SR)
        sf.write(os.path.join(OUT, f"{slug}_synth_fx.wav"), (wet / (wet.abs().max() + 1e-9) * 0.5).cpu().numpy(), SR)
        json.dump({"prompt": prompt, "score": score, "vital_params": vp, "describe": d}, open(os.path.join(OUT, f"{slug}.json"), "w"), indent=1)
        sy = d["synth"]
        print(f"  BEST {score:+.4f} | frame {sy['frame']:.1f}/{F_T-1} cutoff {sy['cutoff_raw']:.2f} res {sy['resonance']:.2f} envamt {sy['fenv_amount']:.2f} "
              f"amp A{sy['attack']*1000:.0f}ms D{sy['decay']:.2f} S{sy['sustain']:.2f} R{sy['release']:.2f} filt D{sy['fdecay']:.2f} S{sy['fsustain']:.2f} "
              f"| eq {d['eq']} drive {d['drive_db']} dB reverb {d['reverb_mix']}", flush=True)
    print("ALL DONE")
