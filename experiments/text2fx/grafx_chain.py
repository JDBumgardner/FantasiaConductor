"""The I2 chain (EQ -> drive -> reverb) ported to GRAFX, H2 recipe.

Loss = -cos(CLAP, prompt) on loudness-matched audio
       + LAM_LOC * MR-STFT distance to source          (locality)
       + LAM_PRI * priors  (|EQ gain| > 6 dB, Q > 2, band crowding, drive > 10 dB, mix > 0.35)
       + LAM_GS  * sum of GRAFX GainStagingRegularization over nodes

Target: reproduce I2 (~0.51) at GRAFX speed. 3 seeds, 600 steps, sequential.
"""
import os, sys, time, json, warnings, math
warnings.filterwarnings("ignore")
import torch, numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from grafx.data import GRAFX, NodeConfigs, convert_to_tensor
from grafx.render import render_grafx, reorder_for_fast_render
from grafx.render.prepare import prepare_render
from grafx import processors as P

STEPS, SEEDS = 600, 3
LAM_LOC, LAM_PRI = 0.0236, 0.05
LAM_GS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0     # gain-staging weight; 0 = faithful I2
PROMPT = sys.argv[2] if len(sys.argv) > 2 else "this sound is warm, dark and mellow"
SOURCE = sys.argv[3] if len(sys.argv) > 3 else None             # wav path; default = the kalimba
TAG    = sys.argv[4] if len(sys.argv) > 4 else "kalimba"
DEV = C.DEVICE
if DEV == "mps": torch.mps.set_per_process_memory_fraction(0.5)
if SOURCE:
    import soundfile as sf
    y, _ = sf.read(SOURCE, dtype="float32"); y = y.mean(axis=1) if y.ndim > 1 else y
    y = (y / np.abs(y).max() * 0.7).astype(np.float32)[: 10 * C.SR]
else:
    y = C.kalimba()
N = len(y); SR = C.SR
SRC = torch.tensor(y, device=DEV); T = C.text_emb(PROMPT)
DB_PER_LG, DB_PER_LPG = 40 / math.log(10), 20 / math.log(10)     # 17.37, 8.69

def build():
    G = GRAFX(config=NodeConfigs(["eq", "dist", "reverb"]))
    G.add_serial_chain(["in", "eq", "dist", "reverb", "out"])
    G_t = reorder_for_fast_render(convert_to_tensor(G), method="beam")
    rd = prepare_render(G_t)
    rev = P.DryWet(P.FilteredNoiseShapingReverb(sr=SR, processor_channel="mono", zerophase=False,
                                                flashfftconv=False, max_input_len=N), external_param=False)
    procs = torch.nn.ModuleDict({
        "eq":     P.GainStagingRegularization(P.ParametricEqualizer(num_filters=6, processor_channel="mono")),
        "dist":   P.GainStagingRegularization(P.TanhDistortion()),
        "reverb": P.GainStagingRegularization(rev),
    }).to(DEV)
    return procs, rd

def init_params(procs, seed):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, sd=0.3: (sd * torch.randn(*s, generator=g))
    f0 = torch.tensor(np.geomspace(80, 12000, 6), dtype=torch.float32)          # spread the 6 bands
    p = {
        "eq":     {"w0": torch.logit(2 * f0 / SR)[None, None] + r(1, 1, 6, sd=0.1),
                   "q_inv": r(1, 1, 6), "log_gain": r(1, 1, 6, sd=0.2)},
        "dist":   {"log_pre_gain": r(1, 1)},
        # NB: GRAFX's reverb "log_gain" is applied as a LINEAR gain (0 = silent) despite its name
        "reverb": {"log_decay": r(1, 1, 12, sd=0.5), "log_gain": 1.0 + r(1, 1, 12, sd=0.1),
                   "drywet_weight": torch.full((1, 1), -1.0) + r(1, 1)},
    }
    return {t: {k: v.to(DEV).requires_grad_(True) for k, v in d.items()} for t, d in p.items()}

def eq_hz_db_q(p):
    hz = torch.sigmoid(p["eq"]["w0"][0, 0]) * SR / 2
    return hz, DB_PER_LG * p["eq"]["log_gain"][0, 0], torch.exp(-p["eq"]["q_inv"][0, 0])

def prior(p):
    hz, db, q = eq_hz_db_q(p)
    pen = (torch.relu(db.abs() - 6.0) ** 2).sum() / 36.0 + (torch.relu(q - 2.0) ** 2).sum() / 4.0
    lf = torch.log10(hz); d = (lf[:, None] - lf[None, :]).abs() + torch.eye(6, device=lf.device) * 9
    pen = pen + 0.35 * (torch.relu(0.20 - d) * db.abs()[:, None] * db.abs()[None, :]).sum() / 6
    drive_db = DB_PER_LPG * p["dist"]["log_pre_gain"][0, 0]
    pen = pen + torch.relu(drive_db - 10.0) ** 2 / 100.0 + torch.relu(-drive_db) ** 2 / 4.0
    mix = torch.sigmoid(p["reverb"]["drywet_weight"][0, 0])
    pen = pen + (torch.relu(-p["reverb"]["log_gain"]) ** 2).sum()          # keep IR band gains >= 0
    return pen + torch.relu(mix - 0.35) ** 2 / 0.1225

def render(procs, p, rd):
    out, inter, _ = render_grafx(procs, SRC[None, None, :], p, rd)
    gs = sum(v["gain_reg"].sum() for v in inter if isinstance(v, dict) and "gain_reg" in v) \
         if isinstance(inter, list) else torch.zeros((), device=DEV)
    return out[0, 0], gs

def optimise(procs, rd, seed):
    p = init_params(procs, seed)
    opt = torch.optim.Adam([v for d in p.values() for v in d.values()], lr=1e-2)
    for _ in range(STEPS):
        opt.zero_grad()
        w, gs = render(procs, p, rd); w = C.level_match(w, SRC)
        loss = -(C.embed(w) @ T.T).squeeze() + LAM_LOC * C.mrstft(w, SRC) + LAM_PRI * prior(p) + LAM_GS * gs
        loss.backward(); opt.step()
    with torch.no_grad():
        w, gs = render(procs, p, rd)
    return C.score(w, T, SRC), w, {t: {k: v.detach() for k, v in d.items()} for t, d in p.items()}, float(gs)

def describe(p):
    hz, db, q = eq_hz_db_q(p)
    lines = [f"       eq {float(h):8.0f} Hz {float(g):+6.2f} dB  Q={float(qq):5.2f}" for h, g, qq in sorted(zip(hz, db, q))]
    lines.append(f"       drive {float(DB_PER_LPG * p['dist']['log_pre_gain'][0,0]):+.2f} dB")
    lines.append(f"       reverb mix={float(torch.sigmoid(p['reverb']['drywet_weight'][0,0])):.2f}  "
                 f"RT60 lo/mid/hi={[int(50 + 1950*float(torch.sigmoid(p['reverb']['log_decay'][0,0,i]))) for i in (1,6,10)]} ms  "
                 f"band gain mean={float(p['reverb']['log_gain'].mean()):.2f}")
    return "\n".join(lines)

if __name__ == "__main__":
    procs, rd = build()
    print(f"  [{TAG}] '{PROMPT}'  gs={LAM_GS}  dry={C.score(SRC, T, SRC):+.4f}", flush=True)
    best, allsc = None, []
    for sd in range(SEEDS):
        t0 = time.time(); sc, w, p, gs = optimise(procs, rd, sd); allsc.append(sc)
        print(f"    seed{sd} score={sc:+.4f}  gain_reg={gs:.3f}  ({time.time()-t0:.0f}s, {(time.time()-t0)/STEPS*1000:.0f} ms/step)", flush=True)
        if best is None or sc > best[0]: best = (sc, w, p)
        if DEV == "mps": torch.mps.empty_cache()
    sc, w, p = best
    with torch.no_grad(): d = float(C.mrstft(C.level_match(w, SRC), SRC))
    print(f"  BEST={sc:+.4f}  seeds={[round(s,4) for s in allsc]}  dist={d:.3f}\n{describe(p)}", flush=True)
    slug = "".join(ch if ch.isalnum() else "_" for ch in PROMPT.replace("this sound is ", ""))[:40].strip("_")
    sf.write(os.path.join(C.HERE, f"t2fx_{TAG}_{slug}_gs{LAM_GS}.wav"), np.clip(C.level_match(w, SRC).cpu().numpy(), -1, 1), SR)
    sf.write(os.path.join(C.HERE, f"t2fx_{TAG}_dry.wav"), y, SR)
    json.dump(dict(prompt=PROMPT, score=sc, seeds=allsc, dist=d, dry=C.score(SRC, T, SRC)),
              open(os.path.join(C.HERE, f"t2fx_{TAG}_{slug}_gs{LAM_GS}.json"), "w"), indent=1)
