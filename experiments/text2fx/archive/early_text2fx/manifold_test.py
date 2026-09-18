"""GAN-inspired "realness" terms on top of the paper's dasp EQ + cosine loss.

  H1  + locality        : multi-res STFT distance to the loudness-matched source
  H2  + locality + prior: quadratic penalty on |gain| beyond 6 dB and Q beyond 2

Neither can be attacked in the audio domain: one is fixed DSP, the other lives
in parameter space. Locality weight is calibrated so the hand shelf pays ~0.03
of cosine and the degenerate F resonance pays proportionally more.
"""
import os, sys, time, json
import numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

STEPS, SEEDS, LAM_PRI = 600, 2, 0.05
PROMPT = "this sound is warm, dark and mellow"
y = C.kalimba(); SRC = torch.tensor(y); T = C.text_emb(PROMPT)
eq = C.dasp_eq(); NP = eq.num_params; names = list(eq.param_ranges)
GAIN = [i for i, n in enumerate(names) if n.endswith("gain_db")]
QS   = [i for i, n in enumerate(names) if n.endswith("q_factor")]

def prior(p01):
    d = C.eq_denorm(eq, p01)
    return (torch.relu(d[GAIN].abs() - 6.0) ** 2).sum() / 36.0 + (torch.relu(d[QS] - 2.0) ** 2).sum() / 4.0

# calibration against the two known points
hand = C.hand_shelf(y)
with torch.no_grad(): d_hand = float(C.mrstft(C.level_match(hand, SRC), SRC))
LAM_LOC = 0.03 / d_hand
line = f"  MR-STFT hand shelf={d_hand:.3f}  lambda_loc={LAM_LOC:.4f}"
f_path = os.path.join(C.HERE, "dasp_F.wav")
if os.path.exists(f_path):
    F_au = torch.tensor(sf.read(f_path, dtype="float32")[0])
    with torch.no_grad(): d_F = float(C.mrstft(F_au, SRC))
    line += f"   F resonance={d_F:.3f} -> pays {LAM_LOC*d_F:.3f} vs hand {0.03:.3f}"
print(line, flush=True)

def run(use_prior, seed):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(NP, generator=g).requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=1e-2)
    for _ in range(STEPS):
        opt.zero_grad()
        w = C.level_match(C.eq_apply(eq, raw, SRC), SRC)
        loss = -(C.embed(w) @ T.T).squeeze() + LAM_LOC * C.mrstft(w, SRC)
        if use_prior: loss = loss + LAM_PRI * prior(torch.sigmoid(raw))
        loss.backward(); opt.step()
    out = C.eq_apply(eq, raw, SRC).detach()
    return C.score(out, T, SRC), out, torch.sigmoid(raw).detach()

if __name__ == "__main__":
    print(f"  dry={C.score(SRC, T, SRC):+.4f}   hand shelf={C.score(hand, T, SRC):+.4f}  <-- bar\n", flush=True)
    res = {}
    for name, use_prior in (("H1 locality", False), ("H2 locality+prior", True)):
        best = None
        for sd in range(SEEDS):
            t0 = time.time(); sc, au, pr = run(use_prior, sd)
            print(f"    {name:18s} seed{sd} score={sc:+.4f} ({time.time()-t0:.0f}s)", flush=True)
            if best is None or sc > best[0]: best = (sc, au, pr)
        sc, au, pr = best
        with torch.no_grad(): d = float(C.mrstft(C.level_match(au, SRC), SRC))
        lvl = 20 * np.log10(float(C.rms(au)) / float(C.rms(SRC)))
        print(f"  {name:18s} BEST={sc:+.4f}  level={lvl:+.1f} dB  dist={d:.3f}", flush=True)
        dn = C.eq_denorm(eq, pr)
        for j, n in enumerate(names):
            if not n.endswith("q_factor"): print(f"       {n:24s} {float(dn[j]):9.2f}", flush=True)
        sf.write(os.path.join(C.HERE, f"manifold_{name.split()[0]}.wav"),
                 np.clip(C.level_match(au, SRC).numpy(), -1, 1), C.SR)
        res[name] = dict(score=sc, level_db=lvl, dist=d)
    json.dump(res, open(os.path.join(C.HERE, "manifold_results.json"), "w"), indent=1)
