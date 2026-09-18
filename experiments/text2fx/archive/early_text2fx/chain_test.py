"""H2's recipe (cosine + loudness match + locality + per-stage priors), chain
extended beyond EQ:

  I1  EQ -> drive              (19 params)
  I2  EQ -> drive -> reverb    (44 params)

Priors: EQ |gain|>6 dB and Q>2 (as H2); drive > 10 dB; reverb mix > 0.35.
"""
import os, sys, time, json
import numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from dasp_pytorch.modules import ParametricEQ, Distortion, NoiseShapedReverb

STEPS, SEEDS = 600, 3                    # flat 600 like H2 / the paper
LAM_LOC, LAM_PRI = 0.0236, 0.05          # from manifold_test calibration
PROMPT = "this sound is warm, dark and mellow"
DEV = C.DEVICE
if DEV == "mps":
    torch.mps.set_per_process_memory_fraction(0.50)      # of torch's ~5.3 GB working set = ~2.7 GB: OOM the process, never the box
y = C.kalimba(); SRC = torch.tensor(y, device=DEV); T = C.text_emb(PROMPT)

class Chain:
    def __init__(self, procs):
        self.procs = procs
        self.n = [p.num_params for p in procs]
        self.total = sum(self.n)
    def split(self, raw):                       # raw: (B, total) -> list of (B, k)
        out, i = [], 0
        for k in self.n: out.append(torch.sigmoid(raw[:, i:i + k])); i += k
        return out
    def apply(self, raw, w):                    # w: (samples,) -> (B, samples)
        B = raw.shape[0]
        x = w[None, None, :].expand(B, 1, -1)
        for p, q in zip(self.procs, self.split(raw)):
            x = p.process_normalized(x, q)
        return x[:, 0]
    def prior(self, raw):                       # -> (B,)
        pen = torch.zeros(raw.shape[0], device=raw.device)
        for p, q in zip(self.procs, self.split(raw)):
            d = C.eq_denorm(p, q); names = list(p.param_ranges)
            if isinstance(p, ParametricEQ):
                gi = [i for i, n in enumerate(names) if n.endswith("gain_db")]
                qi = [i for i, n in enumerate(names) if n.endswith("q_factor")]
                pen = pen + (torch.relu(d[:, gi].abs() - 6.0) ** 2).sum(1) / 36.0 \
                          + (torch.relu(d[:, qi] - 2.0) ** 2).sum(1) / 4.0
            elif isinstance(p, Distortion):
                pen = pen + (torch.relu(d[:, 0] - 10.0) ** 2) / 100.0
            elif isinstance(p, NoiseShapedReverb):
                pen = pen + (torch.relu(d[:, names.index("mix")] - 0.35) ** 2) / 0.1225
        return pen
    def describe(self, raw):                    # raw: (total,) single item
        lines = []
        for p, q in zip(self.procs, self.split(raw[None, :])):
            q = q[0]
            d = C.eq_denorm(p, q); names = list(p.param_ranges)
            if isinstance(p, ParametricEQ):
                for i, n in enumerate(names):
                    if n.endswith("gain_db"):
                        f = d[names.index(n.replace("gain_db", "cutoff_freq"))]
                        lines.append(f"       eq {n[:-8]:11s} {float(f):8.0f} Hz {float(d[i]):+6.2f} dB")
            elif isinstance(p, Distortion):
                lines.append(f"       drive {float(d[0]):+.2f} dB")
            elif isinstance(p, NoiseShapedReverb):
                mix = float(d[names.index("mix")])
                dec = [float(d[names.index(f'band{i}_decay')]) for i in range(12)]
                lines.append(f"       reverb mix={mix:.2f}  decay lo/mid/hi={dec[1]:.2f}/{dec[6]:.2f}/{dec[10]:.2f}")
        return "\n".join(lines)

def _optimise(chain, raw):
    """raw: (B, total) leaf. Returns (out (B, samples), raw)."""
    opt = torch.optim.Adam([raw], lr=1e-2)
    for _ in range(STEPS):
        opt.zero_grad()
        w = C.level_match(chain.apply(raw, SRC), SRC[None])
        per = -(C.embed(w) @ T.T).squeeze(-1) + LAM_LOC * C.mrstft(w, SRC[None].expand_as(w)) \
              + LAM_PRI * chain.prior(raw)
        per.sum().backward(); opt.step()
    with torch.no_grad(): return chain.apply(raw, SRC), raw.detach()

def run(chain, batch):
    """SEEDS inits in batches of `batch`. Returns best (score, audio, raw), steps, all scores."""
    g = torch.Generator().manual_seed(0)
    init = torch.randn(SEEDS, chain.total, generator=g)
    outs, raws = [], []
    for i in range(0, SEEDS, batch):
        raw = init[i:i + batch].to(DEV).requires_grad_(True)
        o, r = _optimise(chain, raw); outs.append(o); raws.append(r)
        if DEV == "mps": torch.mps.empty_cache()
    out, raw = torch.cat(outs), torch.cat(raws)
    with torch.no_grad(): scores = [C.score(out[i], T, SRC) for i in range(SEEDS)]
    i = int(np.argmax(scores))
    return scores[i], out[i], raw[i], STEPS, scores

if __name__ == "__main__":
    print(f"  dry={C.score(SRC, T, SRC):+.4f}   H2 (EQ only) was +0.3717\n", flush=True)
    def drive():                      # dasp bug: Distortion lacks the attr its base class reads
        d = Distortion(); d.sample_rate = C.SR
        d.param_ranges = {"drive_db": (0.0, 24.0)}   # module says gain_db, function wants drive_db
        return d
    CH = {"I0 eq only (=H2)":   (Chain([ParametricEQ(C.SR)]), 1),
          "I1 eq+drive":        (Chain([ParametricEQ(C.SR), drive()]), 1),
          "I2 eq+drive+reverb": (Chain([ParametricEQ(C.SR), drive(), C.fast_reverb(C.SR)]), 1)}
    res = {}
    for name, (chain, batch) in CH.items():
        t0 = time.time(); sc, au, raw, steps, all_sc = run(chain, batch)
        print(f"    {name:20s} seeds={[round(v,4) for v in all_sc]}  {steps} steps  {time.time()-t0:.0f}s", flush=True)
        with torch.no_grad(): d = float(C.mrstft(C.level_match(au, SRC), SRC))
        print(f"  {name:20s} BEST={sc:+.4f}  dist={d:.3f}\n{chain.describe(raw)}", flush=True)
        sf.write(os.path.join(C.HERE, f"chain_{name.split()[0]}.wav"),
                 np.clip(C.level_match(au, SRC).cpu().numpy(), -1, 1), C.SR)
        res[name] = dict(score=sc, dist=d)
    json.dump(res, open(os.path.join(C.HERE, "chain_results.json"), "w"), indent=1)
