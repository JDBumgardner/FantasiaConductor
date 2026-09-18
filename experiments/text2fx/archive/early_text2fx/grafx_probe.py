"""Minimal GRAFX graph on our audio: EQ -> tanh drive -> reverb, 10 s mono.

Checks the three things that bit us with dasp: does it run at this length,
does it fit in memory, and are the MPS gradients correct at FULL length.
"""
import os, sys, time, warnings, resource
warnings.filterwarnings("ignore")
import torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from grafx.data import GRAFX, NodeConfigs, convert_to_tensor
from grafx.render.prepare import prepare_render
from grafx.render import render_grafx, reorder_for_fast_render
from grafx import processors as P

y = C.kalimba(); SR = C.SR; N = len(y)

def build(dev):
    G = GRAFX(config=NodeConfigs(["eq", "dist", "reverb"]))
    first, last = G.add_serial_chain(["in", "eq", "dist", "reverb", "out"])
    G_t = convert_to_tensor(G)
    G_t = reorder_for_fast_render(G_t, method="beam")
    rd = prepare_render(G_t)
    procs = {
        "eq":     P.ParametricEqualizer(num_filters=6, processor_channel="mono"),
        "dist":   P.TanhDistortion(),
        "reverb": P.FilteredNoiseShapingReverb(sr=SR, processor_channel="mono", zerophase=False,
                                               flashfftconv=False, max_input_len=N),
    }
    procs = torch.nn.ModuleDict(procs).to(dev)
    g = torch.Generator().manual_seed(0)
    params = {}
    for t, p in procs.items():
        params[t] = {k: (0.1 * torch.randn(1, *([s] if isinstance(s, int) else list(s)), generator=g)).to(dev).requires_grad_(True)
                     for k, s in p.parameter_size().items()}
    return procs, params, rd

def run(dev, n_time=3):
    procs, params, rd = build(dev)
    x = torch.tensor(y, device=dev)[None, None, :]          # |V0|=1 input, C=1, L
    T = C.text_emb("this sound is warm, dark and mellow").to(dev)
    C._model.to(dev); C._mel.to(dev); C._todb.to(dev); C.DEVICE = dev
    def step():
        for d in params.values():
            for v in d.values(): v.grad = None
        out = render_grafx(procs, x, params, rd)
        out = out if isinstance(out, torch.Tensor) else out[0]
        w = C.level_match(out[0, 0], x[0, 0])
        loss = -(C.embed(w) @ T.T).squeeze() + 0.0236 * C.mrstft(w, x[0, 0])
        loss.backward()
        if dev == "mps": torch.mps.synchronize()
        return float(loss)
    l = step(); t = time.time()
    for _ in range(n_time): step()
    ms = (time.time() - t) / n_time * 1000
    grads = torch.cat([v.grad.flatten().cpu() for d in params.values() for v in d.values()])
    return l, ms, grads

if __name__ == "__main__":
    if torch.backends.mps.is_available(): torch.mps.set_per_process_memory_fraction(0.5)
    lc, msc, gc = run("cpu")
    print(f"  CPU: loss={lc:+.5f}  {msc:.0f} ms/step  peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.2f} GB")
    lm, msm, gm = run("mps")
    cos = float(torch.nn.functional.cosine_similarity(gc, gm, dim=0))
    print(f"  MPS: loss={lm:+.5f}  {msm:.0f} ms/step")
    print(f"  grad cos(cpu, mps) over {gc.numel()} params = {cos:+.4f}   |cpu|={gc.norm():.3e} |mps|={gm.norm():.3e}")
