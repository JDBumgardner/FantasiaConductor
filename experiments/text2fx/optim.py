"""Optimiser for text -> synth + FX searches.

The loss is nearly deterministic given the parameters (CLAP has no augmentation; the twin re-draws each note's
start phase per render, which barely moves a mel-magnitude embedding), and the restart-to-restart spread of the
plain 3 x 300-step Adam search is large (0.16 / 0.45 / 0.21 on one prompt): the landscape is multimodal, not noisy.  So spend the budget on breadth first and depth last:

  1. successive halving over many inits: short bursts, keep the best half, repeat;
  2. cosine learning-rate decay in the final stage;
  3. an L-BFGS polish on the survivors (valid because the loss is deterministic);
  4. return the whole final population as a frontier, not just the argmax, with a robustness score
     (mean CLAP cosine over a few circular time shifts) alongside the raw one.
"""
import math, time, functools, contextlib, torch

log_flush = functools.partial(print, flush=True)

def cosine_lr(opt, base, step, total, floor=0.1):
    for g in opt.param_groups: g["lr"] = base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * step / total)))

def mem_note():
    if not torch.backends.mps.is_available(): return ""
    return f"   [mps live {torch.mps.current_allocated_memory()/2**20:.0f} MB, pool {torch.mps.driver_allocated_memory()/2**20:.0f} MB]"

def free_cache():
    """The MPS caching allocator counts cached blocks against the process cap; a phase with a different
    allocation pattern (L-BFGS line search, a longer render) leaves blocks the next phase cannot reuse."""
    if torch.backends.mps.is_available(): torch.mps.synchronize(); torch.mps.empty_cache()

def adam_burst(cand, loss_fn, steps, lr=0.02, decay=False):
    """Run `steps` Adam steps on the candidate's tensors (in place). loss_fn(cand) -> scalar loss.
    An out-of-memory step is retried once after emptying the allocator cache."""
    params = [v for v in cand.leaves() if v.requires_grad]
    opt = torch.optim.Adam(cand.param_groups(lr), lr=lr); free_cache()
    base_lrs = [g["lr"] for g in opt.param_groups]
    for i in range(steps):
        if decay:
            for g, b in zip(opt.param_groups, base_lrs): g["lr"] = b * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * i / steps)))
        for attempt in (0, 1):
            try:
                opt.zero_grad(); loss = loss_fn(cand); loss.backward(); opt.step(); break
            except RuntimeError as e:
                if "out of memory" not in str(e) or attempt: raise
                loss = None; free_cache()
    return float(loss)

def lbfgs_polish(cand, loss_fn, iters=15, max_eval=60):
    """Deterministic-loss polish. Falls back to the pre-polish parameters if the line search made things worse."""
    params = [v for v in cand.leaves() if v.requires_grad]
    before = [v.detach().clone() for v in params]; l0 = float(loss_fn(cand)); free_cache()   # L-BFGS allocates differently from Adam: start it on an empty pool
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters, max_eval=max_eval, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-9)
    def closure():
        opt.zero_grad(); loss = loss_fn(cand); loss.backward(); return loss
    failed = None
    try: opt.step(closure)
    except RuntimeError as e:                       # an OOM mid line-search: give the parameters back and say so
        failed = str(e).split("\n")[0][:80]; free_cache()
    with torch.no_grad():
        l1 = float(loss_fn(cand)) if failed is None else float("inf")
        if not math.isfinite(l1) or l1 > l0:
            for v, b in zip(params, before): v.copy_(b)
            l1 = l0
    del opt
    return l0, l1, failed

class Candidate:
    """A point in the search space: a synth raw-param dict and an FX param dict-of-dicts, plus a label.
    lr_mult: optional {param name -> multiplier} (synth keys as 'synth/<k>', fx as '<type>/<k>') applied to Adam's lr."""
    def __init__(self, ps, pf, label="", lr_mult=None): self.ps, self.pf, self.label, self.lr_mult = ps, pf, label, lr_mult or {}
    def leaves(self): return list(self.ps.values()) + [v for d in self.pf.values() for v in d.values()]
    def named_leaves(self): return [(f"synth/{k}", v) for k, v in self.ps.items()] + [(f"{t}/{k}", v) for t, d in self.pf.items() for k, v in d.items()]
    def param_groups(self, lr):
        groups = {}
        for name, v in self.named_leaves():
            if not v.requires_grad: continue
            m = self.lr_mult.get(name, 1.0); groups.setdefault(m, []).append(v)
        return [{"params": vs, "lr": lr * m} for m, vs in groups.items()]
    def detach(self): return Candidate({k: v.detach().clone() for k, v in self.ps.items()}, {t: {k: v.detach().clone() for k, v in d.items()} for t, d in self.pf.items()}, self.label)

def successive_halving(cands, loss_fn, eval_fn, rounds=((30, 4), (60, 2), (220, 1)), lr=0.02, polish=True, polish_ctx=contextlib.nullcontext, log=log_flush, stage_hook=None):
    """cands: list[Candidate]. rounds: (steps, survivors) per stage; the last stage gets cosine decay and the polish.
    eval_fn(cand) -> dict with at least 'score' (higher is better). polish_ctx: context manager factory entered around
    the L-BFGS polish (e.g. to freeze the synth's random phases so the line search sees a deterministic loss).
    Returns the final frontier sorted best first: list of (eval dict, Candidate)."""
    t0 = time.time(); total = 0
    for si, (steps, keep) in enumerate(rounds):
        last = si == len(rounds) - 1
        if stage_hook: stage_hook(si, last)                     # e.g. switch the loss to a short excerpt early, the full clip last
        scored = []
        for c in cands:
            adam_burst(c, loss_fn, steps, lr=lr, decay=last); total += steps
            if last and polish:
                with polish_ctx(): l0, l1, failed = lbfgs_polish(c, loss_fn)
                total += 15; free_cache()
                log(f"      polish {c.label}: loss {l0:.4f} -> {l1:.4f}" + (f"  FAILED: {failed}" if failed else "") + mem_note())
            free_cache()
            with torch.no_grad(): ev = eval_fn(c)
            scored.append((ev, c))
        scored.sort(key=lambda t: -t[0]["score"])
        log(f"    stage {si} ({steps} steps x {len(cands)}): " + "  ".join(f"{c.label}:{ev['score']:+.3f}" for ev, c in scored) + f"   [{time.time()-t0:.0f}s]" + mem_note())
        cands = [c for _, c in scored[:keep]] if not last else [c for _, c in scored]
        frontier = scored
    return frontier, total


def sensitivity_lr(cand, embed_fn, step=0.02, lo=0.25, hi=1.0, floor=1e-4, knee=5.0):
    """Measure how far the embedding moves for one Adam step of each parameter and slow down the outliers:
    mult = clip(knee * median / s, lo, hi). Measured on the word grid: fully equalising (hi = 4, knee = 1) moved work
    to drive and resonance but cost 0.03-0.10 on the cello because it also slowed its best tools; slowing only the
    hyper-sensitive parameters (the delay time everywhere, the e-piano's cutoff) kept the e-piano's +0.05 gain.
    Parameters that barely register (s < floor) keep multiplier 1. Cost: two renders per scalar parameter element."""
    with torch.no_grad():
        e0 = embed_fn(cand.ps, cand.pf); sens = {}
        for name, v in cand.named_leaves():
            if not v.requires_grad: continue
            flat = v.detach().flatten(); ds = []
            for i in range(flat.numel()):
                d = 0.0
                for sgn in (1, -1):
                    v2 = flat.clone(); v2[i] += sgn * step; v2 = v2.view_as(v)
                    if name.startswith("synth/"): ps2 = dict(cand.ps); ps2[name[6:]] = v2; d += float((embed_fn(ps2, cand.pf) - e0).norm()) / 2
                    else:
                        t, k = name.split("/"); pf2 = {tt: dict(dd) for tt, dd in cand.pf.items()}; pf2[t][k] = v2; d += float((embed_fn(cand.ps, pf2) - e0).norm()) / 2
                ds.append(d)
            sens[name] = sum(ds) / len(ds)
    live = sorted(s for s in sens.values() if s >= floor); ref = live[len(live) // 2] if live else 1.0
    return {n: (min(hi, max(lo, knee * ref / s)) if s >= floor else 1.0) for n, s in sens.items()}, sens
