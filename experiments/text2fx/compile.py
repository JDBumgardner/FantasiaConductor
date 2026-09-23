"""Compile an app track graph -- source + FxInsert list (+ FxWire DAG) -- into a differentiable SearchGraph the
optimiser can hill-climb: render(params) -> audio, init at the CURRENT settings, priors, describe, amount-scale
against the current settings, and export(params) -> {insert_id: app params} plus the source's patch.

The DAG runs in the app's own topological order (fantasia_core.document.fx_insert.topo_order / effective_wires),
summing at merges as FxHost does; "mix" inserts blend their inputs by `wet`. Inserts we have no twin for are
frozen (identity, flagged). The source is clip audio (fixed) or the Vital twin (synth params searchable)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import appnodes as AN
from fantasia_core.document.fx_insert import SOURCE, OUT, as_dict, effective_wires, insert_id, topo_order

class SearchGraph:
    def __init__(self, inserts, wires, N, device, source_audio=None, synth=None, notes=None, checkpoint=True, search_ids=None):
        self.specs = [as_dict(s) for s in inserts]; self.by_id = {insert_id(s) or s.get("type"): s for s in self.specs}
        self.wires = effective_wires(inserts, wires or []); self.order = topo_order(inserts, wires or [])
        self.N, self.device, self.checkpoint = N, device, checkpoint
        self.source_audio, self.synth, self.notes = (None if source_audio is None else torch.as_tensor(source_audio, dtype=torch.float32).to(device)), synth, notes
        self.twins = {nid: AN.twin_for(self.by_id[nid]) for nid in self.order if self.by_id[nid].get("type") != "mix"}
        self.procs = torch.nn.ModuleDict({nid: tw.make(N) for nid, tw in self.twins.items()}).to(device)
        self.frozen = [nid for nid, tw in self.twins.items() if isinstance(tw, AN.FrozenNode)]
        # Inserts the caller did not pick are rendered at their current settings but never moved: a track carries
        # devices the user tuned by hand, and "make it warmer" is no licence to touch them.
        # A bypassed insert is skipped in the render, so searching it optimises nothing and its export would write
        # parameters onto a device the user switched off.
        self.searchable = {nid for nid in self.twins if (search_ids is None or nid in set(search_ids))
                           and nid not in self.frozen and not self.by_id[nid].get("bypassed")}
        self.held = [nid for nid in self.twins if nid not in self.searchable and nid not in self.frozen]
        self.raw0 = self.init_from_app()
    # ---- parameters
    def init_from_app(self):
        """The user's current settings, as raw tensors (the search starts here; amount 0 means 'as it was')."""
        return {nid: {k: v.to(self.device) for k, v in tw.from_app(self.by_id[nid].get("params") or {}).items()} for nid, tw in self.twins.items()}
    def init(self, seed=0, jitter=0.0):
        g = torch.Generator().manual_seed(seed)
        out = {}
        for nid, d in self.raw0.items():
            live = nid in self.searchable
            out[nid] = {k: (v.clone() + (jitter * torch.randn(v.shape, generator=g).to(self.device) if (jitter and live) else 0)).requires_grad_(live) for k, v in d.items()}
        return out
    def flat(self, params): return [v for d in params.values() for v in d.values()]
    def prior(self, params): return sum((tw.prior(params[nid]) for nid, tw in self.twins.items() if nid in self.searchable), torch.zeros((), device=self.device))
    def describe(self, params): return {nid: tw.describe({k: v.detach() for k, v in params[nid].items()}) for nid, tw in self.twins.items()}
    def export(self, params, ps=None):
        """{insert_id: app params} for every insert that has a twin, in the app's units; frozen inserts are untouched.
        With a synth source and its parameters `ps`, also "vital": the Vital raw parameters of the searched patch."""
        out = {nid: tw.to_app({k: v.detach().cpu() for k, v in params[nid].items()}) for nid, tw in self.twins.items() if nid in self.searchable}
        if self.synth is not None and ps is not None: out["vital"] = self.synth.vital_params({k: v.detach().cpu() for k, v in ps.items()})
        return out
    def scale(self, params, a): return {nid: self.twins[nid].scale(params[nid], self.raw0[nid], a) for nid in params}
    # ---- rendering
    def source(self, ps=None):
        if self.synth is not None: return self.synth.render(ps, self.notes, self.N)
        return self.source_audio[None, None]
    def render(self, params, ps=None):
        x = self.source(ps); bufs = {SOURCE: x}
        for nid in self.order:
            spec = self.by_id.get(nid)
            if spec is None: continue
            incoming = [w.src for w in self.wires if w.dst == nid]; mixed = self._sum(bufs, incoming, x)
            if spec.get("bypassed"): bufs[nid] = mixed; continue
            if spec.get("type") == "mix":
                p = spec.get("params") or {}; wet = float(p.get("wet", 0.5)); dry_src, wet_src = p.get("dry_src") or (incoming[0] if incoming else SOURCE), p.get("wet_src") or (incoming[-1] if incoming else SOURCE)
                bufs[nid] = (1 - wet) * bufs.get(dry_src, x) + wet * bufs.get(wet_src, x); continue
            proc, p = self.procs[nid], params.get(nid, {})
            def f(inp, *vals, proc=proc, keys=tuple(p.keys())):
                out = proc(inp, **dict(zip(keys, vals))); return out[0] if isinstance(out, tuple) else out
            if self.checkpoint and torch.is_grad_enabled() and p: bufs[nid] = torch.utils.checkpoint.checkpoint(f, mixed, *p.values(), use_reentrant=False)
            else: bufs[nid] = f(mixed, *p.values())
        outgoing = [w.src for w in self.wires if w.dst == OUT]
        return (self._sum(bufs, outgoing, x) if outgoing else x)[0, 0], x[0, 0]
    def _sum(self, bufs, srcs, x):
        acc = None
        for s in srcs:
            b = bufs.get(s)
            if b is None: continue
            acc = b if acc is None else acc + b
        return acc if acc is not None else torch.zeros_like(x)

def compile_track(inserts, wires=None, source_audio=None, synth=None, notes=None, N=480000, device="cpu", search_ids=None):
    """A track's insert graph as a searchable twin. Source: `source_audio` (a recording, fixed) or `synth` + `notes` (a
    Vital instrument as its twin: build the WavetableSynth, load the patch with synth.raw_from_vital, check
    synth.vital_coverage; the search then moves patch and inserts together and export() returns both)."""
    return SearchGraph(inserts, wires, N, device, source_audio=source_audio, synth=synth, notes=notes, search_ids=search_ids)
