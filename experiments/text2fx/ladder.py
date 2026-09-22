"""The ladder: 'make it more <word>' at fixed distances from the original, for a twin instrument or a recording.

Stops at target distances d* (band-energy distance of the heard output from the untouched source) are found with a
soft constraint  mu * relu(dist - d*)^2  instead of a locality weight, each warm-started from the previous stop, the
last stop unconstrained. Artefact penalties (within-note flux, tremolo, crest drop, jumps -- against the same render's
dry signal) are in the loss at every stop. Candidates are ranked by the objective. Every effect starts at bypass.
Output per stop: wav, CLAP score on the word, distance, identity ('a <instrument>'), artefact flags; plus one
ladder clip (the first bar: original then every stop) for listening."""
import os, sys, json, time, contextlib
import numpy as np, torch, soundfile as sf
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, text2synth as T2, artefacts as A
from optim import Candidate, successive_halving
SR = 48000; L = 10 * SR; DEV = C.DEVICE
DISTS = (0.3, 0.6, 1.0, 1.6, 2.5, None)                       # None = unconstrained. 0.3 (~3 dB average band deviation) is the first stop a listener can tell from the original
MU, ART_W = 5.0, float(os.environ.get("T2_ART_W", "1.0"))
ROUNDS_FIRST, ROUNDS_NEXT, ROUNDS_LAST = ((20, 4), (40, 2), (100, 1)), ((80, 1),), ((150, 1),)

def level_match(y, ref, peak=0.89):
    """y at the K-weighted loudness of ref, then the whole clip scaled down if it would clip (never up): the sounds
    are compared at the loudness the optimiser judged them at, and a punchier stop is not quieter for its peaks."""
    y = y.detach(); ref = ref.detach(); g = C.loudness(ref) / (C.loudness(y) + 1e-9); y = y * g
    ref_peak = float(ref.abs().max()); y = y * min(1.0, peak / max(float(y.abs().max()), 1e-9), 0.5 / max(ref_peak, 1e-9) * max(ref_peak, 1e-9))    # cap at 0.89 FS
    return y

def run(word, source, notes, out_dir, tag, anchor_text, synth=None, p_inst=None, n_start=8, log=print, graph=None, jitter=0.15):
    """source: (L,) audio of the untouched sound (the recording, or the twin's dry render). Either synth/p_inst (the twin
    through text2fx's default chain) or `graph`: a compile.SearchGraph of an APP track -- its inserts at the user's
    current settings, exported back as app parameters per insert id. With a graph, FX candidates start at the current
    settings plus a small jitter, and the amount knob is relative to those settings."""
    torch.manual_seed(0)                                   # the synth's per-note phase and noise draws come from the global RNG: seeded, a run repeats bit for bit (2026-09-21)
    os.makedirs(out_dir, exist_ok=True)
    T = C.text_emb("this sound is " + word); T_self = C.text_emb("this sound is " + anchor_text)
    src_n = T2.loudness_norm(source); x = source[None, None]
    if graph is not None: synth = graph.synth
    def render(c):
        if graph is not None:
            out, y = graph.render(c.pf, c.ps if synth is not None else None); return y, out, torch.zeros((), device=out.device)
        if synth is None: y = x
        else: y = synth.render(c.ps, notes, L)
        out, gs = T2.render_fx(y, c.pf); return y[0, 0], out, gs
    state = dict(d=DISTS[0])
    def loss_of(c):
        y, out, gs = render(c); w = T2.loudness_norm(out); dist = C.band_energy_loss(w, src_n)
        con = MU * torch.relu(dist - state["d"]) ** 2 if state["d"] is not None else 0.0
        pri = graph.prior(c.pf) if graph is not None else T2.fx_prior(c.pf)
        return -(C.embed(w) @ T.T).squeeze() + T2.LAM_PRI * pri + T2.LAM_GS * gs + con + ART_W * A.penalty(out, y, notes)
    def evaluate(c):
        with torch.no_grad():
            if synth is not None: synth.random_phase = False
            y, out, gs = render(c); w = T2.loudness_norm(out); e = C.embed(w)
            if synth is not None: synth.random_phase = True
            clap = float((e @ T.T).squeeze()); dist = float(C.band_energy_loss(w, src_n)); con = MU * max(0.0, dist - state["d"]) ** 2 if state["d"] is not None else 0.0
            art = A.measure_notes(out.cpu(), y.cpu(), notes)
            return dict(score=clap - con - ART_W * float(A.penalty(out, y, notes)), clap=clap, dist=dist, self=float((e @ T_self.T).squeeze()), flags=A.flags_notes(art), art={k: round(v, 3) for k, v in art.items()}), out
    with torch.no_grad(): e0 = C.embed(src_n); stops = [dict(clap=float((e0 @ T.T).squeeze()), dist=0.0, self=float((e0 @ T_self.T).squeeze()), flags=[], target=0.0)]
    def cand(r):
        pf = graph.init(r, jitter=0.0 if r == 0 else jitter) if graph is not None else T2.init_fx(r)
        if synth is None: return Candidate({}, pf, label=f"c{r}")
        g = torch.Generator().manual_seed(r); noise = [0, 0.3, 0.15, 0.5, 0.3, 0.5, 0.15, 0.8][r % 8]
        ps = {k: (v.clone() + (noise * torch.randn((), generator=g).to(DEV) if noise else 0)).requires_grad_(k != "level") for k, v in p_inst.items()}
        if "noise" not in ps: ps["noise"] = torch.logit(torch.tensor(0.005, device=DEV)).requires_grad_(True)
        return Candidate(ps, pf, label=f"c{r}")
    cands = [cand(r) for r in range(n_start)]; t0 = time.time(); wavs = []
    from optim import free_cache
    for i, d in enumerate(DISTS):
        state["d"] = d; rounds = ROUNDS_FIRST if i == 0 else (ROUNDS_LAST if d is None else ROUNDS_NEXT)
        if i > 0: cands = cands + [cand(100 * i + r) for r in range(2)]; rounds = ((rounds[0][0] // 2, 1), (rounds[0][0] - rounds[0][0] // 2, 1))   # continue the last optimum against two fresh starts: a tight stop is a poor init for a loose one
        for attempt in (0, 1):
            try: frontier, _ = successive_halving(cands, loss_of, lambda c: evaluate(c)[0], rounds=rounds, polish=False, log=lambda *a: None); break
            except RuntimeError as e:
                if "out of memory" not in str(e) or attempt or synth is None: raise
                log("    OOM — retrying with synth checkpointing"); free_cache(); synth.checkpoint = True
        ev, c = frontier[0]; ev, out = evaluate(c); ev["target"] = d; stops.append(ev)
        f = os.path.join(out_dir, f"{tag}__{word}__stop{i+1}_d{d if d is not None else 'inf'}.wav"); sf.write(f, level_match(out, source).cpu().numpy(), SR); wavs.append(f)
        desc = graph.describe(c.pf) if graph is not None else T2.CHAIN.describe(c.pf)
        json.dump({"word": word, "tag": tag, "target": d, **{k: v for k, v in ev.items() if k != "score"}, "describe": desc, **({"export": graph.export(c.pf), "frozen": graph.frozen} if graph is not None else {}),
                   **({"raw": {k: float(v) for k, v in c.ps.items()}} if synth is not None else {}),
                   "fx_raw": {t: {k: v.detach().cpu().tolist() for k, v in dd.items()} for t, dd in c.pf.items()}}, open(f[:-4] + ".json", "w"), indent=1)
        log(f"    stop {i+1} (target {d}): word {ev['clap']:+.3f} dist {ev['dist']:.2f} '{anchor_text}' {ev['self']:+.2f} flags {ev['flags'] or '-'}   [{time.time()-t0:.0f}s]")
        cands = [Candidate({k: v.detach().clone().requires_grad_(v.requires_grad) for k, v in c.ps.items()}, {t: {k: v.detach().clone().requires_grad_(True) for k, v in dd.items()} for t, dd in c.pf.items()}, label=f"d{d}")]
    # the listening ladder: first bar of the original, then every stop -- all at the ORIGINAL's loudness (peak-normalising made the punchier stops sound quieter)
    seg, fade, gap = int(2.4 * SR), int(0.01 * SR), np.zeros(int(0.25 * SR), dtype=np.float32)
    def take(y):
        y = np.asarray(y[:seg], dtype=np.float32); w = np.ones_like(y); w[:fade] = np.linspace(0, 1, fade); w[-fade:] = np.linspace(1, 0, fade); return y * w
    lad = [take(level_match(source, source).cpu().numpy()), gap]
    for f in wavs: lad += [take(sf.read(f)[0]), gap]
    sf.write(os.path.join(out_dir, f"{tag}__{word}__LADDER.wav"), np.concatenate(lad), SR)
    json.dump(stops, open(os.path.join(out_dir, f"{tag}__{word}__stops.json"), "w"), indent=1)
    return stops
