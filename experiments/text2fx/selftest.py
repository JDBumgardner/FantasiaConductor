"""The checks to run after touching anything in text2fx: `python selftest.py` (about two minutes on MPS).

1. fxgraph.selftest -- every research node: amount 0 is the identity, CPU-vs-MPS gradient cosine, non-zero gradient.
2. The judge's gradients -- every differentiable artefact term has a non-zero gradient on a signal that exhibits the
   artefact, and the gradient agrees between CPU and MPS. The flux term had a ZERO gradient on MPS from the day the
   penalties were written until 2026-09-20 (torch.stft's reflect pad), and nothing here would have said so: the values
   were right, only the gradients were dead.
3. Repeatability -- a full optimiser step (render, CLAP, locality, penalty, backward) run twice gives bit-identical
   gradients on every route: the recording chain, a compiled app graph, the twin with its seeded phase draws. Every
   GPU op with a colliding scatter in its backward (torch.stft's framing, CLAP's bicubic resize, the chorus gather,
   the wavetable read) has been replaced or moved; this is what keeps them out (2026-09-21).
Exit code 1 on any failure.
"""
import os, sys, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", ".."))
if torch.backends.mps.is_available(): torch.mps.set_per_process_memory_fraction(0.5)
import fxgraph as FG, common as C, artefacts as A, text2synth as T2, compile as CP
SR = 48000; L = 10 * SR; DEV = "mps" if torch.backends.mps.is_available() else "cpu"; FAILS = []
def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg, flush=True)
    if not ok: FAILS.append(msg)

def source():
    f = os.path.join(HERE, "words", "instruments", "cello_ref.wav")
    if os.path.exists(f):
        import soundfile as sf; a, _ = sf.read(f); return torch.tensor(a[:L].astype(np.float32))
    torch.manual_seed(0); t = torch.arange(L) / SR                                # no soundfont handy: a plucked-ish tone
    return (torch.sin(2 * np.pi * 220 * t) * torch.exp(-((t % 1.0) * 3))).float()

def judge_gradients(a):
    print("judge gradients (non-zero on an artefacted signal, CPU vs MPS):")
    notes = [(60, 0.0, 9.9, 100)]; t = torch.arange(L) / SR
    trem = 0.55 + 0.45 * torch.sign(torch.sin(2 * np.pi * 6 * t))                # a 6 Hz square tremolo: flux, tremolo, drops, jumps
    terms = {"flux": lambda x, d: A.within_note_flux_t(x, notes), "tremolo": lambda x, d: A.within_note_tremolo_t(x, notes),
             "drops": lambda x, d: A.within_note_drops_t(x, notes), "jump": lambda x, d: A.jump_t(x), "crest": lambda x, d: A.crest_t(x),
             "penalty": lambda x, d: A.penalty(x, d, notes)}
    for name, fn in terms.items():
        g = {}
        for dev in sorted({"cpu", DEV}):
            x = (a * trem).to(dev).requires_grad_(True); fn(x, a.to(dev)).backward(); g[dev] = x.grad.cpu()
        norm = float(g["cpu"].norm()); check(norm > 0, f"{name:8s} |grad| {norm:.2e} > 0")
        if DEV != "cpu" and norm > 0:
            cos = float(torch.nn.functional.cosine_similarity(g["cpu"], g[DEV], dim=0)); ratio = float(g[DEV].norm() / norm)
            check(cos > 0.99 and 0.9 < ratio < 1.1 if name != "jump" else ratio > 0.5, f"{name:8s} CPU/MPS cos {cos:+.4f} norm ratio {ratio:.3f}" + ("  (a quantile picks one sample: only the norm is comparable)" if name == "jump" else ""))

def repeatability(a):
    print("repeatability (two runs of a full step, bit-identical):")
    T = C.text_emb("this sound is dark"); x = a.to(DEV); xn = T2.loudness_norm(x); notes = [(60, 0.0, 9.9, 100)]
    def same(fn, n=2):
        outs = [[t.detach().cpu().clone() for t in fn()] for _ in range(n)]
        return all(all(torch.equal(p, q) for p, q in zip(outs[0], o)) for o in outs[1:]), max([float((p - q).abs().max()) for o in outs[1:] for p, q in zip(outs[0], o)] or [0.0])
    def step_rec():
        pf = T2.init_fx(0); out, gs = T2.render_fx(x[None, None], pf); w = T2.loudness_norm(out)
        (-(C.embed(w) @ T.T).squeeze() + C.band_energy_loss(w, xn) + A.penalty(out, x, notes) + T2.LAM_GS * gs + T2.LAM_PRI * T2.fx_prior(pf)).backward()
        return [v.grad for v in T2.CHAIN.params_flat(pf)]
    ok, d = same(step_rec); check(ok, f"recording route step{'' if ok else f' (max diff {d:.2e})'}")
    from fantasia_core.document.fx_insert import FxInsert
    ins = [FxInsert(id="e", type="eq", params=dict(bands=[dict(type="eq_peak", freq=900.0, gain=0.0, q=1.0)])), FxInsert(id="k", type="compressor", params=dict(threshold=-18.0, ratio=1.5, attack=20.0, release=150.0, makeup=0.0)),
           FxInsert(id="c", type="chorus", params=dict(rate=1.0, depth=0.3, centre_delay=7.0, mix=0.3)), FxInsert(id="g", type="gate", params=dict(threshold=-35.0, ratio=3.0, attack=5.0, release=80.0)),
           FxInsert(id="d", type="delay", params=dict(time=0.25, feedback=0.2, mix=0.1)), FxInsert(id="r", type="reverb", params=dict(room_size=0.5, damping=0.5, wet=0.15, dry=0.85))]
    g = CP.compile_track(ins, None, source_audio=a, N=L, device=DEV)
    def step_app():
        p = g.init(0, jitter=0.1); out, y = g.render(p); w = T2.loudness_norm(out)
        (-(C.embed(w) @ T.T).squeeze() + C.band_energy_loss(w, xn) + A.penalty(out, y, notes) + g.prior(p)).backward()
        return [v.grad for d_ in p.values() for v in d_.values()]
    ok, d = same(step_app); check(ok, f"app graph step (eq comp chorus gate delay reverb){'' if ok else f' (max diff {d:.2e})'}")
    patch = os.path.join(HERE, "words", "instruments", "cello_patch.json")
    if os.path.exists(patch):
        import house_session as H
        from synth import WavetableSynth
        raw = json.load(open(patch))["raw"]; s_ = WavetableSynth(H.CL.TABLE, SR, voices=1, bounds=H.B, interpolation=H.CL.TBL.get("interpolation", 1)).to(DEV)
        def step_twin():
            torch.manual_seed(0); p = {k: torch.tensor(float(v), device=DEV).requires_grad_(True) for k, v in raw.items()}; pf = T2.init_fx(0)
            y = s_.render(p, H.NOTES, L); out, gs = T2.render_fx(y, pf); w = T2.loudness_norm(out)
            (-(C.embed(w) @ T.T).squeeze() + C.band_energy_loss(w, xn) + A.penalty(out, y[0, 0], notes) + T2.LAM_GS * gs).backward()
            return [v.grad for v in p.values() if v.grad is not None] + [v.grad for v in T2.CHAIN.params_flat(pf)]
        ok, d = same(step_twin); check(ok, f"twin route step, seeded phases{'' if ok else f' (max diff {d:.2e})'}")
    else: print("  skip twin route (no cello patch on disk)")

if __name__ == "__main__":
    print("research nodes:"); FG.selftest()
    a = source(); judge_gradients(a); repeatability(a)
    print("\nALL OK" if not FAILS else f"\n{len(FAILS)} FAILURE(S):\n  " + "\n  ".join(FAILS)); sys.exit(1 if FAILS else 0)
