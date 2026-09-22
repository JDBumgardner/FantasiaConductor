"""`tune_toward` runner: one subprocess per job, so torch, CLAP and the twin never live in the app's process (8 GB).

    python tune.py job.json

job.json (written by fantasia_core.tune.start):
  text        the word or phrase ("dark", "a warmer electric piano")
  anchor      what the sound is, for the identity score ("an electric piano"); optional
  out_dir     where stops, previews and result.json go
  inserts     [{id, type, params, bypassed}]  the track's FX graph, wires [{src, dst}] (empty = serial)
  source      {"kind": "audio", "path": wav}                                      -- a recording (or a bounced track)
              {"kind": "vital", "params": {vital raw 0-1}, "notes": [[pitch, start, dur, vel]], "wavetable": name,
               "preset": {...vital JSON...} (optional, for the coverage report), "voices": 1}
  objective   "dir" (default: the change in embedding aligned with X minus not-X, two-sided stops) or "cos"
  stops       distance targets (default 0.3/0.6/1.0/1.6 for dir; 0.3/0.6/1.0/1.6/2.5 + unconstrained for cos); n_start (8); device
Progress goes to stdout, one JSON per line: {"event": "start"|"route"|"stop"|"done"|"error", ...}. The result file lists
every stop with its scores, flags, exported app parameters and preview path. A Vital source the twin cannot model
(see synth.vital_coverage; unknown wavetable) is reported and, when the job carries a "fallback_audio", taken as a
recording instead -- the inserts are still searched, the patch is left alone.
"""
import os, sys, json, glob, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", ".."))

def emit(**kw): print(json.dumps(kw), flush=True)

def main(job_path):
    job = json.load(open(job_path)); out_dir = job["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    import numpy as np, torch, soundfile as sf
    if torch.backends.mps.is_available(): torch.mps.set_per_process_memory_fraction(0.5)
    import common as C, ladder, compile as CP
    from fantasia_core.document.fx_insert import FxInsert, FxWire
    SR = 48000; DEV = job.get("device") or C.DEVICE; text = job["text"]; anchor = job.get("anchor") or "this sound"
    ladder.OBJ = job.get("objective") or ladder.OBJ; ladder.EQ = bool(job.get("two_sided", ladder.OBJ == "dir"))
    if job.get("stops"):
        if ladder.OBJ == "dir": ladder.DISTS_DIR = tuple(job["stops"])
        else: ladder.DISTS = tuple(job["stops"]) + (None,)
    emit(event="start", text=text, device=DEV)
    inserts = [FxInsert(id=i["id"], type=i["type"], params=i.get("params") or {}, bypassed=bool(i.get("bypassed", False))) for i in job["inserts"]]
    wires = [FxWire(w["src"], w["dst"]) for w in job.get("wires") or []]
    src = job["source"]; synth = p_inst = None; notes = None; route = "recording"; coverage = []
    if src["kind"] == "vital":
        from synth import WavetableSynth, raw_from_vital, vital_coverage
        coverage = vital_coverage(src["params"], src.get("preset"))
        tables = {torch.load(f)["name"]: f for f in glob.glob(os.path.join(HERE, "tables", "*.pt"))}
        tbl = tables.get(src.get("wavetable", ""))
        if tbl is None: coverage.append(f"wavetable {src.get('wavetable')!r} has no extracted twin table (have: {', '.join(tables) or 'none'})")
        if not coverage:
            T = torch.load(tbl); voices = int(src.get("voices", 1))
            import house_session as H                                                    # the bounds the twins were recovered with
            synth = WavetableSynth(T["table"], SR, voices=voices, bounds=H.B, interpolation=T.get("interpolation", 1)).to(DEV)
            p_inst = {k: v.to(DEV) for k, v in raw_from_vital(synth, src["params"]).items()}
            notes = [tuple(n) for n in src["notes"]]; route = "vital"
            L = int(max(s + d for _, s, d, _ in notes) * SR) + 2 * SR; L = min(L, 10 * SR)   # the twin's clip: up to 10 s
            with torch.no_grad(): synth.random_phase = False; source = synth.render(p_inst, notes, L)[0, 0]; synth.random_phase = True
        elif src.get("fallback_audio"): src = {"kind": "audio", "path": src["fallback_audio"]}
        else: emit(event="error", message="the twin cannot model this patch and no fallback audio was given", coverage=coverage); return 2
    if src["kind"] == "audio":
        a, sr = sf.read(src["path"], dtype="float32"); a = a.mean(axis=1) if a.ndim == 2 else a
        if sr != SR:
            import torchaudio; a = torchaudio.functional.resample(torch.tensor(a), sr, SR).numpy()
        a = a[:10 * SR]; L = len(a); source = torch.tensor(a, device=DEV)
        notes = [tuple(n) for n in job["notes"]] if job.get("notes") else [(60, 0.0, L / SR - 0.05, 100)]   # the artefact detectors want note spans; without MIDI, the whole clip is one
    emit(event="route", route=route, coverage=coverage, seconds=L / SR)
    graph = CP.compile_track(inserts, wires, source_audio=None if synth is not None else source, synth=synth, notes=notes, N=L, device=DEV)
    if graph.frozen: emit(event="note", message="inserts without a twin are held fixed: " + ", ".join(graph.frozen))
    def log(s):
        if "stop" in s: emit(event="progress", message=s.strip())
    t0 = time.time()
    stops = ladder.run(text, source, notes, out_dir, "tune", anchor, synth=synth, p_inst=p_inst, n_start=int(job.get("n_start", 8)), log=log, graph=graph)
    result = {"text": text, "anchor": anchor, "route": route, "objective": ladder.OBJ, "coverage": coverage, "frozen": graph.frozen, "seconds": round(time.time() - t0), "stops": []}
    best_dir = 0.0
    for i, st in enumerate(stops):
        best_dir = max(best_dir, st.get("dir", 0.0))
        row = {"stop": i, "target": st.get("target"), "word": round(st["clap"], 4), "direction": round(st.get("dir", 0.0), 3), "distance": round(st["dist"], 3), "identity": round(st["self"], 3), "flags": st.get("flags", []),
               "past_range": bool(i > 1 and st.get("dir", 0.0) < 0.5 * best_dir)}      # the change stopped pointing the word's way: the amount exceeds what this word can do here
        if i > 0:
            f = os.path.join(out_dir, f"tune__{text}__stop{i}_d{st['target'] if st['target'] is not None else 'inf'}")
            d = json.load(open(f + ".json")); row.update(export=d.get("export", {}), preview=f + ".wav", describe=d.get("describe"))
        result["stops"].append(row)
    path = os.path.join(out_dir, "result.json"); json.dump(result, open(path, "w"), indent=1)
    emit(event="done", result=path, stops=len(stops) - 1, seconds=result["seconds"]); return 0

if __name__ == "__main__":
    try: sys.exit(main(sys.argv[1]))
    except Exception as e:
        import traceback; emit(event="error", message=str(e), trace=traceback.format_exc()); sys.exit(1)
