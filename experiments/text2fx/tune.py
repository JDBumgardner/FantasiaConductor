"""`tune_toward` runner: one subprocess per job, so torch, CLAP and the twin never live in the app's process (8 GB).

    python tune.py job.json

job.json (written by fantasia_core.tune.start):
  text        the word or phrase ("dark", "a warmer electric piano")
  anchor      what the sound is, for the identity score ("an electric piano"); optional
  out_dir     where stops, previews and result.json go
  inserts     [{id, type, params, bypassed}]  the track's FX graph, wires [{src, dst}] (empty = serial)
  use         insert ids or types that may move (default: all). The rest are rendered as they are, never touched.
  add         effect types to append for this search (see fantasia_core.tune.ADDABLE_FX); each starts inaudible
  source      {"kind": "audio", "path": wav}                                      -- a recording (or a bounced track)
              {"kind": "vital", "params": {vital raw 0-1}, "notes": [[pitch, start, dur, vel]], "wavetable": name,
               "preset": {...vital JSON...} (optional, for the coverage report), "voices": 1}
  objective   "dir" (default: the change in embedding aligned with X minus not-X, two-sided stops) or "cos"
  amount      one attempt at this distance (default 0.6: audible, still the same sound). "ladder": true walks the
              full set of stops instead; "stops": [...] names them
  quality     "thorough" (8 starts, default) or "quick" (4): the surface is multimodal -- the leader after the first
              stage is overtaken 37 %% of the time -- so starts are the search's only diversity. n_start overrides; device
Progress goes to stdout, one JSON per line: {"event": "start"|"route"|"stop"|"done"|"error", ...}. The result file lists
every stop with its scores, flags, exported app parameters and preview path. A Vital source the twin cannot model
(see synth.vital_coverage; unknown wavetable) is reported and, when the job carries a "fallback_audio", taken as a
recording instead -- the inserts are still searched, the patch is left alone.
"""
import os, sys, json, glob, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", ".."))

def emit(**kw): print(json.dumps(kw), flush=True)

# Does an effect do anything at these settings? Each type's "amount" parameter, at the point a listener would notice.
_AUDIBLE = {"reverb": ("wet", 0.02), "delay": ("mix", 0.02), "chorus": ("mix", 0.02), "saturator": ("drive", 0.5),
            "distortion": ("drive", 0.5), "gain": ("gain", 0.3), "compressor": ("ratio", 1.1), "gate": ("threshold", -70.0),
            "lowpass": ("cutoff", 19000.0), "highpass": ("cutoff", 25.0)}
def _audible(kind, params):
    key, thr = _AUDIBLE.get(kind, (None, None))
    if key is None:                                          # eq and anything else: any band moved off flat
        bands = params.get("bands") if isinstance(params, dict) else None
        return any(abs(float(b.get("gain", 0.0))) > 0.3 for b in bands) if bands else True
    v = float(params.get(key, 0.0))
    if kind == "gate": return v > thr                        # a gate below -70 dB never opens
    if kind == "lowpass": return v < thr
    if kind == "highpass": return v > thr
    return abs(v) >= thr if kind != "compressor" else v >= thr

def main(job_path):
    job = json.load(open(job_path)); out_dir = job["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    import numpy as np, torch, soundfile as sf
    if torch.backends.mps.is_available(): torch.mps.set_per_process_memory_fraction(0.5)
    import common as C, ladder, compile as CP
    from fantasia_core.document.fx_insert import FxInsert, FxWire
    SR = 48000; DEV = job.get("device") or C.DEVICE; text = job["text"]; anchor = job.get("anchor") or "this sound"
    ladder.OBJ = job.get("objective") or ladder.OBJ; ladder.EQ = bool(job.get("two_sided", ladder.OBJ == "dir"))
    stops = tuple(job["stops"]) if job.get("stops") else (None if job.get("ladder") else (float(job.get("amount", 0.6)),))
    if stops:                                                  # one attempt by default: a single distance, ~2 min
        if ladder.OBJ == "dir": ladder.DISTS_DIR = stops
        else: ladder.DISTS = stops if len(stops) > 1 else stops
    emit(event="start", text=text, device=DEV)
    inserts = [FxInsert(id=i["id"], type=i["type"], params=i.get("params") or {}, bypassed=bool(i.get("bypassed", False))) for i in job["inserts"]]
    wires = [FxWire(w["src"], w["dst"]) for w in job.get("wires") or []]
    # Which devices are in play. `use`: insert ids or types on the track that may move (default: all of them).
    # `add`: effect types to append for this search, each starting inaudible, so one only survives if it earns its place.
    from fantasia_core.tune import neutral_params
    added = []
    for kind in (job.get("add") or []):
        nid = f"tune_{kind}"; n = 1
        while any(i.id == nid for i in inserts): n += 1; nid = f"tune_{kind}{n}"
        inserts.append(FxInsert(id=nid, type=str(kind), params=neutral_params(str(kind)))); added.append(nid)
    if wires and added:                                    # a wired graph: hang the new devices off the final node
        last = next((w.src for w in wires if w.dst == "out"), None)
        wires = [w for w in wires if not (w.dst == "out" and w.src == last)]
        chain = ([last] if last else []) + added
        for a, b in zip(chain, chain[1:]): wires.append(FxWire(a, b))
        wires.append(FxWire(chain[-1], "out"))
    inserts_by_id = {i.id: i for i in inserts}
    use = job.get("use")
    if use is None: search_ids = None
    else:
        want = {str(u) for u in use}
        search_ids = [i.id for i in inserts if i.id in want or i.type in want] + added
    emit(event="chain", inserts=[{"id": i.id, "type": i.type, "added": i.id in added,
                                  "searched": (search_ids is None or i.id in (search_ids or []))} for i in inserts])
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
    graph = CP.compile_track(inserts, wires, source_audio=None if synth is not None else source, synth=synth, notes=notes, N=L, device=DEV, search_ids=search_ids)
    if graph.frozen: emit(event="note", message="no twin, held fixed: " + ", ".join(graph.frozen))
    if graph.held: emit(event="note", message="left alone at their current settings: " + ", ".join(graph.held))
    if not graph.searchable: emit(event="error", message="nothing to search: every device is held or has no twin"); return 2
    def log(s):
        if "stop" in s: emit(event="progress", message=s.strip())
    t0 = time.time()
    n_start = int(job.get("n_start") or {"quick": 4, "thorough": 8}.get(str(job.get("quality", "thorough")), 8))
    emit(event="note", message=f"{n_start} starts ({'quick' if n_start < 8 else 'thorough'})")
    stops = ladder.run(text, source, notes, out_dir, "tune", anchor, synth=synth, p_inst=p_inst, n_start=n_start, log=log, graph=graph)
    result = {"text": text, "anchor": anchor, "route": route, "objective": ladder.OBJ, "coverage": coverage, "frozen": graph.frozen,
              "held": graph.held, "added": added, "searched": sorted(graph.searchable), "seconds": round(time.time() - t0), "stops": []}
    best_dir = 0.0
    for i, st in enumerate(stops):
        best_dir = max(best_dir, st.get("dir", 0.0))
        row = {"stop": i, "target": st.get("target"), "word": round(st["clap"], 4), "direction": round(st.get("dir", 0.0), 3), "distance": round(st["dist"], 3), "identity": round(st["self"], 3), "flags": st.get("flags", []),
               "past_range": bool(i > 1 and st.get("dir", 0.0) < 0.5 * best_dir)}      # the change stopped pointing the word's way: the amount exceeds what this word can do here
        if i > 0:
            f = os.path.join(out_dir, f"tune__{text}__stop{i}_d{st['target'] if st['target'] is not None else 'inf'}")
            d = json.load(open(f + ".json")); ex = dict(d.get("export", {}))
            dropped = [nid for nid in added if nid in ex and not _audible(inserts_by_id[nid].type, ex[nid])]
            for nid in dropped: ex.pop(nid)                 # an added effect that stayed inaudible is not worth an insert
            row.update(export=ex, preview=f + ".wav", describe=d.get("describe"), dropped_added=dropped)
        result["stops"].append(row)
    path = os.path.join(out_dir, "result.json"); json.dump(result, open(path, "w"), indent=1)
    emit(event="done", result=path, stops=len(stops) - 1, seconds=result["seconds"]); return 0

if __name__ == "__main__":
    try: sys.exit(main(sys.argv[1]))
    except Exception as e:
        import traceback; emit(event="error", message=str(e), trace=traceback.format_exc()); sys.exit(1)
