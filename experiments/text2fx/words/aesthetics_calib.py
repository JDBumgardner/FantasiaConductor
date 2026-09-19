"""Does Audiobox-Aesthetics' production quality track the artefact flags (and the ear)? Score every rendered result and
its dry render; report PQ/CE deltas by flag class; save per-file scores."""
import os, sys, json, glob, collections, statistics
sys.path.insert(0, "experiments/text2fx"); import aesthetics as AE
W = "experiments/text2fx/words"; rows = json.load(open(f"{W}/artefacts.json"))
paths = []
for f, m, fl in rows:
    fx = f"{W}/{f}"; dry = fx.replace("_fx.wav", "_dry.wav") if not f.startswith("source/") else f"{W}/instruments/{os.path.basename(f).split('__')[0]}_ref.wav"
    paths += [fx, dry]
paths = sorted(set(paths)); print(f"scoring {len(paths)} files")
scores = {}
for i in range(0, len(paths), 16):
    for p, s in zip(paths[i:i+16], AE.score(paths[i:i+16])): scores[p] = s
    print(f"  {min(i+16, len(paths))}/{len(paths)}", flush=True)
json.dump(scores, open(f"{W}/aesthetics_scores.json", "w"), indent=1)
by = collections.defaultdict(list)
for f, m, fl in rows:
    fx = f"{W}/{f}"; dry = fx.replace("_fx.wav", "_dry.wav") if not f.startswith("source/") else f"{W}/instruments/{os.path.basename(f).split('__')[0]}_ref.wav"
    d = {k: scores[fx][k] - scores[dry][k] for k in ("PQ", "CE", "PC", "CU")}
    for k in (fl or ["clean"]): by[k].append(d)
print("\nΔ vs the dry render (fx − dry), mean [median] per flag class:")
for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
    print(f"  {k:36s} n={len(v):3d}  ΔPQ {statistics.mean(x['PQ'] for x in v):+.2f} [{statistics.median(x['PQ'] for x in v):+.2f}]  ΔCE {statistics.mean(x['CE'] for x in v):+.2f}  ΔPC {statistics.mean(x['PC'] for x in v):+.2f}")
# the ear's offenders
for f in ("contrast/epiano__a_metallic_electric_piano_eq-comp-dist-delay-reverb_fx.wav", "named/cello__a_metallic_cello_eq-comp-dist-delay-reverb_fx.wav", "balance/A/cello__metallic_eq-comp-drive-delay-reverb_fx.wav", "balance/D/cello__dark_eq-comp-drive-pwtanh-chorus-transient-gate-delay-reverb_fx.wav", "source/epiano__punchy_fx.wav", "contrast/brass__a_punchy_brass_section_eq-comp-dist-delay-reverb_fx.wav", "balance/D/cello__metallic_eq-comp-drive-pwtanh-chorus-transient-gate-delay-reverb_fx.wav"):
    fx = f"{W}/{f}"; dry = fx.replace("_fx.wav", "_dry.wav") if not f.startswith("source/") else f"{W}/instruments/{os.path.basename(f).split('__')[0]}_ref.wav"
    if fx in scores: print(f"  {f[:75]:75s} PQ {scores[fx]['PQ']:.2f} (dry {scores[dry]['PQ']:.2f}, Δ {scores[fx]['PQ']-scores[dry]['PQ']:+.2f})  CE {scores[fx]['CE']:.2f} (Δ {scores[fx]['CE']-scores[dry]['CE']:+.2f})")
