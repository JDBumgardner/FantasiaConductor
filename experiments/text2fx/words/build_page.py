"""Build the 'Ten Words, Four Instruments' page: mp3-encode every clip into words/site/mp3, write words/site/index.html.
Publish the html with the mp3 folder as supporting files (no base64: 100+ clips)."""
import os, sys, json, glob, html, subprocess
D = os.path.dirname(os.path.abspath(__file__)); SITE = os.path.join(D, "site"); MP3 = os.path.join(SITE, "mp3"); os.makedirs(MP3, exist_ok=True)
sys.path.insert(0, D); from word_grid import INSTRUMENTS, WORDS, DISPLAY, art
SCRATCH = "/private/tmp/claude-502/-Users-jacobbumgardner-ClaudeProjects-FantasiaCondutcor/ee355064-e438-48c3-aff6-13109780eb95/scratchpad"
CSS = open(os.path.join(SCRATCH, "roadmap_css.txt")).read().replace("</style>", "").replace("<style>", "")
TAG = "_eq-comp-dist-delay-reverb"
base = json.load(open(os.path.join(D, "baseline.json"))); cross = json.load(open(os.path.join(D, "cross_scores.json"))) if os.path.exists(os.path.join(D, "cross_scores.json")) else {}

def mp3(wav, name):
    out = os.path.join(MP3, name + ".mp3")
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(wav):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-ac", "1", "-b:a", "64k", out], check=True)
    return f"mp3/{name}.mp3"
def clip(src, role, name, note=""):
    label = {"ref": "reference", "clone": "twin", "fx": "twin + fx", "vital": "Vital", "rec": "recording + fx", "app": "app graph"}[role]
    return f'''<div class="clip"><span class="tag {role}">{label}</span><div class="clip-body"><div class="clip-name">{html.escape(name)}</div>{f'<div class="mono">{html.escape(note)}</div>' if note else ''}</div><audio controls preload="none" src="{src}"></audio></div>'''
def load(folder, name, prompt):
    for f in glob.glob(os.path.join(folder, f"{name}__*{TAG}.json")):
        d = json.load(open(f))
        if d["prompt"] == prompt: return d, f[:-5]
    return None, None
def fxline(d, inst):
    s, de = d["describe"]["synth"], d["describe"]; parts = []
    dc = s["cutoff_raw"] - inst["cutoff_raw"]; da = (s["attack"] - inst["attack"]) * 1000; dr = s["release"] - inst["release"]; dres = s["resonance"] - inst["resonance"]
    if abs(dc) >= 0.08: parts.append(f"cutoff {dc:+.2f}")
    if abs(dres) >= 0.1: parts.append(f"res {dres:+.2f}")
    if abs(da) >= 20: parts.append(f"attack {da:+.0f} ms")
    if abs(dr) >= 0.5: parts.append(f"release {dr:+.1f} s")
    if de["drive_db"] >= 2: parts.append(f"drive {de['drive_db']:.0f} dB")
    if de["comp"]["ratio"] >= 2.5: parts.append(f"comp {de['comp']['ratio']:.1f}:1")
    if de["delay_mix"] >= 0.2: parts.append(f"delay {de['delay_mix']:.2f}")
    if de["reverb"]["mix"] >= 0.2: parts.append(f"reverb {de['reverb']['mix']:.2f}")
    eq = [f"{h/1000:.1f}k{g:+.0f}" if h >= 1000 else f"{h}{g:+.0f}" for h, g in de["eq"] if abs(g) >= 4]
    if eq: parts.append("eq " + " ".join(eq))
    return " · ".join(parts)

names = [n for n, _ in INSTRUMENTS]; inst_json = {n: json.load(open(os.path.join(D, "instruments", f"{n}_patch.json"))) for n in names}
plain = {(n, w): load(os.path.join(D, "plain"), n, w) for n in names for w in WORDS}
named = {(n, w): (load(os.path.join(D, "named"), n, f"a {w} {DISPLAY[n]}")[0] or load(os.path.join(D, "named"), n, art(f"{w} {DISPLAY[n]}"))[0],
                  load(os.path.join(D, "named"), n, f"a {w} {DISPLAY[n]}")[1] or load(os.path.join(D, "named"), n, art(f"{w} {DISPLAY[n]}"))[1]) for n in names for w in WORDS}
contrast = {(n, w): load(os.path.join(D, "contrast"), n, art(f"{w} {DISPLAY[n]}")) for n in names for w in WORDS} if os.path.isdir(os.path.join(D, "contrast")) else {}

def heat(v, lo=-0.15, hi=0.45):
    t = max(0.0, min(1.0, (v - lo) / (hi - lo))); return f"background:color-mix(in oklab, var(--accent) {int(t * 55)}%, var(--surface))"
# ---------------------------------------------------------------------------------------------------------------
P = []
P.append(f'''<meta charset="utf-8"><title>Ten Words, Four Instruments</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Chivo:wght@500;600;700;800&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}
.heat td{{text-align:center;font-family:"JetBrains Mono",ui-monospace,monospace;font-size:.8rem;font-variant-numeric:tabular-nums;padding:7px 6px}} .heat td:first-child{{text-align:left;font-family:"Chivo",sans-serif;font-size:.9rem}}
.heat td small{{display:block;color:var(--muted);font-size:.68rem;margin-top:1px}}
.words{{display:grid;grid-template-columns:1fr;gap:1px;background:var(--rule);border:1px solid var(--rule);margin:14px 0 22px}}
.words .clip{{grid-template-columns:auto 1fr;grid-template-areas:"tag body" "audio audio"}} @media (min-width:640px){{.words .clip{{grid-template-columns:110px 1fr 220px;grid-template-areas:"tag body audio"}}}}
.tag.word{{background:var(--accent);color:var(--accent-ink);min-width:76px;text-align:center}} .tag.word small{{display:block;font-size:.62rem;opacity:.85;letter-spacing:.02em;text-transform:none}}
.pair{{display:grid;grid-template-columns:1fr;gap:1px;background:var(--rule);border:1px solid var(--rule);margin:12px 0 18px}} @media (min-width:700px){{.pair{{grid-template-columns:1fr 1fr}}}}
.pair .clip{{grid-template-columns:1fr;grid-template-areas:"tag" "body" "audio";gap:6px}}
.axis{{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:.72rem;color:var(--muted);letter-spacing:.05em;text-transform:uppercase}}
.finding{{border-left:3px solid var(--accent);padding:2px 0 2px 14px;margin:18px 0}} .finding b{{font-family:"Chivo",sans-serif}}
.tag.rec{{background:var(--tint);color:var(--ink)}} .tag.app{{background:var(--ink);color:var(--bg)}}
.ab td:nth-child(n+2){{text-align:right;font-family:"JetBrains Mono",ui-monospace,monospace;font-size:.8rem;font-variant-numeric:tabular-nums}} .ab td.over{{color:var(--accent);font-weight:600}}
</style>
<main>
<h1>Ten Words, Four Instruments</h1>
<p class="lede">One adjective at a time, pushed onto a cello, a brass section, an electric piano and a flute — forty searches through the differentiable twin and its effects chain, every result playable, every score against the untouched instrument.</p>
<p class="meta">Fantasia Conductor · Sept 2026 · companion to <em>The Vital Twin</em> and its <em>Roadmap</em> · clips 10 s, level-matched · <span class="tag ref">reference</span> soundfont · <span class="tag clone">twin</span> recovered patch, no effects · <span class="tag fx">twin + fx</span> the result</p>
''')
# 01 setup
P.append('''<h2><span class="num">01</span>The setup</h2>
<p>Each instrument is a General MIDI soundfont playing the same two-bar house hook. The twin recovers a patch for it from the audio alone (the <span class="tag clone">twin</span> clips below are that patch, dry). Then, for each word, the search starts from the recovered patch and moves synth and effects together toward <em>“this sound is ⟨word⟩”</em> as heard by CLAP, with a small locality term that keeps the result recognisably the instrument. Chain: EQ → compressor → drive → delay → reverb. Budget: successive halving over eight starts, 460 steps plus an L-BFGS polish, about three minutes a cell.</p>
<div class="stack">''')
for n, prog in INSTRUMENTS:
    ij = inst_json[n]; ph = ij["physical"]
    P.append(clip(mp3(os.path.join(D, "instruments", f"{n}_ref.wav"), f"{n}_ref"), "ref", f"{DISPLAY[n].capitalize()} — GM program {prog + 1}"))
    P.append(clip(mp3(os.path.join(D, "instruments", f"{n}_clone.wav"), f"{n}_clone"), "clone", f"{DISPLAY[n].capitalize()} — the twin's patch", f"cutoff {ph['cutoff_raw']:.2f} · res {ph['resonance']:.2f} · attack {ph['attack']*1000:.0f} ms · release {ph['release']:.2f} s · CLAP “a {DISPLAY[n]}” {base[n]['_self']:+.2f}"))
P.append('''</div>
<p>The ten words were chosen one or two per axis of the space a sound designer actually works in:</p>
<div class="defs">
<div class="def"><b>Spectrum</b><span>bright · dark · warm</span></div>
<div class="def"><b>Texture</b><span>harsh · metallic · wooden</span></div>
<div class="def"><b>Dynamics</b><span>punchy · soft</span></div>
<div class="def"><b>Space</b><span>airy · distant</span></div>
</div>
<p><strong>Reading a score.</strong> CLAP cosine between the audio and the sentence, roughly −0.2 … 0.6 in practice. What matters is the <em>gain</em> over the untouched instrument, because CLAP already has opinions about the starting sound: the e-piano twin scores +0.18 on “bright” before anything is done to it, the cello −0.14 on “soft”. Every cell below carries both.</p>
''')
# 02 grid
P.append('''<h2><span class="num">02</span>The grid</h2>
<p>Score of the result, with the untouched instrument's score → gain underneath. Darker cells are higher.</p>
<div class="tw"><table class="heat"><tr><th>word</th>''' + "".join(f"<th>{DISPLAY[n]}</th>" for n in names) + "</tr>")
for w in WORDS:
    row = f"<tr><td>{w}</td>"
    for n in names:
        d, _ = plain[(n, w)]
        if d is None: row += "<td>·</td>"; continue
        b = base[n][w]; row += f'<td style="{heat(d["score"])}">{d["score"]:+.2f}<small>{b:+.2f} → {d["score"]-b:+.2f}</small></td>'
    P.append(row + "</tr>")
P.append("</table></div>")
for n in names:
    ph = inst_json[n]["physical"]
    P.append(f'<h3>{DISPLAY[n].capitalize()}</h3><div class="words">')
    for w in WORDS:
        d, stem = plain[(n, w)]
        if d is None: continue
        P.append(clip(mp3(stem + "_fx.wav", f"{n}_{w}"), "fx", f"{w}  —  {d['score']:+.2f}  (gain {d['score']-base[n][w]:+.2f}, half amount {d['amount50']['score']:+.2f})", fxline(d, ph)))
    P.append("</div>")
# 03 findings
P.append('''<h2><span class="num">03</span>What the words did</h2>
<div class="finding"><b>The word class matters more than the instrument.</b> Spectrum and texture words gain +0.15 … +0.39 on every instrument. <em>Soft, airy</em> and <em>distant</em> gain about the same (+0.15 … +0.22) but start from −0.1 … −0.15 — CLAP is actively sure a sustained single-oscillator tone is none of those things — so they only reach zero. The three words that fail are the three that describe what a synth without a noise or breath source cannot make, not the three the optimiser cannot move toward.</div>
<div class="finding"><b>The synth's filter is barely used.</b> Across all forty cells the cutoff moved at most ±0.2 and resonance hardly at all; the spectral work is done by the EQ pinned at its ±6 dB rails plus the compressor. “Dark” on the brass is five EQ cuts from 1.3 kHz up with the cutoff untouched. The EQ is the cheaper gradient path, so it wins — a prior-weighting problem, and the one to fix if the point is a Vital patch rather than an insert chain.</div>
<div class="finding"><b>Two solution families for “punchy” and “bright”.</b> On the sustained instruments: attack shortened by 40–70 ms and the compressor at its 8:1 rail. On the e-piano, already percussive: resonance cut by half and 600 Hz – 5 kHz lifted. Both are what a sound designer would do; the optimiser picked by what the instrument already had.</div>
<div class="finding"><b>Half amount tells you whether the answer is a path or a corner.</b> The e-piano keeps 50–70 % of its gain at half amount on every word: the words are reachable along a smooth path from it. The sustained instruments keep 20–50 % on the words that work and go <em>negative</em> on the words that fail — those solutions live in a corner of the constraints.</div>
<div class="finding"><b>“Distant” on the cello found a transmission, not a room.</b> +10 dB of drive and 8:1 compression — the one cell that used the drive at all. A one-word prompt carries no reverb bias, and CLAP's “distant” evidently includes a bad radio.</div>
''')
# 04 naming
have_named = [n for n in names if all(named[(n, w)][0] for w in WORDS)]
P.append('''<h2><span class="num">04</span>Naming the instrument</h2>
<p>The three failing words suggested a fix: say <em>“a soft cello”</em> instead of <em>“soft”</em>, so CLAP's notion of the word is conditioned on the instrument rather than free to drift toward any soft sound. Same patches, same chain, same optimiser, same locality weight; only the sentence changed. Each result is then scored three ways — on the sentence it optimised, on the bare word, and on “a ⟨instrument⟩” alone.</p>''')
for n in have_named:
    cs = {w: cross.get(f"{n}/{w}") for w in WORDS}
    if not all(cs.values()): continue
    P.append(f'''<h3>{DISPLAY[n].capitalize()} <span class="axis">— untouched twin scores {base[n]["_self"]:+.2f} on “{art(DISPLAY[n])}”</span></h3>
<div class="tw"><table><tr><th>word</th><th>plain result → bare word</th><th>plain result → “{art(DISPLAY[n])}”</th><th>named result → bare word</th><th>named result → “a ⟨word⟩ {DISPLAY[n]}”</th><th>named result → “{art(DISPLAY[n])}”</th></tr>''')
    for w in WORDS:
        r = cs[w]; p_, q_ = r["plain"], r["named"]
        P.append(f'<tr><td>{w}</td><td>{p_["word"]:+.2f}</td><td>{p_["self"]:+.2f}</td><td>{q_["word"]:+.2f}</td><td><strong>{q_["named"]:+.2f}</strong></td><td>{q_["self"]:+.2f}</td></tr>')
    P.append("</table></div>")
    P.append('<div class="words">')
    for w in WORDS:
        d, stem = named[(n, w)]
        P.append(clip(mp3(stem + "_fx.wav", f"{n}_{w}_named"), "fx", f"{d['prompt']}  —  {d['score']:+.2f} on its sentence · {cs[w]['named']['word']:+.2f} on “{w}” · {cs[w]['named']['self']:+.2f} on “{art(DISPLAY[n])}”", fxline(d, inst_json[n]["physical"])))
    P.append("</div>")
P.append('''<div class="finding"><b>It works, and it exposes the next problem.</b> With the instrument named, the results stay — indeed become more — the instrument: the cello twin goes from 0.18 to about 0.5 on “a cello”, the flute from 0.13 to about 0.47. But on those two, that is also where most of the gain comes from: the bare-word scores collapse (cello “dark” 0.40 → 0.10, flute “punchy” 0.33 → 0.04), and for <em>soft, airy, distant</em> the named result scores <em>higher</em> on “a cello” than on “a soft cello”. The optimiser found that the cheapest way to satisfy “a soft cello” was to be more of a cello, not softer. The brass, already recognised at 0.41, shows the same: every named result scores ~0.55 on “a brass section” and about zero on its adjective.</div>
<div class="finding"><b>The e-piano is the counter-example, and it explains the mechanism.</b> CLAP already hears the e-piano twin as an electric piano (0.39), so the instrument term is close to saturated and the adjective has to do the work: the named results keep most of their bare-word score (“bright” 0.36 vs 0.38 plain, “warm” 0.38 vs 0.39) <em>and</em> gain identity (0.39 → 0.57). Naming the instrument works exactly when the twin already sounds like the instrument — which, for the cello and flute, is the missing noise and breath source again.</div>
<p>The obvious remedy is the roadmap's <em>directional loss</em>, made concrete: optimise <span class="mono">cos(“a soft cello”) − cos(“a cello”)</span>, the adjective's direction with the instrument's identity subtracted out. That is the third grid.</p>
''')
if contrast and any(v[0] for v in contrast.values()):
    P.append('''<h2><span class="num">05</span>The contrastive objective</h2>
<p>Same patches, chain and budget; the optimised quantity is now <span class="mono">cos(“a ⟨word⟩ ⟨instrument⟩”) − cos(“a ⟨instrument⟩”)</span>. The table puts the three objectives side by side on the two things we actually care about: does the result carry the adjective (its score on the bare word), and is it still the instrument (its score on “a ⟨instrument⟩”, with the untouched twin's value for reference).</p>''')
    for n in names:
        rows = [(w, contrast[(n, w)]) for w in WORDS if contrast[(n, w)][0]]
        cs = {w: cross.get(f"{n}/{w}") for w in WORDS}
        if not rows or not all(cs.get(w) and cs[w].get("contrast") for w, _ in rows): continue
        P.append(f'''<h3>{DISPLAY[n].capitalize()} <span class="axis">— untouched twin: {base[n]["_self"]:+.2f} on “{art(DISPLAY[n])}”</span></h3>
<div class="tw"><table><tr><th rowspan="2">word</th><th colspan="3">adjective: score on “⟨word⟩”</th><th colspan="3">identity: score on “{art(DISPLAY[n])}”</th></tr>
<tr><th>plain</th><th>named</th><th>contrast</th><th>plain</th><th>named</th><th>contrast</th></tr>''')
        for w, _ in rows:
            r = cs[w]; P.append(f'<tr><td>{w}</td><td>{r["plain"]["word"]:+.2f}</td><td>{r["named"]["word"]:+.2f}</td><td>{r["contrast"]["word"]:+.2f}</td><td>{r["plain"]["self"]:+.2f}</td><td>{r["named"]["self"]:+.2f}</td><td>{r["contrast"]["self"]:+.2f}</td></tr>')
        P.append("</table></div>")
        P.append('<div class="words">')
        for w, (d, stem) in rows:
            P.append(clip(mp3(stem + "_fx.wav", f"{n}_{w}_contrast"), "fx", f"{d['prompt']}  —  contrast {d['score']:+.2f} · “{w}” {cs[w]['contrast']['word']:+.2f} · “{art(DISPLAY[n])}” {cs[w]['contrast']['self']:+.2f}", fxline(d, inst_json[n]["physical"])))
        P.append("</div>")
    P.append('''<div class="finding"><b>Subtracting the instrument throws the instrument away.</b> The contrastive results do carry more adjective than the named ones (brass “dark” 0.34 on the bare word against −0.01 named; cello “punchy” 0.24 against 0.11), but never as much as the plain prompt, and they pay for it in identity: the brass twin falls from 0.41 on “a brass section” to 0.05–0.28, the e-piano from 0.39 to 0.05–0.37. The gradient of <span class="mono">−cos(“a cello”)</span> points away from being a cello, and the optimiser followed it — the locality term was the only thing holding the sound near the instrument. Three objectives, three failure modes: the plain word drifts toward the adjective's generic sense, the named sentence collapses onto the instrument, the difference sacrifices the instrument.</div>
<div class="finding"><b>What the three grids say the objective should be.</b> Maximise the adjective and <em>hold</em> the identity rather than trade against it: <span class="mono">cos(“a soft cello”) + λ · min(0, cos(“a cello”) − cos₀)</span>, a hinge that costs nothing while the result stays at least as much a cello as the twin started, and bites only when identity drops. On the e-piano, where the twin already reads as its instrument, the plain named sentence already behaves this way — which is the strongest argument that the cello and flute problem is the twin's missing noise and breath, not the objective.</div>
''')
# ---- 06 the recording route, 07 curves not points, 08 the judge and the noise source
FR = json.load(open(os.path.join(D, "frontier", "frontier.json"))) if os.path.exists(os.path.join(D, "frontier", "frontier.json")) else {}
SRC = {}
for f in glob.glob(os.path.join(D, "source", "*.json")):
    d = json.load(open(f)); SRC[(os.path.basename(f).split("__")[0], d["prompt"])] = d
BAL = {}
for f in glob.glob(os.path.join(D, "balance", "D", "*.json")):
    d = json.load(open(f)); BAL[(os.path.basename(f).split("__")[0], d["prompt"])] = d
P.append('''<h2><span class="num">06</span>The same words on the recording itself</h2>
<p>Text2FX's own setting, on the fixed chain: the nine-node effects chain on the soundfont recording, no twin, same words, same optimiser, a locality term to the recording. This is what the tool can do on an audio track rather than a Vital instrument.</p>
<div class="tw"><table><tr><th>cell</th><th>untouched</th><th>FX on the recording</th><th>twin synth + FX</th></tr>''')
for n, w in (("cello", "dark"), ("cello", "soft"), ("cello", "punchy"), ("cello", "metallic"), ("epiano", "dark"), ("epiano", "punchy")):
    s, b = SRC.get((n, w)), BAL.get((n, w))
    if s and b: P.append(f'<tr><td>{DISPLAY[n]} {w}</td><td>{s["baseline"]:+.2f}</td><td>{"<strong>" if s["score"] > b["score"] else ""}{s["score"]:+.2f}{"</strong>" if s["score"] > b["score"] else ""}</td><td>{"<strong>" if b["score"] >= s["score"] else ""}{b["score"]:+.2f}{"</strong>" if b["score"] >= s["score"] else ""}</td></tr>')
P.append('''</table></div>
<p>It works — the earlier "cheats or no-ops" verdict was reached on the buggy chain and an EQ-only setup — and the chain reaches for what a person would: the transient shaper at +1.0 for <em>punchy</em>, chorus 0.54 for <em>metallic</em>, a 5–6:1 compressor for <em>punchy</em> and <em>soft</em>. The twin wins where the word is about the source (<em>punchy</em> needs the envelope; <em>dark</em> on the e-piano needs the filter); the recording wins on e-piano <em>punchy</em>, where the shaper on a real hammer beats a synthetic one. But this table ranks endpoints, and the next section is why that is the wrong question.</p>
''')
if FR:
    P.append('''<h2><span class="num">07</span>A curve, not a point</h2>
<p>Every table above ranks the highest CLAP score, and the highest score for "dark" is the darkest sound the constraints allow — not a darker cello. "Make it darker" asks for a <em>direction and an amount</em>, so a route should be judged by how much word it buys per unit of change from the original, and whether the sound is still the instrument along the way. Scaling a finished result to 50 % is not that curve (the half-amount scores collapsed to zero for a third of the grid: a scaled-down extreme was never optimised). So each stop below is <strong>its own optimum at that distance</strong>: the locality weight is swept from tight to loose, each run warm-started from the previous, every effect starting from bypass, distance measured on what you hear (band-energy, phase-invariant) against the untouched instrument, and candidates ranked by the objective rather than by score alone. Getting this right took four attempts, each of which exposed an asymmetry in how the routes had been compared — the last was that the twin's effects were not being counted as distance.</p>''')
    for cell in ("cello/dark", "cello/punchy", "epiano/dark"):
        n, w = cell.split("/"); tw, fx = FR.get(f"{cell}/twin"), FR.get(f"{cell}/fx")
        if not (tw and fx): continue
        P.append(f'''<h3>{DISPLAY[n].capitalize()}, “{w}”</h3>
<div class="tw"><table><tr><th>stop</th><th colspan="3">twin: word · distance · “{art(DISPLAY[n])}”</th><th colspan="3">FX on the recording: word · distance · “{art(DISPLAY[n])}”</th></tr>''')
        for i in range(len(tw)):
            a, b = tw[i], fx[i]; lab = "start" if i == 0 else f"stop {i}"
            P.append(f'<tr><td>{lab}</td><td>{a["clap"]:+.2f}</td><td>{a["dist"]:.2f}</td><td>{a["self"]:.2f}</td><td>{b["clap"]:+.2f}</td><td>{b["dist"]:.2f}</td><td>{b["self"]:.2f}</td></tr>')
        P.append("</table></div><div class=\"pair\">")
        for route, lab in (("twin", "twin"), ("fx", "FX on the recording")):
            f = os.path.join(D, "listen_frontier", "ab", f"{n}_{w}__{route}__ladder_original_then_stops3-6.wav")
            if os.path.exists(f): P.append(clip(mp3(f, f"{n}_{w}_{route}_ladder"), "fx", f"{lab} — ladder: the same bar, original then stops 3 → 6", "2.4 s each; hear the direction accumulate"))
        P.append("</div>")
    P.append('''<div class="finding"><b>At matched distance the twin is above the recording route on all three cells</b>, by a lot on the cello: at distance ≈ 0.5 the twin sits at +0.12 (dark) and +0.23 (punchy) where the effects alone sit at −0.01 and +0.05. The recording route saturates — "dark" on the cello never passes −0.01 however loose the leash — because effects can only take away from a recording that has no darkness in it, while the twin can change the source. Identity <em>rises</em> along the recording route (0.31 → 0.43) since the effects that darken also hide the soundfont's artefacts.</div>
<div class="finding"><b>The musical stops are 3–5, not 6.</b> Cello punchy stop 4 (+0.17 at distance 0.33, identity unchanged) is "the cello, punchier"; stop 6 (+0.28 at 0.66, identity slipping) is on its way to something else. The first ~0.1 of distance buys nothing on any word: the effects have to switch on before the word can move. Adjacent stops are hard to tell apart because they are spaced by the locality weight, not by how different they sound — the tool should place its stops at fixed distances instead (≈ 0.15 / 0.4 / 0.8 / 1.5) and return the ladder; the amount you choose is the stop you liked. (Done — §09.)</div>
''')
ART = json.load(open(os.path.join(D, "artefacts.json"))) if os.path.exists(os.path.join(D, "artefacts.json")) else []
P.append('''<h2><span class="num">08</span>Is it a good sound? First layer of a judge</h2>
<p>A semantic score is blind to two failure classes a listener hears at once: <em>breaking</em> (clipping, clicks, gate chops) and <em>wobble</em> (chorus depth, delay-time jitter, pumping) — CLAP's mel front end barely registers a click and may like the "movement". The first layer is six signal-level detectors, each measured <strong>inside the sustained part of each note</strong> and as a <strong>delta against the same notes without the effects</strong> (a naive modulation detector flags every melody, because notes move the spectral centroid at note rate): clipped-sample fraction, sample-to-sample jumps relative to local level, within-note spectral flux (chorus / flanger / pumping), within-note level modulation at 2–14 Hz (tremolo / pumping), crest-factor change (squashed) and energy chops that recover within 200 ms (gate stutter).</p>''')
if ART:
    import collections; cnt = collections.Counter(fl for r in ART for fl in r[2])
    P.append("<p>Over the 168 rendered results: " + ", ".join(f"<strong>{v}</strong> flagged for {k}" for k, v in cnt.most_common()) + f"; {len(ART) - sum(1 for r in ART if r[2])} clean. The worst case of each class:</p><div class=\"stack\">")
    worst = [("d_trem", "contrast/epiano__a_metallic_electric_piano_eq-comp-dist-delay-reverb_fx.wav", "pumping — 8.6 dB of level modulation inside notes"), ("d_flux", "balance/A/cello__metallic_eq-comp-drive-delay-reverb_fx.wav", "chorus — the highest spectral churn, +2.4 dB per frame"), ("d_stutter", "balance/D/cello__dark_eq-comp-drive-pwtanh-chorus-transient-gate-delay-reverb_fx.wav", "gate chops — six inside notes; a 60 dB range biting the tails"), ("d_crest", "contrast/brass__a_punchy_brass_section_eq-comp-dist-delay-reverb_fx.wav", "squashed — crest factor down 7.6 dB")]
    for k, f, note in worst:
        if os.path.exists(os.path.join(D, f)): P.append(clip(mp3(os.path.join(D, f), "art_" + k), "fx", f.split("/")[-1].split("_eq-")[0].replace("__", " · "), note))
    P.append("</div>")
P.append('''<p>What comes next is mechanical once the ear agrees with the flags: the four differentiable ones (flux, tremolo depth, crest change, jumps) become penalties against the dry render so wobble and chops cannot buy score, and the gate's 60 dB range prior comes down to a musical 20–30 dB. A learned aesthetics judge (Audiobox-Aesthetics' production-quality axis) was the candidate second layer; §09 has both outcomes.</p>
<h3>The noise source, and what it did for the flute</h3>
<p>Every failure in the grid pointed the same way — <em>soft / airy / distant</em> reaching only zero, CLAP not hearing the cello or flute twins as their instruments — so Vital's sample oscillator is now in the twin: measured on the plugin (its default sample is white noise within 0.4 dB; output rms 0.207 × level²; it takes the voice envelope; "FILTER 1" routes it through the filter), calibrated to 3 % including the drive saturation at full level. Recovering it needed a loss the fine STFT terms could not provide — two noise realisations never match sample for sample, which had the twin quietly preferring less noise than the target — so a phase- and realisation-invariant band-energy term went into the recovery. Re-recovered with breath noise, the flute twin goes from 0.13 to <strong>0.20</strong> on “a flute” (the soundfont itself: 0.33) and survives the round trip into Vital; the cello stays at 0.19 — bow noise is pitched and granular, not white at the filter input.</p>
<div class="pair">''')
for f, role, name, note in ((os.path.join(D, "instruments", "flute_clone.wav"), "clone", "Flute twin before — no noise source", "“a flute” 0.13"), (os.path.join(D, "noise", "flute_clone.wav"), "clone", "Flute twin after — breath noise 0.31", "“a flute” 0.20; Vital playing the patch 0.18")):
    if os.path.exists(f): P.append(clip(mp3(f, "noise_" + os.path.basename(f)[:-4] + ("_before" if "instruments" in f else "_after")), role, name, note))
P.append("</div>")
# ---- 09 fixed stops with penalties, 10 the app's own graphs, 11 the A/B and the variance
LAD = {}
for f in glob.glob(os.path.join(D, "ladder", "*__stops.json")):
    tag, w = os.path.basename(f)[:-12].split("__"); LAD[(tag, w)] = json.load(open(f))
P.append('''<h2><span class="num">09</span>Fixed stops, with a judge in the loop</h2>
<p>The ladders above were spaced by the locality weight; these are spaced by <em>distance</em>. Each stop is a soft constraint at a target band-energy distance (0.15 / 0.4 / 0.8 / 1.5 / 3.0, then unconstrained), continued from the previous stop against two fresh starts, every effect from bypass. And the judge's four differentiable detectors are now penalties in the loss — within-note spectral flux, 2–14 Hz level modulation, crest drop, sample jumps — plus a gate-drop term for the chops the flags kept finding in release tails, all measured against the same render's dry signal. Weights had to triple after the first ladders still bought their last +0.05 with wobble; the gate's range prior came down from 60 to 25 dB. Result: every stop below is clean, and the scores held or improved (cello punchy on the recording: +0.43 with chops → +0.47 without). Clips are loudness-matched — peak matching had made the punchy stops 6–10 dB quieter and read as "fading".</p>
<div class="tw"><table><tr><th>cell</th><th>route</th><th>start</th><th>s1</th><th>s2</th><th>s3</th><th>s4</th><th>s5</th><th>free</th></tr>''')
for n, w in (("cello", "dark"), ("cello", "punchy"), ("cello", "soft"), ("epiano", "dark")):
    for route, lab in (("twin", "twin"), ("fx", "recording")):
        s = LAD.get((f"{n}_{route}", w))
        if not s: continue
        P.append(f'<tr><td>{DISPLAY[n]} {w}</td><td>{lab}</td>' + "".join(f'<td>{st["clap"]:+.2f}<small>{st["dist"]:.2f} · {st["self"]:.2f}</small></td>' for st in s) + "</tr>")
P.append('''</table></div>
<p class="axis">each cell: word score, then distance · identity</p>
<div class="stack">''')
for n, w, route, role, lab in (("cello", "punchy", "twin", "fx", "cello punchy — twin"), ("cello", "punchy", "fx", "rec", "cello punchy — recording"), ("epiano", "dark", "twin", "fx", "e-piano dark — twin"), ("cello", "dark", "fx", "rec", "cello dark — recording: nothing to darken until the effects switch on")):
    f = os.path.join(D, "listen_frontier", "ladder_final_loudness_matched", f"{n}_{route}__{w}__LADDER.wav")
    if os.path.exists(f): P.append(clip(mp3(f, f"{n}_{w}_{route}_fixed_ladder"), role, lab, "the first bar: original, then stops 1 → 6, 2.4 s each, loudness-matched"))
P.append('''</div>
<div class="finding"><b>The learned judge was measured and rejected.</b> Audiobox-Aesthetics' production-quality axis, over 168 results against their dry renders: clean results −0.28, pumping −0.43, gate stutter −0.66 — the right order — but chorus/flanger <em>+0.15</em>, and its enjoyment axis rises more for pumping (+0.73) than for clean (+0.39). It reads modulation and pumping as production, not damage, and per file it is too noisy to gate with (a clean control lost 1.36 PQ; the worst chorus offender lost nothing). It stays as an extra signal, not a judge.</div>
<div class="finding"><b>Still open: the stops are hard to tell apart by ear</b> at the low end, and the far ones drift toward "something else". Parked; the graph work below came first.</div>
''')
# 10 the app's own graphs
RT = (("gain", 149), ("saturator", 156), ("distortion", 159), ("lowpass", 125), ("highpass", 122), ("limiter", 109), ("compressor", 107), ("reverb", 94), ("delay", 87), ("eq", 82), ("gate", 50), ("chorus", 49))
APP = json.load(open(os.path.join(D, "appgraph", "cello_appgraph__dark__stops.json"))) if os.path.exists(os.path.join(D, "appgraph", "cello_appgraph__dark__stops.json")) else []
APPLOG = []
for lf in ("appgraph_exact.log", "appgraph.log"):
    f = os.path.join(D, "logs", lf)
    if os.path.exists(f) and "APPGRAPH DONE" in open(f).read():
        for line in open(f):
            parts = line.split("|")
            if len(parts) >= 4 and parts[0].strip().isdigit(): APPLOG.append((int(parts[0]), float(parts[1]), float(parts[2].split()[0]), float(parts[2].split()[1].replace("dB", "")), float(parts[2].split()[3])))
        break
P.append('''<h2><span class="num">10</span>Any track graph in the app</h2>
<p>Everything so far ran on one hand-built chain. The tool has to run on whatever the user has patched: the app's inserts and wires, in any order, with parallel branches, a VST in the middle. So there is now a compiler: it takes a track's <code>FxInsert</code> graph, walks the app's own topological order, sums at merges the way the app's engine does, gives every stock insert a differentiable twin <em>in the app's own parameter units</em>, freezes what it cannot twin (a VST passes through, flagged), starts every search at the user's current settings, and exports app parameters per insert id when it is done. The twins were then calibrated against the app's engine (pedalboard, i.e. JUCE) on the same audio until each one reproduced it sample for sample — which took measuring rather than reading: JUCE's ballistics run on a time constant of <em>ms / 2π</em>, its delay truncates to whole samples, Freeverb sizes its lines by integer division, its chorus LFO drifts by up to 0.1 % because its phase is a float32 accumulator. Two of our own bugs fell out on the way — the compressor twin started fully clamped, and the effects chain's compressor had been 6× too slow.</p>
<p class="axis">sample snr vs the app engine, per twin: ''' + " · ".join(f"{n} {s} dB" for n, s in RT) + '''<br>whole graphs: serial eq → comp → sat → delay 78 dB · parallel eq ∥ highpass → sat 99 dB · gain → unknown VST (frozen) 152 dB</p>''')
if APP and APPLOG:
    P.append('''<p>End to end, on an app graph — EQ → compressor → delay → reverb on the cello recording, the word “dark”, the fixed-distance ladder: the exported parameters, rendered by the <em>app's</em> engine, score what the twin promised.</p>
<div class="tw"><table class="ab"><tr><th>stop</th><th>target</th><th>distance</th><th>twin score</th><th>app engine, exported params</th><th>envelope corr</th></tr>''')
    for i, tw, ap, lvl, ec in APPLOG:
        st = APP[i]; P.append(f'<tr><td>{i}</td><td>{st["target"] if st["target"] is not None else "free"}</td><td>{st["dist"]:.2f}</td><td>{tw:+.3f}</td><td>{ap:+.3f}</td><td>{ec:.3f}</td></tr>')
    P.append("</table></div><div class=\"stack\">")
    f = os.path.join(D, "appgraph", "cello_appgraph__dark__LADDER.wav")
    if os.path.exists(f): P.append(clip(mp3(f, "appgraph_cello_dark_ladder"), "app", "cello dark — the app's own graph, eq → compressor → delay → reverb", "the first bar: original, then stops 1 → 6"))
    P.append("</div>")
P.append('''<div class="finding"><b>What this buys:</b> the search no longer lives in a research chain. Point it at a track, it climbs the user's graph from the user's settings and hands back parameters the app already knows how to render — and the round-trip check means a discrepancy is a bug, not a shrug. Left: a Vital instrument as the compiled source (the twin at the head of the graph), pre-rendering a VST that sits first, and wiring it into the app as <code>tune_toward</code>.</div>
''')
# 11 the A/B and the variance
FLUX = {}
for arm in ("nograd", "nograd_rep", "fixed", "fixed_rep"):
    f = os.path.join(D, "ladder_flux", arm, "epiano_twin__dark__stops.json")
    if os.path.exists(f): FLUX[arm] = json.load(open(f))
CELLO2 = {}
for arm in ("nograd", "nograd2"):
    f = os.path.join(D, "ladder_flux", arm, "cello_fx__punchy__stops.json")
    if os.path.exists(f): CELLO2[arm] = json.load(open(f))
P.append('''<h2><span class="num">11</span>A penalty that was not pulling, and how much a run moves on its own</h2>
<p>Checking the new twins' gradients turned up a bug in the judge: the flux penalty — the largest of the four, the one against chorus and pumping — had <em>no gradient at all</em> on the GPU. Its spectrogram used the library's own reflect padding, whose backward pass is broken on Apple's MPS past 65 k samples; a median's gradient is one frame's worth, far into the clip, and it was swallowed whole. The penalty was still <em>scored</em>, so the optimiser could be pruned at the line but never steered away from it. Fixed, then tested the honest way: the same cell twice per arm, same seeds, the flux term detached in one arm (the old behaviour) and live in the other.</p>''')
if len(FLUX) == 4:
    P.append('''<div class="tw"><table class="ab"><tr><th>e-piano dark, twin route</th><th colspan="2">without the gradient (two runs)</th><th colspan="2">with it (two runs)</th></tr><tr><th>stop target</th><th>word</th><th>within-note flux vs dry</th><th>word</th><th>within-note flux vs dry</th></tr>''')
    for i in range(1, 7):
        row = f'<tr><td>{FLUX["nograd"][i]["target"] if FLUX["nograd"][i]["target"] is not None else "free"}</td>'
        for a, b in (("nograd", "nograd_rep"), ("fixed", "fixed_rep")):
            c = (FLUX[a][i]["clap"], FLUX[b][i]["clap"]); fl = (FLUX[a][i]["art"]["d_flux"], FLUX[b][i]["art"]["d_flux"])
            row += f'<td>{c[0]:+.2f} / {c[1]:+.2f}</td><td>' + " / ".join(f'<span class="{"over" if v > 1.0 else ""}">{v:.2f}</span>' for v in fl) + "</td>"
        P.append(row + "</tr>")
    P.append('''</table></div>
<p class="axis">the penalty's hinge is at 1.0 — values past it are highlighted</p>
<div class="pair">''')
    for arm, lab in (("nograd", "without the flux gradient — the old behaviour"), ("fixed", "with it")):
        f = os.path.join(D, "ladder_flux", arm, "epiano_twin__dark__LADDER.wav")
        if os.path.exists(f): P.append(clip(mp3(f, f"flux_{arm}_epiano_dark_ladder"), "fx", f"e-piano dark, twin — {lab}", "original, then stops 1 → 6"))
    P.append('''</div>
<div class="finding"><b>The mechanism works; the score effect is modest.</b> Without the gradient, four of twelve stops sat just past the hinge (1.03, 1.16, 1.19, 1.27) — parked at the line, which is exactly what a scored-but-not-backpropagated penalty looks like. With it, one of twelve, at 1.00. The constrained stops also reach their distance targets (1.43–1.54 for targets 1.6 / 2.5, against 1.10–1.17 without: steering around the hinge instead of being pruned at it), and the free stop scores 0.378 ± 0.012 against 0.305 ± 0.019. At matched distance the gain is about +0.03 around distance 0.9 and within noise elsewhere. On the recording route the hinge never engages at all (cello punchy: flux 0.42–0.47 at every stop, both arms), so the fix changes nothing there.</div>''')
if len(CELLO2) == 2:
    P.append('''<h3>The same run, twice</h3>
<p>The control arm was also run twice on the cello, identical seeds, identical code. The two agree to three decimals for two stops and then part ways for good:</p>
<div class="tw"><table class="ab"><tr><th>cello punchy, recording route</th>''' + "".join(f'<th>{st["target"] if st["target"] is not None else "free"}</th>' for st in CELLO2["nograd"][1:]) + "</tr>")
    for arm, lab in (("nograd", "run 1"), ("nograd2", "run 2")):
        P.append(f'<tr><td>{lab}</td>' + "".join(f'<td>{st["clap"]:+.3f}<br><small>{st["dist"]:.2f}</small></td>' for st in CELLO2[arm][1:]) + "</tr>")
    P.append("</table></div><div class=\"pair\">")
    for arm, lab, note in (("nograd", "run 1 — free stop, +0.20", "drive −6 dB: EQ and transient shaper doing the work"), ("nograd2", "run 2 — free stop, +0.55", "piecewise-tanh drive +22.5 dB, threshold 0.01: hard clipping, which CLAP calls very punchy")):
        f = os.path.join(D, "ladder_flux", arm, "cello_fx__punchy__stop6_dinf.wav")
        if os.path.exists(f): P.append(clip(mp3(f, f"bifurcation_{arm}"), "rec", lab, note))
    P.append('''</div>
<div class="finding"><b>Ladders are not repeatable on the GPU past the second stop.</b> At stop 3 one run's successive halving kept a candidate that had found the clipping route and the other never did; MPS kernels are not bit-deterministic, and the keep/prune decisions amplify the difference into two different sounds. Two consequences. Every single-run comparison on these pages carries a spread of that order on cells that can bifurcate (the e-piano cell was tamer, ±0.01–0.04): from here, repeats or a deterministic CPU mode before saying one route beats another. And the clipping route got past the judge — crest +9.8 dB and jumps 1.7 sat under the thresholds — so "punchy" still has a cheat available.</div>
''')
P.append('''<h2><span class="num">12</span>Method notes</h2>
<p>Every run now re-renders its saved parameters through a freshly built synth and chain and checks that the score reproduces; a first pass at this analysis mis-read sixteen cells because the re-scoring script built the synth with linear frame interpolation while the cello table is stepped. The twins were recovered with 250-step searches from six starts (correlation to the soundfont: cello 0.81, brass 0.93, e-piano 0.96, flute 0.86). The named-prompt grid reused the same patches. Scores are CLAP cosines with per-note phases fixed; random phases move a score by a standard deviation of 0.006–0.01. Since the grid: every node type is in the chain by default (+0.02 mean on six cells, +10 % time; per-node contribution eq 0.09, delay 0.08, transient shaper 0.05, reverb 0.03, compressor 0.03, gate 0.02, chorus 0.02, drive 0.01); the drive and piecewise-tanh nodes are level-referenced (the absolute-level tanh was never used); every effect initialises at bypass; a retrieval warm start (the nearest earlier prompts on the same instrument) runs 370 steps instead of 700 for equal-or-better results (mean 0.341 vs 0.322, 120 s vs 214 s per prompt); early pruning of the halving was measured over 155 runs and rejected (the runner-up overtakes 37 % of the time). The step is 290 ms: synth 100, chain 90, CLAP 46, locality 17 — the optimiser's cost is not CLAP.</p>
</main>''')
open(os.path.join(SITE, "index.html"), "w").write("\n".join(P))
n_mp3 = len(glob.glob(os.path.join(MP3, "*.mp3"))); size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(MP3, "*.mp3"))) / 2**20
print(f"page {os.path.getsize(os.path.join(SITE, 'index.html'))/1024:.0f} KB, {n_mp3} mp3s, {size:.1f} MB")
