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
    label = {"ref": "reference", "clone": "twin", "fx": "twin + fx", "vital": "Vital"}[role]
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

names = [n for n, _ in INSTRUMENTS]; inst_json = {n: json.load(open(os.path.join(D, f"{n}_patch.json"))) for n in names}
plain = {(n, w): load(D, n, w) for n in names for w in WORDS}
named = {(n, w): (load(os.path.join(D, "named"), n, f"a {w} {DISPLAY[n]}")[0] or load(os.path.join(D, "named"), n, art(f"{w} {DISPLAY[n]}"))[0],
                  load(os.path.join(D, "named"), n, f"a {w} {DISPLAY[n]}")[1] or load(os.path.join(D, "named"), n, art(f"{w} {DISPLAY[n]}"))[1]) for n in names for w in WORDS}
contrast = {(n, w): load(os.path.join(D, "contrast"), n, art(f"{w} {DISPLAY[n]}")) for n in names for w in WORDS} if os.path.isdir(os.path.join(D, "contrast")) else {}

def heat(v, lo=-0.15, hi=0.45):
    t = max(0.0, min(1.0, (v - lo) / (hi - lo))); return f"background:color-mix(in oklab, var(--accent) {int(t * 55)}%, var(--surface))"
# ---------------------------------------------------------------------------------------------------------------
P = []
P.append(f'''<title>Ten Words, Four Instruments</title>
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
    P.append(clip(mp3(os.path.join(D, f"{n}_ref.wav"), f"{n}_ref"), "ref", f"{DISPLAY[n].capitalize()} — GM program {prog + 1}"))
    P.append(clip(mp3(os.path.join(D, f"{n}_clone.wav"), f"{n}_clone"), "clone", f"{DISPLAY[n].capitalize()} — the twin's patch", f"cutoff {ph['cutoff_raw']:.2f} · res {ph['resonance']:.2f} · attack {ph['attack']*1000:.0f} ms · release {ph['release']:.2f} s · CLAP “a {DISPLAY[n]}” {base[n]['_self']:+.2f}"))
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
P.append('''<h2><span class="num">06</span>Method notes</h2>
<p>Every run now re-renders its saved parameters through a freshly built synth and chain and checks that the score reproduces; a first pass at this analysis mis-read sixteen cells because the re-scoring script built the synth with linear frame interpolation while the cello table is stepped. The twins were recovered with 250-step searches from six starts (correlation to the soundfont: cello 0.81, brass 0.93, e-piano 0.96, flute 0.86). The named-prompt grid reused the same patches. Scores are CLAP cosines with per-note phases fixed; random phases move a score by a standard deviation of 0.006–0.01.</p>
</main>''')
open(os.path.join(SITE, "index.html"), "w").write("\n".join(P))
n_mp3 = len(glob.glob(os.path.join(MP3, "*.mp3"))); size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(MP3, "*.mp3"))) / 2**20
print(f"page {os.path.getsize(os.path.join(SITE, 'index.html'))/1024:.0f} KB, {n_mp3} mp3s, {size:.1f} MB")
