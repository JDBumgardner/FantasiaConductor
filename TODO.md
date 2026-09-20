# TODO

## Roadmap — toward `tune_toward(track, target, amount, scope)`

The feature: "make this sound more like X" for a human or the agent, given any FX
graph (and the Vital twin if the track is a Vital track). Pages: *The Vital Twin*
(writeup, with sounds) and *Vital Twin Roadmap* (this list, expanded).

**Critical path:** noise source → compressor → graph compiler → directional loss.
The first two widen what a prompt can reach; the second two make it a tool.

### Effects the chain is missing (GRAFX has most; some are ours to write)
Registry + serial chain builder: `experiments/text2fx/fxgraph.py` (node type →
make / init / prior / describe; per-node activation checkpointing so eight
FFT-conv nodes fit under the 8 GB MPS cap). `text2synth` / `house_session` take
the chain from `T2_CHAIN=eq,comp,dist,delay,reverb` (default). Every node and
both chains: CPU-vs-MPS gradient cosine 1.0000 at full 10 s length.
- [x] **Compressor** — ours (`fxgraph.Compressor`): GRAFX's applies its
      ln-energy gain straight to the waveform (ratio 2 is already a limiter,
      >2 inverts) and shifts the threshold −26 dB, so we kept only its FFT
      smoother and wrote a dB-domain law (verified to 0.1 dB on sine bursts).
      torchcomp attack/release ballistics run on the CPU (8 ms; 1.3 s on MPS).
      On the three cello prompts (fixed chain) it settled at ratio 1.7–8.2,
      threshold −17…−25 dB, release 60–370 ms; no loudness cheat seen. The
      ratio prior (≤ 8) is the rail it leans on for "airy".
- [x] **Delay** — GRAFX `MultitapDelay`, 8 surrogate taps, uncoloured, dry/wet;
      its `radii_reg` goes into the loss. Used at mix 0.26–0.52 on all three
      cello prompts once the dry/wet bug (below) was fixed. [ ] tempo-sync taps.
- [x] **Chorus / flanger** — ours: LFO-modulated fractional delay (1–30 ms,
      0.05–10 Hz, linear interp). [ ] feedback path (needs the recursion below);
      [ ] reuse as the synth LFO.
- [x] **Saturation flavours** — GRAFX `PiecewiseTanhDistortion` (asymmetric
      hardness/threshold) as `pwtanh`. GRAFX `ChebyshevDistortion` has a broken
      backward (in-place op) — skipped. [ ] measure Vital's distortion types.
- [x] **Transient shaper** — ours: 1 ms vs 20–200 ms RMS followers, ±1 dB/dB
      attack and sustain gains (kalimba peak/RMS 2.26 → 3.81 at attack +1).
      Behaviour under prompts still to measure on the fixed chain (the one
      eight-node run, attack +0.9 on "burbling", predates the fixes).
- [x] **Gate** — ours: soft expander on a smoothed RMS level (GRAFX's
      `NoiseGate` shares the compressor's gain-law problem).
- [ ] **Stereo** — the twin is mono; a mono judge cannot hear *wide*. Needs a
      stereo chain and a stereo-aware loss (GRAFX has mid/side tools).
- [x] **Graph compiler v1** (`text2fx/compile.py`, `appnodes.py`, `roundtrip.py`,
      2026-09-20): `compile_track(inserts, wires, source) → SearchGraph` walks
      the app's own `topo_order`/`effective_wires`, sums at merges like
      `FxHost`, handles `mix` nodes, freezes inserts without a twin (`vst`),
      starts at the user's current settings and `export()`s app-unit params per
      insert id. Twins in app units, round-tripped against pedalboard on the
      same audio: gain / saturator / distortion / lowpass / highpass / eq /
      delay **exact** (SNR 82–159 dB), **reverb exact** (Freeverb evaluated in
      the frequency domain: 8 damped combs, 4 JUCE 'allpasses', stereo spread,
      input gain 0.015·(L+R), wet ×3 / dry ×2), compressor & limiter within
      0.2 dB / env corr 0.93–0.99, chorus and gate approximate. Whole parallel
      graph exact (99 dB). End to end on eq→comp→delay→reverb for "dark": the
      exported parameters rendered by the app's engine score within ±0.035 of
      the twin at every ladder stop. `ladder.run(..., graph=)` searches any
      compiled graph. [ ] chorus (JUCE depth/LFO law), gate (ratio expander),
      first-node VST pre-render, Vital instrument as the source through the
      compiler, hook into the app as `tune_toward`.
- **Found by the round trip: our compressor's ballistics were never applied.**
  torchcomp's `compressor_core` takes the update fraction 1−α (we passed α ≈
  0.999) and treats "attack" as the coefficient for a *falling* input (it
  smooths gains). Fed a level, attack and release were effectively
  instantaneous — a sample-by-sample waveshaper with a hard knee — in every run
  before 2026-09-20 (the HF splatter, the pumping flags, the appetite for 8:1).
  Fixed (1−α, roles swapped); a 0.5 sine now reads 0.44 as JUCE does.
- **Four upstream bugs found on the way (three GRAFX, one torchcomp), all fixed
  in `fxgraph.py`, the first three affecting every result before 2026-09-15:**
  1. `DryWet` documents mix = sigmoid(z) but uses z raw: "reverb mix 0.5" in the
     earlier runs and the writeup meant z = 0 = *no reverb*; the −1 init meant
     −1·wet + 2·dry. [ ] re-run the writeup's experiments and correct the page.
  2. `convolve` calls `irfft` without `n=`: an odd x+h−1 (ours) comes back one
     sample short and the whole output is time-stretched by that sample, so
     dry+wet combed at the top of the band in EQ, compressor, delay and reverb.
     Patched (exact length, power-of-two padding: 3–10× faster too).
  3. The reverb draws its noise bank from the *unseeded* numpy RNG and re-draws a
     random offset every forward: the reverb heard was never the one optimised.
     Now seeded and `"fixed"`.
  4. (torchcomp) `CompressorFunction.setup_context` saves its own output with
     `save_for_forward` → a grad_fn→y→grad_fn cycle the GC cannot see: every
     forward whose graph is never backpropagated (an evaluation in grad mode,
     an aborted line search) leaked the whole upstream graph, ~230 MB. This was
     the "MPS OOM after the L-BFGS polish". `fxgraph._Ballistics` drops it.
  Lesson: a bypass test (every node at "amount 0" must return the input) and a
  forward-only leak test belong next to the gradient test for every node
  (`python fxgraph.py`).

### Why the clone isn't a perfect Vital (measured residuals)
- [ ] **Recursive time-varying filter** via torchlpc (installed): fixes the
      ~4 dB pluck-onset error (frame-domain smearing) AND lets the saturation
      sit inside the filter loop like Vital's analog model (±1.5 dB residual).
- [ ] **Other filter models** — Ladder, Dirty, Digital, Diode, Formant, Comb,
      Phase, 24 dB variants. Only Analog 12 dB (with LP→BP→HP morph) is measured.
- [ ] Unison phase: blend ±20% is inherent (per-note random phases). Accept.
- [ ] Wavetables: magnitudes only, 128 harmonics; add phase-distortion /
      spectral-morph modes only when presets need them.

### Vital features to add (ranked by palette bought)
1. [~] **Noise / sample source** — built 2026-09-17 as Vital's sample oscillator:
       measured on the plugin (default sample is white noise within 0.4 dB;
       output rms = 0.207·sample_level²; takes the voice amp envelope; destination
       FILTER 1 sends it through the filter; keytrack negligible). Twin: `noise`
       raw = linear amplitude fraction a, summed at the filter input with
       NOISE_GAIN 0.28; export sample_level = √a. Level law matches Vital within
       3 % over the range incl. the drive saturation at a = 1; filtered spectrum
       within 1 dB/band; CPU-vs-MPS gradient cosine 1.0. On by default (quiet
       start, `T2_NOISE=0` to disable) in recovery, house search and text2synth.
       [x] Loss fixed: `common.band_energy_loss` (32 log bands, 80 ms smoothing)
       is noise-realisation-invariant (0.12 equal level vs 0.61 half level; the
       fine STFT floor is 0.73 → 0.78) and enters `closed_loop_filter.loss_fn`
       at weight 3. Secret pluck: noise 0.44 → 0.68 recovered for 0.60.
       [x] Re-recovered cello and flute with noise (`words/noise/`). CLAP "a
       flute": twin 0.13 → 0.20 (soundfont 0.33), survives the Vital round trip
       (0.18); noise 0.31. Cello: 0.18 → 0.19 — bow noise is pitched and
       granular, not white at the filter input. [ ] re-run the soft / airy /
       distant column on the noise patches.
2. [ ] **LFO** (8 + random) — vibrato, wobble, tremolo; chorus with a delay line.
3. [ ] **Oscillator 2 with FM/RM** — bells and metallic sounds need it.
4. [ ] **Mod matrix as parameters** — routings + amounts searchable, not one
       hand-wired env→cutoff.
5. [ ] **Vital's own effects** (chorus, delay, reverb, compressor, EQ) so the
       whole result can live inside one preset.
6. [ ] **Filter 2, routing, filter models.**
Each: measure on the plugin → CPU-vs-MPS gradient cosine at full length →
closed-loop recovery.

### Experiments
- [x] **Prompt → parameter map** — `experiments/text2fx/words/` (page: *Ten
      Words, Four Instruments*): 10 single words × cello / brass / e-piano /
      flute × 3 objectives = 120 cells. Findings: spectrum/texture words gain
      +0.15…+0.39 on every instrument; *soft / airy / distant* start at −0.1…
      −0.15 (CLAP is sure a sustained single-osc tone is none of them) and only
      reach zero — the noise/breath source, not the optimiser. The filter is
      barely used; the EQ at its ±6 dB rails and the compressor do the spectral
      work (prior weighting to fix). Half-amount scores separate paths (e-piano)
      from corners (sustained instruments). Untouched twins read as their
      instrument for brass (0.41) and e-piano (0.39) but not cello (0.18) or
      flute (0.13).
- [x] **Naming the instrument** ("a dark cello") — anchors identity (cello 0.18
      → ~0.5 on "a cello") but the adjective evaporates on cello/flute/brass;
      works on the e-piano, whose identity term is already saturated.
- [x] **Directional loss, first form** — cos(named) − cos(instrument): more
      adjective than the named sentence, less than the plain word, and it throws
      the identity away (brass 0.41 → 0.05–0.28). Three objectives, three
      failure modes.
- [ ] **Hinge objective** — cos("a soft cello") + λ·min(0, cos("a cello") − cos₀):
      hold identity, don't trade against it.
- [x] **Text → FX on the recording itself** (`words/source/`, 2026-09-17): works on
      the fixed chain (cello dark −0.10 → 0.39); twin ahead where the word is
      about the source (punchy, e-piano dark), recording ahead on e-piano punchy.
      But: endpoints — see the frontier.
- [x] **The frontier (`words/frontier.py`, 2026-09-18)** — "make it darker" is a
      direction and an amount, so a route is judged by its score-vs-distance
      curve: each stop its own optimum at that distance (locality weight swept
      3 → 0.01 by continuation, 80 steps per stop), every effect starting at
      bypass, distance = band-energy of the HEARD output vs the untouched
      instrument (phase-invariant; the fine mrstft has a 0.32 phase floor),
      candidates ranked by the objective (ranking by score alone let the
      locality have no say in which start wins — also true of prompt_on until
      now). Four attempts, each exposing an asymmetry; the last was that the
      twin's effects were not counted as distance. Result on cello dark / punchy,
      e-piano dark: at matched distance the twin is above the recording route on
      all three (cello @0.5: +0.12 / +0.23 vs −0.01 / +0.05); the recording route
      saturates on cello dark (never past −0.01). Musical stops are 3–5, not 6.
      Listening: `words/listen_frontier/` (+ `ab/` ladders and A/Bs — adjacent
      stops spaced by λ are hard to tell apart).
- [x] **Stops at fixed distances** — `text2fx/ladder.py` (2026-09-18): soft
      constraint μ·relu(dist − d*)² at d* = 0.15 / 0.4 / 0.8 / 1.5 / 3.0 / ∞,
      each stop warm-started from the previous AND two fresh starts (a tight
      optimum is a poor init for a loose one: without them the far end stalled
      at +0.10 where a cold search reached +0.39), artefact penalties in the
      loss, ranking by the objective, every effect from bypass. Stops land on
      their targets (0.20 / 0.43 / 0.82 / 1.14 for 0.15 / 0.4 / 0.8 / 1.5).
      Final ladders, all stops clean: cello dark twin +0.35 @1.30, recording
      +0.20 @0.90; cello punchy twin +0.33 @0.97, recording +0.47 @1.77; cello
      soft twin +0.11 @1.18; e-piano dark twin +0.39 @1.68, recording +0.35
      @0.79. Listening: `words/listen_frontier/ladder_final/`.
      [ ] `prompt_on` / `tune_toward` return the ladder; rank by the objective
      and use the band-energy locality there too.
- [ ] **Amount as a knob** — candidates at 25/50/100% of the parameter delta,
      ear chooses (the paper's best numbers relied on exactly this).
- [ ] **Transfer across phrases** — does a patch tuned on the hook survive a
      different melody? If not, the feature doesn't either.
- [ ] **Reference + text together** — magnitude from the clip, direction from words.
- [ ] **Listening protocol** — ten pairs, forced choice, written down.
- [ ] **Reranking judge** — CLAP for gradients; an audio-LM or Audealize's
      word→EQ data (magnitude prior for adjectives) to pick finalists.
- [~] **Speed** — done: the synth renders each note's tail only as long as its
      release (quantised to 0.25 s, capped at 2 s with a fade; the old fixed 1 s
      tail also chopped long releases mid-curve) and runs the short-frame onset
      filter on the first 160 ms only: synth fwd+bwd 151 → 100 ms, full house
      step 348 → 291 ms on MPS. Left: batch restarts, cache text embeddings,
      5 s excerpts; the compressor's CPU ballistics round trip forces a sync.

### The optimiser
Today (`optim.py`, default in `text2synth` and `house_session`, `T2_OPT=adam`
for the old path): **successive halving** over 8 inits (30 steps × 8 → 60 × 4 →
220 × 1, cosine decay in the last stage), an **L-BFGS polish** with the synth's
phases frozen, and a **frontier** (every finalist with score, time-shift-robust
score and distance to the instrument) plus a **50 %-amount render**. Measured
against 3 × 300 Adam at equal budget on the cello hook (fixed chain):

| prompt | Adam 3 × 300 (restarts) | halving 8 → 4 → 2 → 1 |
|---|---|---|
| under water, burbling | 0.358 (0.36 / 0.15 / 0.24) | **0.458** (robust 0.439) |
| airy, whistling through the trees | 0.185 (0.12 / 0.18 / 0.17) | 0.188 |
| catholic orchestral, in chorus | **0.468** (0.47 / 0.44 / 0.42) | 0.439 |

Breadth wins where restarts disagree (a rugged landscape), depth wins where they
agree; the final rounds now give the winner 310 steps so the downside is small.
The polish is worth 0–0.02 and 15 s per finalist. The loss is nearly
deterministic (CLAP has no augmentation; per-note random phases move the score
by a std of 0.006 over six draws), so the restart spread is multimodality, not noise. "At 50 %
amount" scores show the corners are real: under water 0.458 → 0.295 at half
amount, airy 0.188 → −0.004.
1. [~] **Amortised inference** — the cheapest form is in and on by default
       (2026-09-17): a **retrieval warm start**. Two candidates are the finished
       results of the nearest earlier prompts on the same instrument (CLAP text
       similarity; synth params plus the FX params of shared node types), and
       the budget drops to 370 steps. Held-out on six cells: mean 0.341 vs 0.322
       cold at 700 steps, 120 s vs 214 s per prompt; never worse than −0.03,
       +0.07 / +0.09 on metallic / e-piano punchy. Every result the tool makes
       speeds up the next; a fresh instrument still starts cold. The learned
       version (mel→parameters on twin-rendered patches) remains for later.
2. [ ] **Batch the restarts** — the synth's saved activations are ~400 MB per
       candidate at 20 notes; two fit, eight do not, on 8 GB.
3. [~] **Average phase draws** — measured: std 0.006 per draw; not worth it for 1 voice.
4. [ ] **Coarse-to-fine** — 3 s excerpts first; envelope/coarse terms first,
       fine STFT terms later (frame ridges live in the fine terms).
5. [ ] **Outer loop for discrete axes** — stepped frame, voices, filter model,
       table choice: grid / CMA-ES around the gradient inner loop.
6. [x] **L-BFGS finish** with phases fixed (small, kept). As the *main* optimiser it
       loses to Adam from the same init (0.085 vs 0.367, 0.289 vs 0.334): the
       line search stalls on the piecewise-linear parts (table lookups, relus).
7. [x] **Return the frontier**, with a robust score and the half-amount render.
8. [x] **Adaptive halving — measured, rejected.** Over 155 logged runs the
       stage-1 runner-up overtook the leader 37 % of the time, and 14 % even when
       the stage-1 gap exceeded 0.05; pruning would save ≤ 3 % of the budget.
       Keeping 3 instead of 4 after stage 0 would have lost the winner in 13/155.
       The landscape is too rugged to prune early.
9. [x] **Rebalancing A/B** (2026-09-17, six cells, `words/balance/`): per-step
       embedding sensitivity showed the cutoff moves CLAP *more* than an EQ band,
       so the EQ's dominance is effectiveness, not weighting. Level-referenced
       drive (A) and a locality term on the heard output (B): no change (±0.01).
       Fully equalised per-parameter lr (C): moves work to drive/resonance but
       −0.03…−0.10 on the cello, +0.05 on the e-piano. **Wide chain (D, all nine
       nodes): +0.02 mean, cello soft 0.08 → 0.14, +10 % time — now the default.**
       Per-node contribution (bypass): eq 0.09, delay 0.08, transient 0.05,
       reverb 0.03, comp 0.03, gate 0.02, chorus 0.02, drive 0.01, pwtanh 0
       (was absolute-level; now level-referenced). Outlier-only lr rule (E): ≈ D,
       costs 45 s of measurement — off by default. One-bar excerpt for the early
       stages (F): −0.02 and only 8 % faster — off. Step profile on the wide
       chain: synth 102 ms, chain 90 (comp ballistics 21, transient 22, gate 13,
       reverb 12, ~30 of checkpoint recompute; the FFT convolutions are 2–5 ms
       each), CLAP 46, locality 43 → 17 after dropping to two resolutions with
       cached anchor spectra.

### The judge
- [x] **Signal-level artefact detectors** (`text2fx/artefacts.py`, 2026-09-18):
      clip fraction, sample jumps vs local level, within-note spectral flux
      (chorus / flanger / pumping), within-note 2–14 Hz level modulation
      (tremolo / pumping), crest change (squashed), gate chops — each inside
      sustained notes and as a delta vs the same notes without the effects (a
      naive modulation detector flagged 119/168 because melodies move the
      centroid at note rate). Over 168 results: pumping 39, chorus/flanger 12,
      stutter 7, squashed 1, clipping 0. Worst offenders sent for listening.
- [x] **Artefact penalties in the loss** (`artefacts.penalty`): hinges on
      within-note flux, 2–14 Hz tremolo, crest drop, sample jumps, plus a
      differentiable gate-drop term (falls > 6 dB per 10 ms anywhere within
      45 dB of the peak — the chops the flags found were in the release
      tails, which a note mask excluded), all against the same render's
      detached dry signal; weights ×3 after the first ladders still bought
      their last +0.05 with wobble; gate range prior 60 → 25 dB. Result: every
      ladder stop clean, scores held or improved (cello punchy on the
      recording +0.43 with chops → +0.47 clean).
- [x] **Audiobox-Aesthetics — measured, rejected** (`text2fx/aesthetics.py`
      kept as an extra signal). Over 168 results vs their dry renders: ΔPQ clean
      −0.28, pumping −0.43, gate stutter −0.66, but chorus/flanger **+0.15**
      and its enjoyment axis rises *more* for pumping (+0.73) than for clean
      (+0.39): it reads modulation and pumping as production, not damage. Per
      file it is too noisy to gate with (the clean control lost 1.36 PQ, the
      chorus offender nothing). Note torch.median leaks driver memory on MPS.
- [ ] **Parameter-space realness prior** from real Vital presets (75 installed,
      thousands online): density model / real-vs-random classifier over the
      searched parameters. Cannot be fooled by audio tricks; supplies magnitude.
      Build before any audio-domain discriminator.
- [ ] **Randomise the judge's input** (EOT): random time shift, small EQ tilt,
      tiny pitch shift before CLAP each step.
- [ ] **Preference head** on CLAP embeddings trained from the listening protocol's
      A/B choices (a few hundred pairs).
- [ ] **Ensemble** a second text–audio embedding model.
- [ ] **Audio-LM reranker** over the 3 finalists (not in the gradient loop).
- [ ] **Zero-shot sanity gate**: rank the result against ~50 adjectives; flag a
      cheat if the prompt's word isn't near the top.
- [ ] NOT: an audio-domain GAN discriminator — frozen it is one more exploitable
      network (adversarial examples transfer); adaptive per sound it overfits;
      trained on real-vs-twin it learns twin artifacts. Our failures were
      magnitude and semantics, not realness.

---

### The feature itself
- [ ] Compile graph → twins (frozen pass-through for untwinned nodes; synth twin
      at the head for Vital tracks).
- [ ] Target = text (directional) | reference (spectral + envelope) | both.
- [ ] Sanity stack: loudness-matched judge, level fixed, priors at musical knees,
      locality to the current sound, structural bounds.
- [ ] Propose, don't apply: 3 candidates at increasing amount, each rendered for
      A/B with a plain-language diff; the pick lands as undoable Commands
      (`set_eq_band`, `set_fx_param`, `set_plugin_param`).
- [ ] Background run with progress; cache the compiled twin per graph.
- [ ] Tool description carries the limits: proposes a direction, amount is the
      user's, *a* patch not *the* patch, only moves what has a twin.

---

Open work, roughly in priority order. Experiments live in `experiments/text2fx/`;
measured plugin behaviour is recorded in `fantasia_core/plugin_notes/` and the
memory notes.

## Torch synth twin (`experiments/text2fx/synth.py`)

- [x] **Measure Vital's unison blend law.** Done 2026-09-13: every outer voice
      has the same weight vs the centre regardless of voice count,
      r(b) = 0.503 b + 0.202 b^2 (mono sum; outers are hard-panned); detune
      spread is (k/K)^1.5, not linear. Blend recovers 0.535 vs 0.5 on random-phase
      Vital audio via the in-note beating-depth term (`common.beat_depth`).
- [x] **Frame ridge.** `common.harmonic_loss`: per-note log-power at each
      harmonic of the known pitch, window wide enough for detuned sidebands,
      shape-normalised, -60 dB floor. Clean minimum at the true frame with a
      10x margin; no ridges.
- [x] **Multi-frame wavetables.** Done 2026-09-13: `extract_table.py` loads a
      preset with routings stripped and osc-only settings pinned (morph type,
      frame spread, detune power, volume — every one of these was a silent
      confound), sweeps 33 frame positions, FFTs harmonic magnitudes, calibrates
      absolute scale against Vital, saves `tables/*.pt` with the interpolation
      style. Closed loop with frame free: Basic Shapes (stepped) 0.273, Harmonic
      Series (smooth, single-partial frames) 0.301, Classic Fade (smooth,
      musical) 0.323 — all for a secret 0.300. The pitch-aware `harmonic_loss`
      is what made frame findable; use `LOSS=full`.
- [ ] **Stepped tables.** Basic Shapes has `interpolation: 0`; the twin still
      blends linearly between extracted rows. Snap to the keyframe when the
      table metadata says so.
- [ ] **Level calibration per voice count / blend.** Currently a single constant
      (`LEVEL_CAL = 0.694`). Vital pans outer voices, so the mono sum drifts
      ~1 dB with blend. Either model the panning or tabulate the constant.
- [x] **Filter + filter envelope.** Done 2026-09-13. Measured on Vital: cutoff
      f = 261.6 * 2^((128 raw - 52)/12) ("0 semitones" = middle C; keytrack raw
      0.5 = 0%, raw 0.0 = -100% — pinning it to 0.0 shifted every cutoff by 3
      octaves), resonance -> (passband dB, Q) tabulated, mod amount = 2 raw - 1
      adds amount*128 semitones, env 2 applies LINEARLY (amp env is squared).
      Twin: time-varying 2-pole LP in the STFT domain (NF 4096: -29 dB error at
      275 Hz / Q 5; near self-oscillation is out of reach). Closed loop, 15
      params: 1 voice 0.994 corr / bands within 0.5 dB; 3 voices 0.947.
      Base cutoff and filter sustain trade off (same trajectory) — inherent.
- [x] **Envelope powers.** Vital's curve is (1 - e^{p x}) / (1 - e^p), display
      power = 40 (raw - 0.5); matches at -2 / -12 / 0 to 3 dp. Modelled for both
      envelopes, exported, and recovered in the loop — up to an inherent
      degeneracy: a steeper power with a longer time is the same curve as a
      gentler power with a shorter one (found -10.2 / 0.40 s for -5.54 / 0.25 s,
      identical trajectory, 0.992 round-trip).
- [x] **Filter blend + drive.** Blend (display 0..2 = 2 raw): d<=1 -> (1-d) LP +
      sqrt(1-(1-d)^2) BP; d>1 -> (d-1) HP + sqrt(1-(d-1)^2) BP, SVF outputs summed
      as complex responses — max error 1 dB across the morph. Drive: display
      20 raw dB; the filter input ALWAYS passes tanh(2 g x)/(2 sqrt g), g = 10^raw
      (THD -33 dB at level 0.5 even at 0 dB drive; matches Vital within 1-3 dB at
      six level/drive points). Osc level is applied BEFORE the filter (0.52 per
      unit display level at the filter input). Only the Analog 12 dB model;
      24 dB / other models still unmodelled.
- [x] **Spectral unison** is a no-op for a plain wavetable (identical sidebands
      on/off); it only acts with a spectral morph, which is not modelled.
- [x] **Unison frame spread** verified on Classic Fade: exports exactly (Vital
      shows 40 / -40 / 100), bands within 0.6 dB, sign assignment confirmed
      (flipping voices drops corr 0.85 -> 0.73).
- [x] **Level calibration** re-measured end to end after the level chain moved
      before the filter: a constant -2.5 dB folded into OUT_GAIN. Residual +/-1.5 dB
      that grows with resonance and drive at high level — Vital saturates INSIDE
      the filter loop (the resonant peak hits the nonlinearity), the twin only at
      the input. Second order; the level parameter absorbs it in recovery.
- [x] **Discrete frame search.** Sweeping frame inside a run was unreliable (1 in
      5 restarts); frame on a stepped table is a discrete choice. Now one restart
      starts at each keyframe (8 spread positions for smooth tables), best-of-N.
      Filter loop: env amount 0.501, decay 0.272, sustain 0.132, power -4.6 for
      0.5 / 0.25 / 0.15 / -5.54; export corr 0.986. ~7 min per recovery.
- [ ] **Blend under random phases**: accepted at +/-20% (only signal is beating
      depth). Exact with deterministic phases.
- [x] **Filter keytrack.** Measured: cutoff += kt (note - 60) semitones, kt = 2 raw - 1,
      exact to a semitone at 50 / 100 / -50 %. Modelled and exported.
- [x] **Velocity.** Measured: shape = 1 - vt (1 - vel/127), squared by the amp
      path (0.1 dB at every velocity incl. negative tracking). The twin had used
      vel/127 linearly — wrong at the preset default of 0%. Now a constructor
      arg (`velocity_track`), exported.
- [x] **Stepped tables** snap to the keyframe below when `interpolation == 0`.
- [x] **Unison frame spread** modelled (outer voices at frame +/- offset *
      spread; units are display frames -> table rows). Forward check only,
      env corr 0.74 vs Vital — the sign/assignment of spread to voices may
      differ; verify before relying on it.
- [x] **Filter onset resolution / modulation smoothing.** Measured: Vital does
      NOT smooth env->cutoff (modulated and static traces identical from the
      first ms). Two frame sizes crossfaded over the first 100 ms, 1024 for the
      onset; worst band error ~4 dB. Nothing further to model here.
- [x] **Resonance vs cutoff.** Re-measured at 250-2000 Hz: cutoff-INDEPENDENT
      (rows agree to 0.1 dB / few % Q). The earlier "drift" was the 119 Hz row,
      where a 33 Hz harmonic grid leaves one harmonic in the passband — a
      measurement artefact. Table refreshed (`res_law_2d.json` has the sweep).
- [ ] **Filter degeneracies to document in the tool**: base cutoff <-> filter
      sustain, decay time <-> decay power, and (if bipolar) negative amount +
      attack <-> positive amount + decay. All produce the same trajectory; the
      twin picks one. `bipolar_fenv=False` by default for that reason.
- [ ] **Velocity tracking.** Twin scales amplitude by vel/127 linearly; Vital's
      `velocity_track` curve is unmeasured. Small, but it is in the loop.
- [x] **Longer test clip.** `synth.sustained_notes()` (1.5 s notes, 0.5 s gaps).
      Release and blend are unidentifiable on the 8-notes/s kalimba figure and
      recover on this one. Use `closed_loop.py 3 multi+mod rand sustained`.
- [ ] **Move the twin out of `experiments/`** once the above land: a
      `fantasia_core/synth_twin/` package with tests, and an agent tool that
      takes a reference clip + track and returns a Vital patch as
      `set_plugin_param` commands.

## Reference-guided FX matching (GRAFX)

- [ ] **`match_reference` on the stock EQ.** Tune the 8-band EQ to a reference
      clip via MR-STFT, land as `set_eq_band` commands. Synthetic recovery test
      first (apply a known curve, recover it).
- [ ] **Closed-form FDN reverb** as a GRAFX node — cheaper and more musical than
      the 60k-tap noise-shaped IR (which is 80% of the chain's step time).
- [ ] **Decide gain-staging default** from the listening tests: `LAM_GS=0`
      builds EQ tilts and drops the reverb; `0.05` keeps EQ energy-neutral and
      uses reverb at the mix knee.

## Text-guided FX (CLAP) — deprioritised, see memory notes

- [ ] Report the five-prompt guitar test (underwater / woody / church bells /
      african dance on fire / folkish) — running as of this commit.
- [ ] If any prompt family works: text supplies *shape*, expose an *amount*
      knob; never auto-apply. If none do: Audealize lookup for its 294 words,
      CLAP only to *find* references.
- [ ] Guard: `level_match` floors the normaliser so silence cannot score 0.27.
      Audit the other losses for the same hole.

## Infrastructure

- [ ] **Tighten the memory cap** for every MPS script (`set_per_process_memory_fraction`,
      currently 0.5 of torch's working set). Two OOM kills leaked ~4 GB of GPU
      memory that only a reboot freed.
- [ ] **Full-length gradient check** (CPU vs MPS cosine at 480k samples) as a
      test for any new differentiable node — the reflect-pad bug passed every
      short-signal check.
- [ ] Pin the GRAFX install recipe in `pyproject.toml` (torchlpc==0.6, then
      `--no-deps` for grafx/torchcomp/torch-geometric/xxhash).
- [ ] Agent panel: a "New conversation" control (`ClaudeCodeSession.reset()`
      exists, no UI).
- [ ] Playback still skips occasionally — revisit the render pool / gate
      (user noted "might go back to efficiency at some point").
