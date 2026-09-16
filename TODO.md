# TODO

## Roadmap — toward `tune_toward(track, target, amount, scope)`

The feature: "make this sound more like X" for a human or the agent, given any FX
graph (and the Vital twin if the track is a Vital track). Pages: *The Vital Twin*
(writeup, with sounds) and *Vital Twin Roadmap* (this list, expanded).

**Critical path:** noise source → compressor → graph compiler → directional loss.
The first two widen what a prompt can reach; the second two make it a tool.

### Effects the chain is missing (GRAFX has most; some are ours to write)
- [ ] **Compressor** — dynamics is the envelope dimension prompts act on most
      (*punchy, glued, smooth*). GRAFX `Compressor` on torchcomp (installed).
      Compress-then-normalise raises quiet parts: expect a new cheat, watch it.
- [ ] **Delay** — GRAFX `MultitapDelay`; tempo-sync taps to the clip BPM.
- [ ] **Chorus / flanger / phaser** — write an LFO-modulated delay line; the
      same node doubles as the synth's LFO.
- [ ] **Stereo** — the twin is mono; a mono judge cannot hear *wide*. Needs a
      stereo chain and a stereo-aware loss (GRAFX has mid/side tools).
- [ ] **Saturation flavours** — Chebyshev / piecewise tanh (GRAFX); measure the
      Vital distortion types presets use.
- [ ] **Transient shaper** — nothing exists; small to write (fast/slow env pair).
- [ ] **Gate** — GRAFX `NoiseGate`; low priority.
- [ ] **Graph compiler** — app FX DAG → differentiable graph, one twin per app
      node TYPE (stock EQ, inserts…), frozen pass-through where none exists.
      This is what makes the feature work on the user's actual graphs.

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
1. [ ] **Noise / sample source** — the mallet click, the breath, the pluck.
       Every imitation that fell short fell short here. Highest value per line.
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
- [ ] **Prompt → parameter sensitivity map** (~40 prompts): which params move,
      which direction; turns "resonance is an attractor" into a table; warm starts.
- [ ] **Directional loss, revisited** — "more like X" from the current sound;
      no-op on a fixed EQ, natural with a synth + starting patch.
- [ ] **Amount as a knob** — candidates at 25/50/100% of the parameter delta,
      ear chooses (the paper's best numbers relied on exactly this).
- [ ] **Transfer across phrases** — does a patch tuned on the hook survive a
      different melody? If not, the feature doesn't either.
- [ ] **Reference + text together** — magnitude from the clip, direction from words.
- [ ] **Listening protocol** — ten pairs, forced choice, written down.
- [ ] **Reranking judge** — CLAP for gradients; an audio-LM or Audealize's
      word→EQ data (magnitude prior for adjectives) to pick finalists.
- [ ] **Speed** — batch restarts, cache text embeddings, 5 s excerpts.

### The optimiser (today: Adam + reparameterisation + best-of-N restarts)
1. [ ] **Amortised inference** — train mel→parameters on twin-rendered random
       patches (free, exact labels); use as the descent start. Minutes → seconds.
       Refs: Masuda & Saito DDSP sound matching, InverSynth, DDSP autoencoders.
2. [ ] **Batch the restarts** — 8 fit in one pass at 0.7 GB/chain.
3. [ ] **Average 2–4 phase draws per step** — smooths the stochastic loss.
4. [ ] **Coarse-to-fine** — 3 s excerpts first; envelope/coarse terms first,
       fine STFT terms later (frame ridges live in the fine terms).
5. [ ] **Outer loop for discrete axes** — stepped frame, voices, filter model,
       table choice: grid / CMA-ES around the gradient inner loop.
6. [ ] **L-BFGS finish** with phases fixed.
7. [ ] **Return the score-vs-distance frontier**, not a single point.

### The judge
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
