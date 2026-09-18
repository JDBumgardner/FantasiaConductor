# words — single-word prompt grids

Layout (results folders hold one `<instrument>__<slug>_<chain>.json` + `_dry/_fx/_fx_amt50.wav` per cell):

- `instruments/` — the four GM soundfont references (`*_ref.wav`), the twin's recovered patches (`*_patch.json`, `*_clone.wav`) and real Vital playing them (`*_vital.wav`)
- `plain/` — "this sound is ⟨word⟩", five-node chain (the first grid)
- `named/` — "this sound is a ⟨word⟩ ⟨instrument⟩"
- `contrast/` — cos(named) − cos("a ⟨instrument⟩")
- `balance/{A..F,W}/` — the rebalancing A/B on six cells (see `balance_test.py`), `W` = retrieval warm start
- `noise/` — cello and flute re-recovered with the noise source
- `source/` — text → FX on the soundfont recording itself, no twin
- `site/` — the "Ten Words, Four Instruments" page (`build_page.py`; `site/mp3` is generated)
- `logs/`, `scripts/` — run logs and the shell drivers

Scripts: `word_grid.py` (a grid; `T2_NAMED=1`, `T2_OBJ=contrast`), `summarize.py` (`RESULTS.md`), `cross_score.py`
(every result on every sentence), `who_did_the_work.py` (synth vs per-node contributions), `sensitivity.py`
(embedding displacement per Adam step per parameter), `balance_test.py` / `balance_summary.py`, `warm_test.py`,
`recover_noise.py`, `source_fx.py`, `build_page.py`. Instrument names, words and prompt wording live in `word_grid.py`.
