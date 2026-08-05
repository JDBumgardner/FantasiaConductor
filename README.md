# Fantasia Conductor

A native desktop DAW (Python + PySide6) for defining and composing music, with
first-class AI **agent hooks** into every editing tool and a **dual vector-DB
sound search**.

## Architecture

- **`fantasia_core/`** — headless, UI-agnostic core (no Qt imports).
  - `document/` — the source of truth: `Project` / `Track` / `Clip`, serialization.
  - `commands/` — every edit is a reversible `Command` on a `CommandBus`
    (this is also exactly what the agent calls → undo/redo for free).
  - `engine/` — audio buffers, mixer, `sounddevice` playback, FX, bounce.
  - `search/` — CLAP embeddings + LanceDB stores + one search service.
  - `agent/` — Claude tool-calling over the CommandBus + search.
- **`ui/`** — PySide6 views; depends on `fantasia_core`, never the reverse.
- **`app.py`** — entry point.

Keeping the core Qt-free is deliberate: a TypeScript/localhost frontend can later
bind to the same core over a local API without a rewrite.

## Quick start

Gets the app running — UI, document model, audio engine, editing.

```bash
# One-time toolchain (macOS / Homebrew):
brew install python@3.12 portaudio

# Environment:
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# Run:
python app.py
```

## Full setup

The quick start deliberately skips the heavy optional stacks. Everything below
is opt-in — install only the extras you need.

### 1. System libraries

```bash
brew install fluid-synth    # MIDI synthesis  (extra: midi)
brew install rubberband     # time-stretch    (extra: stretch)
```

### 2. Python extras

| Extra | Install | Gives you |
| --- | --- | --- |
| `midi` | `pip install -e '.[midi]'` | Soundfont MIDI playback |
| `search` | `pip install -e '.[search]'` | Semantic sound search (CLAP + LanceDB) |
| `generate` | `pip install -e '.[generate]'` | Text→audio via MusicGen |
| `separate` | `pip install -e '.[separate]'` | Stem separation via Demucs |
| `transcribe` | `pip install -e '.[transcribe]'` | Audio→MIDI via basic-pitch |
| `voice` | `pip install -e '.[voice]'` | TTS / singing via Kokoro — **Apple Silicon only** |
| `stretch` | `pip install -e '.[stretch]'` | Time-stretch without pitch change |
| `agent` | `pip install -e '.[agent]'` | Claude tool-calling |
| `bridge` | `pip install -e '.[bridge]'` | MCP server (drive the app from Claude Code) |
| `dev` | `pip install -e '.[dev]'` | pytest |

Combine them: `pip install -e '.[midi,search,bridge]'`

### 3. A soundfont (required for MIDI audio)

Soundfonts are **not** in the repo — they're large and separately licensed.
Without one, MIDI tracks render silent. Install any General MIDI `.sf2`:

```bash
mkdir -p assets/soundfonts
# e.g. GeneralUser GS — https://schristiancollins.com/generaluser.php
curl -L -o assets/soundfonts/GeneralUser-GS.sf2 <url-to-a-gm-sf2>
```

`default_soundfont()` resolves in this order, so any of these works:

1. `$FANTASIA_SOUNDFONT` (explicit path — wins if set)
2. repo-local `assets/soundfonts/*.sf2`
3. a Homebrew-installed font
4. an OS-provided font

### 4. Build the sound library (required for search)

The generated sample library and its vector index live in `.fantasia_cache/`,
which is gitignored — a fresh clone starts empty and `find_sound` returns
nothing. Rebuild it locally (needs `midi` + a soundfont; MusicGen textures need
`generate`):

```bash
.venv/bin/python tools/build_sample_library.py
.venv/bin/python tools/build_sample_library.py --no-musicgen   # soundfont only, fast
```

### 5. API key (only for the `agent` extra)

```bash
mkdir -p .fantasia_cache
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .fantasia_cache/secrets.env
```

### 6. Model weights

Downloaded automatically on first use into the library caches
(`~/.cache/huggingface`), **not** into this repo:

- MusicGen — ~2 GB, on first generation
- Demucs `htdemucs` — ~300 MB, on first separation
- basic-pitch — bundled with the pip package, no download

Expect a slow first run of each feature.

## Example projects

`tracks_megadope/` holds a few `.fcp` saves. They open fine, but any **audio**
clip inside them stores an absolute `source_path` into `.fantasia_cache/` —
which is gitignored, so those clips resolve to nothing on a fresh clone. MIDI
tracks, synth patches, FX chains, and the arrangement all load normally. Treat
them as structural references rather than playable demos.

## MCP server

`.mcp.json` registers the `fantasia` server with paths relative to the repo
root, so it works from any clone — no edits needed. Requires the `bridge`
extra and a `.venv` at the repo root. Launch Claude Code from this directory
and the DAW tools become available.

## Tests

```bash
pip install -e '.[dev]'
pytest
```

## Status

Building the MVP milestone by milestone — see the plan for M0–M6. Symbolic/MIDI
tracks, stem separation, AI audio generation, and the TS frontend are Phase 2.
