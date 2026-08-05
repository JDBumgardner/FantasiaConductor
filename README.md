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

## Development

```bash
# One-time toolchain (macOS / Homebrew):
brew install python@3.12 portaudio

# Environment:
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .            # core (M0–M4)
# pip install -e '.[search]'  # add M5 sound search (CLAP + LanceDB, heavy)
# pip install -e '.[agent]'   # add M6 agent (Anthropic SDK)

# Run:
python app.py
```

## Status

Building the MVP milestone by milestone — see the plan for M0–M6. Symbolic/MIDI
tracks, stem separation, AI audio generation, and the TS frontend are Phase 2.
