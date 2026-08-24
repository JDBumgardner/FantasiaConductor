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
# One-time toolchain — pick your OS:

# macOS / Homebrew:
brew install python@3.12 portaudio fluid-synth

# Linux / WSL (Debian/Ubuntu):
sudo apt install python3.12 python3.12-venv portaudio19-dev \
  libxcb-cursor0 libasound2-plugins pulseaudio-utils \
  libfluidsynth-dev fluid-soundfont-gm
# Fedora:
#   sudo dnf install python3-devel portaudio-devel fluidsynth-libs
# On WSL with WSLg audio, point ALSA at Pulse once:
#   printf 'pcm.!default { type pulse }\nctl.!default { type pulse }\n' > ~/.asoundrc
# If the app says "No audio output device available" on WSL, Pulse is often
# wedged — from **Windows** PowerShell run `wsl --shutdown`, reopen WSL, then
# `./tools/run_app.sh`. Diagnose with `./tools/check_audio.sh`.

# Environment (uv keeps the lockfile + venv in sync):
uv sync

# Always launch through the wrapper — Fedora/WSL library paths + Pulse.
# Bare `uv run python app.py` skips those paths.
./tools/run_app.sh
```

## Full setup

The quick start deliberately skips the heavy optional stacks. Everything below
is opt-in — install only the extras you need.

### 1. System libraries

```bash
# macOS:
brew install fluid-synth    # MIDI synthesis  (extra: midi)
brew install rubberband     # time-stretch    (extra: stretch)

# Linux / WSL (Debian/Ubuntu):
sudo apt install libfluidsynth-dev fluid-soundfont-gm   # MIDI (extra: midi)
sudo apt install librubberband-dev rubberband-cli       # stretch (extra: stretch)
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
4. an OS-provided font (`/usr/share/sounds/sf2/` on Debian/Ubuntu,
   `/usr/share/soundfonts/` on Arch/Fedora)

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

## Project files

Projects save as `.fcp` (JSON — see `fantasia_core/document/serialize.py`).
Saves are gitignored, including the `tracks_megadope/` working directory:
they're personal work rather than part of the codebase, and audio clips inside
them store **absolute** `source_path`s into `.fantasia_cache/`, so they don't
travel between machines anyway.

To hear the app without building a project first, use **File ▸ Load Demo
Arrangement** (run `tools/make_demo_audio.py` once to generate the samples).

## Driving the DAW from an agent (MCP)

Claude Code or Cursor can compose directly in the app — add tracks, write MIDI,
design synth patches, search sounds, render vocals — by calling the same tools
the in-app agent uses. Every call goes through the `CommandBus`, so **everything
the agent does is undoable** with ⌘Z / Ctrl+Z like any manual edit.

This path uses your Claude Code or Cursor subscription — no separate
`ANTHROPIC_API_KEY` (that key is only for the optional in-app Agent panel).

### How it connects

```
Claude Code / Cursor  →  tools/mcp_server.py  →  HTTP bridge :8765  →  running app
                    (.mcp.json / .cursor/mcp.json)   (fantasia_core/bridge.py)
```

The MCP server is a thin forwarder. It holds no tool definitions of its own —
it fetches them live from the running app, so the MCP tool list always matches
the in-app agent's. **The app must be running**; the bridge starts
automatically with it.

### Setup

```bash
pip install -e '.[bridge]'
```

`.mcp.json` (Claude Code) uses repo-relative paths; `.cursor/mcp.json` (Cursor)
uses `${workspaceFolder}` so Cursor can spawn the server reliably. Both expect
a `.venv` at the repo root with the `bridge` extra installed
(`uv pip install -e '.[bridge]'` or `pip install -e '.[bridge]'` **inside** the
venv — not with system `pip`).

### Use it

```bash
# 1. Start the DAW — the control bridge comes up with it on 127.0.0.1:8765
.venv/bin/python app.py   # or: tools/run_app.sh
```

Then, in a second session from this same directory:

- **Claude Code:** run `claude`, approve the `fantasia` server on first run.
- **Cursor:** open this repo, enable the `fantasia` MCP server if prompted, and
  ask the agent to compose — it will call the same tools over the bridge.

Ask for music:

> *Make a house track at 125 BPM in D major — four-on-the-floor kick, offbeat
> bass, and a Clair de Lune melody on celesta over it.*

Keep the app visible while it works: tracks, clips, and notes appear live on
the timeline as the tools fire.

### Notes

- **Port conflicts** — the bridge binds `127.0.0.1:8765`. A second instance of
  the app won't own the port (`ControlBridge.start()` returns `False`).
  Override with `FANTASIA_BRIDGE_URL` on both sides if needed.
- **Slow tools** — generation and stem separation run on CPU and can take
  minutes; the client timeout is 600s.
- **Localhost only** — the bridge binds the loopback interface and has no
  authentication. Don't expose it beyond your machine.

**"Fantasia Conductor doesn't appear to be running"** means the app isn't up,
or the bridge lost the port. Start `app.py` first, then retry.

## Tests

```bash
pip install -e '.[dev]'
pytest
```

## Status

Building the MVP milestone by milestone — see the plan for M0–M6. Symbolic/MIDI
tracks, stem separation, AI audio generation, and the TS frontend are Phase 2.
