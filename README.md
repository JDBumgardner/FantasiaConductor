# Fantasia Conductor

A native desktop DAW (Python + PySide6) for defining and composing music, with
first-class AI **agent hooks** into every editing tool and a **dual vector-DB
sound search**.

## Architecture

- **`fantasia_core/`** — headless, UI-agnostic core (no Qt imports).
  - `document/` — the source of truth: `Project` / `Track` / `Clip`, serialization.
  - `commands/` — every edit is a reversible `Command` on a `CommandBus`
    (this is also exactly what the agent calls → undo/redo for free).
  - `engine/` — audio buffers, mixer, `sounddevice` playback, FX graph, bounce,
    background clip rendering, and playback diagnostics.
  - `plugins.py` + `plugin_notes/` — VST3/AU hosting, and measured notes on how
    each hosted plugin's parameters actually behave.
  - `svs.py` — DiffSinger singing synthesis from OpenUtau voicebanks.
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
| `voice` | `pip install -e '.[voice]'` | TTS / speech via Kokoro — **Apple Silicon only** |
| `svs` | `pip install -e '.[svs]'` | Singing synthesis from DiffSinger voicebanks |
| `voiceconv` | `pip install -e '.[voiceconv]'` | Zero-shot voice conversion (Seed-VC) |
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

### 5. Agent backend — a key, or your Claude Code subscription

The in-app Agent panel can run either way:

| Backend | Billing | Needs |
| --- | --- | --- |
| Claude Code | your existing subscription | the CLI + `pip install claude-agent-sdk` |
| Anthropic API | per token, separately | an API key (below) |

Claude Code is used automatically when it is installed; otherwise the panel
falls back to the API key. Force the key with `FANTASIA_AGENT_BACKEND=api`.

```bash
npm i -g @anthropic-ai/claude-code
pip install claude-agent-sdk
```

The Claude Code backend does not declare its own tools — it reaches the DAW
through `tools/mcp_server.py`, the same server an outside MCP client uses, so
there is one definition of what the agent can do. That also means the app must
be running with its bridge up, and each request is a fresh session: it shares
your authentication, not the conversation of a Claude Code window you have open
elsewhere.

#### API key (only for the `agent` extra)

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

## Singing

`svs` renders sung vocals from a MIDI clip plus lyrics, using **DiffSinger
voicebanks** in OpenUtau format. Banks are not in the repo — import one and it
is unpacked into `.fantasia_cache/`:

```bash
pip install -e '.[svs]'
```

Then **Agent ▸ Singing Voicebanks…** (or `import_voicebank` from an agent),
pointing at a downloaded `.zip`. `list_voicebanks` shows what is installed.
Lyrics are one syllable per note, hyphenated across notes for multi-syllable
words (`fan-ta-si-a`).

`voice` (Kokoro) is a different thing — spoken TTS, not singing. `voiceconv`
converts an existing vocal into another voice.

## Instrument plugins (VST3 / AU)

Any installed VST3 or AU instrument can play a MIDI track — set a track's
`plugin` and its notes render through it. Hosting is `pedalboard`, which is
already a base dependency, so there is nothing extra to install.

Right-click a track header:

```
Plugin Instrument…            pick from what is installed
Open <plugin> Interface…      the plugin's own window
```

Plugins are found in the usual OS locations; add more with
`FANTASIA_PLUGIN_PATH`. `list_plugins` reports what was found.

**Each track keeps its own patch.** The patch lives on the track as
`plugin_state` and is saved with the project, while rendering runs through one
shared instance with that patch swapped in — so twelve plugin tracks cost one
instance, not twelve. A track whose editor is open gets its own instance for as
long as the window is up.

Clips render on background worker threads, nearest the playhead first. While
that is catching up a clip can be briefly silent; `playback_health` says which
clips are still waiting.

### Plugin notes

`fantasia_core/plugin_notes/` records what a plugin's parameters *actually*
accept, measured against the running plugin rather than assumed. This matters
more than it sounds: on Vital, sending `"On"` to a switch sets it Off and still
reports success, envelope times follow `32 × raw⁴` rather than anything linear,
and several values read back as the square of what you sent. Read the note for a
plugin before automating it, and read back what `set_plugin_param` echoes.

## FX and signal routing

Every channel — and the Master bus — carries a chain of **inserts**, each with a
stable id, so "this compressor" stays addressable across reorders.

By default a chain is serial (`in → fx[0] → … → out`). Tracks can also carry
explicit **wires** describing a directed graph, which the engine renders for
real: copy on a split, mix on a join. That is how you get parallel wet/dry —
the dry signal reaching the fader untouched alongside a reverb — rather than
running everything in line.

```
in ─┬─► reverb ──► out          a send, not an insert: the dry path is intact
    └────────────► out
```

Put a gain node at the head of each wet branch. A parallel branch summed at
full level is a second copy of the part, not an effect.

The node editor edits the graph by hand; `get_fx_routing` / `set_fx_routing`
do it from an agent. A stock 8-band EQ (`get_eq` / `set_eq_band`) is available
on every channel including Master, with a live analyzer.

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

M0–M6 are in: document model, undoable commands, audio engine, classical editing
tools, sound search, and agent hooks. Since then, MIDI tracks, stem separation,
audio generation, singing synthesis and plugin hosting have all landed too.

Still ahead:

- **Gapless transport.** Pressing play mid-song can briefly leave a clip silent
  while its audio is still rendering. `playback_health` measures it; the fix is
  to refuse to open the stream until the first window is cached.
- **Audio in its own process.** Not needed for stutter — the callback runs at
  ~12ms of its 186ms budget — but it is what would let the buffer drop from
  186ms to ~46ms, which matters for playing live into the app.
- **TypeScript frontend** over a local API onto `fantasia_core`.
