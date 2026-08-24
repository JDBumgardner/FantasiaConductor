#!/usr/bin/env bash
# The supported way to launch Fantasia Conductor (Fedora, Ubuntu/WSL, macOS).
#
# Do not use bare ``uv run python app.py`` as your daily launcher: ``uv run``
# syncs only core extras and used to strip pyfluidsynth, and it never sets
# Linux library paths (FluidSynth / PortAudio / Pulse).
#
# On Linux/WSL this also wires PortAudio + ALSA→Pulse and a user-local
# FluidSynth extract under ~/.local/lib/fantasia-deps. On macOS it is a thin
# wrapper around the venv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv sync
  else
    echo "No .venv yet. Create one first (see README Quick start)." >&2
    exit 1
  fi
fi

# ---- Linux / WSL audio environment --------------------------------------
if [[ "$(uname -s)" == "Linux" ]]; then
  DEPS="${FANTASIA_LOCAL_DEPS:-$HOME/.local/lib/fantasia-deps/root}"
  # Debian/WSL extract uses …/root/usr/lib/x86_64-linux-gnu; a Fedora RPM
  # extract may live at …/fantasia-deps/usr/lib64 (no "root" prefix).
  if [[ ! -d "$DEPS/usr/lib/x86_64-linux-gnu" && ! -d "$DEPS/usr/lib64" ]]; then
    if [[ -d "$HOME/.local/lib/fantasia-deps/usr/lib64" ]]; then
      DEPS="$HOME/.local/lib/fantasia-deps"
    fi
  fi
  for libdir in usr/lib/x86_64-linux-gnu usr/lib64 usr/lib; do
    if [[ -d "$DEPS/$libdir" ]]; then
      export LD_LIBRARY_PATH="$DEPS/$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
  done
  if [[ -d "$DEPS/usr/lib/x86_64-linux-gnu/alsa-lib" ]]; then
    export ALSA_PLUGIN_DIR="${ALSA_PLUGIN_DIR:-$DEPS/usr/lib/x86_64-linux-gnu/alsa-lib}"
  fi
  if [[ -d "$DEPS/usr/bin" ]]; then
    export PATH="$DEPS/usr/bin${PATH:+:$PATH}"
  fi

  # WSLg: route ALSA's "default" through the host Pulse server.
  if [[ -S /mnt/wslg/PulseServer ]]; then
    export PULSE_SERVER="${PULSE_SERVER:-unix:/mnt/wslg/PulseServer}"
    if [[ ! -f "$HOME/.asoundrc" ]]; then
      printf 'pcm.!default { type pulse }\nctl.!default { type pulse }\n' > "$HOME/.asoundrc"
      echo "Wrote $HOME/.asoundrc (ALSA → Pulse for WSLg)." >&2
    fi
    # Fail soft with a clear hint if Pulse is wedged (common after long WSL uptime).
    if command -v pactl >/dev/null 2>&1; then
      if ! pactl info >/dev/null 2>&1; then
        cat >&2 <<'EOF'
WARNING: WSLg PulseAudio is not accepting connections.
  PortAudio will report no output devices until Pulse is healthy.
  From Windows PowerShell (outside WSL), run:
    wsl --shutdown
  Then reopen this distro and launch again with:  ./tools/run_app.sh
EOF
      fi
    fi
  fi
fi

exec .venv/bin/python app.py "$@"
