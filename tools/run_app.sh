#!/usr/bin/env bash
# Launch Fantasia Conductor with the repo venv.
#
# On Linux/WSL, wires up PortAudio + ALSA→Pulse when system packages aren't
# installed yet (optional user-local tree under ~/.local/lib/fantasia-deps).
# On macOS this is a thin wrapper around ``.venv/bin/python app.py``.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "No .venv yet. Create one first (see README Quick start)." >&2
  exit 1
fi

# ---- Linux / WSL audio environment --------------------------------------
if [[ "$(uname -s)" == "Linux" ]]; then
  DEPS="${FANTASIA_LOCAL_DEPS:-$HOME/.local/lib/fantasia-deps/root}"
  if [[ -d "$DEPS/usr/lib/x86_64-linux-gnu" ]]; then
    export LD_LIBRARY_PATH="$DEPS/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if [[ -d "$DEPS/usr/lib/x86_64-linux-gnu/alsa-lib" ]]; then
      export ALSA_PLUGIN_DIR="${ALSA_PLUGIN_DIR:-$DEPS/usr/lib/x86_64-linux-gnu/alsa-lib}"
    fi
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
