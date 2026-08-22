#!/usr/bin/env bash
# Quick audio diagnostics for Linux/WSL (safe no-op path on macOS).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "uname: $(uname -a)"
echo "PULSE_SERVER=${PULSE_SERVER:-unset}"
echo "XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-unset}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Not Linux — use the system audio device picker in the app."
  exit 0
fi

DEPS="${FANTASIA_LOCAL_DEPS:-$HOME/.local/lib/fantasia-deps/root}"
if [[ -d "$DEPS/usr/lib/x86_64-linux-gnu" ]]; then
  export LD_LIBRARY_PATH="$DEPS/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export ALSA_PLUGIN_DIR="${ALSA_PLUGIN_DIR:-$DEPS/usr/lib/x86_64-linux-gnu/alsa-lib}"
  export PATH="$DEPS/usr/bin${PATH:+:$PATH}"
  echo "using local deps: $DEPS"
fi
[[ -S /mnt/wslg/PulseServer ]] && export PULSE_SERVER="${PULSE_SERVER:-unix:/mnt/wslg/PulseServer}"

echo "--- Pulse ---"
if command -v pactl >/dev/null 2>&1; then
  if pactl info 2>&1 | head -20; then
    echo "Pulse: OK"
    pactl list short sinks 2>&1 | head -10 || true
  else
    echo "Pulse: FAILED (connection refused/timeout is common when WSLg audio wedges)"
    echo "Fix: in Windows PowerShell run  wsl --shutdown  then reopen WSL."
  fi
else
  echo "pactl not found (optional: sudo apt install pulseaudio-utils)"
fi

echo "--- PortAudio / sounddevice ---"
if [[ -x .venv/bin/python ]]; then
  .venv/bin/python - <<'PY'
import sounddevice as sd
print("hostapis:", sd.query_hostapis())
print("devices:", sd.query_devices())
print("default:", sd.default.device)
outs = [i for i, d in enumerate(sd.query_devices()) if d.get("max_output_channels", 0) > 0]
print("output indexes:", outs)
PY
else
  echo "No .venv — skip sounddevice probe"
fi
