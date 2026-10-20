#!/usr/bin/env bash
# Headless AppImage smoke test. Run on a Linux host with xvfb-run and pgrep.
#
# Checks:
#   1. --version prints the configured version and exits 0.
#   2. The GUI starts, the voice service is alive, and IPC connects.
#   3. Terminating the shell leaves no verbix.cli process behind.
#   4. If tauri-plugin-single-instance is present, a second launch forwards and
#      exits while the first shell keeps running. Otherwise the step is
#      reported as deferred to #8 and is not claimed as a pass.
#
# The smoke host has no sound card, so when ALSA has no hardware device the
# script points ALSA at a null PCM. That lets the voice service open its output
# stream and bring up IPC; it does not test audio rendering.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <AppImage> [timeout-seconds]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
TIMEOUT="${2:-45}"
VERSION="$(sed -n 's/.*"version": "\([^"]*\)".*/\1/p' "$ROOT/src-tauri/tauri.conf.json" | head -n 1)"

if [[ ! -f "$APP" ]]; then
  echo "AppImage not found: $APP" >&2
  exit 2
fi
if [[ -z "$VERSION" ]]; then
  echo "could not read the version from $ROOT/src-tauri/tauri.conf.json" >&2
  exit 2
fi
if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "xvfb-run is required" >&2
  exit 2
fi
if ! command -v pgrep >/dev/null 2>&1; then
  echo "pgrep is required" >&2
  exit 2
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/verbix-smoke.XXXXXX")"
REAL_HOME="$HOME"
SMOKE_HOME="$WORK/home"
mkdir -p "$SMOKE_HOME/.verbix/logs"
export HOME="$SMOKE_HOME"
STT_MODEL="${VERBIX_STT_MODEL:-$REAL_HOME/.verbix/voice/stt/models/nemotron-gguf/nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf}"
if [[ ! -f "$STT_MODEL" ]]; then
  echo "STT model not found: $STT_MODEL" >&2
  echo "run 'uv run scripts/fetch_voice_models.py' or set VERBIX_STT_MODEL" >&2
  exit 2
fi
export VERBIX_STT_MODEL="$STT_MODEL"
if [[ -z "${ALSA_CONFIG_PATH:-}" && ! -e /dev/snd/controlC0 ]]; then
  cat >"$WORK/asound.conf" <<'EOF'
pcm.!default { type null }
ctl.!default { type null }
EOF
  export ALSA_CONFIG_PATH="$WORK/asound.conf"
fi
RUNNER_PID=""
SHELL_PID=""
VOICE_PID=""

cleanup() {
  if [[ -n "$SHELL_PID" ]] && kill -0 "$SHELL_PID" 2>/dev/null; then
    kill -TERM "$SHELL_PID" 2>/dev/null || true
  fi
  if [[ -n "$VOICE_PID" ]] && kill -0 "$VOICE_PID" 2>/dev/null; then
    kill -TERM "$VOICE_PID" 2>/dev/null || true
  fi
  if [[ -n "$RUNNER_PID" ]] && kill -0 "$RUNNER_PID" 2>/dev/null; then
    if [[ "$(ps -o pgid= -p "$RUNNER_PID" | tr -d ' ')" == "$RUNNER_PID" ]]; then
      kill -TERM -- "-$RUNNER_PID" 2>/dev/null || true
    else
      kill -TERM "$RUNNER_PID" 2>/dev/null || true
    fi
  fi
  sleep 1
  if [[ -n "$VOICE_PID" ]] && kill -0 "$VOICE_PID" 2>/dev/null; then
    kill -KILL "$VOICE_PID" 2>/dev/null || true
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT INT TERM

fail() {
  echo "SMOKE FAILED: $*" >&2
  echo "--- app output ---" >&2
  cat "$WORK/app.log" 2>/dev/null >&2 || true
  echo "--- shell log ---" >&2
  cat "$HOME/.verbix/logs/shell.log" 2>/dev/null >&2 || true
  echo "--- voice log ---" >&2
  cat "$HOME/.verbix/logs/voice-service.log" 2>/dev/null >&2 || true
  exit 1
}

wait_for() {
  local description="$1"
  local seconds="$2"
  shift 2
  local deadline=$((SECONDS + seconds))
  while ((SECONDS < deadline)); do
    if "$@"; then
      return 0
    fi
    sleep 0.25
  done
  echo "timed out waiting for $description" >&2
  return 1
}

# pgrep matches zombie processes, and a container init may leave one behind
# from an earlier run. Only live processes count for start and shutdown checks.
live_pids() {
  local pid state
  for pid in $(pgrep -x "$1"); do
    state="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
    [[ "$state" == Z* ]] && continue
    echo "$pid"
  done
}

live_matches() {
  local pid state
  for pid in $(pgrep -f "$1"); do
    state="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
    [[ "$state" == Z* ]] && continue
    echo "$pid"
  done
}

shell_running() {
  [[ -n "$(live_pids verbix-shell)" ]]
}

voice_running() {
  [[ -n "$(live_matches 'verbix.cli --serve --voice')" ]]
}

echo "==> version check"
version_output="$(APPIMAGE_EXTRACT_AND_RUN=1 xvfb-run -a "$APP" --version 2>"$WORK/version.err")" || {
  cat "$WORK/version.err" >&2
  fail "--version exited non-zero"
}
if [[ "$version_output" != "Verbix $VERSION" ]]; then
  fail "--version printed '$version_output', expected 'Verbix $VERSION'"
fi
echo "    $version_output"

echo "==> launching the GUI headless"
APPIMAGE_EXTRACT_AND_RUN=1 setsid xvfb-run -a "$APP" --show >"$WORK/app.log" 2>&1 &
RUNNER_PID=$!

wait_for "verbix-shell" "$TIMEOUT" shell_running || fail "shell process did not start"
SHELL_PID="$(live_pids verbix-shell | head -n 1)"
echo "    shell pid $SHELL_PID"

wait_for "voice service" "$TIMEOUT" voice_running ||
  fail "voice service did not start"
VOICE_PID="$(live_matches 'verbix.cli --serve --voice' | head -n 1)"
echo "    voice service pid $VOICE_PID"

wait_for "IPC connection" "$TIMEOUT" grep -q 'connected' "$HOME/.verbix/logs/shell.log" 2>/dev/null ||
  fail "shell did not report an IPC connection"
echo "    IPC connected"

if grep -q 'tauri-plugin-single-instance' "$ROOT/src-tauri/Cargo.toml"; then
  echo "==> single-instance forwarding"
  before_count="$(live_pids verbix-shell | wc -l)"
  if timeout 30 env APPIMAGE_EXTRACT_AND_RUN=1 xvfb-run -a "$APP" --show >"$WORK/second.log" 2>&1; then
    after_count="$(live_pids verbix-shell | wc -l)"
    if [[ "$before_count" -ne 1 || "$after_count" -ne 1 ]]; then
      fail "second launch did not forward (shell count $before_count -> $after_count)"
    fi
    echo "    second launch forwarded and exited; one shell remains"
  else
    fail "second launch did not exit with forwarding"
  fi
else
  echo "==> single-instance forwarding: DEFERRED to #8"
  echo "    tauri-plugin-single-instance is not in src-tauri/Cargo.toml."
  echo "    Forwarding was not tested and is not claimed as passing."
fi

echo "==> quitting the shell"
kill -TERM "$SHELL_PID"
for _ in $(seq 1 40); do
  if ! kill -0 "$SHELL_PID" 2>/dev/null; then
    break
  fi
  sleep 0.25
done
if kill -0 "$SHELL_PID" 2>/dev/null; then
  kill -KILL "$SHELL_PID" 2>/dev/null || true
fi

for _ in $(seq 1 40); do
  if [[ -z "$(live_matches 'verbix.cli --serve --voice')" ]]; then
    break
  fi
  sleep 0.25
done
if [[ -n "$(live_matches 'verbix.cli --serve --voice')" ]]; then
  live_matches 'verbix.cli --serve --voice' >&2 || true
  fail "voice service survived the shell"
fi
VOICE_PID=""
echo "    shell stopped, no voice service process remains"

echo "smoke test PASSED"
