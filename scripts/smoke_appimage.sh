#!/usr/bin/env bash
# Headless AppImage smoke test. Run on a Linux host with xvfb-run, pgrep, and
# Mesa GLES (libGLESv2.so.2, provided by mesa-libGLES on Fedora).
#
# Checks:
#   1. --version prints the configured version and exits 0.
#   2. A fresh home opens the first-run setup and starts no voice service.
#   3. The GUI starts, the voice service is alive, and IPC connects.
#   4. Terminating the shell leaves no verbix.cli process behind.
#   5. A second launch forwards its arguments and exits while the first shell
#      keeps running, and an unknown argument exits 2 with the usage.
#
# The AppImage is copied to a path with spaces before it runs, so the launch
# path is exercised on every run.
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

# The single-instance plugin finds the running shell over the D-Bus session
# bus. A headless host has none, so run the whole smoke inside one.
if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ]]; then
  if ! command -v dbus-run-session >/dev/null 2>&1; then
    echo "dbus-run-session is required for the single-instance check" >&2
    exit 2
  fi
  SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  exec dbus-run-session -- "$SCRIPT" "$@"
fi

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
# Exercise the AppImage from a path containing spaces on every run, including
# CI where the artifact path itself has none.
APP_RUN="$WORK/Verbix AppImage with spaces.AppImage"
cp --reflink=auto "$APP" "$APP_RUN" 2>/dev/null || cp "$APP" "$APP_RUN"
REAL_HOME="$HOME"
SMOKE_HOME="$WORK/home"
mkdir -p "$SMOKE_HOME/.verbix/logs"
# The setup is complete for the service checks: the marker, a placeholder key,
# and the model path. The placeholder is never sent to a provider.
printf '{"setup_complete": true}\n' >"$SMOKE_HOME/.verbix/shell.json"
printf '{"openrouter":{"type":"api","key":"sk-smoke-placeholder"}}\n' \
  >"$SMOKE_HOME/.verbix/auth.json"
chmod 600 "$SMOKE_HOME/.verbix/shell.json" "$SMOKE_HOME/.verbix/auth.json"
export HOME="$SMOKE_HOME"
# Headless WebKit needs software rendering; compositing can hang panel creation
# on distributions whose WebKit build differs from the one used to build the
# AppImage.
export WEBKIT_DISABLE_COMPOSITING_MODE="${WEBKIT_DISABLE_COMPOSITING_MODE:-1}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export GDK_BACKEND="${GDK_BACKEND:-x11}"
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
version_output="$(APPIMAGE_EXTRACT_AND_RUN=1 xvfb-run -a "$APP_RUN" --version 2>"$WORK/version.err")" || {
  cat "$WORK/version.err" >&2
  fail "--version exited non-zero"
}
if [[ "$version_output" != "Verbix $VERSION" ]]; then
  fail "--version printed '$version_output', expected 'Verbix $VERSION'"
fi
echo "    $version_output"

echo "==> fresh home opens the setup and starts no service"
FRESH_HOME="$WORK/fresh-home"
mkdir -p "$FRESH_HOME"
HOME="$FRESH_HOME" APPIMAGE_EXTRACT_AND_RUN=1 setsid xvfb-run -a "$APP_RUN" --show \
  >"$WORK/fresh.log" 2>&1 &
FRESH_RUNNER=$!
if ! wait_for "fresh-home shell" "$TIMEOUT" shell_running; then
  fail "the shell did not start with a fresh home"
fi
sleep 5
if [[ -n "$(live_matches 'verbix.cli --serve --voice')" ]]; then
  fail "the voice service started with an incomplete setup"
fi
if ! grep -q 'setup required' "$FRESH_HOME/.verbix/logs/shell.log" 2>/dev/null; then
  cat "$FRESH_HOME/.verbix/logs/shell.log" >&2 || true
  fail "the fresh home did not report an incomplete setup"
fi
for pid in $(live_pids verbix-shell); do
  kill -TERM "$pid" 2>/dev/null || true
done
for _ in $(seq 1 40); do
  if [[ -z "$(live_pids verbix-shell)" ]]; then
    break
  fi
  sleep 0.25
done
for pid in $(live_pids verbix-shell); do
  kill -KILL "$pid" 2>/dev/null || true
done
if [[ -n "$FRESH_RUNNER" ]] && kill -0 "$FRESH_RUNNER" 2>/dev/null; then
  kill -TERM "$FRESH_RUNNER" 2>/dev/null || true
fi
echo "    fresh home: setup required, no voice service"

echo "==> launching the GUI headless"
APPIMAGE_EXTRACT_AND_RUN=1 setsid xvfb-run -a "$APP_RUN" --show >"$WORK/app.log" 2>&1 &
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

echo "==> bundled PortAudio check"
appdir="$(tr '\0' '\n' <"/proc/$SHELL_PID/environ" 2>/dev/null |
  sed -n 's/^APPDIR=//p' | head -n 1)"
if [[ -z "$appdir" || ! -f "$appdir/usr/lib/libportaudio.so.2" ]]; then
  fail "bundled PortAudio not found in the AppImage (APPDIR='$appdir')"
fi
portaudio_version="$(env -u LD_LIBRARY_PATH \
  LD_LIBRARY_PATH="$appdir/usr/lib" \
  PYTHONPATH="$appdir/usr/lib/Verbix/voice/site-packages" \
  VERBIX_PORTAUDIO_LIB="$appdir/usr/lib/libportaudio.so.2" \
  "$appdir/usr/lib/Verbix/voice/python/bin/python3" \
  -c 'from verbix.voice.portaudio import ensure_loadable; ensure_loadable(); import sounddevice; print(sounddevice.get_portaudio_version())' 2>&1)" ||
  fail "bundled PortAudio failed to load: $portaudio_version"
if [[ "$portaudio_version" != *"PortAudio"* ]]; then
  fail "unexpected PortAudio version output: $portaudio_version"
fi
echo "    $portaudio_version (bundled, no system package)"

if ! grep -q 'tauri-plugin-single-instance' "$ROOT/src-tauri/Cargo.toml"; then
  fail "tauri-plugin-single-instance is missing from src-tauri/Cargo.toml"
fi

second_launch() {
  timeout 30 env APPIMAGE_EXTRACT_AND_RUN=1 xvfb-run -a "$APP_RUN" "$@" \
    >"$WORK/second.log" 2>&1
}

echo "==> single-instance forwarding"
before_count="$(live_pids verbix-shell | wc -l)"
if [[ "$before_count" -ne 1 ]]; then
  fail "expected one shell before the second launch, found $before_count"
fi
if ! second_launch; then
  cat "$WORK/second.log" >&2
  fail "second launch without arguments did not exit 0"
fi
if [[ "$(live_pids verbix-shell | wc -l)" -ne 1 ]]; then
  fail "second launch without arguments started a second shell"
fi
echo "    a launch without arguments forwarded and exited; one shell remains"

if ! second_launch --hidden; then
  cat "$WORK/second.log" >&2
  fail "second launch with --hidden did not exit 0"
fi
if [[ "$(live_pids verbix-shell | wc -l)" -ne 1 ]]; then
  fail "second launch with --hidden started a second shell"
fi
echo "    --hidden forwarded and did nothing; one shell remains"

toggle_count() {
  local count
  count="$(grep -c 'toggle sent' "$HOME/.verbix/logs/shell.log" 2>/dev/null || true)"
  echo "${count:-0}"
}

toggle_before="$(toggle_count)"
if ! second_launch --toggle; then
  cat "$WORK/second.log" >&2
  fail "second launch with --toggle did not exit 0"
fi
forwarded_toggle=0
for _ in $(seq 1 80); do
  if [[ "$(toggle_count)" -gt "$toggle_before" ]]; then
    forwarded_toggle=1
    break
  fi
  sleep 0.25
done
if [[ "$forwarded_toggle" -ne 1 ]]; then
  fail "the forwarded --toggle did not reach the voice service"
fi
if [[ "$(live_pids verbix-shell | wc -l)" -ne 1 ]]; then
  fail "the forwarded --toggle started a second shell"
fi
if [[ "$(live_matches 'verbix.cli --serve --voice' | wc -l)" -ne 1 ]]; then
  fail "the forwarded --toggle started a second voice service"
fi
echo "    --toggle forwarded to the running shell; one shell and one voice service remain"

echo "==> argument handling before the GUI"
set +e
APPIMAGE_EXTRACT_AND_RUN=1 "$APP_RUN" --wat >"$WORK/invalid.out" 2>"$WORK/invalid.err"
invalid_status=$?
set -e
if [[ "$invalid_status" -ne 2 ]]; then
  fail "an unknown argument exited with $invalid_status, expected 2"
fi
if ! grep -q 'Usage: verbix' "$WORK/invalid.err"; then
  fail "an unknown argument did not print the usage on stderr"
fi
if [[ "$(live_pids verbix-shell | wc -l)" -ne 1 ]]; then
  fail "an unknown argument started a second shell"
fi
help_output="$(APPIMAGE_EXTRACT_AND_RUN=1 "$APP_RUN" --help 2>/dev/null)" || {
  fail "--help exited non-zero"
}
if [[ "$help_output" != *"Usage: verbix"* ]]; then
  fail "--help did not print the usage"
fi
echo "    unknown arguments exit 2 with usage; --help prints usage"

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
