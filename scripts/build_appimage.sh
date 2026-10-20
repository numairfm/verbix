#!/usr/bin/env bash
# Builds the Verbix AppImage: Python runtime, wheel, legal files, then Tauri.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESOURCES="$ROOT/src-tauri/resources"
VOICE="$RESOURCES/voice"

rm -rf "$VOICE"
mkdir -p "$VOICE"
rm -rf "$RESOURCES/licenses"
mkdir -p "$RESOURCES/licenses"

echo "==> verifying bundled TTS assets"
uv run scripts/fetch_voice_models.py --bundle-tts

echo "==> building the Verbix wheel"
rm -rf "$ROOT/dist"
uv build --wheel

echo "==> provisioning the relocatable Python runtime"
uv python install --install-dir "$VOICE" 3.13
python_dir="$(find "$VOICE" -mindepth 1 -maxdepth 1 -type d -name 'cpython-*' | head -n 1)"
if [[ -z "$python_dir" ]]; then
  echo "no CPython installation found under $VOICE" >&2
  exit 1
fi
mv "$python_dir" "$VOICE/python"
find "$VOICE" -maxdepth 1 -type l -name 'cpython-*' -delete
rm -rf "$VOICE/.temp" "$VOICE/.lock" "$VOICE/.gitignore"
"$VOICE/python/bin/python3" --version

echo "==> installing the wheel and dependencies"
uv pip install --python "$VOICE/python/bin/python3" --target "$VOICE/site-packages" "$ROOT"/dist/verbix-*.whl

echo "==> copying the downloader and legal files"
cp scripts/fetch_voice_models.py "$VOICE/fetch_voice_models.py"
cp THIRD-PARTY-NOTICES.md DISTRIBUTION-TERMS.md "$RESOURCES/licenses/"
mkdir -p "$RESOURCES/licenses/LICENSES"
cp -a LICENSES/. "$RESOURCES/licenses/LICENSES/"

echo "==> smoke-checking the bundled runtime"
PYTHONPATH="$VOICE/site-packages" "$VOICE/python/bin/python3" -c "import verbix.cli"

echo "==> building the AppImage"
cd "$ROOT/src-tauri"
export APPIMAGE_EXTRACT_AND_RUN=1
export NO_STRIP=true
cargo tauri build --bundles appimage

# Keep the resource paths present for cargo check on a clean checkout without
# bundling these placeholders into the AppImage.
touch "$VOICE/.gitkeep" "$RESOURCES/licenses/.gitkeep"
