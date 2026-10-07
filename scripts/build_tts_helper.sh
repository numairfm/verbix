#!/usr/bin/env bash
# Builds the PocketTTS.cpp helper from a pinned source commit into a runtime
# dir layout (bin/, lib/, models/, voices/) and fetches the pinned assets.
# Usage: build_tts_helper.sh <source-commit-or-tag> <out-dir>
# Dev fast path for local builds; release bundles come from CI instead.
set -euo pipefail

COMMIT="${1:?usage: $0 <source-commit> <out-dir>}"
OUT="${2:?usage: $0 <source-commit> <out-dir>}"
SRC="$(mktemp -d)/PocketTTS.cpp"

git clone --quiet https://github.com/numairfm/PocketTTS.cpp "$SRC"
git -C "$SRC" checkout --quiet "$COMMIT"
cmake -S "$SRC" -B "$SRC/.build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$SRC/.build" -j"$(nproc)" --target pocket-tts

mkdir -p "$OUT/bin" "$OUT/lib" "$OUT/models" "$OUT/voices"
cp "$SRC/.build/pocket-tts" "$OUT/bin/verbix-tts"
cp "$SRC/.build"/libonnxruntime.so* "$OUT/lib/" 2>/dev/null || true
rm -rf "$SRC"
echo "helper built at $OUT/bin/pocket-tts"
echo "next: fetch models + voices into $OUT (see tts_assets PINNED_FILES)"
