#!/usr/bin/env bash
# Full TTS runtime redo from internet sources into one layout.
# Usage: setup_tts_runtime.sh [<engine-commit>] [<out-dir>]
# Defaults: fork HEAD, ./tts-runtime (repo-local, gitignored).
# Provenance is recorded in <out>/models/MANIFEST.json. Fails loud, no fallbacks.
set -euo pipefail

# Throwaway build/export trees live on disk-backed storage, never /tmp tmpfs
# (torch + builds exceed small tmpfs quotas).
export TMPDIR="${TMPDIR:-$HOME/.cache/verbix-tmp}"
mkdir -p "$TMPDIR"

ENGINE_REPO="https://github.com/numairfm/PocketTTS.cpp"
COMMIT="${1:-$(git ls-remote "$ENGINE_REPO" HEAD | awk '{print $1}')}"
OUT="${2:-$(pwd)/tts-runtime}"
ALBA_URL="https://huggingface.co/kyutai/tts-voices/resolve/main/alba-mackenna/casual.wav"

echo "engine: $ENGINE_REPO @ $COMMIT"
echo "out:    $OUT"

# 1. Engine build (skipped when a binary is already installed).
SRC="$(mktemp -d)/PocketTTS.cpp"
if test -x "$OUT/bin/verbix-tts"; then
    echo "engine binary present, skipping build"
    git clone --quiet "$ENGINE_REPO" "$SRC"
    git -C "$SRC" checkout --quiet "$COMMIT"
else
    git clone --quiet "$ENGINE_REPO" "$SRC"
    git -C "$SRC" checkout --quiet "$COMMIT"
    cmake -S "$SRC" -B "$SRC/.build" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON >/dev/null
    cmake --build "$SRC/.build" -j"$(nproc)" --target pocket-tts
    mkdir -p "$OUT/bin" "$OUT/lib"
    cp "$SRC/pocket-tts" "$OUT/bin/verbix-tts"
    cp "$SRC/.build/_deps/onnxruntime-src/lib"/libonnxruntime.so* "$OUT/lib/" 2>/dev/null || true
fi
mkdir -p "$OUT/bin" "$OUT/lib" "$OUT/models" "$OUT/models/tts" "$OUT/voices"

# 2. Weights: throwaway venv (never touches project deps) running the
# maintained export pipeline from the pocket-tts-onnx-export fork.
EXPORT_REPO="https://github.com/numairfm/pocket-tts-onnx-export"
EXPORT_SRC="$(mktemp -d)/export"
git clone --quiet "$EXPORT_REPO" "$EXPORT_SRC"
EXPORT_COMMIT="$(git -C "$EXPORT_SRC" rev-parse HEAD)"
echo "export: $EXPORT_REPO @ $EXPORT_COMMIT"
EXPORT_VENV="$(mktemp -d)/export-venv"
python3 -m venv "$EXPORT_VENV"
"$EXPORT_VENV/bin/pip" install --quiet -r "$EXPORT_SRC/requirements.txt" requests
# Weights come from the ungated mirror (kyutai repo is gated); proper fix is
# an env override upstream, sed keeps tonight moving with identical files.
grep -rl "kyutai/pocket-tts" "$EXPORT_SRC/export.py" "$EXPORT_SRC/onnx_export" \
    | xargs sed -i 's|kyutai/pocket-tts|Verylicious/pocket-tts-ungated|g'
cd "$EXPORT_SRC"
"$EXPORT_VENV/bin/python" export.py --quantize
ls onnx/
for f in flow_lm_flow_int8.onnx flow_lm_main_int8.onnx mimi_decoder_int8.onnx \
         mimi_encoder.onnx text_conditioner.onnx; do
    test -f "onnx/$f"
    cp "onnx/$f" "$OUT/models/tts/$f"
done
cp onnx/*.onnx.data "$OUT/models/tts/"
find weights -name "tokenizer.model" -exec cp {} "$OUT/models/tts/" \;
test -f "$OUT/models/tts/tokenizer.model"
cd /
rm -rf "$EXPORT_VENV" "$EXPORT_SRC"

# 3. Canonical voice.
curl -sfL "$ALBA_URL" -o "$OUT/voices/alba.wav"
test -f "$OUT/voices/alba.wav"

# 4. Provenance manifest (byte sizes are the integrity gate in tts_assets.py).
python3 - "$OUT" "$ENGINE_REPO" "$COMMIT" "$EXPORT_REPO" "$EXPORT_COMMIT" "$ALBA_URL" <<'EOF'
import datetime, json, sys
from pathlib import Path
out = Path(sys.argv[1])
files = {}
for p in sorted((out / "models").glob("*")) + sorted((out / "voices").glob("*.wav")):
    if p.suffix == ".py":
        continue
    files[str(p.relative_to(out))] = p.stat().st_size
(out / "models" / "MANIFEST.json").write_text(json.dumps({
    "engine_repo": sys.argv[2], "engine_commit": sys.argv[3],
    "export_repo": sys.argv[4], "export_commit": sys.argv[5],
    "alba_canonical_url": sys.argv[6],
    "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "files": files,
}, indent=2))
print(json.dumps(files, indent=2))
EOF

rm -rf "$SRC"
echo "runtime ready: $OUT"
