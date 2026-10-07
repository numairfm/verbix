#!/usr/bin/env bash
# Full TTS runtime redo from internet sources into one layout.
# Usage: setup_tts_runtime.sh [<engine-commit>] [<out-dir>]
# Defaults: fork HEAD, ./tts-runtime (repo-local, gitignored).
# Provenance is recorded in <out>/models/MANIFEST.json. Fails loud, no fallbacks.
set -euo pipefail

ENGINE_REPO="https://github.com/numairfm/PocketTTS.cpp"
COMMIT="${1:-$(git ls-remote "$ENGINE_REPO" HEAD | awk '{print $1}')}"
OUT="${2:-$(pwd)/tts-runtime}"
ALBA_URL="https://huggingface.co/kyutai/tts-voices/resolve/main/alba-mackenna/casual.wav"

echo "engine: $ENGINE_REPO @ $COMMIT"
echo "out:    $OUT"

# 1. Engine build.
SRC="$(mktemp -d)/PocketTTS.cpp"
git clone --quiet "$ENGINE_REPO" "$SRC"
git -C "$SRC" checkout --quiet "$COMMIT"
cmake -S "$SRC" -B "$SRC/.build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$SRC/.build" -j"$(nproc)" --target pocket-tts
mkdir -p "$OUT/bin" "$OUT/lib" "$OUT/models" "$OUT/voices"
cp "$SRC/pocket-tts" "$OUT/bin/verbix-tts"
cp "$SRC"/libonnxruntime.so* "$OUT/lib/" 2>/dev/null || true
cp "$SRC/export_onnx.py" "$OUT/models/export_onnx.py"

# 2. Weights: throwaway venv (never touches project deps), export + quantize.
EXPORT_VENV="$(mktemp -d)/export-venv"
python3 -m venv "$EXPORT_VENV"
"$EXPORT_VENV/bin/pip" install --quiet pocket-tts onnx onnxruntime
cd "$(mktemp -d)"
"$EXPORT_VENV/bin/python" "$OUT/models/export_onnx.py"
for f in flow_lm_flow_int8.onnx flow_lm_main_int8.onnx mimi_decoder_int8.onnx \
         mimi_encoder.onnx text_conditioner.onnx tokenizer.model; do
    test -f "models/$f"
    cp "models/$f" "$OUT/models/$f"
done
# BOS conditioning file ships with the pocket-tts package data.
BOS="$("$EXPORT_VENV/bin/python" -c "import pocket_tts, pathlib; print(pathlib.Path(pocket_tts.__file__).parent)")"
find "$BOS" -name "bos_before_voice.npy" -exec cp {} "$OUT/models/" \;
test -f "$OUT/models/bos_before_voice.npy"
rm -rf "$EXPORT_VENV"
cd - >/dev/null

# 3. Canonical voice.
curl -sfL "$ALBA_URL" -o "$OUT/voices/alba.wav"
test -f "$OUT/voices/alba.wav"

# 4. Provenance manifest (byte sizes are the integrity gate in tts_assets.py).
python3 - "$OUT" "$ENGINE_REPO" "$COMMIT" "$ALBA_URL" <<'EOF'
import datetime, json, sys
from pathlib import Path
out, repo, commit, alba = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
files = {}
for p in sorted((out / "models").glob("*")) + sorted((out / "voices").glob("*.wav")):
    if p.suffix == ".py":
        continue
    files[str(p.relative_to(out))] = p.stat().st_size
(out / "models" / "MANIFEST.json").write_text(json.dumps({
    "engine_repo": repo, "engine_commit": commit,
    "weight_origin": "Verylicious/pocket-tts-ungated via export_onnx.py (same commit)",
    "alba_canonical_url": alba,
    "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "files": files,
}, indent=2))
print(json.dumps(files, indent=2))
EOF

rm -rf "$SRC"
echo "runtime ready: $OUT"
