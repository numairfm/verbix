from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# Pinned bundle content (bytes), re-minted 2026-10-08 via
# scripts/setup_tts_runtime.sh (engine + export commits in MANIFEST.json).
# URLs are recorded in MANIFEST.json by whoever fetches; sizes are the
# integrity gate here.
PINNED_FILES: dict[str, int] = {
    "models/flow_lm_flow_int8.onnx": 9962530,
    "models/flow_lm_main_int8.onnx": 76341079,
    "models/mimi_decoder_int8.onnx": 22695710,
    "models/mimi_encoder.onnx": 585023,
    "models/mimi_encoder.onnx.data": 72613888,
    "models/text_conditioner.onnx": 465,
    "models/text_conditioner.onnx.data": 16388096,
    "models/tokenizer.model": 59339,
    "voices/alba.wav": 958542,
}

# Canonical Alba reference (upstream seed; production mirrors under own control).
ALBA_URL = "https://huggingface.co/kyutai/tts-voices/resolve/main/alba-mackenna/casual.wav"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def ensure_files(
    root: Path, files: dict[str, int], base_url: str | None = None
) -> dict[str, Path]:
    """Ensures each relpath exists under root with the pinned byte size.

    Missing or wrong-sized files are (re)downloaded from base_url. Returns
    relpath -> absolute path. Raises RuntimeError naming the file when no
    base_url is set or the download fails.
    """
    out: dict[str, Path] = {}
    for rel, size in files.items():
        dest = root / rel
        if dest.is_file() and dest.stat().st_size == size:
            out[rel] = dest
            continue
        if not base_url:
            raise RuntimeError(
                f"TTS asset missing or wrong size: {rel} "
                f"(want {size} bytes). Set a bundle source or run "
                "scripts/build_tts_helper.sh."
            )
        try:
            _download(f"{base_url.rstrip('/')}/{rel}", dest)
        except OSError as e:
            raise RuntimeError(f"TTS asset download failed: {rel}: {e}") from e
        if dest.stat().st_size != size:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"TTS asset download failed: {rel} has wrong size "
                f"(want {size} bytes). Source mismatch — refusing to load."
            )
        out[rel] = dest
    return out


def _resolve_binary(root: Path) -> Path | None:
    """Helper binary for this runtime dir, or None. One name, no guessing."""
    import sys

    suffix = ".exe" if sys.platform == "win32" else ""
    candidate = root / "bin" / ("verbix-tts" + suffix)
    return candidate if candidate.is_file() else None


def default_root() -> Path:
    if env := os.environ.get("VERBIX_TTS_DIR"):
        return Path(env)
    local = Path.cwd() / "tts-runtime"
    if _resolve_binary(local) is not None:
        return local
    return Path.home() / ".verbix" / "tts"


def ensure_tts_runtime(
    root: Path | None = None,
    base_url: str | None = None,
    files: dict[str, int] | None = None,
) -> dict[str, Path]:
    """Ensures the helper runtime layout; returns bin/models/voices dirs.

    Binary: must already exist (CI prebuilt, local build, or dev override).
    If missing and base_url serves one, it is fetched; otherwise a clear
    error points at scripts/build_tts_helper.sh. Compilation is never
    triggered implicitly — explicit script runs and CI own that step.
    """
    root = root or default_root()
    binary = _resolve_binary(root)
    if binary is None:
        if base_url:
            try:
                _download(f"{base_url.rstrip('/')}/bin/verbix-tts", root / "bin" / "verbix-tts")
                binary = root / "bin" / "verbix-tts"
            except OSError as e:
                raise RuntimeError(
                    f"TTS helper binary missing under {root / 'bin'}: {e}. Build one with "
                    "scripts/build_tts_helper.sh."
                ) from e
        else:
            raise RuntimeError(
                f"TTS helper binary missing under {root / 'bin'}. Build one with "
                "scripts/build_tts_helper.sh or point VERBIX_TTS_DIR at a runtime dir."
            )
    ensure_files(root, files if files is not None else PINNED_FILES, base_url)
    manifest = {
        "binary": str(binary),
        "files": files if files is not None else PINNED_FILES,
        "base_url": base_url,
        "alba_canonical_url": ALBA_URL,
    }
    (root / "models" / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return {"bin": root / "bin", "models": root / "models", "voices": root / "voices"}
