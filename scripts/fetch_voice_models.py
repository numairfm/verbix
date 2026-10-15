#!/usr/bin/env python3
"""Downloads and verifies pinned INT8 speech models for sherpa-onnx.

Zero external dependencies: uses stdlib urllib and hashlib.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# Pinned user-managed STT models, static SHA256 hashes, and byte sizes
MANIFEST: dict[str, dict] = {
    "asr": {
        "id": "nemotron-gguf",
        "license": "OpenMDW-1.1",
        "repo": "handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf",
        "files": {
            "nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf": (
                "41c99fa5fb6f3d35f68e79adc3e755eca2232a8d921178bd647b71194792b8fd",
                495831520,
            ),
        },
    },
}

# Fixed bundled application TTS asset specification (packaged under verbix/voice/tts/supertonic-int8)
BUNDLED_TTS_SPEC: dict = {
    "id": "supertonic-int8",
    "license": "BigScience OpenRAIL-M",
    "repo": "csukuangfj2/sherpa-onnx-supertonic-3-tts-int8-2026-05-11",
    "files": {
        "duration_predictor.int8.onnx": (
            "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db",
            3700147,
        ),
        "text_encoder.int8.onnx": (
            "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff",
            36416150,
        ),
        "vector_estimator.int8.onnx": (
            "20cd86fa5c6effedfda0e7cffe5b0569ca401c440a0c3a1d72bf39286c0db3fd",
            78400833,
        ),
        "vocoder.int8.onnx": (
            "e923d60f53f95eb1ce235f1dc33ec56d9c057823c96fa6f8acf98f32b0da6152",
            25991073,
        ),
        "tts.json": (
            "42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09",
            8253,
        ),
        "unicode_indexer.bin": (
            "8402ca48e5189a8950138580b0fff64db6f072f24ac07cd54ba8b2fbb9883b30",
            262144,
        ),
        "voice.bin": (
            "67d5209b0ee8ce6c74105ffbe12fe6a7628aea3b4ba2fcb308a4a67938a93ce8",
            517168,
        ),
        "LICENSE": (
            "0dfe0d0ba84416fe3879d9a34f4909d8d0137c78d1e95834177b0414ac096fa2",
            1070,
        ),
    },
}


def default_root() -> Path:
    """Resolves target root directory: VERBIX_VOICE_DIR -> ./voice-runtime -> ~/.verbix/voice."""
    if env := os.environ.get("VERBIX_VOICE_DIR"):
        return Path(env).resolve()
    local = Path.cwd() / "voice-runtime"
    if local.is_dir():
        return local.resolve()
    return (Path.home() / ".verbix" / "voice").resolve()


def _verify_sha256(path: Path, expected_sha: str) -> bool:
    """Verifies SHA256 of file on disk in chunks to avoid large memory allocations."""
    if not path.is_file():
        return False
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower() == expected_sha.lower()


def _download_file(url: str, dest_path: Path) -> None:
    """Downloads a remote URL to dest_path with atomic write via temp file."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")
    req = urllib.request.Request(
        url, headers={"User-Agent": "verbix-model-fetcher/1.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp, temp_path.open(
        "wb"
    ) as out:
        while chunk := resp.read(65536):
            out.write(chunk)
    temp_path.replace(dest_path)


def _ensure_spec_files(spec: dict, model_target_dir: Path) -> None:
    """Ensures all files in a model spec exist in model_target_dir and match SHA256."""
    model_id = spec["id"]
    repo = spec["repo"]
    model_target_dir.mkdir(parents=True, exist_ok=True)

    for filename, file_spec in spec["files"].items():
        if isinstance(file_spec, dict):
            expected_sha = file_spec["sha256"]
            url = file_spec.get("url")
        else:
            expected_sha, _ = file_spec
            url = None
        file_dest = model_target_dir / filename
        if file_dest.is_file() and _verify_sha256(file_dest, expected_sha):
            continue  # already verified and matches checksum

        if url is None:
            url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        sys.stdout.write(f"Fetching {model_id}/{filename} ...\n")
        sys.stdout.flush()
        _download_file(url, file_dest)

        if not _verify_sha256(file_dest, expected_sha):
            file_dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {model_id}/{filename}: expected {expected_sha}"
            )


def fetch_models(root: Path | None = None) -> dict[str, Path]:
    """Fetches and verifies user-managed STT voice models into <root>/stt/models."""
    from verbix.voice.transcribe_stt import migrate_legacy_stt_model

    target_root = (root or default_root()).resolve()
    models_dir = target_root / "stt" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    result_paths: dict[str, Path] = {}
    provenance_record: dict[str, dict] = {}

    for kind, spec in MANIFEST.items():
        model_id = spec["id"]
        repo = spec["repo"]
        migrate_legacy_stt_model(target_root, model_id=model_id)
        model_target_dir = models_dir / model_id
        _ensure_spec_files(spec, model_target_dir)

        result_paths[f"{kind}_dir"] = model_target_dir
        provenance_record[kind] = {
            "id": model_id,
            "repo": repo,
            "license": spec["license"],
            "fetched_at": time.time(),
        }

    # Record provenance
    prov_file = models_dir / "provenance.json"
    prov_file.write_text(json.dumps(provenance_record, indent=2))

    return result_paths


def fetch_bundled_tts(dest_dir: Path | None = None) -> Path:
    """Provisions the bundled Supertonic INT8 TTS assets into the application package resource tree."""
    import shutil

    from verbix.voice.sherpa_tts import bundled_tts_dir

    target_dir = (dest_dir or bundled_tts_dir()).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    # Seed from local legacy cache if verified, avoiding redundant network fetches on fresh checkouts
    legacy_dir = default_root() / "models" / BUNDLED_TTS_SPEC["id"]
    if dest_dir is None and legacy_dir.is_dir():
        for filename, file_spec in BUNDLED_TTS_SPEC["files"].items():
            expected_sha = (
                file_spec["sha256"] if isinstance(file_spec, dict) else file_spec[0]
            )
            dest_file = target_dir / filename
            legacy_file = legacy_dir / filename
            if (
                not (dest_file.is_file() and _verify_sha256(dest_file, expected_sha))
                and legacy_file.is_file()
                and _verify_sha256(legacy_file, expected_sha)
            ):
                shutil.copy2(legacy_file, dest_file)

    _ensure_spec_files(BUNDLED_TTS_SPEC, target_dir)
    return target_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch user-managed STT voice models (or provision bundled TTS assets) for Verbix"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Target voice root directory for STT models (defaults to VERBIX_VOICE_DIR, ./voice-runtime, or ~/.verbix/voice)",
    )
    parser.add_argument(
        "--bundle-tts",
        action="store_true",
        help="Provision only the bundled application TTS assets under src/verbix/voice/tts/supertonic-int8",
    )
    args = parser.parse_args(argv)

    try:
        paths: dict[str, Path] = {}
        if args.bundle_tts:
            sys.stdout.write("Provisioning bundled TTS assets ...\n")
            paths["bundled_tts_dir"] = fetch_bundled_tts()
        else:
            root = (args.root or default_root()).resolve()
            sys.stdout.write(
                f"Provisioning STT models into {root / 'stt' / 'models'} ...\n"
            )
            paths = fetch_models(root=root)
            paths["bundled_tts_dir"] = fetch_bundled_tts()
        sys.stdout.write("Models provisioned successfully:\n")
        for k, p in paths.items():
            sys.stdout.write(f"  {k}: {p}\n")
        return 0
    except (RuntimeError, OSError, ValueError) as e:
        sys.stderr.write(f"Error provisioning voice models: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
