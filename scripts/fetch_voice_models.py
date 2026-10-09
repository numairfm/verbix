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

# Pinned models, static SHA256 hashes, and byte sizes
MANIFEST: dict[str, dict] = {
    "tts": {
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
    },
    "asr": {
        "id": "canary-int8",
        "license": "CC-BY-4.0",
        "repo": "csukuangfj/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8",
        "files": {
            "encoder.int8.onnx": (
                "7a75b4e2a5857a6dcc0819503bbe3fad66943db4a3ccf21d3f27c633667d303f",
                132678643,
            ),
            "decoder.int8.onnx": (
                "e41a2ab9c0c2fe81a1e8ade5a45fb02a74bc4db7d1f91b89a54a25e2cf79cba2",
                74437848,
            ),
            "tokens.txt": (
                "2dae6fc7815f9640645e0c765522b278ee0cef49b482d91f6913e334628d3e77",
                53555,
            ),
        },
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


def fetch_models(root: Path | None = None) -> dict[str, Path]:
    """Fetches and verifies all pinned voice models into root directory."""
    target_root = (root or default_root()).resolve()
    models_dir = target_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    result_paths: dict[str, Path] = {}
    provenance_record: dict[str, dict] = {}

    for kind, spec in MANIFEST.items():
        model_id = spec["id"]
        repo = spec["repo"]
        model_target_dir = models_dir / model_id
        model_target_dir.mkdir(parents=True, exist_ok=True)

        for filename, (expected_sha, expected_size) in spec["files"].items():
            file_dest = model_target_dir / filename
            if file_dest.is_file() and _verify_sha256(file_dest, expected_sha):
                continue  # already verified and matches checksum

            url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
            sys.stdout.write(f"Fetching {model_id}/{filename} ...\n")
            sys.stdout.flush()
            _download_file(url, file_dest)

            if not _verify_sha256(file_dest, expected_sha):
                file_dest.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Checksum mismatch for {model_id}/{filename}: expected {expected_sha}"
                )

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch voice models for Verbix"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Target root directory (defaults to VERBIX_VOICE_DIR, ./voice-runtime, or ~/.verbix/voice)",
    )
    args = parser.parse_args()

    root = (args.root or default_root()).resolve()
    sys.stdout.write(f"Provisioning voice models into {root} ...\n")
    try:
        paths = fetch_models(root=root)
        sys.stdout.write("Models provisioned successfully:\n")
        for k, p in paths.items():
            sys.stdout.write(f"  {k}: {p}\n")
        return 0
    except (RuntimeError, OSError, ValueError) as e:
        sys.stderr.write(f"Error provisioning voice models: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
