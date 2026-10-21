"""PEP 517 build backend wrapper over uv_build that verifies bundled TTS assets."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

REQUIRED_TTS_FILES = (
    "duration_predictor.int8.onnx",
    "text_encoder.int8.onnx",
    "vector_estimator.int8.onnx",
    "vocoder.int8.onnx",
    "tts.json",
    "unicode_indexer.bin",
    "voice.bin",
    "LICENSE",
    "LICENSE-OpenRAIL-M",
)


def verify_bundled_tts_assets(tts_dir: Path | None = None) -> None:
    """Fails the distribution build if any required Supertonic INT8 TTS asset is missing or empty."""
    target = tts_dir or (
        Path(__file__).resolve().parent / "voice" / "tts" / "supertonic-int8"
    )
    missing = [
        fname
        for fname in REQUIRED_TTS_FILES
        if not (target / fname).is_file() or (target / fname).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            f"Cannot build Verbix distribution: missing bundled TTS assets under {target}: {missing}. "
            "Run 'uv run scripts/fetch_voice_models.py --bundle-tts' before building."
        )


def _run_uv_backend(args: list[str]) -> str:
    uv_bin = shutil.which("uv-build") or shutil.which("uv")
    if not uv_bin:
        raise RuntimeError("uv executable not found on PATH")
    cmd = (
        [uv_bin, *args]
        if Path(uv_bin).name in ("uv-build", "uv-build.exe")
        else [uv_bin, "build-backend", *args]
    )
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        detail = (
            res.stderr.strip()
            or res.stdout.strip()
            or f"uv build-backend failed ({res.returncode})"
        )
        raise RuntimeError(detail)
    return res.stdout.strip().splitlines()[-1]


def build_sdist(
    sdist_directory: str, config_settings: dict[str, Any] | None = None
) -> str:
    verify_bundled_tts_assets()
    return _run_uv_backend(["build-sdist", str(sdist_directory)])


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    verify_bundled_tts_assets()
    args = ["build-wheel", str(wheel_directory)]
    if metadata_directory is not None:
        args.extend(["--metadata-directory", str(metadata_directory)])
    return _run_uv_backend(args)


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    args = ["build-editable", str(wheel_directory)]
    if metadata_directory is not None:
        args.extend(["--metadata-directory", str(metadata_directory)])
    return _run_uv_backend(args)


def get_requires_for_build_sdist(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return []


def get_requires_for_build_wheel(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return []


def get_requires_for_build_editable(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return []


def prepare_metadata_for_build_wheel(
    metadata_directory: str, config_settings: dict[str, Any] | None = None
) -> str:
    return _run_uv_backend(
        ["prepare-metadata-for-build-wheel", str(metadata_directory)]
    )


def prepare_metadata_for_build_editable(
    metadata_directory: str, config_settings: dict[str, Any] | None = None
) -> str:
    return _run_uv_backend(
        ["prepare-metadata-for-build-editable", str(metadata_directory)]
    )
