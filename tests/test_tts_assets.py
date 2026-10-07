"""Task 2 tests: asset fetch with cache, size checks, and layout.

All downloads use file:// fixtures. No network, no real binary.
"""

from pathlib import Path

import pytest


def _fixture_repo(tmp_path: Path) -> str:
    repo = tmp_path / "repo"
    (repo / "models").mkdir(parents=True)
    (repo / "voices").mkdir(parents=True)
    (repo / "models" / "mimi_encoder.onnx").write_bytes(b"E" * 64)
    (repo / "voices" / "alba.wav").write_bytes(b"W" * 128)
    return repo.as_uri()


def test_missing_or_wrong_size_file_refetches(tmp_path):
    from verbix.voice import tts_assets as mod

    base = _fixture_repo(tmp_path)
    files = {"models/mimi_encoder.onnx": 64, "voices/alba.wav": 128}
    root = tmp_path / "tts"

    out = mod.ensure_files(root, files, base_url=base)
    assert out["models/mimi_encoder.onnx"].read_bytes() == b"E" * 64

    # Corrupt one file, re-ensure must redownload it.
    out["models/mimi_encoder.onnx"].write_bytes(b"junk")
    out2 = mod.ensure_files(root, files, base_url=base)
    assert out2["models/mimi_encoder.onnx"].read_bytes() == b"E" * 64


def test_no_network_fails_clearly(tmp_path, monkeypatch):
    from verbix.voice import tts_assets as mod

    def boom(url, dest):
        raise OSError("network down")

    monkeypatch.setattr(mod, "_download", boom)
    with pytest.raises(RuntimeError, match="download failed.*mimi_encoder.onnx"):
        mod.ensure_files(tmp_path / "tts", {"models/mimi_encoder.onnx": 64}, base_url="file:///x")


def test_layout_contains_bin_models_voices(tmp_path):
    from verbix.voice import tts_assets as mod

    base = _fixture_repo(tmp_path)
    root = tmp_path / "tts"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "verbix-tts").write_bytes(b"fakebinary")

    layout = mod.ensure_tts_runtime(
        root, base_url=base, files={"models/mimi_encoder.onnx": 64, "voices/alba.wav": 128}
    )
    assert (layout["bin"] / "verbix-tts").is_file()
    assert layout["models"].is_dir()
    assert layout["voices"].is_dir()
    assert (layout["models"] / "MANIFEST.json").is_file()
