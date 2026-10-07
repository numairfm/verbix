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


def _bucket_repo(tmp_path: Path) -> str:
    repo = tmp_path / "bucket" / "v1"
    (repo / "models" / "tts").mkdir(parents=True)
    (repo / "voices").mkdir(parents=True)
    (repo / "models" / "tts" / "flow_lm_flow_int8.onnx").write_bytes(b"F" * 32)
    (repo / "voices" / "alba.wav").write_bytes(b"W" * 16)
    return (tmp_path / "bucket" / "v1").as_uri()


def test_first_run_downloads_from_default_bundle(monkeypatch, tmp_path):
    from verbix.voice import tts_assets as mod

    base = _bucket_repo(tmp_path)
    monkeypatch.setattr(mod, "DEFAULT_BUNDLE_URL", base)
    files = {"models/tts/flow_lm_flow_int8.onnx": 32, "voices/alba.wav": 16}
    root = tmp_path / "tts"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "verbix-tts").write_bytes(b"fakebinary")

    layout = mod.ensure_tts_runtime(root, files=files)
    assert (layout["models"] / "tts" / "flow_lm_flow_int8.onnx").read_bytes() == b"F" * 32
    assert (layout["voices"] / "alba.wav").read_bytes() == b"W" * 16


def test_missing_binary_points_at_setup_script(tmp_path):
    from verbix.voice import tts_assets as mod

    with __import__("pytest").raises(RuntimeError, match="setup_tts_runtime"):
        mod.ensure_tts_runtime(tmp_path / "empty", files={})


def test_default_root_is_universal_home(monkeypatch, tmp_path):
    from verbix.voice import tts_assets as mod

    monkeypatch.delenv("VERBIX_TTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert mod.default_root() == Path.home() / ".verbix" / "tts"


def test_repo_local_runtime_wins_when_present(monkeypatch, tmp_path):
    from verbix.voice import tts_assets as mod

    monkeypatch.delenv("VERBIX_TTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert mod.default_root() == Path.home() / ".verbix" / "tts"
    local_bin = tmp_path / "tts-runtime" / "bin"
    local_bin.mkdir(parents=True)
    (local_bin / "verbix-tts").write_bytes(b"fakebinary")
    assert mod.default_root() == tmp_path / "tts-runtime"


def test_download_sends_browser_like_user_agent(monkeypatch, tmp_path):
    from verbix.voice import tts_assets as mod

    seen = {}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"data"

    def fake_urlopen(req, *a, **k):
        seen["ua"] = req.get_header("User-agent")
        return FakeResp()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    mod._download("https://example.com/y.bin", tmp_path / "y.bin")
    assert (tmp_path / "y.bin").read_bytes() == b"data"
    assert seen["ua"] and "Python-urllib" not in seen["ua"]
