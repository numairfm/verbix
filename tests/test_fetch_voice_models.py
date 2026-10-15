from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_default_root_prefers_env(monkeypatch, tmp_path):
    env_dir = tmp_path / "custom_voice"
    monkeypatch.setenv("VERBIX_VOICE_DIR", str(env_dir))
    import scripts.fetch_voice_models as mod

    assert mod.default_root() == env_dir


def test_default_root_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("VERBIX_VOICE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    import scripts.fetch_voice_models as mod

    # When no local dir exists, falls back to ~/.verbix/voice
    expected_fallback = Path.home() / ".verbix" / "voice"
    assert mod.default_root() == expected_fallback

    # When local voice-runtime exists, prefers local
    local_dir = tmp_path / "voice-runtime"
    local_dir.mkdir()
    assert mod.default_root() == local_dir


def test_fetch_downloads_and_verifies_stt_files_only(tmp_path):
    import scripts.fetch_voice_models as mod

    assert "vad" not in mod.MANIFEST
    assert "tts" not in mod.MANIFEST
    test_content = b"valid model content"
    test_sha = hashlib.sha256(test_content).hexdigest()

    fake_manifest = {
        "asr": {
            "id": "nemotron-gguf",
            "license": "OpenMDW-1.1",
            "repo": "handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf",
            "files": {
                "test_asr.gguf": (test_sha, len(test_content)),
            },
        },
    }

    downloaded_urls = []

    def fake_download(url, dest_path):
        downloaded_urls.append(url)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(test_content)

    with patch.object(mod, "MANIFEST", fake_manifest), patch.object(
        mod, "_download_file", side_effect=fake_download
    ):
        paths = mod.fetch_models(root=tmp_path)
        assert (tmp_path / "stt" / "models" / "nemotron-gguf" / "test_asr.gguf").exists()
        assert (tmp_path / "stt" / "models" / "provenance.json").exists()
        assert not (tmp_path / "models" / "supertonic-int8").exists()
        assert paths["asr_dir"] == tmp_path / "stt" / "models" / "nemotron-gguf"
        assert "tts_dir" not in paths
        assert "vad_dir" not in paths

        # Check provenance json
        prov = json.loads((tmp_path / "stt" / "models" / "provenance.json").read_text())
        assert prov["asr"]["id"] == "nemotron-gguf"
        assert prov["asr"]["license"] == "OpenMDW-1.1"
        assert "tts" not in prov
        assert "vad" not in prov


def test_fetch_migrates_existing_legacy_stt_without_redownload(tmp_path):
    import scripts.fetch_voice_models as mod

    test_content = b"already downloaded nemotron gguf"
    test_sha = hashlib.sha256(test_content).hexdigest()

    legacy_file = tmp_path / "models" / "nemotron-gguf" / "test_asr.gguf"
    legacy_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_file.write_bytes(test_content)

    fake_manifest = {
        "asr": {
            "id": "nemotron-gguf",
            "license": "OpenMDW-1.1",
            "repo": "handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf",
            "files": {
                "test_asr.gguf": (test_sha, len(test_content)),
            },
        },
    }

    downloaded_urls = []

    def fake_download(url, dest_path):
        downloaded_urls.append(url)

    with patch.object(mod, "MANIFEST", fake_manifest), patch.object(
        mod, "_download_file", side_effect=fake_download
    ):
        paths = mod.fetch_models(root=tmp_path)
        canonical_file = tmp_path / "stt" / "models" / "nemotron-gguf" / "test_asr.gguf"
        assert canonical_file.is_file()
        assert canonical_file.read_bytes() == test_content
        assert downloaded_urls == []
        assert paths["asr_dir"] == tmp_path / "stt" / "models" / "nemotron-gguf"


def test_fetch_raises_on_checksum_mismatch(tmp_path):
    import scripts.fetch_voice_models as mod

    bad_content = b"corrupted file content"
    expected_sha = "0000000000000000000000000000000000000000000000000000000000000000"

    fake_manifest = {
        "tts": {
            "id": "supertonic-int8",
            "license": "BigScience OpenRAIL-M",
            "repo": "test-repo",
            "files": {
                "corrupted.bin": (expected_sha, 100),
            },
        }
    }

    def fake_download(url, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(bad_content)

    with patch.object(mod, "MANIFEST", fake_manifest), patch.object(
        mod, "_download_file", side_effect=fake_download
    ), pytest.raises(RuntimeError, match="(?i)checksum mismatch"):
        mod.fetch_models(root=tmp_path)


def test_fetch_bundled_tts_provisions_package_dir_only(tmp_path, monkeypatch):
    import scripts.fetch_voice_models as mod

    test_content = b"bundled tts asset"
    test_sha = hashlib.sha256(test_content).hexdigest()
    fake_tts_spec = {
        "id": "supertonic-int8",
        "license": "BigScience OpenRAIL-M",
        "repo": "csukuangfj2/sherpa-onnx-supertonic-3-tts-int8-2026-05-11",
        "files": {
            "tts.json": (test_sha, len(test_content)),
            "voice.bin": (test_sha, len(test_content)),
        },
    }
    dest = tmp_path / "pkg_tts"
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)

    def fake_download(url, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(test_content)

    with patch.object(mod, "BUNDLED_TTS_SPEC", fake_tts_spec), patch.object(
        mod, "_download_file", side_effect=fake_download
    ):
        out = mod.fetch_bundled_tts(dest_dir=dest)
        assert out == dest.resolve()
        assert (dest / "tts.json").read_bytes() == test_content
        assert (dest / "voice.bin").read_bytes() == test_content
        assert not (home / ".verbix").exists()
