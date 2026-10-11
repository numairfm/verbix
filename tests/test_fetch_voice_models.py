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


def test_fetch_downloads_and_verifies_files(tmp_path):
    import scripts.fetch_voice_models as mod

    test_content = b"valid model content"
    test_sha = hashlib.sha256(test_content).hexdigest()

    fake_manifest = {
        "tts": {
            "id": "supertonic-int8",
            "license": "BigScience OpenRAIL-M",
            "repo": "csukuangfj2/sherpa-onnx-supertonic-3-tts-int8-2026-05-11",
            "files": {
                "test_file.bin": (test_sha, len(test_content)),
            },
        },
        "vad": {
            "id": "silero-vad",
            "license": "MIT",
            "repo": "k2-fsa/sherpa-onnx",
            "files": {
                "test_vad.onnx": {
                    "sha256": test_sha,
                    "url": "https://custom.url/test_vad.onnx",
                },
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
        assert (tmp_path / "models" / "supertonic-int8" / "test_file.bin").exists()
        assert (tmp_path / "models" / "silero-vad" / "test_vad.onnx").exists()
        assert "https://custom.url/test_vad.onnx" in downloaded_urls
        assert (tmp_path / "models" / "provenance.json").exists()
        assert paths["tts_dir"] == tmp_path / "models" / "supertonic-int8"
        assert paths["vad_dir"] == tmp_path / "models" / "silero-vad"

        # Check provenance json
        prov = json.loads((tmp_path / "models" / "provenance.json").read_text())
        assert prov["tts"]["id"] == "supertonic-int8"
        assert prov["tts"]["license"] == "BigScience OpenRAIL-M"
        assert prov["vad"]["id"] == "silero-vad"


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
