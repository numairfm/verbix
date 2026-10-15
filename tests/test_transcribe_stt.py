from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


def _fake_transcribe_cpp(monkeypatch, text: str = "Hello world."):
    mock_result = MagicMock()
    mock_result.text = text

    mock_session = MagicMock()
    mock_session.run.return_value = mock_result
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=None)

    mock_model = MagicMock()
    mock_model.session.return_value = mock_session
    mock_model.__enter__ = MagicMock(return_value=mock_model)
    mock_model.__exit__ = MagicMock(return_value=None)

    fake_module = MagicMock()
    fake_module.Model.return_value = mock_model
    monkeypatch.setattr("verbix.voice.transcribe_stt.transcribe_cpp", fake_module)
    return fake_module, mock_model, mock_session


def test_missing_model_file_raises_clear_error(tmp_path):
    from verbix.voice.transcribe_stt import TranscribeSTT

    fake_path = tmp_path / "missing_model.gguf"
    with pytest.raises(FileNotFoundError, match="STT model not found"):
        TranscribeSTT(model_path=fake_path)


def test_transcribe_returns_cleaned_text(monkeypatch, tmp_path):
    from verbix.voice.transcribe_stt import TranscribeSTT

    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"dummy")

    _fake_module, _mock_model, mock_session = _fake_transcribe_cpp(
        monkeypatch, text="  hello from nemotron  \n"
    )

    stt = TranscribeSTT(model_path=model_file)

    # 0.5s of voiced 300Hz tone
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    wave = (np.sin(2 * np.pi * 300 * t) * 8000).astype(np.int16).tobytes()

    text = stt.transcribe(wave)
    assert text == "hello from nemotron"
    assert mock_session.run.call_count == 1


def test_short_audio_rejects_without_calling_model(monkeypatch, tmp_path):
    from verbix.voice.transcribe_stt import TranscribeSTT

    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"dummy")

    _fake_module, _mock_model, mock_session = _fake_transcribe_cpp(monkeypatch)
    stt = TranscribeSTT(model_path=model_file)

    # 100ms audio (< 150ms minimum)
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    wave = (np.sin(2 * np.pi * 300 * t) * 8000).astype(np.int16).tobytes()

    assert stt.transcribe(wave) == ""
    assert mock_session.run.call_count == 0


def test_keypress_click_rejects_without_calling_model(monkeypatch, tmp_path):
    """A keypress / transient click without sustained voicing must be discarded immediately."""
    from verbix.voice.transcribe_stt import TranscribeSTT

    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"dummy")

    _fake_module, _mock_model, mock_session = _fake_transcribe_cpp(monkeypatch)
    stt = TranscribeSTT(model_path=model_file)

    # 0.5s of audio, but only a 10ms click impulse in the middle
    click_audio = np.zeros(8000, dtype=np.int16)
    click_audio[2000:2160] = 12000

    assert stt.transcribe(click_audio.tobytes()) == ""
    assert mock_session.run.call_count == 0


def test_swappable_model_path_via_env(monkeypatch, tmp_path):
    from verbix.voice.transcribe_stt import TranscribeSTT

    custom_model = tmp_path / "custom_parakeet.gguf"
    custom_model.write_bytes(b"parakeet")

    _fake_transcribe_cpp(monkeypatch)
    monkeypatch.setenv("VERBIX_STT_MODEL", str(custom_model))
    stt = TranscribeSTT()
    assert Path(stt.model_path).resolve() == custom_model.resolve()


def test_project_verbix_models_scope_is_ignored_in_favor_of_global_or_voice_runtime(
    tmp_path, monkeypatch
):
    from verbix.voice.transcribe_stt import find_default_stt_model

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    proj_model = ws / ".verbix" / "models" / "stt" / "a.gguf"
    proj_model.parent.mkdir(parents=True, exist_ok=True)
    proj_model.write_bytes(b"dummy")
    voice_root = tmp_path / "voice-runtime-fake"
    fallback = voice_root / "models" / "nemotron-gguf" / "fallback.gguf"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    fallback.write_bytes(b"dummy")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("VERBIX_VOICE_DIR", str(voice_root))
    monkeypatch.delenv("VERBIX_STT_MODEL", raising=False)
    assert find_default_stt_model(workspace=ws).resolve() == fallback.resolve()


def test_global_verbix_models_scope_used(tmp_path, monkeypatch):
    from verbix.voice.transcribe_stt import find_default_stt_model

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    global_model = home / ".verbix" / "models" / "stt" / "b.gguf"
    global_model.parent.mkdir(parents=True, exist_ok=True)
    global_model.write_bytes(b"dummy")
    voice_root = tmp_path / "voice-runtime-fake"
    fallback = voice_root / "models" / "nemotron-gguf" / "fallback.gguf"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    fallback.write_bytes(b"dummy")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("VERBIX_VOICE_DIR", str(voice_root))
    monkeypatch.delenv("VERBIX_STT_MODEL", raising=False)
    assert find_default_stt_model(workspace=ws).resolve() == global_model.resolve()


def test_canonical_stt_model_path_under_voice_stt_models_and_lazy(tmp_path, monkeypatch):
    from verbix.voice.transcribe_stt import TranscribeSTT, find_default_stt_model

    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("VERBIX_VOICE_DIR", raising=False)
    monkeypatch.delenv("VERBIX_STT_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)

    default_path = find_default_stt_model()
    expected = (
        home
        / ".verbix"
        / "voice"
        / "stt"
        / "models"
        / "nemotron-gguf"
        / "nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf"
    )
    assert default_path == expected.resolve()
    # Must not create ~/.verbix/voice/stt/models merely on lookup
    assert not (home / ".verbix" / "voice").exists()

    # When model exists at canonical path, TranscribeSTT loads it
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_bytes(b"nemotron-weights")
    _fake_transcribe_cpp(monkeypatch)
    stt = TranscribeSTT()
    assert stt.model_path == expected.resolve()


def test_legacy_stt_model_migrates_safely_without_touching_legacy_tts(
    tmp_path, monkeypatch
):
    from verbix.voice.transcribe_stt import find_default_stt_model

    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("VERBIX_VOICE_DIR", raising=False)
    monkeypatch.delenv("VERBIX_STT_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)

    legacy_stt = (
        home
        / ".verbix"
        / "voice"
        / "models"
        / "nemotron-gguf"
        / "nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf"
    )
    legacy_stt.parent.mkdir(parents=True, exist_ok=True)
    legacy_stt.write_bytes(b"legacy-nemotron-gguf")

    legacy_tts = home / ".verbix" / "voice" / "models" / "supertonic-int8" / "tts.json"
    legacy_tts.parent.mkdir(parents=True, exist_ok=True)
    legacy_tts.write_text('{"legacy": true}', encoding="utf-8")

    resolved = find_default_stt_model()
    canonical = (
        home
        / ".verbix"
        / "voice"
        / "stt"
        / "models"
        / "nemotron-gguf"
        / "nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf"
    ).resolve()
    assert resolved == canonical
    assert canonical.is_file()
    assert canonical.read_bytes() == b"legacy-nemotron-gguf"
    # Legacy TTS files must not be deleted
    assert legacy_tts.is_file()


def test_legacy_stt_migration_never_overwrites_existing_destination(
    tmp_path, monkeypatch
):
    from verbix.voice.transcribe_stt import find_default_stt_model

    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("VERBIX_VOICE_DIR", raising=False)
    monkeypatch.delenv("VERBIX_STT_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)

    canonical = (
        home
        / ".verbix"
        / "voice"
        / "stt"
        / "models"
        / "nemotron-gguf"
        / "nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf"
    )
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(b"new-canonical-weights")

    legacy = (
        home
        / ".verbix"
        / "voice"
        / "models"
        / "nemotron-gguf"
        / "nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf"
    )
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(b"old-legacy-weights")

    resolved = find_default_stt_model()
    assert resolved == canonical.resolve()
    assert canonical.read_bytes() == b"new-canonical-weights"
    assert legacy.read_bytes() == b"old-legacy-weights"




