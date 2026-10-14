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


@pytest.mark.asyncio
async def test_transcribe_session_drops_pre_speech_noise_outside_vad_spans():
    from verbix.voice.transcribe_stt import TranscribeSession

    transcribed_buffers = []
    fake_stt = MagicMock()
    fake_stt.transcribe = lambda audio, sr: (
        transcribed_buffers.append(audio) or "hey what's up"
    )

    speech_span = b"\x10\x00" * 1600  # 100ms span (3200 bytes)
    fake_seg = MagicMock()
    fake_seg.pop_segments = MagicMock(side_effect=[[], [], [speech_span]])
    fake_seg.flush = MagicMock(return_value=[])

    sess = TranscribeSession(stt=fake_stt, segmenter=fake_seg, sample_rate=16000)
    pre_noise = b"\x01\x00" * 1600
    sess.feed(pre_noise)
    sess.feed(pre_noise)
    sess.feed(speech_span)

    res = await sess.finalize()
    assert res == "hey what's up"
    assert transcribed_buffers == [speech_span]
    assert len(sess._buf) == 0
    assert len(sess.fed_chunks) == 0

    # TranscribeSession also exposes is_speech_detected delegating to segmenter
    fake_seg.is_speech_detected = True
    sess2 = TranscribeSession(stt=fake_stt, segmenter=fake_seg, sample_rate=16000)
    assert sess2.is_speech_detected is True


