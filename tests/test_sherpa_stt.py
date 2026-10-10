from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


def test_missing_model_files_raises_guidance(tmp_path):
    from verbix.voice.sherpa_stt import SherpaSTT

    stt = SherpaSTT(model_dir=tmp_path / "nonexistent")
    with pytest.raises(RuntimeError, match="fetch_voice_models.py"):
        import asyncio
        asyncio.run(stt.load())


def test_transcribe_returns_text(monkeypatch, tmp_path):
    model_dir = tmp_path / "canary-int8"
    model_dir.mkdir(parents=True)
    for fname in ["encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"]:
        (model_dir / fname).write_bytes(b"dummy")

    fake_stream = MagicMock()
    fake_stream.result.text = " Hello world. "

    fake_recognizer = MagicMock()
    fake_recognizer.create_stream.return_value = fake_stream

    fake_module = types.ModuleType("sherpa_onnx")
    fake_offline_rec = MagicMock()
    fake_offline_rec.from_nemo_canary.return_value = fake_recognizer
    fake_module.OfflineRecognizer = fake_offline_rec

    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_module)

    from verbix.voice.sherpa_stt import SherpaSTT

    stt = SherpaSTT(model_dir=model_dir)

    # 1 second of loud sine tone (loud enough to pass energy pre-gate)
    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    tone = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)

    import asyncio
    res = asyncio.run(stt.transcribe(tone.tobytes()))
    assert res == "Hello world."


def test_transcribe_silence_returns_empty(monkeypatch, tmp_path):
    model_dir = tmp_path / "canary-int8"
    model_dir.mkdir(parents=True)
    for fname in ["encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"]:
        (model_dir / fname).write_bytes(b"dummy")

    fake_module = types.ModuleType("sherpa_onnx")
    fake_offline_rec = MagicMock()
    fake_module.OfflineRecognizer = fake_offline_rec
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_module)

    from verbix.voice.sherpa_stt import SherpaSTT

    stt = SherpaSTT(model_dir=model_dir)

    # Pure digital silence
    silence = np.zeros(16000, dtype=np.int16)
    import asyncio
    res = asyncio.run(stt.transcribe(silence.tobytes()))
    assert res == ""
    # Should not even call recognizer on pure silence
    fake_offline_rec.from_nemo_canary.assert_not_called()


def test_lazy_import_does_not_require_sherpa_onnx(monkeypatch):
    import importlib

    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    import verbix.voice.sherpa_stt as mod
    importlib.reload(mod)

    stt = mod.SherpaSTT(model_dir=Path("/any"))
    assert not stt.is_loaded


def test_single_char_soup_is_hallucination():
    from verbix.voice.sherpa_stt import _looks_like_hallucination

    assert _looks_like_hallucination("1 I 1")
    assert _looks_like_hallucination("a b c d")
    assert not _looks_like_hallucination("I")
    assert not _looks_like_hallucination("a")
    assert not _looks_like_hallucination("OK")
    assert not _looks_like_hallucination("Hello world")


def test_open_session_creates_functional_session(monkeypatch, tmp_path):
    from verbix.voice.sherpa_stt import SherpaSTT, STTSession

    stt = SherpaSTT(model_dir=tmp_path)
    session = stt.open_session()
    assert isinstance(session, STTSession)
