from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


class FakeSegmenter:
    """Scripted spans for engine tests: no model files needed."""

    def __init__(self, scripts: list[list[bytes]] | None = None) -> None:
        self.scripts = list(scripts or [])

    def feed(self, pcm: bytes) -> None:
        pass

    def pop_segments(self) -> list[bytes]:
        return []

    def flush(self) -> list[bytes]:
        rest = self.scripts
        self.scripts = []
        return [b for spans in rest for b in spans]

    def reset(self) -> None:
        pass


def _model_dir(tmp_path):
    model_dir = tmp_path / "canary-int8"
    model_dir.mkdir(parents=True)
    for fname in ["encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"]:
        (model_dir / fname).write_bytes(b"dummy")
    return model_dir


def _fake_sherpa(monkeypatch, text=" Hello world. "):
    fake_stream = MagicMock()
    fake_stream.result.text = text
    fake_recognizer = MagicMock()
    fake_recognizer.create_stream.return_value = fake_stream
    fake_module = types.ModuleType("sherpa_onnx")
    fake_offline_rec = MagicMock()
    fake_offline_rec.from_nemo_canary.return_value = fake_recognizer
    fake_module.OfflineRecognizer = fake_offline_rec
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_module)
    return fake_offline_rec


def test_missing_model_files_raises_guidance(tmp_path):
    from verbix.voice.sherpa_stt import SherpaSTT

    stt = SherpaSTT(model_dir=tmp_path / "nonexistent")
    with pytest.raises(RuntimeError, match="fetch_voice_models.py"):
        import asyncio
        asyncio.run(stt.load())


def test_transcribe_returns_text(monkeypatch, tmp_path):
    from verbix.voice.sherpa_stt import SherpaSTT

    model_dir = _model_dir(tmp_path)
    _fake_sherpa(monkeypatch)

    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    tone = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    seg = FakeSegmenter(scripts=[[tone.tobytes()]])
    stt = SherpaSTT(model_dir=model_dir, segmenter=seg)

    import asyncio
    assert asyncio.run(stt.transcribe(tone.tobytes())) == "Hello world."


def test_transcribe_silence_returns_empty_without_model(monkeypatch, tmp_path):
    from verbix.voice.sherpa_stt import SherpaSTT

    model_dir = _model_dir(tmp_path)
    fake_offline_rec = _fake_sherpa(monkeypatch)

    stt = SherpaSTT(model_dir=model_dir, segmenter=FakeSegmenter())
    silence = np.zeros(16000, dtype=np.int16)
    import asyncio
    assert asyncio.run(stt.transcribe(silence.tobytes())) == ""
    # No spans: the recognizer is never even constructed
    fake_offline_rec.from_nemo_canary.assert_not_called()


def test_blip_span_dies_at_turn_gate(monkeypatch, tmp_path):
    """A sub-threshold span decodes to nothing: the turn gate, not the model."""
    from verbix.voice.sherpa_stt import SherpaSTT

    model_dir = _model_dir(tmp_path)
    fake_offline_rec = _fake_sherpa(monkeypatch)

    rng = np.random.default_rng(0)
    blip = (rng.normal(0, 4000, 1600)).astype(np.int16).tobytes()  # 100 ms
    stt = SherpaSTT(model_dir=model_dir, segmenter=FakeSegmenter(scripts=[[blip]]))

    import asyncio
    assert asyncio.run(stt.transcribe(blip)) == ""
    fake_offline_rec.from_nemo_canary.assert_not_called()


def test_sustained_span_decodes(monkeypatch, tmp_path):
    """500 ms of voiced-level tone decodes (model faked)."""
    from verbix.voice.sherpa_stt import SherpaSTT

    model_dir = _model_dir(tmp_path)
    _fake_sherpa(monkeypatch, text="yes")

    t = np.arange(8000) / 16000
    tone = (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16)
    stt = SherpaSTT(model_dir=model_dir, segmenter=FakeSegmenter(scripts=[[tone.tobytes()]]))

    import asyncio
    assert asyncio.run(stt.transcribe(tone.tobytes())) == "yes"


def test_lazy_import_does_not_require_sherpa_onnx(monkeypatch):
    """sherpa_stt must import without sherpa_onnx installed (lazy inside methods)."""
    import importlib.util

    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    # Load an isolated copy from the file location: importlib.reload() would
    # rebind the live module's classes mid-suite and break isinstance checks
    # in other test files holding the original references.
    path = Path("src/verbix/voice/sherpa_stt.py").resolve()
    spec = importlib.util.spec_from_file_location("sherpa_stt_isolated", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    stt = mod.SherpaSTT(model_dir=Path("/any"))
    assert not stt.is_loaded


def test_open_session_uses_injected_segmenter(monkeypatch, tmp_path):
    from verbix.voice.sherpa_stt import SherpaSTT, STTSession

    seg = FakeSegmenter()
    stt = SherpaSTT(model_dir=tmp_path, segmenter=seg)
    session = stt.open_session()
    assert isinstance(session, STTSession)
    assert session._segmenter is seg


def test_open_session_without_segmenter_or_model_raises(monkeypatch, tmp_path):
    from verbix.voice.sherpa_stt import SherpaSTT

    # Point the default voice root at an empty dir so no real model can satisfy it
    monkeypatch.setenv("VERBIX_VOICE_DIR", str(tmp_path / "empty-root"))
    stt = SherpaSTT(model_dir=tmp_path / "nonexistent")
    with pytest.raises(RuntimeError, match="fetch_voice_models.py"):
        stt.open_session()


@pytest.mark.asyncio
async def test_concurrent_transcribe_raw_preserves_stderr(monkeypatch, tmp_path):
    """Multiple parallel transcribe_raw calls must not corrupt process stderr (FD 2)."""
    import asyncio
    import os
    import time
    from verbix.voice.sherpa_stt import SherpaSTT

    model_dir = _model_dir(tmp_path)
    fake_stream = MagicMock()
    fake_stream.result.text = "test"
    fake_recognizer = MagicMock()

    def slow_decode(stream):
        time.sleep(0.05)

    fake_recognizer.create_stream.return_value = fake_stream
    fake_recognizer.decode_stream.side_effect = slow_decode

    fake_module = types.ModuleType("sherpa_onnx")
    fake_offline_rec = MagicMock()
    fake_offline_rec.from_nemo_canary.return_value = fake_recognizer
    fake_module.OfflineRecognizer = fake_offline_rec
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_module)

    stt = SherpaSTT(model_dir=model_dir)
    audio = np.ones(1600, dtype=np.float32)

    # Initial check: FD 2 points to a valid file/pipe/terminal
    initial_stat = os.fstat(2)

    # Launch two concurrent decodes
    t1 = asyncio.create_task(stt.transcribe_raw(audio))
    t2 = asyncio.create_task(stt.transcribe_raw(audio))
    await asyncio.gather(t1, t2)

    # After concurrent execution, FD 2 must match initial_stat (not pointing to closed or /dev/null)
    final_stat = os.fstat(2)
    assert (initial_stat.st_ino, initial_stat.st_dev) == (final_stat.st_ino, final_stat.st_dev)


@pytest.mark.asyncio
async def test_transcribe_handles_odd_bytes(tmp_path):
    """transcribe() must not raise ValueError when given odd-length bytes."""
    from verbix.voice.sherpa_stt import SherpaSTT
    stt = SherpaSTT(model_dir=tmp_path)
    odd_data = b"\x01" * 641  # 641 bytes (odd length, > 320 threshold)
    # Should safely truncate to 640 and process (without ValueError)
    res = await stt.transcribe(odd_data)
    assert isinstance(res, str)


