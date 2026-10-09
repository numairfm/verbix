from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


def test_missing_model_files_raises_guidance(tmp_path):
    from verbix.voice.sherpa_tts import SherpaTTS

    tts = SherpaTTS(model_dir=tmp_path / "nonexistent")
    with pytest.raises(RuntimeError, match="fetch_voice_models.py"):
        import asyncio
        asyncio.run(tts.load())


def test_synthesize_yields_24k_f32_chunks(monkeypatch, tmp_path):
    model_dir = tmp_path / "supertonic-int8"
    model_dir.mkdir(parents=True)
    for fname in [
        "duration_predictor.int8.onnx",
        "text_encoder.int8.onnx",
        "vector_estimator.int8.onnx",
        "vocoder.int8.onnx",
        "tts.json",
        "unicode_indexer.bin",
        "voice.bin",
    ]:
        (model_dir / fname).write_bytes(b"dummy")

    # Mock sherpa_onnx
    fake_audio = MagicMock()
    # 44100 samples of 1 kHz tone at 44.1 kHz = 1 second
    t = np.linspace(0, 1.0, 44100, endpoint=False, dtype=np.float32)
    fake_audio.samples = (np.sin(2 * np.pi * 1000 * t) * 0.5).astype(np.float32)
    fake_audio.sample_rate = 44100

    fake_offline_tts = MagicMock()
    fake_offline_tts.generate.return_value = fake_audio

    fake_module = types.ModuleType("sherpa_onnx")
    fake_module.OfflineTts = MagicMock(return_value=fake_offline_tts)
    fake_module.OfflineTtsConfig = MagicMock()
    fake_module.OfflineTtsModelConfig = MagicMock()
    fake_module.OfflineTtsSupertonicModelConfig = MagicMock()

    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_module)

    from verbix.voice.sherpa_tts import SherpaTTS

    tts = SherpaTTS(model_dir=model_dir, voice_id="M1")

    async def _run():
        await tts.load()
        assert tts.is_loaded
        chunks = []
        async for chunk in tts.synthesize("Hello world"):
            chunks.append(chunk)
        return chunks

    import asyncio
    chunks = asyncio.run(_run())

    assert len(chunks) == 1
    pcm = chunks[0]
    # Resampled to 24 kHz: 1 second = 24000 float32 samples = 96000 bytes
    assert len(pcm) == 24000 * 4
    arr = np.frombuffer(pcm, dtype=np.float32)
    assert arr.max() <= 1.0 and arr.min() >= -1.0


def test_synthesize_empty_string_yields_nothing(tmp_path):
    from verbix.voice.sherpa_tts import SherpaTTS

    tts = SherpaTTS(model_dir=tmp_path)
    tts._loaded = True
    tts._tts = MagicMock()

    async def _run():
        chunks = []
        async for chunk in tts.synthesize("   \n\t  "):
            chunks.append(chunk)
        return chunks

    import asyncio
    chunks = asyncio.run(_run())
    assert chunks == []
    tts._tts.generate.assert_not_called()


def test_lazy_import_does_not_import_sherpa_onnx_at_module_load(monkeypatch):
    import importlib

    # Block sherpa_onnx
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)

    import verbix.voice.sherpa_tts as mod
    importlib.reload(mod)

    # Creating instance does not import sherpa_onnx
    tts = mod.SherpaTTS(model_dir=Path("/any"))
    assert not tts.is_loaded
