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


def test_trim_silence_removes_dead_air():
    from verbix.voice.sherpa_tts import _trim_silence

    # 1000 samples of 0s, 1000 samples of signal, 1000 samples of 0s
    leading = np.zeros(1000, dtype=np.float32)
    signal = np.ones(1000, dtype=np.float32) * 0.5
    trailing = np.zeros(1000, dtype=np.float32)
    raw = np.concatenate([leading, signal, trailing])

    # pad_ms=10 -> 240 samples pad
    trimmed = _trim_silence(raw, rate=24000, threshold=0.01, pad_ms=10)

    # Should have trimmed ~760 samples from both ends
    assert len(trimmed) < len(raw)
    assert len(trimmed) == 1000 + 2 * 240


def test_bundled_tts_dir_resolves_package_resource_not_user_home(tmp_path, monkeypatch):
    from verbix.voice.sherpa_tts import REQUIRED_FILES, SherpaTTS, bundled_tts_dir

    empty_home = tmp_path / "empty_home"
    empty_home.mkdir()
    unrelated_cwd = tmp_path / "unrelated_cwd"
    unrelated_cwd.mkdir()
    monkeypatch.setattr(Path, "home", lambda: empty_home)
    monkeypatch.chdir(unrelated_cwd)
    monkeypatch.delenv("VERBIX_VOICE_DIR", raising=False)

    bdir = bundled_tts_dir()
    assert bdir.parts[-3:] == ("voice", "tts", "supertonic-int8")
    assert not str(bdir).startswith(str(empty_home))
    for fname in REQUIRED_FILES:
        assert (bdir / fname).is_file(), f"Missing bundled TTS asset: {fname}"

    # Legacy ~/.verbix/voice/models/supertonic-int8 must not be touched or deleted
    legacy_tts_file = (
        empty_home / ".verbix" / "voice" / "models" / "supertonic-int8" / "voice.bin"
    )
    legacy_tts_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_tts_file.write_bytes(b"legacy-user-file")

    fake_module = types.ModuleType("sherpa_onnx")
    fake_module.OfflineTts = MagicMock(return_value=MagicMock())
    fake_module.OfflineTtsConfig = MagicMock()
    fake_module.OfflineTtsModelConfig = MagicMock()
    captured_supertonic = MagicMock()
    fake_module.OfflineTtsSupertonicModelConfig = captured_supertonic
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_module)

    import asyncio

    tts = SherpaTTS()
    asyncio.run(tts.load())
    assert tts.is_loaded
    kwargs = captured_supertonic.call_args.kwargs
    assert Path(kwargs["tts_json"]).parent == bdir
    assert legacy_tts_file.is_file()
    assert legacy_tts_file.read_bytes() == b"legacy-user-file"


def test_build_backend_rejects_missing_or_empty_bundled_tts_assets(tmp_path):
    from verbix._build_backend import REQUIRED_TTS_FILES, verify_bundled_tts_assets

    incomplete_dir = tmp_path / "tts" / "supertonic-int8"
    incomplete_dir.mkdir(parents=True)
    (incomplete_dir / "tts.json").write_text("{}", encoding="utf-8")
    (incomplete_dir / "LICENSE").write_text("license", encoding="utf-8")

    with pytest.raises(RuntimeError, match="fetch_voice_models.py --bundle-tts") as exc:
        verify_bundled_tts_assets(incomplete_dir)
    assert "duration_predictor.int8.onnx" in str(exc.value)

    # Empty (0-byte) file is also rejected
    for fname in REQUIRED_TTS_FILES:
        (incomplete_dir / fname).write_bytes(b"ok")
    (incomplete_dir / "vocoder.int8.onnx").write_bytes(b"")
    with pytest.raises(RuntimeError, match="vocoder.int8.onnx"):
        verify_bundled_tts_assets(incomplete_dir)

    # All files present and non-empty passes
    (incomplete_dir / "vocoder.int8.onnx").write_bytes(b"ok")
    verify_bundled_tts_assets(incomplete_dir)

