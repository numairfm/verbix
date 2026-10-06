"""No-speech + hallucination guards at the single decode choke point."""

import pytest

from verbix.voice.stt import FasterWhisperSTT, _looks_like_hallucination


def test_dots_only_is_hallucination():
    assert _looks_like_hallucination(". . . . . .")
    assert _looks_like_hallucination("...")
    assert _looks_like_hallucination("  ..  ..  ")


def test_real_speech_is_not_hallucination():
    assert not _looks_like_hallucination("Hello.")
    assert not _looks_like_hallucination("Okay")
    assert not _looks_like_hallucination("")
    assert not _looks_like_hallucination("No, no, no")


@pytest.mark.asyncio
async def test_vad_gate_skips_model_on_noise(monkeypatch):
    import faster_whisper.vad as vad_mod
    import numpy as np

    monkeypatch.setattr(vad_mod, "get_speech_timestamps", lambda *a, **k: [])
    stt = FasterWhisperSTT()
    noise = (np.random.default_rng(0).uniform(-0.05, 0.05, 16000) * 32768).astype(
        np.int16
    ).tobytes()
    assert await stt.transcribe(noise) == ""
    assert not stt.is_loaded  # model never touched


@pytest.mark.asyncio
async def test_vad_error_fails_open_to_model(monkeypatch):
    import faster_whisper.vad as vad_mod
    import numpy as np

    def boom(*a, **k):
        raise OSError("vad down")

    monkeypatch.setattr(vad_mod, "get_speech_timestamps", boom)
    stt = FasterWhisperSTT()
    tiny = (np.zeros(320, dtype=np.int16)).tobytes()
    # model would load here — stub _load_model to prove fall-through, not model
    def no_model():
        raise RuntimeError("no weights in test")

    monkeypatch.setattr(stt, "_load_model", no_model)
    assert await stt.transcribe(tiny) == ""
