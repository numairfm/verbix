"""No-speech + hallucination guards at the single decode choke point."""

import numpy as np
import pytest

from verbix.voice.sherpa_stt import SherpaSTT, _looks_like_hallucination


def test_dots_only_is_hallucination():
    assert _looks_like_hallucination(". . . . . .")
    assert _looks_like_hallucination("...")
    assert _looks_like_hallucination("  ..  ..  ")


def test_real_speech_is_not_hallucination():
    assert not _looks_like_hallucination("Hello.")
    assert not _looks_like_hallucination("Okay")
    assert not _looks_like_hallucination("No, no, no")


def test_common_silence_phrases_are_hallucination():
    assert _looks_like_hallucination("I'm not sure.")
    assert _looks_like_hallucination("Thank you.")
    assert _looks_like_hallucination("Thanks for watching.")


@pytest.mark.asyncio
async def test_vad_gate_skips_model_on_low_energy_noise(tmp_path):
    stt = SherpaSTT(model_dir=tmp_path)
    # Low-amplitude noise under RMS 200 threshold
    rng = np.random.default_rng(0)
    noise = (rng.uniform(-0.003, 0.003, 16000) * 32768).astype(np.int16).tobytes()
    assert await stt.transcribe(noise) == ""
    assert not stt.is_loaded  # model never loaded/touched
