import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from verbix.agent.models.litellm_model import LitellmModel
from verbix.voice.base import MockAudioSink, MockSpeechToText, MockTextToSpeech
from verbix.voice.capture import PTTCapture, _condition_s16le
from verbix.voice.pipeline import VoicePipeline
from verbix.voice.player import AudioPlayer


def test_capture_reads_20ms_frames():
    assert PTTCapture(sample_rate=16000).frame_bytes == 640


def _speech_like(n: int = 3200, seed: int = 7) -> np.ndarray:
    """A voiced chunk: two partials plus noise, in s16le range."""
    t = np.arange(n) / 16000
    rng = np.random.default_rng(seed)
    x = 3000 * np.sin(2 * np.pi * 220 * t) + 1500 * np.sin(2 * np.pi * 900 * t + 1.0)
    x += rng.normal(0, 300, n)
    return np.clip(x, -32768, 32767).astype(np.int16)


def test_conditioning_strips_dc_bias_byte_exact():
    """Biased speech and clean speech come out byte-equal: the fix is a no-op
    on unbiased audio and exactly cancels a constant offset."""
    src = _speech_like()
    biased = (src.astype(np.int32) + 5757).astype(np.int16)
    got, _ = _condition_s16le(biased.tobytes(), 16000, (0.0, None))
    clean, _ = _condition_s16le(src.tobytes(), 16000, (0.0, None))
    assert got == clean
    assert abs(np.frombuffer(got, dtype=np.int16).mean()) <= 100


def test_conditioning_keeps_silence_silent():
    out, _ = _condition_s16le(np.full(3200, 5757, dtype=np.int16).tobytes(), 16000, (0.0, None))
    assert out.count(0) == len(out)


def test_condition_handles_odd_bytes():
    """An odd-length byte chunk must not raise ValueError in frombuffer."""
    odd_data = b"\x00\x01\x02\x03\x04"
    out, _ = _condition_s16le(odd_data, 16000, (0.0, None))
    assert len(out) == 4  # Truncated to even bytes and processed


@pytest.mark.asyncio
async def test_on_chunk_cleared_on_session_end():
    capture = PTTCapture()
    capture.on_chunk = lambda b: None
    capture.stop_recording = AsyncMock(return_value=b"")
    runtime = MagicMock()
    runtime.is_idle = True
    pipeline = VoicePipeline(
        runtime=runtime,
        stt=MockSpeechToText(predefined_transcriptions=[""]),
        player=AudioPlayer(tts=MockTextToSpeech(), sink=MockAudioSink()),
        capture=capture,
    )
    pipeline._live_stt = MagicMock()
    pipeline._live_stt.finalize = AsyncMock(return_value="")
    await pipeline.stop_recording_and_process()
    await asyncio.sleep(0)
    assert capture.on_chunk is None


@pytest.mark.asyncio
async def test_warm_is_rate_limited_to_60s():
    model = LitellmModel(model_name="fake-model")
    with patch("litellm.acompletion", new=AsyncMock(return_value={})) as ac:
        await model.warm()
        await model.warm()
        assert ac.await_count == 1


@pytest.mark.asyncio
async def test_warm_never_breaks_recording():
    model = LitellmModel(model_name="fake-model")
    with patch("litellm.acompletion", new=AsyncMock(side_effect=RuntimeError("boom"))):
        await model.warm()
