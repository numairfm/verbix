import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from verbix.agent.models.litellm_model import LitellmModel
from verbix.voice.base import MockAudioSink, MockSpeechToText, MockTextToSpeech
from verbix.voice.capture import PTTCapture
from verbix.voice.pipeline import VoicePipeline
from verbix.voice.player import AudioPlayer


def test_capture_reads_20ms_frames():
    assert PTTCapture(sample_rate=16000).frame_bytes == 640


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
