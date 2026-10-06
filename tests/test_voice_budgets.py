import asyncio
import time
import pytest
from verbix.voice.base import MockAudioSink, MockTextToSpeech
from verbix.voice.output import SpeechOutput
from verbix.core.events import TextDelta

@pytest.mark.asyncio
async def test_emit_returns_without_awaiting_playback():
    out = SpeechOutput(tts=MockTextToSpeech(), sink=MockAudioSink(playback_delay=2.0))
    t0 = time.monotonic()
    await out.emit(TextDelta(text="Hello world. This is spoken aloud."))
    assert time.monotonic() - t0 < 0.5  # L1: emit never blocks on audio

@pytest.mark.asyncio
async def test_interrupt_silences_within_60ms():
    out = SpeechOutput(tts=MockTextToSpeech(), sink=MockAudioSink(playback_delay=2.0))
    await out.emit(TextDelta(text="Hello world. This is spoken aloud."))
    await asyncio.sleep(0.05)
    t0 = time.monotonic()
    out.interrupt()
    assert time.monotonic() - t0 < 0.06  # subprocess-class bound (§5.2)
