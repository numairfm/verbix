import asyncio

import pytest

from verbix.core.events import TextDelta, TurnComplete
from verbix.voice.base import MockAudioSink, MockTextToSpeech
from verbix.voice.output import SpeechOutput


@pytest.mark.asyncio
async def test_emit_does_not_block_on_playback():
    out = SpeechOutput(tts=MockTextToSpeech(), sink=MockAudioSink(playback_delay=5.0))
    await out.emit(TextDelta(text="Hello world. This is a test sentence here."))
    await asyncio.sleep(0.05)
    assert out.sink.played_chunks  # audio started without emit awaiting it


@pytest.mark.asyncio
async def test_interrupt_discards_stale_chunks():
    out = SpeechOutput(tts=MockTextToSpeech(), sink=MockAudioSink(playback_delay=0.2))
    await out.emit(TextDelta(text="First sentence here spoken. Second sentence here spoken."))
    await asyncio.sleep(0.05)
    out.interrupt()
    n = len(out.sink.played_chunks)
    await out.emit(TextDelta(text="New sentence here spoken."))
    await asyncio.sleep(0.3)
    assert out.delivered_text().startswith("New")


@pytest.mark.asyncio
async def test_turn_complete_never_respeaks_delivered_text():
    out = SpeechOutput(tts=MockTextToSpeech(), sink=MockAudioSink())
    await out.emit(TextDelta(text="Hello world. This is spoken."))
    await asyncio.sleep(0.2)
    n = len(out.tts.synthesized_texts)
    await out.emit(TurnComplete(final_text="Hello world. This is spoken."))
    await asyncio.sleep(0.2)
    assert len(out.tts.synthesized_texts) == n
