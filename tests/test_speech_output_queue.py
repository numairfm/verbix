import asyncio
import pytest
from verbix.core.events import TextDelta, TurnComplete
from verbix.voice.output import SpeechOutput

@pytest.mark.asyncio
async def test_consecutive_utterances_play_in_strict_fifo_order_during_active_playback():
    played_chunks = []
    first_playing = asyncio.Event()

    class DelayedTTS:
        async def synthesize(self, text):
            if "First" in text:
                first_playing.set()
                # Simulate slow first sentence streaming
                await asyncio.sleep(0.08)
            yield f"audio({text})"

    class FakeSink:
        def __init__(self):
            self.stopped = False
        def stop(self):
            self.stopped = True
        async def play_stream(self, generator):
            async for chunk in generator:
                played_chunks.append(chunk)

    sink = FakeSink()
    out = SpeechOutput(tts=DelayedTTS(), sink=sink)

    # Dispatch first utterance
    await out.emit(TextDelta(text="First sentence."))
    await out.emit(TurnComplete(final_text="First sentence."))

    # Wait until first utterance is actively synthesizing/playing
    await asyncio.wait_for(first_playing.wait(), timeout=1.0)

    # Dispatch second utterance WHILE first is actively playing
    await out.emit(TextDelta(text="Second sentence."))
    await out.emit(TurnComplete(final_text="Second sentence."))

    # Wait for both utterances to drain completely
    for _ in range(50):
        if len(played_chunks) >= 2:
            break
        await asyncio.sleep(0.02)

    # MUST play in exact FIFO sequence: First, then Second
    assert played_chunks == ["audio(First sentence.)", "audio(Second sentence.)"]
    assert sink.stopped is False
