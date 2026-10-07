"""Prefetch slice tests: next sentence synthesizes during current playback,
and barge-in never plays (or records) prefetched audio."""

import asyncio

import pytest

from verbix.core.events import TextDelta, TurnComplete
from verbix.voice.output import SpeechOutput


class GateTTS:
    """Fake TTS with per-text start/release control."""

    def __init__(self):
        self.started = {}
        self.release = {}
        self.calls = []

    async def synthesize(self, text):
        self.calls.append(text)
        ev = self.started.setdefault(text, asyncio.Event())
        ev.set()
        await self.release.setdefault(text, asyncio.Event()).wait()
        yield f"audio:{text}".encode()


class GateSink:
    """Fake sink: records chunks, blocks each chunk on its gate."""

    def __init__(self):
        self.played = []
        self.gate = asyncio.Event()
        self.stopped = False
        self.sample_rate = 24000
        self.audio_format = "f32"

    async def play(self, data):
        self.played.append(data)
        return True

    async def play_stream(self, chunks):
        async for c in chunks:
            self.played.append(c)
            await self.gate.wait()
        return not self.stopped

    def stop(self):
        self.stopped = True


async def _wait_for(cond, timeout=2.0):
    import time

    start = time.monotonic()
    while not cond():
        assert time.monotonic() - start < timeout, "timed out waiting"
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_prefetch_starts_next_sentence_during_playback():
    tts, sink = GateTTS(), GateSink()
    tts.release = {"Sentence one here.": asyncio.Event(), "Sentence two here.": asyncio.Event()}
    out = SpeechOutput(tts=tts, sink=sink)
    await out.emit(TextDelta(text="Sentence one here. "))
    await out.emit(TextDelta(text="Sentence two here. "))
    await out.emit(TurnComplete(final_text=""))
    await _wait_for(lambda: "Sentence one here." in tts.started)
    tts.release["Sentence one here."].set()
    await _wait_for(lambda: sink.played != [])
    # Sentence one is held inside the sink; sentence two must already be synthesizing.
    await _wait_for(lambda: "Sentence two here." in tts.started)
    assert tts.calls[0] == "Sentence one here."
    tts.release["Sentence two here."].set()
    sink.gate.set()
    await _wait_for(lambda: len(sink.played) >= 2)
    assert sink.played[1] == b"audio:Sentence two here."


@pytest.mark.asyncio
async def test_barge_in_never_plays_or_records_prefetched():
    tts, sink = GateTTS(), GateSink()
    tts.release = {"Sentence one here.": asyncio.Event(), "Sentence two here.": asyncio.Event()}
    out = SpeechOutput(tts=tts, sink=sink)
    await out.emit(TextDelta(text="Sentence one here. "))
    await out.emit(TextDelta(text="Sentence two here. "))
    await out.emit(TurnComplete(final_text=""))
    await _wait_for(lambda: "Sentence one here." in tts.started)
    tts.release["Sentence one here."].set()
    await _wait_for(lambda: sink.played != [])
    tts.release["Sentence two here."].set()
    await _wait_for(lambda: len(tts.calls) == 2)
    await asyncio.sleep(0.05)
    heard = out.delivered_text()
    out.interrupt()
    sink.gate.set()
    await asyncio.sleep(0.1)
    assert sink.played == [b"audio:Sentence one here."]
    assert heard == "Sentence one here."


@pytest.mark.asyncio
async def test_stale_buffered_sentence_never_yielded_after_epoch_flip():
    # Stages the exact race: the flip wins over the pump (pump NOT cancelled),
    # so only the new epoch re-check can stop the buffered sentence.
    tts, sink = GateTTS(), GateSink()
    tts.release = {"Sentence one here.": asyncio.Event(), "Sentence two here.": asyncio.Event()}
    out = SpeechOutput(tts=tts, sink=sink)
    await out.emit(TextDelta(text="Sentence one here. "))
    await out.emit(TextDelta(text="Sentence two here. "))
    await out.emit(TurnComplete(final_text=""))
    await _wait_for(lambda: "Sentence one here." in tts.started)
    tts.release["Sentence one here."].set()
    await _wait_for(lambda: sink.played != [])
    tts.release["Sentence two here."].set()
    await _wait_for(lambda: len(tts.calls) == 2)
    await asyncio.sleep(0.05)
    out._epoch += 1
    sink.gate.set()
    await asyncio.sleep(0.1)
    assert sink.played == [b"audio:Sentence one here."]
    assert out.delivered_text() == "Sentence one here."
