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
    tts = MockTextToSpeech()
    sink = MockAudioSink(playback_delay=0.2)
    out = SpeechOutput(tts=tts, sink=sink)
    await out.emit(TextDelta(text="First sentence here spoken. Second sentence here spoken."))
    for _ in range(100):
        if tts.synthesized_texts:
            break
        await asyncio.sleep(0.01)
    out.interrupt()
    n_texts, n_chunks = len(tts.synthesized_texts), len(sink.played_chunks)
    await out.emit(TextDelta(text="New sentence here spoken."))
    for _ in range(200):
        if not out.is_playing and len(sink.played_chunks) >= n_chunks + 2:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)  # let any stale audio surface
    assert out.delivered_text().startswith("New")
    assert len(sink.played_chunks) == n_chunks + 2  # only new-epoch audio
    assert all("First" not in t and "Second" not in t for t in tts.synthesized_texts[n_texts:])


@pytest.mark.asyncio
async def test_backendless_turn_completes_silently(monkeypatch):
    import shutil

    from verbix.voice.sinks import SubprocessAudioSink

    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: None)
    out = SpeechOutput(tts=MockTextToSpeech(), sink=SubprocessAudioSink())
    await out.emit(TextDelta(text="Hello world. This is spoken aloud."))
    await out.emit(TurnComplete(final_text="Hello world. This is spoken aloud."))
    await asyncio.sleep(0.2)
    assert not out.is_playing  # no backend, no crash, pump drained


@pytest.mark.asyncio
async def test_turn_complete_never_respeaks_delivered_text():
    out = SpeechOutput(tts=MockTextToSpeech(), sink=MockAudioSink())
    await out.emit(TextDelta(text="Hello world. This is spoken."))
    await asyncio.sleep(0.2)
    n = len(out.tts.synthesized_texts)
    await out.emit(TurnComplete(final_text="Hello world. This is spoken."))
    await asyncio.sleep(0.2)
    assert len(out.tts.synthesized_texts) == n


@pytest.mark.asyncio
async def test_interrupt_snapshots_heard_and_unspoken_draft():
    """Interrupting multi-sentence speech captures both heard text and unspoken draft."""
    tts = MockTextToSpeech()
    sink = MockAudioSink(playback_delay=0.05)
    out = SpeechOutput(tts=tts, sink=sink)

    text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
    await out.emit(TextDelta(text=text))
    await out.emit(TurnComplete(final_text=text))

    # Wait until first sentence is delivered to sink
    for _ in range(100):
        if len(out._delivered) >= 1:
            break
        await asyncio.sleep(0.01)

    assert len(out._delivered) >= 1
    snapshot = out.interrupt()
    assert snapshot is not None
    assert out.last_interruption is snapshot
    assert "First sentence here." in snapshot.heard_text
    assert "Fourth sentence here." in snapshot.unspoken_draft
    assert snapshot.heard_text not in snapshot.unspoken_draft

