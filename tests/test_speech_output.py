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


@pytest.mark.asyncio
async def test_playback_terminates_when_cue_queued_behind_end():
    """Playback must terminate and is_playing become False even if a cue is enqueued behind end."""
    tts = MockTextToSpeech()
    sink = MockAudioSink(playback_delay=0.01)
    out = SpeechOutput(tts=tts, sink=sink)

    epoch = out._epoch
    out._frames.put_nowait((epoch, "text", "Sentence to speak."))
    out._frames.put_nowait((epoch, "end", ""))
    out._frames.put_nowait((epoch, "cue", b"\x00" * 160))
    out._ensure_pump()
    out._top_up()

    for _ in range(50):
        if not out.is_playing:
            break
        await asyncio.sleep(0.01)

    assert not out.is_playing, "Pump hung because end frame was dropped before cue frame"


@pytest.mark.asyncio
async def test_interrupt_snapshot_distinguishes_played_interrupted_and_unplayed():
    """Snapshot distinguishes confirmed played, partially audible, and unplayed sentences."""
    tts = MockTextToSpeech()

    class TrackingSink(MockAudioSink):
        def __init__(self):
            super().__init__(playback_delay=0.01)
            self._played = 0

        @property
        def played_samples(self):
            return self._played

    sink = TrackingSink()
    out = SpeechOutput(tts=tts, sink=sink)

    text = "First sentence here. Second sentence here. Third sentence here."
    await out.emit(TextDelta(text=text))
    await out.emit(TurnComplete(final_text=text))

    for _ in range(100):
        if len(out._sent_spans) == 3:
            break
        await asyncio.sleep(0.01)

    assert len(out._sent_spans) == 3
    start2 = out._sent_spans[1][1]
    end2 = out._sent_spans[1][2]
    sink._played = (start2 + end2) // 2

    snap = out.interrupt()
    assert snap is not None
    assert snap.confirmed_heard == "First sentence here."
    assert snap.interrupted_sentence == "Second sentence here."
    assert snap.unplayed_text == "Third sentence here."
    assert snap.heard_text == "First sentence here."
    assert snap.unspoken_draft == "Second sentence here. Third sentence here."


@pytest.mark.asyncio
async def test_interrupt_snapshot_relative_to_turn_baseline():
    """Consecutive turns must not leak prior-turn sample counts into new turn snapshots."""
    tts = MockTextToSpeech()

    class PersistentTrackingSink(MockAudioSink):
        def __init__(self):
            super().__init__(playback_delay=0.01)
            self._played = 0

        @property
        def played_samples(self):
            return self._played

    sink = PersistentTrackingSink()
    out = SpeechOutput(tts=tts, sink=sink)

    # Turn 1: 100,000 samples played in turn 1
    sink._played = 100_000
    out.reset()

    # Turn 2: Starts with baseline of 100,000 samples
    text = "First story sentence. Second story sentence. Third story sentence."
    await out.emit(TextDelta(text=text))
    await out.emit(TurnComplete(final_text=text))

    for _ in range(100):
        if len(out._sent_spans) == 3:
            break
        await asyncio.sleep(0.01)

    assert len(out._sent_spans) == 3
    # Interruption happens 500 samples into the first sentence of turn 2
    # In turn 2 relative terms: played 500 samples
    # In cumulative sink terms: played 100_500 samples
    sink._played = 100_500

    snap = out.interrupt()
    assert snap is not None
    # Sentence 1 was interrupted mid-playback, NOT fully heard!
    assert snap.confirmed_heard == ""
    assert snap.interrupted_sentence == "First story sentence."
    assert "Second story sentence." in snap.unplayed_text
    assert "Third story sentence." in snap.unplayed_text


@pytest.mark.asyncio
async def test_clean_reset_leaves_no_interruption():
    """reset() must clear state without creating or setting last_interruption."""
    out = SpeechOutput(tts=MockTextToSpeech(), sink=MockAudioSink())
    out._delivered.append("Completed sentence.")
    out.reset()
    assert out.last_interruption is None
    assert not out._delivered


@pytest.mark.asyncio
async def test_playback_completion_clears_accounting():
    """When playback drains normally, accounting spans must be cleared for next turn."""
    out = SpeechOutput(tts=MockTextToSpeech(), sink=MockAudioSink(playback_delay=0.01))
    await out.emit(TextDelta(text="First sentence. Second sentence."))
    await out.emit(TurnComplete(final_text="First sentence. Second sentence."))
    for _ in range(50):
        if not out.is_playing:
            break
        await asyncio.sleep(0.02)
    assert not out.is_playing
    assert not out._sent_spans
    assert out._enqueued_samples == 0
    # Calling interrupt() when playback completed normally must not leave an interruption snapshot
    snap = out.interrupt()
    assert snap is None
    assert out.last_interruption is None


@pytest.mark.asyncio
async def test_speech_output_never_synthesizes_error_turn_complete_tokens():
    tts = MockTextToSpeech()
    sink = MockAudioSink(playback_delay=0.01)
    out = SpeechOutput(tts=tts, sink=sink)

    await out.emit(
        TurnComplete(
            final_text="RepeatedToolCall",
            exit_reason="repeated_tool_call",
            is_error=True,
        )
    )
    await asyncio.sleep(0.05)
    assert tts.synthesized_texts == []


@pytest.mark.asyncio
async def test_Idle_ptt_toggle_plays_start_and_stop_earcon_cues():
    from unittest.mock import AsyncMock, MagicMock

    from verbix.runtime.floor import FloorArbiter
    from verbix.voice.base import MockAudioSink, MockSpeechToText, MockTextToSpeech
    from verbix.voice.pipeline import VoicePipeline
    from verbix.voice.player import AudioPlayer

    rt = MagicMock()
    rt.floor = FloorArbiter()
    rt.is_idle = True
    rt.submit = AsyncMock(return_value="ok")

    cap = MagicMock()
    cap.sample_rate = 16000
    cap.is_recording = False

    async def start_rec():
        cap.is_recording = True

    async def stop_rec():
        cap.is_recording = False
        return b"\x00\x00" * 100

    cap.start_recording = AsyncMock(side_effect=start_rec)
    cap.stop_recording = AsyncMock(side_effect=stop_rec)

    sink = MockAudioSink(playback_delay=0.0)
    pipeline = VoicePipeline(
        runtime=rt,
        stt=MockSpeechToText(predefined_transcriptions=[""]),
        player=AudioPlayer(tts=MockTextToSpeech(), sink=sink),
        capture=cap,
    )

    cues: list[float] = []
    orig_play_cue = pipeline.turn_output.play_cue

    async def spy_cue(freq: float = 880.0):
        cues.append(freq)
        await orig_play_cue(freq)

    pipeline.turn_output.play_cue = spy_cue  # type: ignore[method-assign]

    r1 = await pipeline.toggle()
    assert r1["action"] == "recording"
    await asyncio.sleep(0.02)
    assert cues == [660.0]

    r2 = await pipeline.toggle()
    assert r2["action"] == "idle"
    await asyncio.sleep(0.02)
    assert cues == [660.0, 880.0]


@pytest.mark.asyncio
async def test_start_recording_prewarms_tts_when_not_loaded():
    from unittest.mock import AsyncMock, MagicMock

    from verbix.runtime.floor import FloorArbiter
    from verbix.voice.base import MockAudioSink, MockSpeechToText, MockTextToSpeech
    from verbix.voice.pipeline import VoicePipeline
    from verbix.voice.player import AudioPlayer

    rt = MagicMock()
    rt.floor = FloorArbiter()
    rt.is_idle = True

    cap = MagicMock()
    cap.is_recording = False
    cap.start_recording = AsyncMock()

    tts = MockTextToSpeech()
    tts.is_loaded = False  # type: ignore[attr-defined]
    loaded_called = False

    async def fake_load():
        nonlocal loaded_called
        loaded_called = True
        tts.is_loaded = True  # type: ignore[attr-defined]

    tts.load = fake_load  # type: ignore[attr-defined]

    pipeline = VoicePipeline(
        runtime=rt,
        stt=MockSpeechToText(),
        player=AudioPlayer(tts=tts, sink=MockAudioSink()),
        capture=cap,
    )
    await pipeline.start_recording()
    await asyncio.sleep(0.02)
    assert loaded_called is True

