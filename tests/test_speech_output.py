import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from verbix.core.events import TextDelta, TurnComplete
from verbix.runtime.floor import FloorArbiter, FloorState
from verbix.voice.base import MockAudioSink, MockTextToSpeech
from verbix.voice.output import SpeechOutput
from verbix.voice.pipeline import VoicePipeline
from verbix.voice.player import AudioPlayer


class _FakeCapture:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.on_chunk = None

    async def start_recording(self):
        self.is_recording = True

    async def stop_recording(self):
        self.is_recording = False
        return b""


class _RecordingAEC:
    """AEC stand-in: keeps pending reference bytes and counts resets."""

    is_available = False

    def __init__(self) -> None:
        self.refs: list[bytes] = []
        self.resets = 0
        self._ref_buffer = bytearray(b"pending")

    def feed_reference(
        self,
        audio: bytes,
        source_sample_rate: int = 24000,
        audio_format: str = "f32",
    ) -> None:
        self._ref_buffer.extend(audio)
        self.refs.append(audio)

    def reset(self) -> None:
        self.resets += 1
        self._ref_buffer.clear()


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
async def test_subprocess_audio_sink_played_samples(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from verbix.voice.sinks import SubprocessAudioSink

    monkeypatch.setattr("shutil.which", lambda _b: "/bin/true")
    sink = SubprocessAudioSink(sample_rate=16000, channels=1, audio_format="s16")
    assert sink.played_samples == 0

    chunk = b"\x00" * 32000

    async def _chunks():
        yield chunk
        await asyncio.sleep(0.05)

    mock_proc = AsyncMock()
    mock_proc.returncode = None
    mock_proc.kill = MagicMock()
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.drain = AsyncMock()
    mock_proc.stdin.wait_closed = AsyncMock()
    mock_proc.wait = AsyncMock(return_value=0)
    monkeypatch.setattr("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc))

    task = asyncio.create_task(sink.play_stream(_chunks()))
    await asyncio.sleep(0.02)
    assert 0 <= sink.played_samples <= 16000
    sink.stop()
    await task
    assert sink.played_samples >= 0


@pytest.mark.asyncio
async def test_gap_silence_reaches_reference_hook():
    """Slice 2: inter-sentence gap silence must reach the on_audio reference hook."""
    seen: list[bytes] = []
    out = SpeechOutput(
        tts=MockTextToSpeech(),
        sink=MockAudioSink(playback_delay=0.01),
        on_audio=seen.append,
    )
    text = "First sentence here. Second sentence here."
    await out.emit(TextDelta(text=text))
    await out.emit(TurnComplete(final_text=text))
    for _ in range(200):
        if not out.is_playing:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    assert bytes(int(24000 * 0.15) * 4) in seen


@pytest.mark.asyncio
async def test_idle_earcon_cue_reaches_reference_hook():
    """Slice 2: an idle earcon cue must feed reference exactly what was heard."""
    seen: list[bytes] = []
    sink = MockAudioSink()
    out = SpeechOutput(tts=MockTextToSpeech(), sink=sink, on_audio=seen.append)
    await out.play_cue()
    assert len(seen) == 1
    assert len(seen[0]) == int(24000 * 0.05) * 4
    assert seen[0] == sink.played_chunks[0]


@pytest.mark.asyncio
async def test_queued_earcon_cue_reaches_reference_hook():
    """Slice 2: an earcon queued behind speech must also reach the reference hook."""
    seen: list[bytes] = []
    sink = MockAudioSink(playback_delay=0.2)
    out = SpeechOutput(tts=MockTextToSpeech(), sink=sink, on_audio=seen.append)
    await out.emit(TextDelta(text="Sentence being spoken aloud right now. "))
    await asyncio.sleep(0.05)
    assert out.is_playing
    await out.play_cue()
    for _ in range(200):
        if not out.is_playing:
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.05)
    assert any(len(c) == int(24000 * 0.05) * 4 for c in seen)


async def _listening_pipeline_with_refs():
    """Pipeline with pending reference audio after speech has flowed to the hook."""
    runtime = MagicMock()
    runtime.floor = FloorArbiter()
    runtime.is_idle = False
    runtime.user_active = False
    runtime.cancel_turn = AsyncMock()
    aec = _RecordingAEC()
    sink = MockAudioSink(playback_delay=0.01)
    player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)
    stt = MagicMock()
    stt.open_session = MagicMock(return_value=MagicMock())
    pipeline = VoicePipeline(
        runtime=runtime, stt=stt, player=player, capture=_FakeCapture(), aec=aec
    )
    pipeline._is_continuous_active = True
    pipeline._loop = asyncio.get_running_loop()
    await pipeline._setup_turn_listening()
    text = "First sentence here. Second sentence here."
    await pipeline.turn_output.emit(TextDelta(text=text))
    await pipeline.turn_output.emit(TurnComplete(final_text=text))
    for _ in range(100):
        if aec.refs:
            break
        await asyncio.sleep(0.01)
    assert aec.refs, "speech audio never reached the reference hook"
    return pipeline, aec


@pytest.mark.asyncio
async def test_interrupt_preserves_pending_reference():
    """Slice 2: barge-in interrupt must not wipe reference whose echo is airborne."""
    pipeline, aec = await _listening_pipeline_with_refs()
    pipeline.runtime.floor.set_state(FloorState.SPEAKING)
    pipeline._interrupt_active_speech()
    await asyncio.sleep(0.02)
    assert aec.resets == 0
    assert aec.refs
    await pipeline.stop_continuous_session()


@pytest.mark.asyncio
async def test_idle_listening_preserves_pending_reference():
    """Slice 2: mic chunks while listening must not wipe pending reference."""
    pipeline, aec = await _listening_pipeline_with_refs()
    while pipeline.turn_output.is_playing:
        await asyncio.sleep(0.01)
    pipeline.runtime.floor.set_state(FloorState.LISTENING)
    pipeline.capture.on_chunk(b"\x00" * 320)
    assert aec.resets == 0
    assert aec.refs
    await pipeline.stop_continuous_session()
