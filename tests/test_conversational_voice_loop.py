import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from verbix.core.events import TextDelta, TurnComplete
from verbix.runtime.floor import FloorArbiter, FloorState
from verbix.voice.base import MockAudioSink, MockTextToSpeech
from verbix.voice.pipeline import VoicePipeline
from verbix.voice.player import AudioPlayer


class FakeCapture:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.on_chunk = None

    async def start_recording(self):
        self.is_recording = True

    async def stop_recording(self):
        self.is_recording = False
        return b""


class FakeLiveSession:
    def __init__(self, text: str, speech_duration: float = 0.5):
        self.fed_chunks = []
        self.text = text
        self.speech_duration = speech_duration
        self.on_endpoint = None

    def feed(self, pcm: bytes):
        self.fed_chunks.append(pcm)

    def cancel(self):
        self.cancelled = True

    async def finalize(self):
        return self.text


class FakeSTT:
    def __init__(self):
        self.sessions = []
        self._texts = ["Hello there", "Second thought"]

    def open_session(self, sample_rate: int = 16000, on_endpoint=None):
        text = self._texts.pop(0) if self._texts else ""
        sess = FakeLiveSession(text)
        sess.on_endpoint = on_endpoint
        self.sessions.append(sess)
        return sess


    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000):
        return "transcribed"


@pytest.mark.asyncio
async def test_continuous_conversational_loop_cycles_turns():
    """Verify continuous conversational loop automatically executes turns on speech endpoint,
    pauses STT feeding while speaking, resumes listening after playback drains,
    and terminates when toggle() closes the session."""
    runtime = MagicMock()
    runtime.floor = FloorArbiter()
    runtime.is_idle = True
    runtime.user_active = False

    turn_submissions = []

    async def fake_submit(text, output):
        turn_submissions.append(text)
        # Emit audio to sink to simulate assistant speaking
        await output.emit(TextDelta(text="Assistant response audio"))
        while output.is_playing:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        return "assistant response"

    runtime.submit = AsyncMock(side_effect=fake_submit)
    runtime.cancel_turn = AsyncMock()

    sink = MockAudioSink(playback_delay=0.05)
    player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)
    stt = FakeSTT()
    capture = FakeCapture()

    pipeline = VoicePipeline(
        runtime=runtime,
        stt=stt,
        player=player,
        capture=capture,
    )

    # 1. Start continuous session
    res = await pipeline.start_continuous_session()
    assert res.get("action") == "continuous_started"
    assert pipeline.is_continuous_active is True
    assert capture.is_recording is True
    assert len(stt.sessions) == 1
    sess1 = stt.sessions[0]

    # Feed some speech bytes to session 1
    capture.on_chunk(b"chunk1")
    assert b"chunk1" in sess1.fed_chunks

    # 2. Trigger speech endpoint (user finished speaking first utterance)
    assert sess1.on_endpoint is not None
    sess1.on_endpoint()

    # Wait for turn 1 to start executing
    await asyncio.sleep(0.02)
    assert turn_submissions == ["Hello there"]

    # STT feed MUST be paused while assistant is speaking (no self-echo)
    # Even if capture receives microphone bytes while assistant speaks, session 1 must NOT receive them
    capture.on_chunk(b"echo_during_playback")
    assert b"echo_during_playback" not in sess1.fed_chunks

    # Wait for audio playback to completely drain and loop to cycle back to listening
    while pipeline.turn_output.is_playing:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.10)
    assert len(stt.sessions) == 2
    sess2 = stt.sessions[1]
    assert b"echo_during_playback" not in sess2.fed_chunks

    # STT feed is unpaused: user speaks second utterance
    capture.on_chunk(b"chunk2")
    assert b"chunk2" in sess2.fed_chunks

    # Trigger second endpoint
    sess2.on_endpoint()
    await asyncio.sleep(0.02)
    assert turn_submissions == ["Hello there", "Second thought"]

    # Wait for turn 2 playback to drain
    while pipeline.turn_output.is_playing:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.20)

    # 3. Stop continuous session (user pressed Enter / toggle)
    stop_res = await pipeline.stop_continuous_session()
    assert stop_res.get("action") == "continuous_stopped"
    assert pipeline.is_continuous_active is False
    assert capture.is_recording is False
    assert runtime.floor.state == FloorState.IDLE


@pytest.mark.asyncio
async def test_continuous_loop_recovers_from_turn_error_and_does_not_resume_prematurely():
    """Verify that a turn exception does not kill continuous loop and mic only resumes after playback."""
    runtime = MagicMock()
    runtime.floor = FloorArbiter()
    runtime.is_idle = True
    runtime.user_active = False

    turn_count = 0

    async def exploding_then_successful_submit(text, output):
        nonlocal turn_count
        turn_count += 1
        if turn_count == 1:
            raise RuntimeError("Model provider explosion")
        await output.emit(TextDelta(text="Recovered output"))
        return "ok"

    runtime.submit = AsyncMock(side_effect=exploding_then_successful_submit)
    runtime.cancel_turn = AsyncMock()

    sink = MockAudioSink(playback_delay=0.1)
    player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)
    stt = FakeSTT()
    capture = FakeCapture()

    pipeline = VoicePipeline(
        runtime=runtime,
        stt=stt,
        player=player,
        capture=capture,
    )

    await pipeline.start_continuous_session()
    sess1 = stt.sessions[0]

    # Turn 1: Explodes -> error speech played (even if speak raises, loop survives) -> listens for Turn 2
    player.speak = AsyncMock(side_effect=RuntimeError("TTS failure"))
    sess1.on_endpoint()
    await asyncio.sleep(0.05)

    while pipeline.turn_output.is_playing or pipeline.player.is_playing:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.20)  # Drain tail delay

    # Loop has smoothly recovered despite turn error and speak error, and opened session 2
    assert len(stt.sessions) == 2
    sess2 = stt.sessions[1]

    # Turn 2: Succeeds
    sess2.on_endpoint()
    await asyncio.sleep(0.05)
    assert turn_count == 2

    await pipeline.stop_continuous_session()


@pytest.mark.asyncio
async def test_endpoint_signaled_from_background_thread_wakes_loop():
    """Verify that calling on_endpoint from a non-asyncio thread (audio capture thread)
    safely signals the captured event loop via call_soon_threadsafe and wakes the loop."""
    import threading

    runtime = MagicMock()
    runtime.floor = FloorArbiter()
    runtime.is_idle = True
    runtime.user_active = False

    turn_triggered = asyncio.Event()

    async def fake_submit(text, output):
        turn_triggered.set()
        return "response"

    runtime.submit = AsyncMock(side_effect=fake_submit)
    runtime.cancel_turn = AsyncMock()

    sink = MockAudioSink()
    player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)
    stt = FakeSTT()
    capture = FakeCapture()

    pipeline = VoicePipeline(
        runtime=runtime,
        stt=stt,
        player=player,
        capture=capture,
    )

    await pipeline.start_continuous_session()
    sess = stt.sessions[0]
    assert sess.on_endpoint is not None

    # Call on_endpoint strictly from a background worker thread (simulating audio callback)
    def _call_from_thread():
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        sess.on_endpoint()

    th = threading.Thread(target=_call_from_thread)
    th.start()
    th.join()

    # The main asyncio loop should be woken by call_soon_threadsafe
    await asyncio.wait_for(turn_triggered.wait(), timeout=2.0)
    assert turn_triggered.is_set()

    await pipeline.stop_continuous_session()


@pytest.mark.asyncio
async def test_announcement_gates_mic_across_multi_sentence_gaps_and_tail_delay():
    """Verify that during an announcement (multi-sentence):
    1. Mic is gated while floor is SPEAKING before audio starts (while tokens generate).
    2. Mic is gated during playback of sentence 1.
    3. Mic does NOT reopen in gaps between sentences of the announcement.
    4. Mic is gated during playback of sentence 2.
    5. Mic remains gated during post-playback tail delay while floor is still SPEAKING.
    6. Mic reopens for user speech once floor returns to IDLE."""
    runtime = MagicMock()
    runtime.floor = FloorArbiter()
    runtime.is_idle = True
    runtime.user_active = False

    sink = MockAudioSink(playback_delay=0.03)
    player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)
    stt = FakeSTT()
    capture = FakeCapture()

    pipeline = VoicePipeline(
        runtime=runtime,
        stt=stt,
        player=player,
        capture=capture,
    )

    await pipeline.start_continuous_session()
    assert len(stt.sessions) == 1
    sess1 = stt.sessions[0]

    # 1. Announcement starts executing: floor acquired (SPEAKING), but audio not started yet
    runtime.floor.set_state(FloorState.SPEAKING)
    capture.on_chunk(b"chunk_before_audio_starts")
    assert b"chunk_before_audio_starts" not in sess1.fed_chunks

    # 2. Sentence 1 begins playing out of sink
    await pipeline.turn_output.emit(TextDelta(text="I checked your workspace directory. "))
    assert pipeline.turn_output.is_playing is True
    capture.on_chunk(b"chunk_during_sentence_1")
    assert b"chunk_during_sentence_1" not in sess1.fed_chunks

    # 3. Inter-sentence gap: LLM is generating sentence 2, floor is SPEAKING
    await asyncio.sleep(0.05)
    assert runtime.floor.is_speaking is True
    capture.on_chunk(b"chunk_during_intersentence_gap")
    assert b"chunk_during_intersentence_gap" not in sess1.fed_chunks

    # 4. Sentence 2 begins playing
    await pipeline.turn_output.emit(TextDelta(text="All desktop files have been saved. "))
    capture.on_chunk(b"chunk_during_sentence_2")
    assert b"chunk_during_sentence_2" not in sess1.fed_chunks

    # 5. Turn completes generation and plays out all audio
    await pipeline.turn_output.emit(
        TurnComplete(
            final_text="I checked your workspace directory. All desktop files have been saved."
        )
    )
    while pipeline.turn_output.is_playing:
        await asyncio.sleep(0.01)

    # 6. Post-playback tail delay in Runtime.submit keeps floor in SPEAKING even after audio drained
    assert pipeline.turn_output.is_playing is False
    assert runtime.floor.is_speaking is True
    capture.on_chunk(b"chunk_during_tail_delay")
    assert b"chunk_during_tail_delay" not in sess1.fed_chunks

    # 7. Announcement completes and floor returns to IDLE after tail delay elapses
    runtime.floor.set_state(FloorState.IDLE)
    await pipeline.stop_continuous_session()


@pytest.mark.asyncio
async def test_empty_stt_resumes_unspoken_draft():
    """When VAD triggers but STT decodes zero words (noise/cough), unspoken draft is resumed."""
    from verbix.voice.output import InterruptionSnapshot

    runtime = MagicMock()
    runtime.floor = FloorArbiter()
    runtime.is_idle = True
    runtime.submit = AsyncMock()
    runtime.cancel_turn = AsyncMock()

    sink = MockAudioSink()
    player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)
    stt = FakeSTT()
    stt._texts = [""]  # STT finalizes with empty string (cough/noise)
    capture = FakeCapture()

    pipeline = VoicePipeline(
        runtime=runtime,
        stt=stt,
        player=player,
        capture=capture,
    )

    await pipeline.start_continuous_session()
    sess = stt.sessions[0]

    # Simulate an interruption that left an unspoken draft
    pipeline._speech_output.last_interruption = InterruptionSnapshot(
        heard_text="Spoken sentence one.",
        unspoken_draft="Remaining sentence two.",
    )

    # User finishes non-speech sound (cough), triggering endpoint
    sess.on_endpoint()
    await asyncio.sleep(0.05)

    # Runtime submit was NOT called (no unnecessary LLM round trip)
    runtime.submit.assert_not_called()

    # The unspoken draft was resumed via speech output
    while pipeline.turn_output.is_playing:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)

    assert "Remaining sentence two." in pipeline.turn_output.delivered_text()
    assert pipeline._speech_output.last_interruption is None

    await pipeline.stop_continuous_session()


@pytest.mark.asyncio
async def test_pipeline_logs_user_transcript_continuous_and_ptt():
    """Pipeline logs finalized recognized transcripts with no duplicates, and ignores empty ones."""
    runtime = MagicMock()
    runtime.floor = FloorArbiter()
    runtime.is_idle = True
    runtime.submit = AsyncMock(return_value="spoken response")

    sink = MockAudioSink()
    player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)
    stt = FakeSTT()
    stt._texts = ["What time is it?", ""]
    capture = FakeCapture()

    printer = MagicMock()
    printer.print_user = MagicMock()

    pipeline = VoicePipeline(
        runtime=runtime,
        stt=stt,
        player=player,
        capture=capture,
        printer=printer,
    )

    # 1. Continuous mode: non-empty transcript is logged once
    await pipeline.start_continuous_session()
    sess1 = stt.sessions[0]
    sess1.on_endpoint()
    await asyncio.sleep(0.05)

    printer.print_user.assert_called_once_with("What time is it?", duration=0.5)
    printer.print_user.reset_mock()

    # 2. Continuous mode: empty transcript is NOT logged
    sess2 = stt.sessions[1]
    sess2.on_endpoint()
    await asyncio.sleep(0.05)

    printer.print_user.assert_not_called()
    await pipeline.stop_continuous_session()

    # 3. PTT mode: non-empty transcript is logged once
    stt.open_session = None
    stt.transcribe = AsyncMock(return_value="Turn the light off")
    capture.stop_recording = AsyncMock(return_value=b"\x00" * 3200)

    res = await pipeline.stop_recording_and_process()
    assert res["transcription"] == "Turn the light off"
    printer.print_user.assert_called_once_with("Turn the light off", duration=0.1)
    printer.print_user.reset_mock()

    # 4. PTT mode: empty audio / empty transcript is NOT logged
    stt.transcribe = AsyncMock(return_value="")
    capture.stop_recording = AsyncMock(return_value=b"")
    res_empty = await pipeline.stop_recording_and_process()
    assert res_empty["transcription"] == ""
    printer.print_user.assert_not_called()



