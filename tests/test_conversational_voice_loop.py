import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from verbix.core.events import TextDelta, TurnComplete
from verbix.runtime.floor import FloorArbiter, FloorState
from verbix.voice.base import MockAudioSink, MockTextToSpeech
from verbix.voice.pipeline import VoicePipeline
from verbix.voice.player import AudioPlayer
from verbix.voice.sherpa_stt import STTSession


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
        self.sample_rate = 16000

    def feed(self, pcm: bytes):
        self.fed_chunks.append(pcm)

    @property
    def is_speech_detected(self) -> bool:
        return False

    def feed_barge_in(self, pcm: bytes) -> bool:
        return False

    def cancel(self):
        self.cancelled = True

    def reset(self):
        pass

    async def finalize(self):
        return self.text


class FakeSTT:
    def __init__(self):
        self.sessions = []
        self._texts = ["Hello there", "Second thought"]

    def open_session(self, sample_rate: int = 16000, on_endpoint=None):
        text = self._texts.pop(0) if self._texts else ""
        sess = FakeLiveSession(text)
        sess.sample_rate = sample_rate
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
    assert runtime.floor.state == FloorState.LISTENING
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


@pytest.mark.asyncio
async def test_barge_in_opens_fresh_session_with_lookback_and_cancels_old():
    """Slice 1: interrupt cancels echo-era session, opens fresh seeded with 300ms look-back.

    Echo-era (early) bytes must not survive into the fresh session; the
    utterance onset (buffered tail) plus post-interrupt speech must.
    """
    runtime = MagicMock()
    runtime.floor = FloorArbiter()
    runtime.is_idle = False
    runtime.user_active = False
    runtime.submit = AsyncMock(return_value="ok")
    runtime.cancel_turn = AsyncMock()

    sink = MockAudioSink(playback_delay=0.05)
    player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)
    stt = FakeSTT()
    stt._texts = ["old echo text", "new user text"]
    capture = FakeCapture()

    pipeline = VoicePipeline(
        runtime=runtime,
        stt=stt,
        player=player,
        capture=capture,
    )
    pipeline._is_continuous_active = True
    pipeline._loop = asyncio.get_running_loop()
    await pipeline._setup_turn_listening()
    assert len(stt.sessions) == 1
    old = stt.sessions[0]

    def _old_barge(chunk: bytes):
        old.fed_chunks.append(chunk)
        return b"C" in chunk

    old.feed_barge_in = _old_barge  # type: ignore[attr-defined]

    # Simulate assistant generating (floor SPEAKING, no audio yet) so mic takes barge path.
    runtime.floor.set_state(FloorState.SPEAKING)

    echo1 = b"A" * 10000
    echo2 = b"B" * 10000
    onset = b"C" * 2000
    post = b"D" * 2000

    capture.on_chunk(echo1)
    assert old.fed_chunks == [echo1]
    capture.on_chunk(echo2)
    assert old.fed_chunks == [echo1, echo2]
    # Onset triggers barge-in interrupt.
    capture.on_chunk(onset)
    await asyncio.sleep(0.02)

    # Fresh session opened with same endpoint hook; old cancelled, not abandoned.
    assert len(stt.sessions) == 2
    fresh = stt.sessions[1]
    assert fresh is not old
    assert getattr(old, "cancelled", False) is True
    assert fresh.on_endpoint is not None
    assert fresh.on_endpoint == pipeline._on_speech_endpoint

    # Fresh seeded with rolling 300ms look-back (last 9600 bytes): onset survives,
    # early echo-era evicted.
    joined_seed = b"".join(fresh.fed_chunks)
    assert b"C" in joined_seed
    assert b"A" not in joined_seed

    # Post-interrupt speech feeds the fresh session (simulate listening resumed).
    runtime.floor.set_state(FloorState.LISTENING)
    fresh.feed_barge_in = lambda chunk: (fresh.fed_chunks.append(chunk), False)[1]  # type: ignore[attr-defined]
    capture.on_chunk(post)
    joined = b"".join(fresh.fed_chunks)
    assert b"C" in joined
    assert b"D" in joined
    assert b"A" not in joined

    # Finalize yields post-interrupt text, not echo-era text.
    assert await fresh.finalize() == "new user text"
    assert await old.finalize() == "old echo text"

    await pipeline.stop_continuous_session()


class MarkerSegmenter:
    """Deterministic VAD stand-in: room tone never commits, marker bytes flag speech."""

    def __init__(self) -> None:
        self.buf = b""
        self._speech = False

    def feed(self, pcm: bytes) -> None:
        if pcm:
            self.buf += pcm
            if b"U" in pcm:
                self._speech = True

    @property
    def is_speech_detected(self) -> bool:
        return self._speech

    def pop_segments(self) -> list[bytes]:
        return []

    def flush(self) -> list[bytes]:
        out = [bytes(self.buf)] if self.buf else []
        self.buf = b""
        return out

    def reset(self) -> None:
        self.buf = b""
        self._speech = False


UTTERANCE_TEXT = "take me downtown"


async def _marker_decode(pcm: bytes, prompt: str = "") -> str:
    return UTTERANCE_TEXT if b"U" in pcm else ""


class ParitySTT:
    """STT factory sharing real STTSession semantics across both voice paths."""

    def __init__(self) -> None:
        self.sessions: list[STTSession] = []

    def open_session(self, sample_rate: int = 16000, on_endpoint=None):
        sess = STTSession(
            _marker_decode,
            MarkerSegmenter(),
            sample_rate=sample_rate,
            on_endpoint=on_endpoint,
        )
        self.sessions.append(sess)
        return sess


@pytest.mark.asyncio
async def test_barge_in_transcript_matches_ptt_transcript_word_for_word():
    """Slice 4 parity gate: a barge-in interrupt turn yields exactly the uttered text.

    The same uttered bytes through the PTT path (stop_recording_and_process)
    decode word for word. Echo-era room tone commits nothing (old session is
    cancelled); the utterance onset survives via the 300ms look-back seed.
    """
    stt = ParitySTT()

    def _pipeline() -> VoicePipeline:
        runtime = MagicMock()
        runtime.floor = FloorArbiter()
        runtime.is_idle = True
        runtime.user_active = False
        runtime.submit = AsyncMock(return_value="ok")
        runtime.cancel_turn = AsyncMock()
        sink = MockAudioSink(playback_delay=0.05)
        player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)
        return VoicePipeline(
            runtime=runtime, stt=stt, player=player, capture=FakeCapture()
        )

    # Continuous path: echo-era tone, then barge-in onset + post-interrupt speech.
    pipeline = _pipeline()
    pipeline.runtime.is_idle = False
    pipeline._is_continuous_active = True
    pipeline._loop = asyncio.get_running_loop()
    await pipeline._setup_turn_listening()
    assert len(stt.sessions) == 1
    old = stt.sessions[0]

    pipeline.runtime.floor.set_state(FloorState.SPEAKING)
    echo = b"\x00" * 320
    onset = b"U" * 2000
    post = b"U" * 2000
    for _ in range(40):
        pipeline.capture.on_chunk(echo)
    pipeline.capture.on_chunk(onset)
    await asyncio.sleep(0.02)

    assert len(stt.sessions) == 2
    fresh = stt.sessions[1]
    assert fresh is not old
    assert fresh.on_endpoint == pipeline._on_speech_endpoint
    # Onset intact: the 300ms look-back seed holds echo tail plus full onset.
    assert fresh._segmenter.buf == b"\x00" * 7600 + onset

    pipeline.runtime.floor.set_state(FloorState.LISTENING)
    pipeline.capture.on_chunk(post)
    interrupt_text = await fresh.finalize()

    # PTT path: the same uttered bytes through stop_recording_and_process.
    ptt = _pipeline()
    ptt.runtime.provider = None
    await ptt.start_recording()
    ptt.capture.on_chunk(onset)
    ptt.capture.on_chunk(post)
    res = await ptt.stop_recording_and_process()
    await asyncio.sleep(0.1)

    assert res["transcription"] == interrupt_text == UTTERANCE_TEXT
    assert interrupt_text.split() == UTTERANCE_TEXT.split()
    assert await old.finalize() == ""

    await pipeline.stop_continuous_session()


@pytest.mark.asyncio
async def test_continuous_loop_cancels_old_session_after_playback_and_preserves_tail_speech():
    sessions = []

    def make_session(**kw):
        s = MagicMock()
        s.polluted_by_barge_in = False
        s.is_speech_detected = False
        s.speech_duration = 1.0
        s.fed_chunks = []
        s.feed = MagicMock(side_effect=s.fed_chunks.append)

        def _barge(chunk):
            s.polluted_by_barge_in = True
            if b"tail_speech" in chunk:
                s.is_speech_detected = True
            return False

        s.feed_barge_in = _barge
        s.finalize = AsyncMock(return_value="hello there")
        s.cancel = MagicMock()
        s.reset = MagicMock()
        sessions.append(s)
        return s

    stt = MagicMock()
    stt.open_session = MagicMock(side_effect=make_session)

    runtime = MagicMock()
    runtime.floor = FloorArbiter()
    runtime.is_idle = True
    runtime.pending_continuation = None

    capture = MagicMock()
    capture.sample_rate = 16000
    capture.is_recording = False
    capture.start_recording = AsyncMock()
    capture.stop_recording = AsyncMock(return_value=b"")

    pipeline = VoicePipeline(
        runtime=runtime,
        stt=stt,
        player=AudioPlayer(tts=MockTextToSpeech(), sink=MockAudioSink()),
        capture=capture,
        aec=MagicMock(is_available=False),
    )

    async def fake_submit(text, output, **kw):
        runtime.floor.set_state("SPEAKING")
        # User starts speaking right at the end of playback (before barge-in interrupts)
        pipeline.capture.on_chunk(b"tail_speech_onset")
        return "Hi!"

    runtime.submit = fake_submit

    await pipeline.start_continuous_session()
    pipeline._speech_endpoint_event.set()
    await asyncio.sleep(0.05)

    # Old session must be cancelled before opening the new post-playback session,
    # and the new session must be unpolluted by barge-in yet seeded with the tail speech onset!
    assert len(sessions) == 2
    assert sessions[0].cancel.called is True
    assert pipeline._live_stt is sessions[1]
    assert pipeline._live_stt.polluted_by_barge_in is False
    assert b"tail_speech_onset" in b"".join(pipeline._live_stt.fed_chunks)
    await pipeline.stop_continuous_session()




