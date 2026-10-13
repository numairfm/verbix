"""Slice 5 — draft shelf: empty transcripts stay silent, bare continue resumes stash."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from verbix.runtime.floor import FloorArbiter, FloorState
from verbix.runtime.runtime import Runtime
from verbix.voice.base import MockAudioSink, MockTextToSpeech
from verbix.voice.output import InterruptionSnapshot
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
    def __init__(self, texts):
        self.sessions = []
        self._texts = list(texts)

    def open_session(self, sample_rate: int = 16000, on_endpoint=None):
        text = self._texts.pop(0) if self._texts else ""
        sess = FakeLiveSession(text)
        sess.sample_rate = sample_rate
        sess.on_endpoint = on_endpoint
        self.sessions.append(sess)
        return sess

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000):
        return "transcribed"


def _pipeline(runtime, stt, capture=None):
    sink = MockAudioSink()
    player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)
    return VoicePipeline(
        runtime=runtime,
        stt=stt,
        player=player,
        capture=capture or FakeCapture(),
    )


def _runtime():
    runtime = MagicMock()
    runtime.floor = FloorArbiter()
    runtime.is_idle = True
    runtime.user_active = False
    runtime.submit = AsyncMock(return_value="ok")
    runtime.cancel_turn = AsyncMock()
    runtime.pending_continuation = None
    return runtime


def test_fresh_runtime_declares_pending_continuation_none(tmp_path):
    """Fresh Runtime carries a typed pending_continuation, initialized to None."""
    runtime = Runtime(workspace=tmp_path, provider=MagicMock())
    assert runtime.pending_continuation is None


@pytest.mark.asyncio
async def test_stash_then_continue_round_trips_on_real_runtime(tmp_path):
    """Stash-then-continue round-trips through the real Runtime attribute."""
    runtime = Runtime(workspace=tmp_path, provider=MagicMock())
    assert runtime.pending_continuation is None
    runtime.submit = AsyncMock(return_value="ok")
    stt = FakeSTT(["continue"])
    pipeline = _pipeline(runtime, stt)

    pipeline._stash_interruption(InterruptionSnapshot(unspoken_draft="Real tail."))
    assert runtime.pending_continuation == "Real tail."

    await pipeline.start_continuous_session()
    stt.sessions[0].on_endpoint()
    await asyncio.sleep(0.05)

    runtime.submit.assert_called_once()
    submitted = runtime.submit.await_args.args[0]
    assert "Real tail." in submitted
    assert runtime.pending_continuation is None
    await pipeline.stop_continuous_session()


@pytest.mark.asyncio
async def test_empty_transcript_stays_silent_and_keeps_snapshot():
    """Empty transcription returns to listening with no speech output (no re-speak)."""
    runtime = _runtime()
    stt = FakeSTT([""])
    pipeline = _pipeline(runtime, stt)

    await pipeline.start_continuous_session()
    sess = stt.sessions[0]
    pipeline._speech_output.last_interruption = InterruptionSnapshot(
        heard_text="Spoken sentence one.",
        unspoken_draft="Remaining sentence two.",
    )

    sess.on_endpoint()
    await asyncio.sleep(0.05)

    runtime.submit.assert_not_called()
    assert pipeline.turn_output.is_playing is False
    assert "Remaining sentence two." not in pipeline.turn_output.delivered_text()
    # Snapshot kept for history repair, not consumed by silence.
    assert pipeline._speech_output.last_interruption is not None
    # Listening resumed.
    assert len(stt.sessions) == 2
    await pipeline.stop_continuous_session()


@pytest.mark.asyncio
async def test_interrupt_stashes_unplayed_outside_history():
    """Barge-in stashes unplayed text on runtime, outside assistant history."""
    runtime = _runtime()
    stt = FakeSTT(["new user text"])
    pipeline = _pipeline(runtime, stt)
    pipeline._is_continuous_active = True
    pipeline._loop = asyncio.get_running_loop()
    await pipeline._setup_turn_listening()
    old = stt.sessions[0]
    old.feed_barge_in = lambda chunk: True  # type: ignore[attr-defined]

    runtime.floor.set_state(FloorState.SPEAKING)
    pipeline._speech_output._delivered.append("Already heard.")
    pipeline.capture.on_chunk(b"trigger")
    await asyncio.sleep(0.02)

    # Fresh session opened; heard goes to history repair via cancel_turn.
    assert len(stt.sessions) == 2
    heard_arg = runtime.cancel_turn.await_args.kwargs.get("heard", None)
    assert heard_arg is not None
    assert "Remaining" not in heard_arg  # unplayed never enters heard/history
    await pipeline.stop_continuous_session()


@pytest.mark.asyncio
async def test_continue_hands_stashed_text_to_agent_and_clears():
    """Bare 'continue' prepends stashed text to the submitted turn and clears it."""
    runtime = _runtime()
    runtime.pending_continuation = "The stashed remainder."
    stt = FakeSTT(["continue"])
    pipeline = _pipeline(runtime, stt)

    await pipeline.start_continuous_session()
    stt.sessions[0].on_endpoint()
    await asyncio.sleep(0.05)

    runtime.submit.assert_called_once()
    submitted = runtime.submit.await_args.args[0]
    assert "The stashed remainder." in submitted
    assert runtime.pending_continuation is None
    await pipeline.stop_continuous_session()


@pytest.mark.asyncio
@pytest.mark.parametrize("utterance", ["Continue", "  CONTINUE  ", "go on", "Go On"])
async def test_continue_variants_match_exactly_case_insensitive(utterance):
    """Case-insensitive stripped exact matches consume the stash; nothing fuzzy."""
    runtime = _runtime()
    runtime.pending_continuation = "Stashed tail."
    stt = FakeSTT([utterance])
    pipeline = _pipeline(runtime, stt)

    await pipeline.start_continuous_session()
    stt.sessions[0].on_endpoint()
    await asyncio.sleep(0.05)

    submitted = runtime.submit.await_args.args[0]
    assert "Stashed tail." in submitted
    assert runtime.pending_continuation is None
    await pipeline.stop_continuous_session()


@pytest.mark.asyncio
@pytest.mark.parametrize("utterance", ["continue please", "please continue", "go on then", "continu"])
async def test_near_continue_does_not_consume_stash(utterance):
    """Anything but the bare exact match leaves the stash alone."""
    runtime = _runtime()
    runtime.pending_continuation = "Stashed tail."
    stt = FakeSTT([utterance])
    pipeline = _pipeline(runtime, stt)

    await pipeline.start_continuous_session()
    stt.sessions[0].on_endpoint()
    await asyncio.sleep(0.05)

    submitted = runtime.submit.await_args.args[0]
    assert "Stashed tail." not in submitted
    assert runtime.pending_continuation == "Stashed tail."
    await pipeline.stop_continuous_session()


@pytest.mark.asyncio
async def test_other_utterance_leaves_stash_alone():
    """A normal turn submits without the stash and retains it."""
    runtime = _runtime()
    runtime.pending_continuation = "Stashed tail."
    stt = FakeSTT(["what time is it"])
    pipeline = _pipeline(runtime, stt)

    await pipeline.start_continuous_session()
    stt.sessions[0].on_endpoint()
    await asyncio.sleep(0.05)

    submitted = runtime.submit.await_args.args[0]
    assert "Stashed tail." not in submitted
    assert runtime.pending_continuation == "Stashed tail."
    await pipeline.stop_continuous_session()


@pytest.mark.asyncio
async def test_newer_interrupt_overwrites_stash():
    """A newer interrupt's unplayed text supersedes the old stash."""
    runtime = _runtime()
    stt = FakeSTT([])
    pipeline = _pipeline(runtime, stt)

    pipeline._stash_interruption(InterruptionSnapshot(unspoken_draft="First tail."))
    assert runtime.pending_continuation == "First tail."
    pipeline._stash_interruption(InterruptionSnapshot(unspoken_draft="Second tail."))
    assert runtime.pending_continuation == "Second tail."
    pipeline._stash_interruption(None)
    assert runtime.pending_continuation is None
