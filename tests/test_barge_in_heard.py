import asyncio
from unittest.mock import MagicMock

import pytest

from tests.conftest import assert_history_valid
from verbix.core.events import TextDelta, TurnComplete
from verbix.provider.models import TextDelta as ProviderTextDelta
from verbix.runtime.runtime import Runtime
from verbix.voice.base import MockAudioSink, MockTextToSpeech
from verbix.voice.output import SpeechOutput
from verbix.voice.pipeline import VoicePipeline
from verbix.voice.player import AudioPlayer


@pytest.mark.asyncio
async def test_cancel_turn_heard_override_repairs_to_spoken_prefix(tmp_path):
    """Barge-in must trim history to what was heard, not wipe or overstate it."""
    entered = asyncio.Event()

    async def hanging_stream(messages, tools=None, **kwargs):
        entered.set()
        yield ProviderTextDelta(text="Sentence one. Sentence two. ")
        await asyncio.Event().wait()

    provider = MagicMock()
    provider.stream = hanging_stream
    runtime = Runtime(workspace=tmp_path, provider=provider)
    speech = SpeechOutput(tts=MockTextToSpeech(), sink=MockAudioSink(playback_delay=0.01))

    turn = asyncio.create_task(runtime.submit("hello", speech))
    await asyncio.wait_for(entered.wait(), timeout=2.0)
    await asyncio.sleep(0.1)

    await runtime.cancel_turn(heard="Sentence one.")

    assistants = [m for m in runtime.foreground.messages if m.get("role") == "assistant"]
    assert assistants, "interrupted turn left no assistant message"
    assert assistants[-1].get("content") == "Sentence one."
    assert_history_valid(runtime.foreground.messages)
    assert turn.done()


class _StubCapture:
    def __init__(self):
        self.is_recording = False
        self.started = False

    async def start_recording(self):
        self.started = True


class _StubRuntime:
    def __init__(self):
        self.is_idle = False
        self.user_active = False
        self.wake_output = None
        self.cancel_kwargs = None

    async def cancel_turn(self, **kwargs):
        self.cancel_kwargs = kwargs


class _StubSession:
    def __init__(self):
        self.on_endpoint = None
        self.speech_duration = 0.0

    def feed(self, pcm: bytes) -> None:
        pass

    @property
    def is_speech_detected(self) -> bool:
        return False

    def feed_barge_in(self, pcm: bytes) -> bool:
        return False

    async def finalize(self) -> str:
        return ""

    def cancel(self) -> None:
        pass

    def reset(self) -> None:
        pass


class _StubSTT:
    def __init__(self):
        self.sessions = []

    def open_session(self, sample_rate: int = 16000, on_endpoint=None):
        sess = _StubSession()
        sess.on_endpoint = on_endpoint
        self.sessions.append(sess)
        return sess


@pytest.mark.asyncio
async def test_toggle_passes_pre_interrupt_heard_text_to_cancel():
    """Snapshot ordering: heard text captured before mute, not after."""
    stub = _StubRuntime()
    capture = _StubCapture()
    player = AudioPlayer(tts=MockTextToSpeech(), sink=MockAudioSink(playback_delay=0.01))
    pipeline = VoicePipeline(runtime=stub, stt=None, player=player, capture=capture)

    await pipeline.turn_output.emit(TextDelta(text="First sentence here. "))
    for _ in range(200):
        if pipeline.turn_output.delivered_text():
            break
        await asyncio.sleep(0.01)
    heard_before = pipeline.turn_output.delivered_text()
    assert heard_before, "setup failed: nothing was delivered"

    await pipeline.toggle()

    assert stub.cancel_kwargs is not None
    assert stub.cancel_kwargs.get("heard") == heard_before


@pytest.mark.asyncio
async def test_interruption_grounding_preserves_three_way_distinction(tmp_path):
    """Runtime submits interruption grounding with 3-way distinction without polluting message history."""
    from verbix.voice.output import InterruptionSnapshot

    entered = asyncio.Event()

    async def hanging_stream(messages, tools=None, **kwargs):
        entered.set()
        yield ProviderTextDelta(
            text="Confirmed played sentence. Partially heard cut sentence. Never started sentence."
        )
        await asyncio.Event().wait()

    provider = MagicMock()
    provider.stream = hanging_stream
    runtime = Runtime(workspace=tmp_path, provider=provider)
    speech = SpeechOutput(tts=MockTextToSpeech(), sink=MockAudioSink(playback_delay=0.01))

    turn = asyncio.create_task(runtime.submit("Tell me a story", speech))
    await asyncio.wait_for(entered.wait(), timeout=2.0)
    await asyncio.sleep(0.1)

    speech.interrupt()

    snap = InterruptionSnapshot(
        confirmed_heard="Confirmed played sentence.",
        interrupted_sentence="Partially heard cut sentence.",
        unplayed_text="Never started sentence.",
    )

    await runtime.cancel_turn(heard=snap.confirmed_heard)
    assert turn.done()
    assert runtime.foreground.messages[-1]["content"] == "Confirmed played sentence."

    received_messages = []

    async def capturing_stream(messages, tools=None, **kwargs):
        received_messages.extend(messages)
        yield ProviderTextDelta(text="Continuing story.")

    provider.stream = capturing_stream

    await runtime.submit("uh-huh", speech, interruption=snap)

    user_prompts = [m.content for m in received_messages if getattr(m, "role", "") == "user"]
    last_prompt = user_prompts[-1]
    assert "Confirmed played sentence." in last_prompt
    assert "Partially heard cut sentence." in last_prompt
    assert "Never started sentence." in last_prompt
    assert "uh-huh" in last_prompt

    assistant_msgs = [m for m in runtime.foreground.messages if m.get("role") == "assistant"]
    assert assistant_msgs[0]["content"] == "Confirmed played sentence."
    assert "Partially heard" not in assistant_msgs[0]["content"]
    assert "Never started" not in assistant_msgs[0]["content"]


@pytest.mark.asyncio
async def test_pipeline_passes_interruption_to_runtime():
    """VoicePipeline passes last_interruption to runtime.submit and clears it."""
    from unittest.mock import AsyncMock

    from verbix.runtime.floor import FloorArbiter
    from verbix.voice.output import InterruptionSnapshot

    runtime = MagicMock()
    runtime.floor = FloorArbiter()
    runtime.submit = AsyncMock(return_value="response")

    sink = MockAudioSink()
    player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)
    pipeline = VoicePipeline(
        runtime=runtime,
        stt=None,
        player=player,
        capture=_StubCapture(),
    )

    snap = InterruptionSnapshot(
        confirmed_heard="Confirmed part.",
        interrupted_sentence="Interrupted part.",
        unplayed_text="Unplayed part.",
    )
    pipeline._speech_output.last_interruption = snap

    await pipeline.execute_voice_turn("continue please")

    runtime.submit.assert_called_once_with(
        "continue please", pipeline._speech_output, interruption=snap
    )
    assert pipeline._speech_output.last_interruption is None


@pytest.mark.asyncio
async def test_interrupt_does_not_pass_unplayed_synthesized_text_to_cancel_turn():
    """Unplayed synthesized text in _delivered must never be passed to cancel_turn as confirmed heard."""

    class TrackingSink(MockAudioSink):
        def __init__(self, playback_delay: float = 0.05):
            super().__init__(playback_delay=playback_delay)
            self.played = 0

        @property
        def played_samples(self) -> int:
            return self.played

    stub = _StubRuntime()
    capture = _StubCapture()
    sink = TrackingSink(playback_delay=0.05)
    player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)
    pipeline = VoicePipeline(runtime=stub, stt=_StubSTT(), player=player, capture=capture)
    pipeline._loop = asyncio.get_running_loop()

    text = "First sentence here. Second sentence here. Third sentence here."
    await pipeline.turn_output.emit(TextDelta(text=text))
    await pipeline.turn_output.emit(TurnComplete(final_text=text))

    # Wait until all 3 sentences are synthesized and spans recorded in SpeechOutput
    for _ in range(100):
        if len(pipeline.turn_output._sent_spans) == 3:
            break
        await asyncio.sleep(0.01)

    assert len(pipeline.turn_output._sent_spans) == 3
    # Before interrupt(), delivered_text() contains all 3 synthesized sentences
    assert pipeline.turn_output.delivered_text() == "First sentence here. Second sentence here. Third sentence here."

    # Case 1: Interruption occurs while 0 samples have played (before first sentence completes)
    sink.played = 0
    pipeline._interrupt_active_speech()
    await asyncio.sleep(0.02)  # allow call_soon_threadsafe task to execute

    assert stub.cancel_kwargs is not None
    # cancel_turn must receive empty string, NOT the 3 synthesized sentences from delivered_text()!
    assert stub.cancel_kwargs.get("heard") == ""

    # Case 2: In a new turn, interruption occurs after sentence 1 has fully played but sentence 2 is interrupted
    sink.played = 0
    pipeline.turn_output.reset()
    stub.cancel_kwargs = None

    await pipeline.turn_output.emit(TextDelta(text=text))
    await pipeline.turn_output.emit(TurnComplete(final_text=text))

    for _ in range(100):
        if len(pipeline.turn_output._sent_spans) == 3:
            break
        await asyncio.sleep(0.01)

    assert len(pipeline.turn_output._sent_spans) == 3
    start2 = pipeline.turn_output._sent_spans[1][1]
    end2 = pipeline.turn_output._sent_spans[1][2]
    # Set played samples mid-way through sentence 2
    sink.played = (start2 + end2) // 2

    pipeline._interrupt_active_speech()
    await asyncio.sleep(0.02)

    assert stub.cancel_kwargs is not None
    assert stub.cancel_kwargs.get("heard") == "First sentence here."


@pytest.mark.asyncio
async def test_cancel_turn_race_repairs_history_after_task_completed(tmp_path):
    """When the turn generation task unwinds before cancel_turn is processed, history repair still runs."""
    provider = MagicMock()

    async def immediate_stream(messages, tools=None, **kwargs):
        yield ProviderTextDelta(text="Unspoken assistant draft text.")

    provider.stream = immediate_stream
    runtime = Runtime(workspace=tmp_path, provider=provider)
    speech = SpeechOutput(tts=MockTextToSpeech(), sink=MockAudioSink(playback_delay=0.01))

    # Run turn. Since playback delay is small and MockTextToSpeech is fast, submit() can complete
    turn_task = asyncio.create_task(runtime.submit("Hello", speech))
    # Interrupt audio output immediately while turn_task is running or finishing
    await asyncio.sleep(0.01)
    speech.interrupt()
    await turn_task

    # The task has now finished and cleared handle.task = None!
    assert runtime._turn_handle.task is None

    # Cancel turn with empty heard (barge-in before user heard anything)
    await runtime.cancel_turn(heard="")

    # The unspoken assistant content must have been removed from foreground.messages!
    assistants = [m for m in runtime.foreground.messages if m.get("role") == "assistant"]
    assert not assistants, f"Expected assistant message to be removed, but found: {assistants}"
    assert runtime.foreground.messages[-1].get("role") == "user"
    assert_history_valid(runtime.foreground.messages)


@pytest.mark.asyncio
async def test_cancel_turn_preserves_unrelated_completed_history(tmp_path):
    """cancel_turn must never corrupt or pop an earlier turn that completed normally."""
    provider = MagicMock()

    async def immediate_stream(messages, tools=None, **kwargs):
        yield ProviderTextDelta(text="Completed turn one.")

    provider.stream = immediate_stream
    runtime = Runtime(workspace=tmp_path, provider=provider)
    speech = SpeechOutput(tts=MockTextToSpeech(), sink=MockAudioSink(playback_delay=0.01))

    # Turn 1 completes normally
    await runtime.submit("First prompt", speech)
    assert runtime._turn_handle.completed is True
    # messages contain system, user, and assistant messages
    assert len(runtime.foreground.messages) == 3
    assert runtime.foreground.messages[-1]["content"] == "Completed turn one."

    # A subsequent cancel_turn (e.g. from idle or redundant event) must not touch completed history
    await runtime.cancel_turn(heard="")
    assert len(runtime.foreground.messages) == 3
    assert runtime.foreground.messages[-1]["content"] == "Completed turn one."


@pytest.mark.asyncio
async def test_request_grounding_marks_unplayed_text_as_discarded(tmp_path):
    """Grounding envelope explicitly marks unplayed text as discarded unless user explicitly continues."""
    from verbix.voice.output import InterruptionSnapshot

    runtime = Runtime(workspace=tmp_path, provider=MagicMock())
    snap = InterruptionSnapshot(
        confirmed_heard="Sentence one.",
        interrupted_sentence="Sentence two.",
        unplayed_text="Sentence three.",
    )
    grounding = runtime._request_grounding(snap)
    assert '- Subsequent text that had not yet started playing (discarded unless the user explicitly requests continuation): "Sentence three."' in grounding



