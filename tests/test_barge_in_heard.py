import asyncio
from unittest.mock import MagicMock

import pytest

from tests.conftest import assert_history_valid
from verbix.core.events import TextDelta
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
