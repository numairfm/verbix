import asyncio
from unittest.mock import MagicMock

import pytest

from verbix.core.events import TextDelta
from verbix.provider.models import Finish as ProviderFinish
from verbix.provider.models import TextDelta as ProviderTextDelta
from verbix.runtime.floor import FloorState
from verbix.runtime.runtime import Runtime


@pytest.mark.asyncio
async def test_task_finishing_while_idle_speaks_immediately(tmp_path):
    spoken = []
    class FakeOutput:
        async def emit(self, event):
            if isinstance(event, TextDelta):
                spoken.append(event.text)
        def delivered_text(self):
            return "".join(spoken)

    provider = MagicMock()
    async def fake_stream(messages, tools=None, reasoning_effort=None):
        yield ProviderTextDelta(text="Task 1 finished successfully.")
        yield ProviderFinish(reason="stop")
    provider.stream = fake_stream

    rt = Runtime(workspace=tmp_path, provider=provider)
    rt.wake_output = FakeOutput()

    assert rt.floor.is_idle is True

    # Complete a task while floor is IDLE
    rt.ledger.register_task("task-1", "check system")
    rt._on_task_finished("task-1", "system is healthy")

    # Allow drain task to run
    for _ in range(20):
        if spoken:
            break
        await asyncio.sleep(0.02)

    assert any("Task 1 finished" in s for s in spoken)
    assert rt.floor.is_idle is True
    await rt.aclose()


@pytest.mark.asyncio
async def test_task_finishing_while_busy_holds_until_floor_returns_to_idle(tmp_path):
    spoken = []
    class FakeOutput:
        async def emit(self, event):
            if isinstance(event, TextDelta):
                spoken.append(event.text)
        def delivered_text(self):
            return "".join(spoken)

    provider = MagicMock()
    turn_count = 0
    async def fake_stream(messages, tools=None, reasoning_effort=None):
        nonlocal turn_count
        turn_count += 1
        last_content = str(messages[-1].content) if messages else ""
        if "joke" in last_content:
            yield ProviderTextDelta(text="Here is a funny joke.")
        else:
            yield ProviderTextDelta(text="Background update: task 2 finished.")
        yield ProviderFinish(reason="stop")
    provider.stream = fake_stream

    rt = Runtime(workspace=tmp_path, provider=provider)
    rt.wake_output = FakeOutput()

    # 1. Simulate an active foreground turn (SPEAKING)
    async with rt.floor.acquire(FloorState.SPEAKING):
        assert rt.floor.is_speaking is True

        # Task completes DURING the active turn
        rt.ledger.register_task("task-2", "run build")
        rt._on_task_finished("task-2", "build succeeded")

        await asyncio.sleep(0.05)
        # MUST NOT have spoken the announcement yet while floor is busy!
        assert len(spoken) == 0

    # 2. Floor exited acquire block and returned to IDLE
    assert rt.floor.is_idle is True

    # Wait for pending announcement to drain on IDLE
    for _ in range(20):
        if spoken:
            break
        await asyncio.sleep(0.02)

    assert any("task 2 finished" in s for s in spoken)
    assert rt.floor.is_idle is True
    await rt.aclose()


@pytest.mark.asyncio
async def test_drain_announcement_turn_is_offered_no_tools_and_spawns_nothing(tmp_path):
    spoken = []
    seen_tools = []

    class FakeOutput:
        async def emit(self, event):
            if isinstance(event, TextDelta):
                spoken.append(event.text)
        def delivered_text(self):
            return "".join(spoken)

    provider = MagicMock()
    async def fake_stream(messages, tools=None, reasoning_effort=None):
        seen_tools.append(tools)
        yield ProviderTextDelta(text="Task 9 finished successfully.")
        yield ProviderFinish(reason="stop")
    provider.stream = fake_stream

    rt = Runtime(workspace=tmp_path, provider=provider)
    rt.wake_output = FakeOutput()
    before = set(rt.tasks)

    rt.ledger.register_task("task-9", "do thing")
    rt._on_task_finished("task-9", "did thing")

    for _ in range(100):
        if spoken:
            break
        await asyncio.sleep(0.02)

    assert any("Task 9 finished" in s for s in spoken)
    assert set(rt.tasks) == before
    assert seen_tools and all(t is None for t in seen_tools)
    await rt.aclose()


@pytest.mark.asyncio
async def test_pending_completion_folded_into_foreground_turn_is_spoken_once_not_twice(tmp_path):
    class FakeOutput:
        def __init__(self):
            self.buf = []

        async def emit(self, event):
            if isinstance(event, TextDelta):
                self.buf.append(event.text)

        def delivered_text(self):
            return "".join(self.buf)

    seen_user_messages = []
    provider = MagicMock()

    async def fake_stream(messages, tools=None, **kw):
        seen_user_messages.append(messages[-1].content if messages else "")
        yield ProviderTextDelta(text="Nvidia agreed to buy Hugging Face for 12.9B.")
        yield ProviderFinish(reason="stop")

    provider.stream = fake_stream
    rt = Runtime(workspace=tmp_path, provider=provider)
    wake_out = FakeOutput()
    rt.wake_output = wake_out

    # Task finishes while floor is busy (SPEAKING) so announcer does not run yet
    rt.floor.set_state(FloorState.SPEAKING)
    rt.ledger.register_task("task-2", "Verify if NVIDIA acquired Hugging Face")
    rt._on_task_finished("task-2", "Agreed to buy for $12.9B.")
    assert rt.ledger.get("task-2").delivery == "pending"

    # User asks "What's the status?" -> completion is folded into the foreground turn
    turn_out = FakeOutput()
    reply = await rt.submit("What's the status?", turn_out)
    assert "12.9B" in reply
    assert "Agreed to buy for $12.9B." in seen_user_messages[0]
    assert rt.ledger.get("task-2").delivery == "heard"

    # Wait after turn ends: announcer MUST NOT speak task-2 a second time!
    await asyncio.sleep(0.1)
    assert wake_out.delivered_text() == ""
    await rt.aclose()


@pytest.mark.asyncio
async def test_interrupted_foreground_turn_leaves_folded_completion_pending(tmp_path):
    provider = MagicMock()

    async def slow_stream(messages, tools=None, **kw):
        yield ProviderTextDelta(text="Starting to tell you...")
        await asyncio.sleep(10.0)

    provider.stream = slow_stream
    rt = Runtime(workspace=tmp_path, provider=provider)
    rt.floor.set_state(FloorState.SPEAKING)
    rt.ledger.register_task("task-2", "Verify deal")
    rt._on_task_finished("task-2", "Deal confirmed.")

    task = asyncio.create_task(rt.submit("What happened?"))
    await asyncio.sleep(0.05)
    await rt.cancel_turn(heard="")
    with pytest.raises(asyncio.CancelledError):
        await task

    # Because the foreground turn was interrupted before completing OK, delivery stays pending!
    assert rt.ledger.get("task-2").delivery == "pending"
    await rt.aclose()


@pytest.mark.asyncio
async def test_status_tool_marks_only_inspected_completed_task_heard(tmp_path):
    from verbix.runtime.worker import WorkerTask

    rt = Runtime(workspace=tmp_path, provider=MagicMock())
    rt.floor.set_state(FloorState.SPEAKING)
    for tid, res in (("task-1", "res 1"), ("task-2", "res 2")):
        t = WorkerTask(id=tid, description=f"job {tid}", session=MagicMock(messages=[]))
        t._status = "completed"
        t._result = res
        rt.tasks[tid] = t
        rt.ledger.register_task(tid, f"job {tid}")
        rt._on_task_finished(tid, res)

    assert rt.ledger.get("task-1").delivery == "pending"
    assert rt.ledger.get("task-2").delivery == "pending"

    out = rt.status(task_id="task-1")
    assert "res 1" in out
    assert rt.ledger.get("task-1").delivery == "heard"
    assert rt.ledger.get("task-2").delivery == "pending"


@pytest.mark.asyncio
async def test_ptt_interrupted_announcement_with_short_capture_is_not_retried_and_folds_into_next_turn(
    tmp_path,
):
    from verbix.voice.base import MockAudioSink, MockTextToSpeech
    from verbix.voice.pipeline import VoicePipeline
    from verbix.voice.player import AudioPlayer

    announcer_calls = 0
    seen_fg_prompts: list[str] = []
    provider = MagicMock()

    async def fake_stream(messages, tools=None, **kw):
        nonlocal announcer_calls
        last_msg = messages[-1].content if messages else ""
        if "[Present these completed findings" in last_msg:
            announcer_calls += 1
            yield ProviderTextDelta(
                text="Long announcement sentence one. Long announcement sentence two."
            )
            yield ProviderFinish(reason="stop")
        else:
            seen_fg_prompts.append(last_msg)
            yield ProviderTextDelta(text="Here is the folded result and your answer.")
            yield ProviderFinish(reason="stop")

    provider.stream = fake_stream
    rt = Runtime(workspace=tmp_path, provider=provider)
    sink = MockAudioSink(playback_delay=0.25)
    player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)

    class ShortThenRealSTT:
        def __init__(self):
            self.Transcripts = ["", "", "", "", "What did it find?"]

        async def transcribe(self, audio_bytes: bytes) -> str:
            return self.Transcripts.pop(0) if self.Transcripts else ""

    class FakePTTCapture:
        sample_rate = 16000
        is_recording = False
        on_chunk = None

        async def start_recording(self):
            self.is_recording = True

        async def stop_recording(self) -> bytes:
            self.is_recording = False
            return b"\x01\x00" * 1600

    pipeline = VoicePipeline(
        runtime=rt,
        stt=ShortThenRealSTT(),
        player=player,
        capture=FakePTTCapture(),
    )

    # PTT-only pipeline must not initialize continuous-only EchoCanceller
    assert pipeline._aec is None

    # 1. Task finishes while idle -> announcement starts playing
    rt.ledger.register_task("task-1", "Check workspace")
    rt._on_task_finished("task-1", "Workspace has 5 files.")
    for _ in range(30):
        if pipeline.turn_output.is_playing:
            break
        await asyncio.sleep(0.01)
    assert pipeline.turn_output.is_playing is True
    assert announcer_calls == 1

    # 2. User presses Enter to interrupt announcement -> starts PTT recording
    r1 = await pipeline.toggle()
    assert r1["action"] == "interrupted_and_recording"
    assert pipeline.turn_output.is_playing is False

    # 3. User presses Enter immediately -> capture too short ("No speech detected"), floor goes IDLE.
    # Repeat 3 more times (4 empty captures total). Announcement MUST NOT replay!
    r2 = await pipeline.toggle()
    assert r2["action"] == "idle"
    await asyncio.sleep(0.05)
    assert announcer_calls == 1

    for _ in range(3):
        await pipeline.toggle()  # start recording
        res_stop = await pipeline.toggle()  # stop recording -> empty transcript -> IDLE
        assert res_stop["action"] == "idle"
        await asyncio.sleep(0.05)

    assert announcer_calls == 1
    assert rt.ledger.get("task-1").announce_attempts == 1
    assert rt.ledger.get("task-1").delivery == "pending"

    # 4. Next real PTT turn folds the interrupted announcement's result into foreground context and marks it heard!
    sink.playback_delay = 0.01
    await pipeline.toggle()  # start recording
    r_real = await pipeline.toggle()  # stop recording -> "What did it find?"
    assert r_real["action"] == "processing"
    if pipeline._active_voice_task:
        await pipeline._active_voice_task

    assert len(seen_fg_prompts) == 1
    assert "Workspace has 5 files." in seen_fg_prompts[0]
    assert rt.ledger.get("task-1").delivery == "heard"
    assert announcer_calls == 1
    assert pipeline._aec is None
    await rt.aclose()


