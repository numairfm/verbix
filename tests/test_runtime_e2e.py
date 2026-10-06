from unittest.mock import MagicMock

import pytest

from verbix.core.events import TextDelta
from verbix.provider.models import Finish as ProviderFinish
from verbix.provider.models import TextDelta as ProviderTextDelta
from verbix.runtime.runtime import Runtime


@pytest.fixture
def mock_provider():
    provider = MagicMock()

    async def fake_stream(messages, tools=None, reasoning_effort=None):
        last_msg = messages[-1].content if messages else ""
        if "prime" in last_msg.lower():
            yield ProviderTextDelta(text="2, 3, 5, 7, 11 are primes.")
            yield ProviderFinish(reason="stop")
        elif "[background task completed]" in last_msg.lower():
            yield ProviderTextDelta(text="The prime calculation finished: 2, 3, 5, 7, 11.")
            yield ProviderFinish(reason="stop")
        else:
            yield ProviderTextDelta(text="Hello! How can I help?")
            yield ProviderFinish(reason="stop")

    provider.stream = fake_stream
    return provider


@pytest.mark.asyncio
async def test_runtime_spawn_and_completion_loop(mock_provider, tmp_path):
    woken = []

    async def on_wake(speech_text):
        woken.append(speech_text)

    runtime = Runtime(
        workspace=tmp_path,
        provider=mock_provider,
        buffer_window=0.05,
        on_wake=on_wake,
    )

    # Spawn worker
    task_id = await runtime.spawn("calculate prime numbers")
    assert task_id == "task-1"
    assert task_id in runtime.tasks

    # Verify task is running or completed
    task = runtime.tasks[task_id]
    await task.wait()
    assert task.status == "completed"
    assert task.result is not None
    assert "primes" in task.result

    # Wait for completion buffer flush and foreground wake
    import asyncio
    await asyncio.sleep(0.1)

    assert len(woken) == 1
    assert "prime" in woken[0].lower()


@pytest.mark.asyncio
async def test_runtime_status_and_cancel(mock_provider, tmp_path):
    runtime = Runtime(
        workspace=tmp_path,
        provider=mock_provider,
        buffer_window=0.05,
    )

    # Status with no tasks
    empty_status = runtime.status()
    assert "no active" in empty_status.lower()

    # Cancel with no tasks
    empty_cancel = await runtime.cancel()
    assert "no active" in empty_cancel.lower()

    # Spawn a slow task
    slow_provider = MagicMock()

    async def hanging_stream(messages, tools=None, reasoning_effort=None):
        import asyncio
        while True:
            yield TextDelta(text="working...")
            await asyncio.sleep(0.01)

    slow_provider.stream = hanging_stream
    runtime.provider = slow_provider

    t1 = await runtime.spawn("slow task 1")
    import asyncio
    await asyncio.sleep(0.02)

    # Status shows active task
    st = runtime.status()
    assert t1 in st
    assert "slow task 1" in st

    # Single active task cancel with no args
    c_res = await runtime.cancel()
    assert "Cancelled" in c_res
    assert t1 in c_res
    assert runtime.tasks[t1].status == "cancelled"


@pytest.mark.asyncio
async def test_runtime_cancelled_task_does_not_fire_announcer(tmp_path):
    import asyncio

    provider = MagicMock()

    async def fake_stream(messages, tools=None, reasoning_effort=None):
        last_msg = messages[-1]["content"] if isinstance(messages[-1], dict) else getattr(messages[-1], "content", "")
        if "[Background task completion update" in last_msg:
            yield ProviderTextDelta(text="Task cancelled.")
            yield ProviderFinish(reason="stop")
        else:
            while True:
                yield ProviderTextDelta(text="working...")
                await asyncio.sleep(0.01)

    provider.stream = fake_stream

    emitted_events = []
    class FakeOutput:
        async def emit(self, event):
            emitted_events.append(event)

    output = FakeOutput()
    runtime = Runtime(workspace=tmp_path, provider=provider, buffer_window=0.02)
    runtime.wake_output = output

    task_id = await runtime.spawn("task to be cancelled")
    await asyncio.sleep(0.02)

    # Cancel the task
    cancel_res = await runtime.cancel(task_id)
    assert "Cancelled" in cancel_res
    assert runtime.tasks[task_id].status == "cancelled"

    # Wait beyond buffer window
    await asyncio.sleep(0.06)

    # Announcer must NOT have emitted any events
    assert len(emitted_events) == 0


@pytest.mark.asyncio
async def test_dual_emission_spoken_ack_and_tool_call(tmp_path):
    from verbix.provider.models import ToolCallDelta as ProviderToolCallDelta

    dual_provider = MagicMock()
    turn_counter = 0

    async def dual_stream(messages, tools=None, reasoning_effort=None):
        nonlocal turn_counter
        turn_counter += 1
        if turn_counter == 1:
            # Dual-emission: spoken acknowledgment + tool call in SAME turn
            yield ProviderTextDelta(text="On it, calculating primes now.")
            yield ProviderToolCallDelta(
                index=0,
                call_id="call_spawn_1",
                name="spawn",
                arguments_delta='{"objective": "calculate primes"}',
            )
            yield ProviderFinish(reason="tool_calls")
        else:
            yield ProviderTextDelta(text="Background worker is on the job.")
            yield ProviderFinish(reason="stop")

    dual_provider.stream = dual_stream
    runtime = Runtime(workspace=tmp_path, provider=dual_provider)

    events = []
    async for ev in runtime.step("Calculate primes in background"):
        events.append(ev)

    # Verify pre-tool text is suppressed and only post-tool speech was emitted
    spoken_deltas = [e.text for e in events if isinstance(e, TextDelta)]
    assert "On it, calculating primes now." not in spoken_deltas
    assert "Background worker is on the job." in spoken_deltas

    # Verify spawn tool was executed and task was created
    assert "task-1" in runtime.tasks
    assert runtime.tasks["task-1"].description == "calculate primes"


@pytest.mark.asyncio
async def test_runtime_prompt_grounding(mock_provider, tmp_path):
    import platform
    runtime = Runtime(workspace=tmp_path, provider=mock_provider)
    assert "# IDENTITY & TONE" in runtime.system_prompt
    assert "# VOICE & AUDIO CONSTRAINTS" in runtime.system_prompt
    assert "# CAPABILITY ROUTING" in runtime.system_prompt
    assert "# TOOL OPERATIONS & BACKGROUND WORK" in runtime.system_prompt
    assert "# ENVIRONMENT & GROUNDING" in runtime.system_prompt
    # Volatile clock content stays out of the cacheable system prompt ...
    assert "Current local date and time:" not in runtime.system_prompt
    assert f"Host operating system: {platform.system()}" in runtime.system_prompt
    assert f"The working directory is {tmp_path.resolve()}" in runtime.system_prompt
    # ... and rides on the user message at request time only.
    events = [e async for e in runtime.step("hello")]
    assert events
    user_msgs = [m for m in runtime.foreground.messages if m.get("role") == "user"]
    assert any("Current local date and time:" in (m.get("content") or "") for m in user_msgs)


@pytest.mark.asyncio
async def test_runtime_layered_custom_system_prompt(mock_provider, tmp_path):
    custom = "Speak like a 1920s radio announcer, see?"
    runtime = Runtime(workspace=tmp_path, provider=mock_provider, system_prompt=custom)
    # Core invariants must remain intact
    assert "# IDENTITY & TONE" in runtime.system_prompt
    assert "# VOICE & AUDIO CONSTRAINTS" in runtime.system_prompt
    assert "# ENVIRONMENT & GROUNDING" in runtime.system_prompt
    # Custom instructions must be layered cleanly at the bottom
    assert "# CUSTOM USER INSTRUCTIONS" in runtime.system_prompt
    assert "Speak like a 1920s radio announcer, see?" in runtime.system_prompt


@pytest.mark.asyncio
async def test_runtime_announcer_layered_custom_system_prompt(mock_provider, tmp_path):
    custom = "Speak like a caveman"
    runtime = Runtime(workspace=tmp_path, provider=mock_provider, system_prompt=custom)
    # Announcer must retain voice constraints and tone
    assert "# IDENTITY & TONE" in runtime.announcer.config.system_template
    assert "# VOICE & AUDIO CONSTRAINTS" in runtime.announcer.config.system_template
    assert "# ENVIRONMENT & GROUNDING" in runtime.announcer.config.system_template
    # Announcer must inherit custom user instructions
    assert "# CUSTOM USER INSTRUCTIONS" in runtime.announcer.config.system_template
    assert "Speak like a caveman" in runtime.announcer.config.system_template



@pytest.mark.asyncio
async def test_runtime_direct_worker_announcement(tmp_path):
    provider = MagicMock()
    stream_calls = 0

    async def fake_stream(messages, tools=None, reasoning_effort=None):
        nonlocal stream_calls
        stream_calls += 1
        last_msg = messages[-1].content if messages else ""
        if "[Background task completion update" in last_msg:
            # Receptionist synthesis turn
            yield ProviderTextDelta(text="I've finished the task: the answer is 42.")
            yield ProviderFinish(reason="stop")
        else:
            # Worker execution
            yield ProviderTextDelta(text="Calculation finished: 42.")
            yield ProviderFinish(reason="stop")

    provider.stream = fake_stream

    emitted_events = []
    class FakeOutput:
        async def emit(self, event):
            emitted_events.append(event)

    output = FakeOutput()
    runtime = Runtime(workspace=tmp_path, provider=provider, buffer_window=0.02)
    runtime.wake_output = output

    # Spawn worker
    task_id = await runtime.spawn("find the answer")
    task = runtime.tasks[task_id]
    await task.wait()
    assert task.status == "completed"

    import asyncio
    await asyncio.sleep(0.05)

    # 1. Synthesized announcement must be emitted through wake_output
    from verbix.core.events import TurnComplete
    turn_completes = [e for e in emitted_events if isinstance(e, TurnComplete)]
    assert len(turn_completes) == 1
    assert "I've finished the task: the answer is 42." in turn_completes[0].final_text

    # 2. Synthesized answer was committed to foreground history
    history_msgs = runtime.foreground.history.messages
    assert any(m.role == "assistant" and "I've finished the task: the answer is 42." in m.content for m in history_msgs)

    # 3. Provider stream was called twice: once by worker, once by foreground synthesis
    assert stream_calls == 2


@pytest.mark.asyncio
async def test_status_includes_completed_task_result(tmp_path):
    mock_provider = MagicMock()
    runtime = Runtime(workspace=tmp_path, provider=mock_provider)
    from verbix.runtime.worker import WorkerTask
    t = WorkerTask(id="task-1", description="check time", session=runtime.foreground)
    t._status = "completed"
    t._result = "The current time is 12:45 AM."
    runtime.tasks["task-1"] = t

    # General status listing
    general_status = runtime.status()
    assert "task-1" in general_status
    assert "completed" in general_status
    assert "The current time is 12:45 AM." in general_status

    # Specific task status query
    task_status = runtime.status(task_id="task-1")
    assert "The current time is 12:45 AM." in task_status

    # Specific full status query
    task_full_status = runtime.status(task_id="task-1", full=True)
    assert "The current time is 12:45 AM." in task_full_status


@pytest.mark.asyncio
async def test_runtime_uses_miniswe_agent(tmp_path):
    from verbix.minisweagent.agents.default import DefaultAgent
    from verbix.provider.client import ProviderClient
    from verbix.runtime.runtime import Runtime
    # Ensure runtime foreground is powered by DefaultAgent
    runtime = Runtime(workspace=tmp_path, provider=ProviderClient())
    assert isinstance(runtime.foreground, DefaultAgent)
    assert hasattr(runtime.foreground, "config")
    assert hasattr(runtime.foreground, "step_stream")


@pytest.mark.asyncio
async def test_runtime_buffer_flush_failure_does_not_blurt_raw_markdown(tmp_path):
    provider = MagicMock()

    async def fake_stream(messages, tools=None, reasoning_effort=None):
        last_msg = messages[-1].content if messages else ""
        if "[Background task completion update" in last_msg:
            # Synthesis fails (e.g. LLM provider error after retries)
            raise RuntimeError("API connection error")
        else:
            # Worker execution succeeds
            yield ProviderTextDelta(text="# Markdown Result\n- Item 1\n- Item 2")
            yield ProviderFinish(reason="stop")

    provider.stream = fake_stream

    emitted_events = []
    class FakeOutput:
        async def emit(self, event):
            emitted_events.append(event)

    output = FakeOutput()
    on_wake_called = []
    runtime = Runtime(
        workspace=tmp_path,
        provider=provider,
        buffer_window=0.02,
        on_wake=lambda text: on_wake_called.append(text),
    )
    runtime.wake_output = output

    task_id = await runtime.spawn("run task")
    task = runtime.tasks[task_id]
    await task.wait()
    assert task.status == "completed"

    import asyncio
    await asyncio.sleep(0.05)

    # Failed synthesis should log and be silent, never emitting raw markdown to wake_output or on_wake
    assert len(emitted_events) == 0
    assert len(on_wake_called) == 0


@pytest.mark.asyncio
async def test_runtime_announcer_clean_history(tmp_path):
    provider = MagicMock()

    async def fake_stream(messages, tools=None, reasoning_effort=None):
        last_msg = messages[-1].content if messages else ""
        if "[Background task completion update" in last_msg:
            # Announcer synthesis turn (must receive NO tools!)
            assert tools is None or len(tools) == 0
            yield ProviderTextDelta(text="Task finished: all 42 items processed.")
            yield ProviderFinish(reason="stop")
        else:
            # Worker execution
            yield ProviderTextDelta(text="Calculation complete: 42.")
            yield ProviderFinish(reason="stop")

    provider.stream = fake_stream

    emitted_events = []
    class FakeOutput:
        async def emit(self, event):
            emitted_events.append(event)

    output = FakeOutput()
    runtime = Runtime(workspace=tmp_path, provider=provider, buffer_window=0.02)
    runtime.wake_output = output

    # 1. Announcer exists and has ZERO tools
    assert hasattr(runtime, "announcer")
    assert runtime.announcer is not None
    assert runtime.announcer.tools is None or len(runtime.announcer.tools.get_definitions()) == 0

    task_id = await runtime.spawn("process items")
    task = runtime.tasks[task_id]
    await task.wait()
    assert task.status == "completed"

    import asyncio
    await asyncio.sleep(0.05)

    # 2. Spoken announcement delivered to wake_output
    from verbix.core.events import TurnComplete
    turn_completes = [e for e in emitted_events if isinstance(e, TurnComplete)]
    assert len(turn_completes) == 1
    assert "Task finished: all 42 items processed." in turn_completes[0].final_text

    # 3. Companion history contains ONLY the assistant speech, NEVER internal wake prompt
    msgs = runtime.foreground.messages
    assert not any(isinstance(m, dict) and "[Background task completion update" in str(m.get("content", "")) for m in msgs)
    assert any(isinstance(m, dict) and m.get("role") == "assistant" and "Task finished: all 42 items processed." in str(m.get("content", "")) for m in msgs)


@pytest.mark.asyncio
async def test_runtime_announcer_multiple_flushes_never_exceeds_limits(tmp_path):
    provider = MagicMock()
    call_idx = 0

    async def fake_stream(messages, tools=None, reasoning_effort=None):
        nonlocal call_idx
        call_idx += 1
        yield ProviderTextDelta(text=f"Announcement number {call_idx}.")
        yield ProviderFinish(reason="stop")

    provider.stream = fake_stream

    emitted_events = []
    class FakeOutput:
        async def emit(self, event):
            emitted_events.append(event)

    output = FakeOutput()
    runtime = Runtime(workspace=tmp_path, provider=provider)
    runtime.wake_output = output

    # Flush 1
    await runtime._on_buffer_flush([("task-1", "Result 1")])
    # Flush 2
    await runtime._on_buffer_flush([("task-2", "Result 2")])
    # Flush 3
    await runtime._on_buffer_flush([("task-3", "Result 3")])

    from verbix.core.events import TurnComplete
    turn_completes = [e for e in emitted_events if isinstance(e, TurnComplete)]
    assert len(turn_completes) == 3
    assert turn_completes[0].final_text == "Announcement number 1."
    assert turn_completes[1].final_text == "Announcement number 2."
    assert turn_completes[2].final_text == "Announcement number 3."
    assert not any(e.final_text == "LimitsExceeded" for e in turn_completes)




