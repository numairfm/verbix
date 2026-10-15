import json
from unittest.mock import MagicMock

import pytest

from verbix.core.events import TextDelta
from verbix.provider.models import Finish as ProviderFinish
from verbix.provider.models import TextDelta as ProviderTextDelta
from verbix.provider.models import ToolCallDelta as ProviderToolCallDelta
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
async def test_runtime_spawn_and_completion_loop(tmp_path):
    woken = []

    async def on_wake(speech_text):
        woken.append(speech_text)

    stream_calls = 0

    async def finishing_stream(messages, tools=None, **kwargs):
        """Worker executes bash then finishes with text; later calls answer normally."""
        nonlocal stream_calls
        stream_calls += 1
        if stream_calls == 1:
            yield ProviderToolCallDelta(
                index=0,
                call_id="c1",
                name="bash",
                arguments_delta=json.dumps({"command": "echo 'primes'"}),
            )
            yield ProviderFinish(reason="tool_calls")
        elif stream_calls == 2:
            yield ProviderTextDelta(text="2, 3, 5, 7, 11 are primes.")
            yield ProviderFinish(reason="stop")
        else:
            yield ProviderTextDelta(text="The prime calculation finished: 2, 3, 5, 7, 11.")
            yield ProviderFinish(reason="stop")

    provider = MagicMock()
    provider.stream = finishing_stream
    runtime = Runtime(
        workspace=tmp_path,
        provider=provider,
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

    # Wait for completion to settle and announce
    import asyncio
    for _ in range(20):
        if woken:
            break
        await asyncio.sleep(0.02)

    # Proactive completion announced to on_wake
    assert len(woken) == 1
    assert "prime" in woken[0].lower()
    # Ledger records completion result
    assert "primes" in runtime.ledger.get_status(task_id)


@pytest.mark.asyncio
async def test_runtime_status_and_cancel(mock_provider, tmp_path):
    runtime = Runtime(
        workspace=tmp_path,
        provider=mock_provider,
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
    runtime = Runtime(workspace=tmp_path, provider=provider)
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
    assert "# DELIVERY" in runtime.system_prompt
    assert "# TOOLS" in runtime.system_prompt
    assert "# INTERRUPTIONS" in runtime.system_prompt
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
    assert "# DELIVERY" in runtime.system_prompt
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
async def test_runtime_direct_worker_completion_is_silent(tmp_path):
    provider = MagicMock()
    stream_calls = 0

    async def fake_stream(messages, tools=None, reasoning_effort=None):
        nonlocal stream_calls
        stream_calls += 1
        if stream_calls == 1:
            yield ProviderToolCallDelta(
                index=0,
                call_id="c1",
                name="read",
                arguments_delta=json.dumps({"path": "."}),
            )
            yield ProviderFinish(reason="tool_calls")
        else:
            yield ProviderTextDelta(text="Calculation finished: 42.")
            yield ProviderFinish(reason="stop")

    provider.stream = fake_stream

    emitted_events = []
    class FakeOutput:
        async def emit(self, event):
            emitted_events.append(event)

    output = FakeOutput()
    runtime = Runtime(workspace=tmp_path, provider=provider)
    runtime.wake_output = output

    # Spawn worker
    task_id = await runtime.spawn("find the answer")
    task = runtime.tasks[task_id]
    await task.wait()
    assert task.status == "completed"

    import asyncio
    for _ in range(20):
        if emitted_events:
            break
        await asyncio.sleep(0.02)

    # 1. Worker completion announced via wake_output
    assert len(emitted_events) > 0

    # 2. Result recorded in ledger
    assert "Calculation finished: 42." in runtime.ledger.get_status(task_id)

    # 3. Context notice recorded in foreground messages
    assert any(f"[Task '{task_id}' completed]" in str(m.get("content", "")) for m in runtime.foreground.messages)


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
async def test_runtime_uses_voice_agent(tmp_path):
    from unittest.mock import MagicMock

    from verbix.agent.voice import VoiceAgent
    from verbix.runtime.runtime import Runtime

    # Ensure runtime foreground is powered by VoiceAgent
    runtime = Runtime(workspace=tmp_path, provider=MagicMock())
    assert isinstance(runtime.foreground, VoiceAgent)
    assert hasattr(runtime.foreground, "config")
    assert hasattr(runtime.foreground, "step_stream")


@pytest.mark.asyncio
async def test_runtime_worker_markdown_result_is_silent(tmp_path):
    provider = MagicMock()
    stream_calls = 0

    async def fake_stream(messages, tools=None, reasoning_effort=None):
        nonlocal stream_calls
        stream_calls += 1
        if stream_calls == 1:
            yield ProviderToolCallDelta(
                index=0,
                call_id="c1",
                name="read",
                arguments_delta=json.dumps({"path": "."}),
            )
            yield ProviderFinish(reason="tool_calls")
        else:
            yield ProviderTextDelta(text="# Markdown Result\n- Item 1\n- Item 2")
            yield ProviderFinish(reason="stop")

    provider.stream = fake_stream

    emitted_events = []
    class FakeOutput:
        async def emit(self, event):
            emitted_events.append(event)

    output = FakeOutput()
    runtime = Runtime(workspace=tmp_path, provider=provider)
    runtime.wake_output = output

    task_id = await runtime.spawn("run task")
    task = runtime.tasks[task_id]
    await task.wait()
    assert task.status == "completed"

    import asyncio
    for _ in range(20):
        if emitted_events:
            break
        await asyncio.sleep(0.02)

    # Worker output announced to wake_output
    assert len(emitted_events) > 0
    assert "# Markdown Result" in runtime.ledger.get_status(task_id)


@pytest.mark.asyncio
async def test_runtime_announcer_is_dedicated_toolless_agent(tmp_path):
    mock_provider = MagicMock()
    runtime = Runtime(workspace=tmp_path, provider=mock_provider)
    assert hasattr(runtime, "announcer")
    assert runtime.announcer is not runtime.foreground
    assert runtime.announcer.get_tool_definitions() is None
    # Announcer history is isolated: sharing the foreground list lets the
    # tool-less announcer echo tool calls it can never execute.
    assert runtime.announcer.messages is not runtime.foreground.messages


def test_task_finished_logs_worker_payload(tmp_path, caplog):
    import logging

    from verbix.runtime.floor import FloorState
    from verbix.runtime.runtime import Runtime

    runtime = Runtime(workspace=tmp_path, provider=MagicMock())
    runtime.floor.set_state(FloorState.LISTENING)
    with caplog.at_level(logging.DEBUG, logger="verbix.runtime"):
        runtime._on_task_finished("task-9", "found two text files: a.txt, b.txt")
    assert "task-9" in caplog.text
    assert "a.txt" in caplog.text


@pytest.mark.asyncio
async def test_runtime_multiple_completions_silently_recorded_in_ledger(tmp_path):
    import asyncio
    provider = MagicMock()
    emitted_events = []
    class FakeOutput:
        async def emit(self, event):
            emitted_events.append(event)

    output = FakeOutput()
    runtime = Runtime(workspace=tmp_path, provider=provider)
    runtime.wake_output = output

    try:
        for i in ("task-1", "task-2", "task-3"):
            runtime.ledger.register_task(i, f"objective {i}")
            runtime._on_task_finished(i, f"Result {i}")

        for _ in range(20):
            if emitted_events:
                break
            await asyncio.sleep(0.02)

        assert "Result task-1" in runtime.ledger.get_status("task-1")
        assert "Result task-2" in runtime.ledger.get_status("task-2")
        assert "Result task-3" in runtime.ledger.get_status("task-3")
        assert len(emitted_events) > 0
    finally:
        await runtime.aclose()






@pytest.mark.asyncio
async def test_cancel_turn_when_idle_leaves_history_unchanged(tmp_path):
    runtime = Runtime(workspace=tmp_path, provider=mock_provider)
    try:
        runtime.foreground.messages.extend(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hello! How can I help?"},
            ]
        )
        before = [dict(m) for m in runtime.foreground.messages]
        await runtime.cancel_turn()
        assert runtime.foreground.messages == before
    finally:
        await runtime.aclose()


def test_finish_log_redacts_keys(tmp_path, caplog):
    import logging

    from verbix.runtime.floor import FloorState

    runtime = Runtime(workspace=tmp_path, provider=MagicMock())
    runtime.floor.set_state(FloorState.LISTENING)
    runtime.ledger.register_task("task-1", "check keys")
    secret = "sk-secret-value"
    with caplog.at_level(logging.DEBUG, logger="verbix.runtime"):
        runtime._on_task_finished(
            "task-1",
            f"done OPENROUTER_API_KEY={secret} and {{\"api_key\": \"{secret}\"}} ok",
        )
    assert secret not in caplog.text
    assert "REDACTED" in caplog.text
    assert runtime.ledger.get("task-1").result_summary.count(secret) == 2


def test_runtime_status_without_task_id_includes_recent_actions_and_honors_full(tmp_path):
    from verbix.runtime.worker import WorkerTask

    runtime = Runtime(workspace=tmp_path, provider=MagicMock())
    session = MagicMock(spec=["messages"])
    session.messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "function": {"name": "read", "arguments": '{"path": "README.md"}'}}
            ],
        },
        {"role": "tool", "content": "showing lines 1-10 of 10\n# Verbix"},
    ]
    t = WorkerTask(id="task-1", description="read docs", session=session)
    t._status = "running"
    runtime.tasks["task-1"] = t

    concise = runtime.status()
    assert "Called read" in concise

    full = runtime.status(full=True)
    assert 'read({"path": "README.md"})' in full


def test_foreground_grounding_and_spawn_tool_derive_worker_capabilities_once_per_runtime(
    tmp_path, monkeypatch
):
    from verbix.tools.sandbox import SandboxStatus
    from verbix.tools.spawn import SpawnTool

    # 1. When sandbox is available, worker capabilities include screenshot, web_search, and bash
    monkeypatch.setattr(
        "verbix.tools.sandbox.probe_sandbox",
        lambda allow_network=False: SandboxStatus(
            available=True, backend="bwrap", reason="ok", capabilities="sandboxed bash"
        ),
    )
    rt_with_bash = Runtime(workspace=tmp_path, provider=MagicMock())
    assert "screenshot" in rt_with_bash.system_prompt
    assert "web_search" in rt_with_bash.system_prompt
    assert "web_fetch" in rt_with_bash.system_prompt
    assert "bash" in rt_with_bash.system_prompt
    assert "screenshot" in SpawnTool(rt_with_bash).description
    assert hasattr(rt_with_bash, "worker_capabilities")
    assert hasattr(rt_with_bash, "worker_tool_names")

    # 2. When sandbox is unavailable, bash is automatically omitted from worker capabilities
    monkeypatch.setattr(
        "verbix.tools.sandbox.probe_sandbox",
        lambda allow_network=False: SandboxStatus(
            available=False, backend=None, reason="no bwrap", capabilities=""
        ),
    )
    rt_no_bash = Runtime(workspace=tmp_path, provider=MagicMock())
    assert "screenshot" in rt_no_bash.system_prompt
    assert "bash" not in rt_no_bash.system_prompt
    assert "bash" not in SpawnTool(rt_no_bash).description


@pytest.mark.asyncio
async def test_scripted_live_session_replay_has_no_duplicate_spawns_and_no_double_spoken_results(
    tmp_path,
):
    import asyncio

    from verbix.provider.models import (
        Finish as ProviderFinish,
    )
    from verbix.provider.models import (
        TextDelta as ProviderTextDelta,
    )
    from verbix.provider.models import (
        ToolCallDelta as ProviderToolCallDelta,
    )
    from verbix.runtime.floor import FloorState

    spawned_objectives: list[str] = []
    seen_turn_prompts: list[str] = []
    announcer_spoken: list[str] = []

    provider = MagicMock()

    async def scripted_stream(messages, tools=None, **kw):
        last_msg = messages[-1].content if messages else ""
        seen_turn_prompts.append(last_msg)

        if "[Present these completed findings" in last_msg:
            text = "Proactively announcing: Nvidia closed at 228 dollars and 52 cents."
            announcer_spoken.append(text)
            yield ProviderTextDelta(text=text)
            yield ProviderFinish(reason="stop")
            return

        # If this is the post-tool continuation after spawn, finish the turn
        if messages and messages[-1].role == "tool":
            yield ProviderFinish(reason="stop")
            return

        if "What's on my screen?" in last_msg:
            yield ProviderTextDelta(text="Checking your screen now.")
            yield ProviderToolCallDelta(
                index=0,
                call_id="c1",
                name="spawn",
                arguments_delta='{"objective": "Capture screenshot and describe screen"}',
            )
            yield ProviderFinish(reason="tool_calls")
        elif "Nvidia bought Hugging Face" in last_msg:
            # Sees active task-1 in grounding, spawns only the new verification task
            assert "[Active background tasks: task-1" in last_msg
            yield ProviderTextDelta(text="Looking into that deal now.")
            yield ProviderToolCallDelta(
                index=0,
                call_id="c2",
                name="spawn",
                arguments_delta='{"objective": "Verify if NVIDIA acquired Hugging Face for $13B"}',
            )
            yield ProviderFinish(reason="tool_calls")
        elif "What's the status?" in last_msg:
            # Sees Newly completed background tasks in grounding -> reports directly without re-spawning!
            assert "[Newly completed background tasks" in last_msg
            assert "No acquisition happened." in last_msg
            yield ProviderTextDelta(
                text="Your screen shows VS Code, and Nvidia did not acquire Hugging Face."
            )
            yield ProviderFinish(reason="stop")
        elif "Nvidia stock price" in last_msg:
            yield ProviderTextDelta(text="Checking Nvidia's stock price.")
            yield ProviderToolCallDelta(
                index=0,
                call_id="c3",
                name="spawn",
                arguments_delta='{"objective": "Get current NVIDIA stock price"}',
            )
            yield ProviderFinish(reason="tool_calls")
        elif "Wait, is that right?" in last_msg:
            # Follows up on most recent exchange (NVDA stock price), does NOT re-spawn Hugging Face
            yield ProviderTextDelta(
                text="Yes, 228 dollars and 52 cents was the latest closing price."
            )
            yield ProviderFinish(reason="stop")

    provider.stream = scripted_stream
    rt = Runtime(workspace=tmp_path, provider=provider)

    async def fake_spawn(task: str) -> str:
        spawned_objectives.append(task)
        tid = f"task-{len(spawned_objectives)}"
        from verbix.runtime.worker import WorkerTask

        wt = WorkerTask(id=tid, description=task, session=MagicMock(messages=[]))
        wt._status = "running"
        rt.tasks[tid] = wt
        rt.ledger.register_task(tid, task)
        return f"Started background task {tid}: {task}"

    rt.spawn = fake_spawn

    # Hold floor in LISTENING (PTT recording active) during turns 1-2
    rt.floor.set_state(FloorState.LISTENING)

    # Turn 1: "What's on my screen?" -> spawns task-1 (screenshot is in worker capabilities)
    assert "screenshot" in rt.system_prompt
    await rt.submit("What's on my screen?")
    assert len(spawned_objectives) == 1

    # Turn 2: "Someone told me Nvidia bought Hugging Face for 13B" -> sees active task-1, spawns task-2
    rt.floor.set_state(FloorState.LISTENING)
    await rt.submit("Someone told me Nvidia bought Hugging Face for 13B, is that true?")
    assert len(spawned_objectives) == 2

    # Tasks 1 and 2 complete while user is recording in PTT (LISTENING) before asking "What's the status?"
    rt.floor.set_state(FloorState.LISTENING)
    rt.tasks["task-1"]._status = "completed"
    rt.tasks["task-1"]._result = "Screen shows VS Code."
    rt._on_task_finished("task-1", "Screen shows VS Code.")
    rt.tasks["task-2"]._status = "completed"
    rt.tasks["task-2"]._result = "No acquisition happened."
    rt._on_task_finished("task-2", "No acquisition happened.")

    # Turn 3: "What's the status?" -> folds both completions into the turn, marks them heard, no announcer double-speak!
    reply3 = await rt.submit("What's the status?")
    assert "did not acquire Hugging Face" in reply3
    assert rt.ledger.get("task-1").delivery == "heard"
    assert rt.ledger.get("task-2").delivery == "heard"
    assert announcer_spoken == []

    # Turn 4: "What's the Nvidia stock price?" -> spawns task-3
    await rt.submit("What's the Nvidia stock price?")
    assert len(spawned_objectives) == 3

    # Task 3 finishes while floor is IDLE -> announced immediately!
    assert rt.floor.state == FloorState.IDLE
    rt.tasks["task-3"]._status = "completed"
    rt.tasks["task-3"]._result = "$228.52"
    rt._on_task_finished("task-3", "$228.52")
    for _ in range(30):
        if rt.ledger.get("task-3").delivery == "heard":
            break
        await asyncio.sleep(0.02)
    assert rt.ledger.get("task-3").delivery == "heard"
    assert len(announcer_spoken) == 1

    # Turn 5: "Wait, is that right?" -> resolves against NVDA stock price, no duplicate spawn!
    reply5 = await rt.submit("Wait, is that right?")
    assert "228" in reply5
    assert len(spawned_objectives) == 3
    await rt.aclose()



