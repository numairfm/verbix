from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.test_regressions import assert_valid_history
from verbix.core.models import WorkerResult
from verbix.provider.client import ProviderClient
from verbix.provider.models import Finish, TextDelta, ToolCallDelta
from verbix.runtime.orchestration import (
    TaskCancelTool,
    TaskSpawnTool,
    TaskStatusTool,
    TaskSteerTool,
    WorkerFinishTool,
)
from verbix.runtime.tasks import DefaultTurnOutput, TaskRuntime
from verbix.runtime.worktree import WorktreeManager
from verbix.tools.base import ToolContext


@pytest.fixture
def mock_provider() -> ProviderClient:
    return ProviderClient(
        base_url="http://mock-provider/v1",
        model="mock-model",
        api_key="mock-key",
    )


@pytest.mark.asyncio
async def test_worktree_fallback(tmp_path: Path):
    manager = WorktreeManager(root_workspace=tmp_path)
    worktree = await manager.create_worktree("task-fallback-1")
    assert worktree.exists()
    assert worktree.is_dir()
    assert "task-fallback-1" in str(worktree)

    await manager.remove_worktree("task-fallback-1")
    assert not worktree.exists()


@pytest.mark.asyncio
async def test_worker_finish_tool(tmp_path: Path):
    tool = WorkerFinishTool()
    ctx = ToolContext(workspace=tmp_path)

    long_summary = "A" * 250
    result = await tool.execute(
        {
            "success": True,
            "speech_summary": long_summary,
            "details": "All unit tests passed with 100% coverage.",
        },
        ctx,
    )
    assert not result.is_error
    assert tool.captured_result is not None
    assert tool.captured_result.success is True
    assert len(tool.captured_result.speech_summary) <= 180
    assert tool.captured_result.details == "All unit tests passed with 100% coverage."


@pytest.mark.asyncio
async def test_task_runtime_spawn_and_finish(
    tmp_path: Path, mock_provider: ProviderClient
):
    runtime = TaskRuntime(workspace=tmp_path, provider=mock_provider)
    receptionist = runtime.create_receptionist()
    out = DefaultTurnOutput()
    runtime.wake_output = out

    task_id = await runtime.spawn_task(title="Build Project", prompt="Run build")
    assert task_id.startswith("task-")
    task = runtime.get_task(task_id)
    assert task is not None
    assert task.status == "running"
    assert task.is_running is True

    # Simulate finish
    worker_res = WorkerResult(
        success=True,
        speech_summary="Build succeeded.",
        details="Compiled 12 files without errors.",
    )
    runtime._finish_task(task_id, worker_res, status="completed")

    assert task.status == "completed"
    assert task.is_running is False
    assert task.result == worker_res

    # Give wake turn time to run
    await asyncio.sleep(0.1)

    # When a foreground turn runs, history records completion notice
    out2 = DefaultTurnOutput()
    await runtime.submit("Status?", out2)

    assert any(
        f"[Task completed: 'Build Project' ({task_id})" in m.content
        for m in receptionist.history.messages
    )
    assert_valid_history(receptionist.history.messages)


@pytest.mark.asyncio
async def test_ephemeral_running_task_prompt_annotation(
    tmp_path: Path, mock_provider: ProviderClient
):
    runtime = TaskRuntime(workspace=tmp_path, provider=mock_provider)
    receptionist = runtime.create_receptionist()
    receptionist.history.append_user("What is the status?")

    task_id = await runtime.spawn_task(title="Compile Core", prompt="Compile")
    task = runtime.get_task(task_id)
    assert task is not None
    task.step_count = 5

    # Render prompt with active running tasks
    active_tasks = runtime.list_active_tasks()
    rendered = receptionist.history.render_turn_prompt(active_tasks)

    # The rendered prompt has the ephemeral annotation with tool calls
    assert len(rendered) == 1
    assert f"[Running: {task_id} 'Compile Core' (5 tool calls)]" in rendered[0].content

    # The underlying history was NOT mutated
    assert "[Running:" not in receptionist.history.messages[0].content

    # Complete task
    runtime._finish_task(
        task_id,
        WorkerResult(success=True, speech_summary="Done", details="Log"),
    )
    # Once finished, it must NEVER appear in running task annotation
    rendered_after = receptionist.history.render_turn_prompt(
        runtime.list_active_tasks()
    )
    assert "[Running:" not in rendered_after[0].content


@pytest.mark.asyncio
async def test_task_runtime_cancellation(tmp_path: Path, mock_provider: ProviderClient):
    runtime = TaskRuntime(workspace=tmp_path, provider=mock_provider)
    runtime.create_receptionist()

    task_id = await runtime.spawn_task(title="Cancel Me", prompt="Neverending")
    cancelled = await runtime.cancel_task(task_id)
    assert cancelled is True

    task = runtime.get_task(task_id)
    assert task is not None
    assert task.status == "cancelled"
    assert task.result is not None
    assert "cancelled" in task.result.speech_summary.lower()
    assert task.recorded is True
    assert task.announced is True


@pytest.mark.asyncio
async def test_orchestration_tools(tmp_path: Path, mock_provider: ProviderClient):
    runtime = TaskRuntime(workspace=tmp_path, provider=mock_provider)
    ctx = ToolContext(workspace=tmp_path)

    spawn_tool = TaskSpawnTool(runtime)
    status_tool = TaskStatusTool(runtime)
    steer_tool = TaskSteerTool(runtime)
    cancel_tool = TaskCancelTool(runtime)

    # 1. Spawn
    res = await spawn_tool.execute({"title": "Test Task", "prompt": "Run test"}, ctx)
    assert not res.is_error
    assert res.is_terminal is True
    assert "Starting task: Test Task." in res.observation

    active_tasks = runtime.list_active_tasks()
    assert len(active_tasks) == 1
    task_id = active_tasks[0].id

    # 2. Status
    res_status = await status_tool.execute({"task_id": task_id}, ctx)
    assert not res_status.is_error
    assert "running" in res_status.observation.lower()

    # 3. Steer
    res_steer = await steer_tool.execute(
        {"task_id": task_id, "instruction": "Hurry up"}, ctx
    )
    assert not res_steer.is_error
    assert "delivered" in res_steer.observation.lower()

    # 4. Cancel
    res_cancel = await cancel_tool.execute({"task_id": task_id}, ctx)
    assert not res_cancel.is_error
    assert "cancelled" in res_cancel.observation.lower()


@pytest.mark.asyncio
async def test_task_completion_sync_before_foreground_turn(
    tmp_path: Path, mock_provider: ProviderClient
):
    runtime = TaskRuntime(workspace=tmp_path, provider=mock_provider)
    receptionist = runtime.create_receptionist()

    # 1. Spawn background task
    task_id = await runtime.spawn_task(title="Analyze repo", prompt="Analyze workspace")
    task = runtime.get_task(task_id)
    assert task is not None
    assert task.is_running is True

    # 2. Task completes
    worker_res = WorkerResult(
        success=True,
        speech_summary="Analyzed repository successfully.",
        details="Found pyproject.toml and src/verbix.",
    )
    runtime._finish_task(task_id, worker_res, status="completed")

    # 3. A new user turn arrives immediately afterward
    out = DefaultTurnOutput()
    await runtime.submit("what did it find?", out)

    # INVARIANT 1: Completed task is injected into history
    assert any(
        "Analyzed repository successfully." in m.content
        for m in receptionist.history.messages
    )

    # INVARIANT 2: render_turn_prompt must NEVER include the completed task
    active_tasks = runtime.list_active_tasks()
    assert len(active_tasks) == 0
    prompt_msgs = receptionist.history.render_turn_prompt(active_tasks)
    user_msg = prompt_msgs[-1]
    assert "[Running:" not in user_msg.content


@pytest.mark.asyncio
async def test_worker_finish_tool_immediate_state_transition(
    tmp_path: Path, mock_provider: ProviderClient
):
    runtime = TaskRuntime(workspace=tmp_path, provider=mock_provider)
    runtime.create_receptionist()
    task_id = await runtime.spawn_task(title="Build", prompt="Build core")
    task = runtime.get_task(task_id)
    assert task is not None
    assert task.is_running is True

    finish_tool = WorkerFinishTool(
        on_finish=lambda res: runtime._finish_task(task_id, res, status="completed")
    )
    ctx = ToolContext(workspace=tmp_path)
    res = await finish_tool.execute(
        {
            "success": True,
            "speech_summary": "Build succeeded.",
            "details": "Compiled without warnings.",
        },
        ctx,
    )
    assert res.is_terminal is True
    # Immediate state transition before any further turns
    assert task.status == "completed"
    assert task.is_running is False
    assert len(runtime.list_active_tasks()) == 0


@pytest.mark.asyncio
async def test_task_spawn_workspace_awareness(
    tmp_path: Path, mock_provider: ProviderClient
):
    runtime = TaskRuntime(workspace=tmp_path, provider=mock_provider)
    spawn_tool = TaskSpawnTool(runtime)
    status_tool = TaskStatusTool(runtime)

    assert str(tmp_path) in spawn_tool.description
    assert str(tmp_path) in status_tool.description

    schema = spawn_tool.parameters_schema
    assert "workspace" not in schema["properties"]

    ctx = ToolContext(workspace=tmp_path)
    res = await spawn_tool.execute(
        {
            "title": "Check files",
            "prompt": "List files",
        },
        ctx,
    )
    assert not res.is_error
    assert res.is_terminal is True
    assert "Starting task: Check files." in res.observation

    active_tasks = runtime.list_active_tasks()
    assert len(active_tasks) == 1
    task = active_tasks[0]
    assert task.workspace == tmp_path


@pytest.mark.asyncio
async def test_task_spawn_terminal_turn_and_history_parity(tmp_path: Path):
    calls_made = 0

    class MockSingleTurnProvider(ProviderClient):
        def __init__(self) -> None:
            super().__init__(base_url="http://mock", model="mock")

        async def stream(self, messages, tools=None, **kwargs):  # type: ignore[override]
            nonlocal calls_made
            is_receptionist = any(
                getattr(getattr(t, "function", None), "name", None) == "task_spawn"
                for t in (tools or [])
            )
            if is_receptionist:
                calls_made += 1
                yield ToolCallDelta(
                    index=0,
                    call_id="call-spawn-1",
                    name="task_spawn",
                    arguments_delta='{"title": "Run full test suite", "prompt": "pytest -v"}',
                )
                yield Finish(reason="tool_calls")
            else:
                await asyncio.sleep(0.2)
                yield ToolCallDelta(
                    index=0,
                    call_id="call-finish-1",
                    name="worker_finish",
                    arguments_delta='{"success": true, "speech_summary": "Done", "details": "Done"}',
                )
                yield Finish(reason="tool_calls")

    provider = MockSingleTurnProvider()
    runtime = TaskRuntime(workspace=tmp_path, provider=provider)
    runtime.create_receptionist()

    out = DefaultTurnOutput()
    await runtime.submit("Run the tests", out)

    assert calls_made == 1
    assert runtime.main_session is not None
    last_msg = runtime.main_session.history.messages[-1]
    assert last_msg.role == "assistant"
    assert last_msg.content == "Starting task: Run full test suite."


@pytest.mark.asyncio
async def test_wake_turn_queues_behind_active_user_turn_and_does_not_cancel_it(
    tmp_path: Path,
):
    """A background task completion that finishes mid-turn must queue behind the active turn

    and never cancel the user's active turn.
    """

    class SlowMockProvider(ProviderClient):
        def __init__(self) -> None:
            super().__init__(base_url="http://mock", model="mock")

        async def stream(self, messages, tools=None, **kwargs):  # type: ignore[override]
            await asyncio.sleep(0.08)
            yield TextDelta(text="User response complete.")
            yield Finish(reason="stop")

    provider = SlowMockProvider()
    runtime = TaskRuntime(workspace=tmp_path, provider=provider)
    runtime.create_receptionist()
    wake_out = DefaultTurnOutput()
    runtime.wake_output = wake_out

    # 1. Start an active user turn
    user_out = DefaultTurnOutput()
    user_task = runtime.start_turn("Analyze codebase", user_out, kind="user")

    # 2. While user turn is active, a background task finishes
    await asyncio.sleep(0.02)
    assert not user_task.done()

    task_id = "task-bg-1"
    runtime.tasks[task_id] = runtime.tasks.get(
        task_id,
        runtime.tasks.setdefault(
            task_id,
            runtime.get_task(task_id)  # type: ignore
            or TaskRuntime.spawn_task.__annotations__,  # placeholder
        ),
    )
    from verbix.core.models import TaskState

    runtime.tasks[task_id] = TaskState(
        id=task_id,
        title="Background Lint",
        status="running",
        step_count=2,
    )
    runtime._finish_task(
        task_id,
        WorkerResult(
            success=True,
            speech_summary="Linting passed cleanly.",
            details="All 10 files ok",
        ),
        status="completed",
    )

    # 3. User turn must complete successfully (NOT cancelled!)
    res = await user_task
    assert res == "User response complete."
    assert not user_task.cancelled()

    # 4. Wait for queued wake turn to complete
    await asyncio.sleep(0.12)
    assert runtime.tasks[task_id].announced is True
    assert runtime.main_session is not None
    assert_valid_history(runtime.main_session.history.messages)


@pytest.mark.asyncio
async def test_user_input_cancels_pending_wake_turn(tmp_path: Path):
    """User input arriving while a wake turn is running must cancel the wake turn immediately."""

    class SlowWakeProvider(ProviderClient):
        def __init__(self) -> None:
            super().__init__(base_url="http://mock", model="mock")

        async def stream(self, messages, tools=None, **kwargs):  # type: ignore[override]
            is_user = any("What time is it?" in (m.content or "") for m in messages)
            if is_user:
                yield TextDelta(text="User response immediate.")
            else:
                await asyncio.sleep(0.15)
                yield TextDelta(text="Wake announcement complete.")
            yield Finish(reason="stop")

    provider = SlowWakeProvider()
    runtime = TaskRuntime(workspace=tmp_path, provider=provider)
    runtime.create_receptionist()
    wake_out = DefaultTurnOutput()
    runtime.wake_output = wake_out

    # 1. Background task finishes while idle -> triggers wake turn
    from verbix.core.models import TaskState

    task_id = "task-bg-wake"
    runtime.tasks[task_id] = TaskState(
        id=task_id,
        title="Compile",
        status="running",
        step_count=1,
    )
    runtime._finish_task(
        task_id,
        WorkerResult(success=True, speech_summary="Build succeeded.", details="OK"),
        status="completed",
    )

    # Give wake turn time to start running
    await asyncio.sleep(0.03)
    wake_task = runtime._turn_task
    assert wake_task is not None
    assert not wake_task.done()

    # 2. User inputs a new prompt -> cancels the wake turn!
    user_out = DefaultTurnOutput()
    user_res = await runtime.submit("What time is it?", user_out)
    assert user_res == "User response immediate."
    assert wake_task.cancelled() or wake_task.done()

    assert runtime.main_session is not None
    assert_valid_history(runtime.main_session.history.messages)


@pytest.mark.asyncio
async def test_wake_batching_simultaneous_completions(tmp_path: Path):
    """Multiple tasks completing concurrently must be batched together without duplicate wake turns."""
    calls_made = 0

    class BatchWakeProvider(ProviderClient):
        def __init__(self) -> None:
            super().__init__(base_url="http://mock", model="mock")

        async def stream(self, messages, tools=None, **kwargs):  # type: ignore[override]
            nonlocal calls_made
            calls_made += 1
            yield TextDelta(text="Both tasks finished.")
            yield Finish(reason="stop")

    provider = BatchWakeProvider()
    runtime = TaskRuntime(workspace=tmp_path, provider=provider)
    runtime.create_receptionist()
    wake_out = DefaultTurnOutput()
    runtime.wake_output = wake_out

    from verbix.core.models import TaskState

    for tid, title in [("t-1", "Task 1"), ("t-2", "Task 2")]:
        runtime.tasks[tid] = TaskState(
            id=tid, title=title, status="running", step_count=1
        )
        runtime._finish_task(
            tid,
            WorkerResult(success=True, speech_summary=f"{title} ok.", details="OK"),
            status="completed",
        )

    # Give wake turn time to execute
    await asyncio.sleep(0.08)

    assert calls_made == 1
    assert runtime.tasks["t-1"].announced is True
    assert runtime.tasks["t-2"].announced is True
    assert runtime.main_session is not None
    assert_valid_history(runtime.main_session.history.messages)


@pytest.mark.asyncio
async def test_provider_error_handling_user_visible_message(tmp_path: Path):
    """A provider error in a user turn must result in a clean user-visible message without crashing."""

    class ErrorProvider(ProviderClient):
        def __init__(self) -> None:
            super().__init__(base_url="http://mock", model="mock")

        async def stream(self, messages, tools=None, **kwargs):  # type: ignore[override]
            raise ConnectionError("Connection refused by provider endpoint")
            yield  # type: ignore[unreachable]

    provider = ErrorProvider()
    runtime = TaskRuntime(workspace=tmp_path, provider=provider)
    runtime.create_receptionist()

    out = DefaultTurnOutput()
    res = await runtime.submit("Hello", out)
    assert res is not None
    assert "Provider connection failed" in res or "Connection refused" in res

    assert runtime.main_session is not None
    msgs = runtime.main_session.history.messages
    assert_valid_history(msgs)
    assert msgs[-1].role == "assistant"
    assert msgs[-1].content == res
