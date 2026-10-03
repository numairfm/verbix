from __future__ import annotations

from pathlib import Path

import pytest

from verbix.core.models import WorkerResult
from verbix.provider.client import ProviderClient
from verbix.runtime.orchestration import (
    TaskCancelTool,
    TaskSpawnTool,
    TaskStatusTool,
    TaskSteerTool,
    WorkerFinishTool,
)
from verbix.runtime.tasks import TaskRuntime
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
    # tmp_path is not a git repo, should fall back to standard subdirectory
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
    # Verify 180 char truncation on speech_summary
    assert len(tool.captured_result.speech_summary) <= 180
    assert tool.captured_result.details == "All unit tests passed with 100% coverage."


@pytest.mark.asyncio
async def test_task_runtime_spawn_and_finish(
    tmp_path: Path, mock_provider: ProviderClient
):
    runtime = TaskRuntime(workspace=tmp_path, provider=mock_provider)
    receptionist = runtime.create_receptionist()

    # Pre-configure a dummy worker finish directly
    task_id = await runtime.spawn_task(title="Build Project", prompt="Run build")
    assert task_id == "task-1"
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

    # Spoken announcement queue received the completion
    assert not runtime.announcement_q.empty()
    queued_task_id, queued_res = runtime.announcement_q.get_nowait()
    assert queued_task_id == task_id
    assert queued_res.speech_summary == "Build succeeded."

    # Immediate turn-boundary injection into idle receptionist history
    assert len(receptionist.history.messages) == 1
    assert receptionist.history.messages[0].role == "system"
    assert (
        "[Task completed: 'Build Project' (task-1) -> Succeeded: Build succeeded."
        in receptionist.history.messages[0].content
    )
    assert (
        "Compiled 12 files without errors." in receptionist.history.messages[0].content
    )


@pytest.mark.asyncio
async def test_turn_boundary_injection_during_active_turn(
    tmp_path: Path, mock_provider: ProviderClient
):
    runtime = TaskRuntime(workspace=tmp_path, provider=mock_provider)
    receptionist = runtime.create_receptionist()

    task_id = await runtime.spawn_task(title="Long Task", prompt="Do work")

    # Start main turn
    runtime.on_main_turn_start()
    assert runtime.is_main_turn_active is True

    # Task finishes while receptionist turn is actively running
    worker_res = WorkerResult(
        success=True,
        speech_summary="Long task done.",
        details="Detailed log.",
    )
    runtime._finish_task(task_id, worker_res, status="completed")

    # Message must NOT be injected mid-turn to preserve tool call/result ordering!
    assert len(receptionist.history.messages) == 0
    assert len(runtime._pending_completions) == 1

    # End main turn -> flushes pending completions
    runtime.on_main_turn_end()
    assert runtime.is_main_turn_active is False
    assert len(runtime._pending_completions) == 0
    assert len(receptionist.history.messages) == 1
    assert "Long task done." in receptionist.history.messages[0].content


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

    # The rendered prompt has the ephemeral annotation
    assert len(rendered) == 1
    assert "[Running: task-1 'Compile Core' (step 5)]" in rendered[0].content

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
    receptionist = runtime.create_receptionist()

    task_id = await runtime.spawn_task(title="Cancel Me", prompt="Neverending")
    cancelled = await runtime.cancel_task(task_id)
    assert cancelled is True

    task = runtime.get_task(task_id)
    assert task is not None
    assert task.status == "cancelled"
    assert task.result is not None
    assert "cancelled" in task.result.speech_summary.lower()

    # History got cancellation record
    assert len(receptionist.history.messages) == 1
    assert (
        "Failed" in receptionist.history.messages[0].content
        or "cancelled" in receptionist.history.messages[0].content.lower()
    )


@pytest.mark.asyncio
async def test_task_runtime_steering(tmp_path: Path, mock_provider: ProviderClient):
    runtime = TaskRuntime(workspace=tmp_path, provider=mock_provider)
    task_id = await runtime.spawn_task(title="Steer Me", prompt="Steerable")

    steered = await runtime.steer_task(task_id, "Please focus on documentation.")
    assert steered is True

    q = runtime._steer_queues[task_id]
    assert not q.empty()
    instruction = q.get_nowait()
    assert instruction == "Please focus on documentation."


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
    assert "task-1" in res.observation

    # 2. Status
    res_status = await status_tool.execute({"task_id": "task-1"}, ctx)
    assert not res_status.is_error
    assert "running" in res_status.observation.lower()

    # 3. Steer
    res_steer = await steer_tool.execute(
        {"task_id": "task-1", "instruction": "Hurry up"}, ctx
    )
    assert not res_steer.is_error
    assert "delivered" in res_steer.observation.lower()

    # 4. Cancel
    res_cancel = await cancel_tool.execute({"task_id": "task-1"}, ctx)
    assert not res_cancel.is_error
    assert "cancelled" in res_cancel.observation.lower()


@pytest.mark.asyncio
async def test_task_completion_sync_before_foreground_turn(
    tmp_path: Path, mock_provider: ProviderClient
):
    """Regression test for task completion synchronization:
    1. Main agent spawns a background task.
    2. Task completes.
    3. Runtime emits task completion.
    4. A new user turn arrives immediately afterward.
    Invariant:
    - The new foreground turn must observe the terminal state.
    - Completed task must NEVER appear in transient running-task prompt annotation.
    - Completed task result (summary + details) is available in history for receptionist.
    """
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

    # 3. Simulate completion occurring while turn state is transitioning or buffered
    # If a completion arrived in _pending_completions:
    runtime._pending_completions.append((task_id, task.title, worker_res))
    task.status = "completed"
    task.result = worker_res

    # 4. A new user turn arrives immediately afterward
    runtime.on_main_turn_start()

    # INVARIANT 1: Pending completions MUST be drained at turn start
    assert len(runtime._pending_completions) == 0
    assert len(receptionist.history.messages) == 1
    assert (
        "Analyzed repository successfully." in receptionist.history.messages[0].content
    )
    assert (
        "Found pyproject.toml and src/verbix."
        in receptionist.history.messages[0].content
    )

    # INVARIANT 2: render_turn_prompt must NEVER include the completed task
    active_tasks = runtime.list_active_tasks()
    assert len(active_tasks) == 0
    receptionist.history.append_user("what did it find?")
    prompt_msgs = receptionist.history.render_turn_prompt(active_tasks)
    user_msg = prompt_msgs[-1]
    assert "[Running:" not in user_msg.content
    assert user_msg.content == "what did it find?"


@pytest.mark.asyncio
async def test_worker_finish_tool_immediate_state_transition(
    tmp_path: Path, mock_provider: ProviderClient
):
    """Verifies that invoking WorkerFinishTool immediately transitions TaskState
    to terminal 'completed' status and signals is_terminal to stop agent loop.
    """
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
    assert not runtime.announcement_q.empty()

    # Dynamic running tasks evaluated in receptionist must be empty
    assert len(runtime.list_active_tasks()) == 0


@pytest.mark.asyncio
async def test_task_spawn_workspace_awareness(
    tmp_path: Path, mock_provider: ProviderClient
):
    """Verifies TaskSpawnTool and TaskStatusTool expose workspace in descriptions and schema."""
    runtime = TaskRuntime(workspace=tmp_path, provider=mock_provider)
    spawn_tool = TaskSpawnTool(runtime)
    status_tool = TaskStatusTool(runtime)

    assert str(tmp_path) in spawn_tool.description
    assert str(tmp_path) in status_tool.description

    schema = spawn_tool.parameters_schema
    # Safety invariant: the runtime, not the model, chooses the workspace path
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
    assert "spawned with id" in res.observation
    task_id = "task-1"
    task = runtime.get_task(task_id)
    assert task is not None
    assert task.workspace == tmp_path
