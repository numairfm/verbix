import asyncio
from unittest.mock import MagicMock

import pytest

from verbix.cli import build_parser
from verbix.core.events import TextDelta, ToolCallFinished
from verbix.provider.models import Finish as ProviderFinish
from verbix.provider.models import TextDelta as ProviderTextDelta
from verbix.runtime.runtime import Runtime, WakeOutputFanOut


def _make_runtime(tmp_path, **kwargs):
    return Runtime(workspace=tmp_path, provider=MagicMock(), **kwargs)


def test_runtime_ipc_methods_exist(tmp_path):
    rt = _make_runtime(tmp_path)
    assert hasattr(rt, "get_task_status_summary")
    assert hasattr(rt, "cancel_task")


def test_get_task_status_summary_empty(tmp_path):
    rt = _make_runtime(tmp_path)
    assert "no active" in rt.get_task_status_summary().lower()
    assert "task-9" in rt.get_task_status_summary("task-9")


def test_get_task_status_summary_lists_ledger(tmp_path):
    rt = _make_runtime(tmp_path)
    rt.ledger.register_task("task-1", "do things")
    assert "task-1" in rt.get_task_status_summary()
    single = rt.get_task_status_summary("task-1")
    assert "task-1" in single
    assert "do things" in single


@pytest.mark.asyncio
async def test_cancel_task_unknown_returns_false(tmp_path):
    rt = _make_runtime(tmp_path)
    assert await rt.cancel_task("task-nope") is False


@pytest.mark.asyncio
async def test_cancel_task_cancels_running_task(tmp_path):
    rt = _make_runtime(tmp_path)
    task = MagicMock()
    task.id = "task-1"
    task.description = "slow job"
    task.status = "running"

    async def _cancel():
        task.status = "cancelled"

    task.cancel = _cancel
    rt.tasks["task-1"] = task
    assert await rt.cancel_task("task-1") is True


@pytest.mark.asyncio
async def test_wake_output_fan_out_reaches_voice_and_broadcast(tmp_path):
    rt = _make_runtime(tmp_path)
    q = rt.subscribe()
    voice_seen = []

    class Voice:
        async def emit(self, event):
            voice_seen.append(event)

    fan = WakeOutputFanOut(Voice(), rt._broadcast_wake_event)
    await fan.emit(TextDelta(text="hello"))
    assert len(voice_seen) == 1
    evt = await asyncio.wait_for(q.get(), timeout=1.0)
    assert evt["type"] == "wake_event"
    assert evt["payload"]["text"] == "hello"
    rt.unsubscribe(q)


@pytest.mark.asyncio
async def test_spawn_broadcasts_ledger_transitions(tmp_path):
    provider = MagicMock()

    async def fake_stream(messages, tools=None, **kwargs):
        yield ProviderTextDelta(text="done fast")
        yield ProviderFinish(reason="stop")

    provider.stream = fake_stream
    rt = Runtime(workspace=tmp_path, provider=provider, buffer_window=0.02)
    q = rt.subscribe()
    try:
        task_id = await rt.spawn("quick job")
        evt = await asyncio.wait_for(q.get(), timeout=2.0)
        assert evt["type"] == "task_registered"
        assert evt["payload"]["task_id"] == task_id

        await rt.tasks[task_id].wait()
        found = False
        for _ in range(10):
            try:
                evt = await asyncio.wait_for(q.get(), timeout=2.0)
            except asyncio.TimeoutError:
                break
            if evt["type"] == "task_completed" and evt["payload"]["task_id"] == task_id:
                found = True
                break
        assert found
    finally:
        await rt.aclose()
        rt.unsubscribe(q)


@pytest.mark.asyncio
async def test_finish_tool_returns_task_completed_observation():
    from verbix.runtime.tools import FinishTool

    res = await FinishTool().execute({"summary": "All 83 tests passed."}, None)
    assert res.observation == "Task completed: All 83 tests passed."


@pytest.mark.asyncio
async def test_finish_tool_observation_has_no_repr_leakage():
    from verbix.agent.agents.default import DefaultAgent
    from verbix.agent.tools.registry import ToolRegistry
    from verbix.core.events import ToolCallDelta
    from verbix.runtime.tools import FinishTool

    registry = ToolRegistry()
    finish_tool = FinishTool()
    registry.register(
        finish_tool.name,
        {
            "type": "function",
            "function": {
                "name": finish_tool.name,
                "description": finish_tool.description,
                "parameters": finish_tool.parameters_schema,
            },
        },
        finish_tool.execute,
    )

    mock_model = MagicMock()

    async def mock_stream(messages, tools=None):
        yield ToolCallDelta(
            index=0,
            call_id="call_finish_1",
            name="finish",
            arguments_delta='{"summary": "All 83 tests passed."}',
        )

    mock_model.query_stream = mock_stream

    agent = DefaultAgent(
        model=mock_model,
        tools=registry,
        tools_list=["finish"],
        system_prompt="Execute tests.",
    )
    events = [e async for e in agent.step_stream("run tests")]
    finished = [e for e in events if isinstance(e, ToolCallFinished)]
    assert len(finished) == 1
    assert finished[0].observation == "Task completed: All 83 tests passed."
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert tool_msgs
    assert "ToolResult(" not in tool_msgs[0]["content"]


def test_cli_dead_flags_removed():
    parser = build_parser()
    args = parser.parse_args([])
    assert not hasattr(args, "isolate_worktrees")
    assert not hasattr(args, "allow_network")
