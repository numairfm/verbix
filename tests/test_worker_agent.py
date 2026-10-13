from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from verbix.agent.exceptions import FormatError
from verbix.agent.tools.registry import ToolRegistry
from verbix.agent.worker import WorkerAgent, WorkerConfig
from verbix.tools.base import ToolResult


@pytest.mark.asyncio
async def test_worker_rejects_chatter_without_tool_call():
    mock_model = MagicMock()
    # First response is plain text (no tool calls), second response calls finish
    mock_model.query.side_effect = [
        {"role": "assistant", "content": "I will create index.html now", "tool_calls": []},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "finish", "arguments": '{"summary": "done"}'}}],
        },
    ]
    tools = ToolRegistry()
    tools.register(
        "finish",
        {"type": "function", "function": {"name": "finish"}},
        lambda a: ToolResult("done", is_terminal=True),
    )

    agent = WorkerAgent(model=mock_model, tools=tools)
    result = await agent.run("build something")

    assert result["status"] == "completed"
    assert result["summary"] == "done"
    # Verify format error was fed back after the first response
    assert any("MUST execute a tool" in m.get("content", "") for m in agent.messages if m.get("role") == "user")


@pytest.mark.asyncio
async def test_worker_multi_turn_executes_to_finish():
    mock_model = MagicMock()
    mock_model.query.side_effect = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": '{"command": "echo hello"}'}}],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c2", "function": {"name": "finish", "arguments": '{"summary": "all done"}'}}],
        },
    ]

    tools = ToolRegistry()
    bash_calls = []

    def fake_bash(args):
        bash_calls.append(args)
        return "hello\n"

    tools.register("bash", {"type": "function", "function": {"name": "bash"}}, fake_bash)
    tools.register(
        "finish",
        {"type": "function", "function": {"name": "finish"}},
        lambda a: ToolResult("Task completed: all done", is_terminal=True),
    )

    agent = WorkerAgent(model=mock_model, tools=tools)
    result = await agent.run("run bash then finish")

    assert result["status"] == "completed"
    assert result["summary"] == "all done"
    assert bash_calls == [{"command": "echo hello"}]
    # Check messages trace
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["content"] == "hello\n"


@pytest.mark.asyncio
async def test_worker_catches_format_error_exception():
    mock_model = MagicMock()
    mock_model.query.side_effect = [
        FormatError({"role": "user", "content": "Tool call error: No tool calls found. Every response MUST execute a tool."}),
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "finish", "arguments": '{"summary": "recovered"}'}}],
        },
    ]

    tools = ToolRegistry()
    tools.register(
        "finish",
        {"type": "function", "function": {"name": "finish"}},
        lambda a: ToolResult("recovered", is_terminal=True),
    )

    agent = WorkerAgent(model=mock_model, tools=tools)
    result = await agent.run("do work")

    assert result["status"] == "completed"
    assert result["summary"] == "recovered"
    assert any("MUST execute a tool" in m.get("content", "") for m in agent.messages if m.get("role") == "user")


@pytest.mark.asyncio
async def test_worker_step_direct_call():
    mock_model = MagicMock()
    mock_model.query.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "function": {"name": "finish", "arguments": '{"summary": "step finished"}'}}],
    }

    tools = ToolRegistry()
    tools.register(
        "finish",
        {"type": "function", "function": {"name": "finish"}},
        lambda a: ToolResult("step finished", is_terminal=True),
    )

    agent = WorkerAgent(model=mock_model, tools=tools)
    res = await agent.step()

    assert isinstance(res, ToolResult)
    assert res.is_terminal is True
    assert res.observation == "step finished"


@pytest.mark.asyncio
async def test_worker_limits_exceeded():
    mock_model = MagicMock()
    mock_model.query.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": '{"command": "echo loop"}'}}],
    }

    tools = ToolRegistry()
    tools.register("bash", {"type": "function", "function": {"name": "bash"}}, lambda a: "loop")

    config = WorkerConfig(step_limit=2)
    agent = WorkerAgent(model=mock_model, tools=tools, config=config)
    result = await agent.run("infinite loop")

    assert result["status"] == "failed"
    assert "LimitsExceeded" in result["summary"]


@pytest.mark.asyncio
async def test_worker_accumulates_cost_on_format_error():
    mock_model = MagicMock()
    mock_model.query.side_effect = [
        FormatError({
            "role": "user",
            "content": "Tool call error: No tool calls found.",
            "extra": {"cost": 0.05},
        }),
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "finish", "arguments": '{"summary": "done"}'}}],
            "extra": {"cost": 0.02},
        },
    ]

    tools = ToolRegistry()
    tools.register(
        "finish",
        {"type": "function", "function": {"name": "finish"}},
        lambda a: ToolResult("done", is_terminal=True),
    )

    agent = WorkerAgent(model=mock_model, tools=tools)
    result = await agent.run("do work")

    assert result["status"] == "completed"
    assert result["summary"] == "done"
    assert agent.cost == pytest.approx(0.07)
    assert result["cost"] == pytest.approx(0.07)


@pytest.mark.asyncio
async def test_worker_cost_limit_halts_on_expensive_format_errors():
    mock_model = MagicMock()
    mock_model.query.side_effect = [
        FormatError({
            "role": "user",
            "content": "Tool call error: No tool calls found.",
            "extra": {"cost": 2.0},
        }),
        FormatError({
            "role": "user",
            "content": "Tool call error: No tool calls found.",
            "extra": {"cost": 2.0},
        }),
    ]

    config = WorkerConfig(cost_limit=3.0)
    agent = WorkerAgent(model=mock_model, config=config)
    result = await agent.run("costly failing task")

    assert result["status"] == "failed"
    assert "LimitsExceeded" in result["summary"]
    assert agent.cost == pytest.approx(4.0)


def test_worker_config_default_max_context_tokens():
    config = WorkerConfig()
    assert config.max_context_tokens == 16000


@pytest.mark.asyncio
async def test_worker_compacts_messages_when_exceeding_max_context_tokens():
    mock_model = MagicMock()
    mock_model.query.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c_fin", "function": {"name": "finish", "arguments": '{"summary": "done"}'}}],
    }
    tools = ToolRegistry()
    tools.register(
        "finish",
        {"type": "function", "function": {"name": "finish"}},
        lambda a: ToolResult("done", is_terminal=True),
    )

    config = WorkerConfig(max_context_tokens=50)
    agent = WorkerAgent(model=mock_model, tools=tools, config=config)

    agent.messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "initial task"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "bash", "content": "x" * 500},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c2", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "name": "bash", "content": "y" * 500},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c3", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c3", "name": "bash", "content": "z" * 500},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c4", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c4", "name": "bash", "content": "w" * 500},
    ]

    res = await agent.step()
    assert res.is_terminal is True
    queried_messages = mock_model.query.call_args[0][0]
    assert len(queried_messages) < 10
    assert len(agent.messages) < 11


@pytest.mark.asyncio
async def test_worker_emits_tool_call_started_and_finished_events():
    from verbix.core.events import ToolCallFinished, ToolCallStarted

    mock_model = MagicMock()
    mock_model.query.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_123", "function": {"name": "calc", "arguments": '{"x": 42}'}}],
    }

    tools = ToolRegistry()
    tools.register(
        "calc",
        {"type": "function", "function": {"name": "calc"}},
        lambda a: ToolResult(f"result is {a.get('x')}", is_error=False, is_terminal=True),
    )

    events = []
    agent = WorkerAgent(model=mock_model, tools=tools)
    agent.on_event = events.append

    res = await agent.step()
    assert res.is_terminal is True
    assert res.observation == "result is 42"

    assert len(events) == 2
    started, finished = events[0], events[1]

    assert isinstance(started, ToolCallStarted)
    assert started.call_id == "call_123"
    assert started.tool_name == "calc"
    assert started.arguments == {"x": 42}

    assert isinstance(finished, ToolCallFinished)
    assert finished.call_id == "call_123"
    assert finished.observation == "result is 42"
    assert finished.is_error is False


@pytest.mark.asyncio
async def test_worker_emits_tool_call_finished_on_error():
    from verbix.core.events import ToolCallFinished, ToolCallStarted

    mock_model = MagicMock()
    mock_model.query.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_err", "function": {"name": "fail_tool", "arguments": "{}"}}],
    }

    tools = ToolRegistry()
    tools.register(
        "fail_tool",
        {"type": "function", "function": {"name": "fail_tool"}},
        lambda a: ToolResult("Tool failure occurred", is_error=True, is_terminal=True),
    )

    events = []
    agent = WorkerAgent(model=mock_model, tools=tools)
    agent.on_event = events.append

    res = await agent.step()
    assert res.is_error is True

    assert len(events) == 2
    started, finished = events[0], events[1]
    assert isinstance(started, ToolCallStarted)
    assert isinstance(finished, ToolCallFinished)
    assert finished.call_id == "call_err"
    assert finished.observation == "Tool failure occurred"
    assert finished.is_error is True


