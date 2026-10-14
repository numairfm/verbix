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
    assert any("You have not executed any tools yet" in m.get("content", "") for m in agent.messages if m.get("role") == "user")


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


@pytest.mark.asyncio
async def test_worker_emits_event_on_format_error():
    from verbix.core.events import TextDelta

    mock_model = MagicMock()
    mock_model.query.side_effect = FormatError({
        "role": "user",
        "content": "No tool call found in the response.",
    })
    events = []
    agent = WorkerAgent(model=mock_model, tools=ToolRegistry())
    agent.on_event = events.append

    res = await agent.step()
    assert res.is_error is True

    text_events = [e for e in events if isinstance(e, TextDelta)]
    assert len(text_events) >= 1
    assert "! format error:" in text_events[0].text


@pytest.mark.asyncio
async def test_worker_emits_event_on_query_exception():
    from verbix.core.events import TextDelta

    mock_model = MagicMock()
    mock_model.query.side_effect = RuntimeError("Openrouter credit balance too low")
    events = []
    agent = WorkerAgent(model=mock_model, tools=ToolRegistry())
    agent.on_event = events.append

    with pytest.raises(RuntimeError, match="Openrouter credit balance too low"):
        await agent.step()

    text_events = [e for e in events if isinstance(e, TextDelta)]
    assert len(text_events) >= 1
    assert "! query error:" in text_events[0].text


@pytest.mark.asyncio
async def test_worker_preserves_assistant_text_on_format_error():
    from types import SimpleNamespace
    from unittest.mock import patch

    from verbix.agent.models.litellm_model import LitellmModel

    model = LitellmModel(model_name="test-model")
    tools = ToolRegistry()
    tools.register(
        "finish",
        {"type": "function", "function": {"name": "finish"}},
        lambda a: ToolResult("done", is_terminal=True),
    )

    first_resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="I could not find foo.txt anywhere.",
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ],
        model_dump=lambda **kw: {"choices": []},
    )
    second_resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="c1",
                            function=SimpleNamespace(name="finish", arguments='{"summary": "done"}'),
                        )
                    ],
                    model_dump=lambda **kw: {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "c1", "function": {"name": "finish", "arguments": '{"summary": "done"}'}}],
                    },
                ),
                finish_reason="tool_calls",
            )
        ],
        model_dump=lambda **kw: {"choices": []},
    )

    with (
        patch("litellm.completion", side_effect=[first_resp, second_resp]),
        patch("litellm.cost_calculator.completion_cost", return_value=0.01),
    ):
        agent = WorkerAgent(model=model, tools=tools)
        result = await agent.run("find foo.txt")

    assert result["status"] == "completed"
    assert result["summary"] == "done"

    assistant_msgs = [
        m for m in agent.messages
        if m.get("role") == "assistant" and m.get("content") == "I could not find foo.txt anywhere."
    ]
    assert len(assistant_msgs) == 1, f"Expected assistant prose to be preserved, got {agent.messages}"
    user_err_msgs = [
        m for m in agent.messages
        if m.get("role") == "user" and "No tool calls found" in m.get("content", "")
    ]
    assert len(user_err_msgs) == 1, f"Expected user format error reminder, got {agent.messages}"

    idx_assistant = agent.messages.index(assistant_msgs[0])
    idx_user_err = agent.messages.index(user_err_msgs[0])
    assert idx_user_err == idx_assistant + 1
    assert agent.cost == pytest.approx(0.02)
    assert result["cost"] == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_worker_circuit_breaker_intercepts_repeated_calls():
    from verbix.core.events import ToolCallFinished

    mock_model = MagicMock()
    mock_model.query.side_effect = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": '{"command": "echo test"}'}}],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c2", "function": {"name": "bash", "arguments": '{"command": "echo test"}'}}],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c3", "function": {"name": "bash", "arguments": '{"command": "echo test"}'}}],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c4", "function": {"name": "finish", "arguments": '{"summary": "recovered"}'}}],
        },
    ]

    tools = ToolRegistry()
    bash_calls = []

    def fake_bash(args):
        bash_calls.append(args)
        return "test\n"

    tools.register("bash", {"type": "function", "function": {"name": "bash"}}, fake_bash)
    tools.register(
        "finish",
        {"type": "function", "function": {"name": "finish"}},
        lambda a: ToolResult("recovered", is_terminal=True),
    )

    agent = WorkerAgent(model=mock_model, tools=tools)
    events = []
    agent.on_event = events.append
    result = await agent.run("run bash command")

    assert result["status"] == "completed"
    assert result["summary"] == "recovered"
    # Tool execution should have been called only twice (the 3rd call was intercepted by circuit breaker)
    assert len(bash_calls) == 2
    # Verify the 3rd tool message has the circuit breaker error observation
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 4
    assert tool_msgs[0]["content"] == "test\n"
    assert tool_msgs[1]["content"] == "test\n"
    assert (
        "Error: Tool 'bash' was called 3 times consecutively with identical arguments without progress"
        in tool_msgs[2]["content"]
    )
    assert "You must try a different approach or output your direct conclusion." in tool_msgs[2]["content"]
    assert tool_msgs[3]["content"] == "recovered"

    cb_events = [e for e in events if isinstance(e, ToolCallFinished) and e.is_error]
    assert len(cb_events) == 1
    assert "Tool 'bash' was called 3 times consecutively" in cb_events[0].observation


@pytest.mark.asyncio
async def test_worker_circuit_breaker_resets_on_different_arguments():
    mock_model = MagicMock()
    mock_model.query.side_effect = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": '{"command": "echo 1"}'}}],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c2", "function": {"name": "bash", "arguments": '{"command": "echo 1"}'}}],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c3", "function": {"name": "bash", "arguments": '{"command": "echo 2"}'}}],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c4", "function": {"name": "bash", "arguments": '{"command": "echo 2"}'}}],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c5", "function": {"name": "finish", "arguments": '{"summary": "done"}'}}],
        },
    ]

    tools = ToolRegistry()
    bash_calls = []

    def fake_bash(args):
        bash_calls.append(args)
        return "out\n"

    tools.register("bash", {"type": "function", "function": {"name": "bash"}}, fake_bash)
    tools.register(
        "finish",
        {"type": "function", "function": {"name": "finish"}},
        lambda a: ToolResult("done", is_terminal=True),
    )

    agent = WorkerAgent(model=mock_model, tools=tools)
    result = await agent.run("execute multiple commands")

    assert result["status"] == "completed"
    assert result["summary"] == "done"
    # All 4 bash calls should have executed because no call hit 3 consecutive identical invocations
    assert len(bash_calls) == 4
    assert bash_calls == [
        {"command": "echo 1"},
        {"command": "echo 1"},
        {"command": "echo 2"},
        {"command": "echo 2"},
    ]


def test_worker_config_step_limit_default():
    from pathlib import Path

    import yaml

    config_path = Path("src/verbix/agent/config/worker.yaml")
    assert config_path.is_file()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["agent"]["step_limit"] == 30


@pytest.mark.asyncio
async def test_worker_completes_on_speech_after_tool_execution():
    mock_model = MagicMock()
    mock_model.query.side_effect = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": '{"command": "echo 42"}'}}],
        },
        {
            "role": "assistant",
            "content": "The calculation result is 42.",
            "tool_calls": None,
        },
    ]

    tools = ToolRegistry()
    tools.register("bash", {"type": "function", "function": {"name": "bash"}}, lambda a: "42\n")

    agent = WorkerAgent(model=mock_model, tools=tools)
    result = await agent.run("calculate something")

    assert result["status"] == "completed"
    assert result["summary"] == "The calculation result is 42."
    assert agent.n_calls == 2


@pytest.mark.asyncio
async def test_worker_completes_on_speech_in_format_error_path():
    mock_model = MagicMock()
    assistant_speech = {"role": "assistant", "content": "Here is the stock summary.", "extra": {}}
    user_err = {"role": "user", "content": "No tool calls found in the response.", "extra": {}}

    mock_model.query.side_effect = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": '{"command": "echo ok"}'}}],
        },
        FormatError(assistant_speech, user_err),
    ]

    tools = ToolRegistry()
    tools.register("bash", {"type": "function", "function": {"name": "bash"}}, lambda a: "ok\n")

    agent = WorkerAgent(model=mock_model, tools=tools)
    result = await agent.run("check stock")

    assert result["status"] == "completed"
    assert result["summary"] == "Here is the stock summary."
    assert agent.n_calls == 2


@pytest.mark.asyncio
async def test_worker_rejects_format_error_chatter_on_turn_one():
    mock_model = MagicMock()
    assistant_speech = {"role": "assistant", "content": "I plan to look at the repo first.", "extra": {}}
    user_err = {"role": "user", "content": "No tool calls found in the response.", "extra": {}}

    mock_model.query.side_effect = [
        FormatError(assistant_speech, user_err),
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
    result = await agent.run("do something")

    assert result["status"] == "completed"
    assert result["summary"] == "done"
    assert agent.n_calls == 2
    # Ensure format error feedback occurred and chatter was not accepted prematurely
    assert any("No tool calls found" in m.get("content", "") for m in agent.messages if m.get("role") == "user")
