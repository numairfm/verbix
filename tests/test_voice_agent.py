from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from verbix.agent.announcer import AnnouncerAgent, AnnouncerConfig
from verbix.agent.tools.registry import ToolRegistry
from verbix.agent.voice import VoiceAgent, VoiceConfig
from verbix.core.events import (
    TextDelta,
    ThoughtDelta,
    ToolCallDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnComplete,
)
from verbix.tools.base import ToolResult


@pytest.mark.asyncio
async def test_voice_agent_streams_text():
    mock_model = MagicMock()

    async def fake_stream(messages, tools=None, **kw):
        yield TextDelta(text="Hello there. How can I help you today?")
        yield TurnComplete(final_text="Hello there. How can I help you today?")

    mock_model.query_stream = fake_stream
    agent = VoiceAgent(model=mock_model)
    events = [e async for e in agent.step_stream("hi")]
    assert any(isinstance(e, TextDelta) and "Hello there" in e.text for e in events)
    assert any(isinstance(e, TurnComplete) for e in events)


@pytest.mark.asyncio
async def test_announcer_agent_has_no_tools():
    agent = AnnouncerAgent(model=MagicMock())
    assert agent.get_tool_definitions() is None or agent.get_tool_definitions() == []


def test_announcer_config_defaults():
    cfg = AnnouncerConfig()
    assert cfg.tools == []
    assert cfg.tool_choice == "none"


def test_voice_agent_unknown_tool_raises_key_error():
    tools = ToolRegistry()
    config = VoiceConfig(tools=["nonexistent_tool"])
    agent = VoiceAgent(model=MagicMock(), tools=tools, config=config)
    with pytest.raises(KeyError):
        agent.get_tool_definitions()


@pytest.mark.asyncio
async def test_voice_agent_execute_tool_encapsulation():
    class MockAdapter:
        async def aexecute(self, name, args):
            return ToolResult(f"executed {name}", is_error=False)

    agent = VoiceAgent(model=MagicMock(), tools=MockAdapter())
    res = await agent._execute_tool("my_tool", {"a": 1})
    assert res.observation == "executed my_tool"


@pytest.mark.asyncio
async def test_voice_agent_speech_segmentation_split_chunks():
    mock_model = MagicMock()

    # Yields two sentences piece by piece. Once the first sentence boundary is reached
    # split_chunks will produce > 1 chunk, releasing the first sentence early.
    async def fake_stream(messages, tools=None, **kw):
        yield TextDelta(text="This is the first sentence. ")
        yield TextDelta(text="And here is the second sentence.")
        yield TurnComplete(final_text="This is the first sentence. And here is the second sentence.")

    mock_model.query_stream = fake_stream
    agent = VoiceAgent(model=mock_model)
    events = [e async for e in agent.step_stream("hi")]

    text_deltas = [e.text for e in events if isinstance(e, TextDelta)]
    assert "".join(text_deltas) == "This is the first sentence. And here is the second sentence."


@pytest.mark.asyncio
async def test_voice_agent_pre_tool_suppression():
    mock_model = MagicMock()

    calls = 0

    async def fake_stream(messages, tools=None, **kw):
        nonlocal calls
        calls += 1
        if calls == 1:
            # Model emits pre-tool chatter, then a tool call
            yield TextDelta(text="Let me spawn that task right away.")
            yield ToolCallDelta(
                index=0,
                call_id="call_spawn",
                name="spawn",
                arguments_delta='{"objective": "build site"}',
            )
            yield TurnComplete(final_text="")
        else:
            # Second iteration after tool result
            yield TextDelta(text="I have spawned the task for you.")
            yield TurnComplete(final_text="I have spawned the task for you.")

    mock_model.query_stream = fake_stream

    tools = ToolRegistry()
    tools.register(
        "spawn",
        {"type": "function", "function": {"name": "spawn"}},
        lambda a: ToolResult("Task task-1 started", is_error=False),
    )

    agent = VoiceAgent(model=mock_model, tools=tools)
    events = [e async for e in agent.step_stream("please build site")]

    # Pre-tool chatter "Let me spawn that task right away." MUST be suppressed
    text_deltas = [e.text for e in events if isinstance(e, TextDelta)]
    delivered = "".join(text_deltas)
    assert "Let me spawn" not in delivered
    assert "I have spawned the task for you." in delivered

    # Tool events must be yielded
    assert any(isinstance(e, ToolCallStarted) and e.tool_name == "spawn" for e in events)
    assert any(isinstance(e, ToolCallFinished) and "Task task-1 started" in e.observation for e in events)


@pytest.mark.asyncio
async def test_announcer_agent_forces_tool_choice_none():
    mock_model = MagicMock()
    passed_kwargs = {}

    async def fake_stream(messages, tools=None, **kw):
        passed_kwargs.update(kw)
        passed_kwargs["tools"] = tools
        yield TextDelta(text="Worker task completed successfully.")
        yield TurnComplete(final_text="Worker task completed successfully.")

    mock_model.query_stream = fake_stream
    agent = AnnouncerAgent(model=mock_model)
    events = [e async for e in agent.step_stream("debrief")]

    assert passed_kwargs.get("tool_choice") == "none"
    assert passed_kwargs.get("tools") is None or passed_kwargs.get("tools") == []
    assert any(isinstance(e, TextDelta) and "Worker task completed" in e.text for e in events)


@pytest.mark.asyncio
async def test_voice_agent_yields_thought_delta():
    mock_model = MagicMock()

    async def fake_stream(messages, tools=None, **kw):
        yield ThoughtDelta(thought="Thinking deeply...")
        yield TextDelta(text="Here is your answer.")
        yield TurnComplete(final_text="Here is your answer.")

    mock_model.query_stream = fake_stream
    agent = VoiceAgent(model=mock_model)
    events = [e async for e in agent.step_stream("query")]

    thoughts = [e.thought for e in events if isinstance(e, ThoughtDelta)]
    assert thoughts == ["Thinking deeply..."]
    assert any(isinstance(e, TextDelta) and "Here is your answer." in e.text for e in events)


@pytest.mark.asyncio
async def test_voice_agent_terminal_tool_finishes_turn():
    mock_model = MagicMock()

    async def fake_stream(messages, tools=None, **kw):
        yield ToolCallDelta(
            index=0,
            call_id="call_cancel",
            name="cancel",
            arguments_delta='{"task_id": "task-0"}',
        )
        yield TurnComplete(final_text="")

    mock_model.query_stream = fake_stream

    tools = ToolRegistry()
    tools.register(
        "cancel",
        {"type": "function", "function": {"name": "cancel"}},
        lambda a: ToolResult("Task cancelled.", is_terminal=True),
    )

    agent = VoiceAgent(model=mock_model, tools=tools)
    events = [e async for e in agent.step_stream("cancel task")]

    assert any(isinstance(e, TurnComplete) and "Task cancelled." in e.final_text for e in events)


@pytest.mark.asyncio
async def test_voice_agent_limits_exceeded():
    mock_model = MagicMock()

    async def fake_stream(messages, tools=None, **kw):
        yield TextDelta(text="talking...")
        yield TurnComplete(final_text="talking...")

    mock_model.query_stream = fake_stream
    config = VoiceConfig(step_limit=1, turn_scoped_limits=False)
    agent = VoiceAgent(model=mock_model, config=config)
    agent.n_calls = 1  # already at limit

    events = [e async for e in agent.step_stream("hi")]
    assert any(isinstance(e, TurnComplete) and "LimitsExceeded" in e.final_text for e in events)


@pytest.mark.asyncio
async def test_voice_agent_handles_stream_error_gracefully():
    mock_model = MagicMock()

    async def fake_stream(messages, tools=None, **kw):
        raise RuntimeError("Connection dropped")
        yield  # make it a generator

    mock_model.query_stream = fake_stream
    agent = VoiceAgent(model=mock_model)
    events = [e async for e in agent.step_stream("hi")]

    # Should catch exception, repair history, and return apologize prompt
    turn_complete = [e for e in events if isinstance(e, TurnComplete)]
    assert len(turn_complete) == 1
    assert "Sorry" in turn_complete[0].final_text
    assert any(m.get("role") == "assistant" and "Sorry" in m.get("content", "") for m in agent.messages)


@pytest.mark.asyncio
async def test_voice_agent_cancellation_repairs_history():
    mock_model = MagicMock()

    async def fake_stream(messages, tools=None, **kw):
        yield TextDelta(text="Partial speech")
        raise asyncio.CancelledError()

    mock_model.query_stream = fake_stream
    agent = VoiceAgent(model=mock_model)

    with pytest.raises(asyncio.CancelledError):
        _ = [e async for e in agent.step_stream("hello")]

    # History should have been repaired cleanly
    assert isinstance(agent.messages, list)


@pytest.mark.asyncio
async def test_voice_agent_renders_system_and_instance_templates():
    mock_model = MagicMock()

    captured_messages = []

    async def fake_stream(messages, tools=None, **kw):
        captured_messages.extend(messages)
        yield TextDelta(text="Done")
        yield TurnComplete(final_text="Done")

    mock_model.query_stream = fake_stream
    config = VoiceConfig(
        system_template="You are {{system_role}}.",
        instance_template="Task: {{task}}",
    )
    agent = VoiceAgent(model=mock_model, config=config)
    agent.extra_template_vars = {"system_role": "VerbixVoice"}

    _ = [e async for e in agent.step_stream("weather")]

    assert captured_messages[0] == {"role": "system", "content": "You are VerbixVoice."}
    assert captured_messages[1] == {"role": "user", "content": "Task: weather"}
