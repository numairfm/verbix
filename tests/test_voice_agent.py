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
    assert any(
        isinstance(e, TurnComplete)
        and e.exit_reason == "limits_exceeded"
        and e.is_error is True
        and "LimitsExceeded" not in e.final_text
        for e in events
    )


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


@pytest.mark.asyncio
async def test_voice_agent_repeat_breaker_never_emits_exit_role_to_api_and_speaks_fallback():
    from verbix.agent.models.litellm_model import LitellmModel

    mock_model = MagicMock()

    async def repeat_stream(messages, tools=None, **kw):
        yield ToolCallDelta(index=0, call_id="c1", name="status", arguments_delta="{}")

    mock_model.query_stream = repeat_stream
    tools = ToolRegistry()
    tools.register(
        "status",
        {"type": "function", "function": {"name": "status"}},
        lambda a: ToolResult("running", is_error=False),
    )

    agent = VoiceAgent(
        model=mock_model,
        tools=tools,
        config=VoiceConfig(max_calls_per_tool_per_turn={}),
    )
    events = [e async for e in agent.step_stream("what?")]

    tc = [e for e in events if isinstance(e, TurnComplete)][-1]
    assert tc.exit_reason == "repeated_tool_call"
    assert tc.is_error is True
    assert "RepeatedToolCall" not in tc.final_text
    assert "Sorry" in tc.final_text

    # Assert on the exact payload built by LitellmModel._prepare_messages_for_api
    litellm_model = LitellmModel(model_name="openrouter/google/gemini-3.5-flash-lite")
    api_payload = litellm_model._prepare_messages_for_api(agent.messages)
    roles = {m.get("role") for m in api_payload}
    assert "exit" not in roles
    assert roles <= {"system", "user", "assistant", "tool"}

    # Announcer must have empty spoken_error_fallback so it never says "Could you say that again?"
    announcer = AnnouncerAgent(
        model=mock_model, config=AnnouncerConfig(step_limit=1, turn_scoped_limits=False)
    )
    announcer.n_calls = 1
    ann_events = [e async for e in announcer.step_stream("debrief")]
    ann_tc = [e for e in ann_events if isinstance(e, TurnComplete)][-1]
    assert ann_tc.exit_reason == "limits_exceeded"
    assert ann_tc.final_text == ""
    assert not any(
        "Could you say that again" in str(m.get("content", ""))
        for m in announcer.messages
    )


@pytest.mark.asyncio
async def test_voice_agent_preserves_already_delivered_speech_on_limit_exit():
    mock_model = MagicMock()

    async def partial_then_tool_stream(messages, tools=None, **kw):
        yield TextDelta(text="I checked your workspace right now. ")
        yield TextDelta(text="Let me also check status.")
        yield ToolCallDelta(index=0, call_id="c1", name="status", arguments_delta="{}")

    mock_model.query_stream = partial_then_tool_stream
    tools = ToolRegistry()
    tools.register(
        "status",
        {"type": "function", "function": {"name": "status"}},
        lambda a: ToolResult("ok"),
    )

    # First turn streams speech + tool call; second step in same turn hits step_limit
    config = VoiceConfig(step_limit=1, turn_scoped_limits=True)
    agent = VoiceAgent(model=mock_model, tools=tools, config=config)
    events = [e async for e in agent.step_stream("check")]

    tc = [e for e in events if isinstance(e, TurnComplete)][-1]
    assert tc.exit_reason == "limits_exceeded"
    assert tc.is_error is True
    assert "LimitsExceeded" not in tc.final_text
    assert "I checked your workspace right now." in tc.final_text

    # Multi-step case: step 1 streams speech + tool call; step 2 emits only a tool call,
    # then step 3 hits step_limit=2. Step 1's delivered speech must STILL be preserved!
    step_num = 0

    async def two_step_stream(messages, tools=None, **kw):
        nonlocal step_num
        step_num += 1
        if step_num == 1:
            yield TextDelta(text="First step spoken sentence. ")
            yield TextDelta(text="More text.")
            yield ToolCallDelta(index=0, call_id="c1", name="spawn", arguments_delta='{"objective":"a"}')
        else:
            yield ToolCallDelta(index=0, call_id="c2", name="spawn", arguments_delta='{"objective":"b"}')

    mock_model2 = MagicMock()
    mock_model2.query_stream = two_step_stream
    tools.register(
        "spawn",
        {"type": "function", "function": {"name": "spawn"}},
        lambda a: ToolResult("started"),
    )
    agent2 = VoiceAgent(
        model=mock_model2,
        tools=tools,
        config=VoiceConfig(step_limit=2, turn_scoped_limits=True),
    )
    events2 = [e async for e in agent2.step_stream("multi")]
    tc2 = [e for e in events2 if isinstance(e, TurnComplete)][-1]
    assert tc2.exit_reason == "limits_exceeded"
    assert "First step spoken sentence." in tc2.final_text
    assert "Sorry" not in tc2.final_text
    assert agent2.messages[1]["role"] == "assistant"
    assert "First step spoken sentence." in agent2.messages[1]["content"]
    assert agent2.messages[3]["role"] == "assistant"
    assert agent2.messages[3]["content"] is None



@pytest.mark.asyncio
async def test_voice_agent_prunes_status_after_one_call_in_same_turn():
    seen_tools_per_call = []

    async def fake_stream(messages, tools=None, **kw):
        tool_names = [t["function"]["name"] for t in (tools or [])]
        seen_tools_per_call.append(tool_names)
        if len(seen_tools_per_call) == 1:
            yield ToolCallDelta(index=0, call_id="c1", name="status", arguments_delta="{}")
        else:
            yield TextDelta(text="Task-1 is currently running pytest.")
            yield TurnComplete(final_text="Task-1 is currently running pytest.")

    mock_model = MagicMock()
    mock_model.query_stream = fake_stream

    tools = ToolRegistry()
    for name in ("spawn", "status", "cancel"):
        tools.register(
            name,
            {"type": "function", "function": {"name": name}},
            lambda a: ToolResult("task-1 running: Called bash"),
        )

    agent = VoiceAgent(
        model=mock_model,
        tools=tools,
        config=VoiceConfig(tools=["spawn", "status", "cancel"]),
    )
    events = [e async for e in agent.step_stream("what is happening?")]

    assert seen_tools_per_call[0] == ["spawn", "status", "cancel"]
    assert "status" not in seen_tools_per_call[1]
    assert "spawn" in seen_tools_per_call[1]
    tc = [e for e in events if isinstance(e, TurnComplete)][-1]
    assert tc.exit_reason == "ok"
    assert "Task-1 is currently running pytest." in tc.final_text


@pytest.mark.asyncio
async def test_voice_agent_canonicalizes_json_args_in_repeat_history():
    calls = 0

    async def reordered_stream(messages, tools=None, **kw):
        nonlocal calls
        calls += 1
        if calls > 10:
            yield TurnComplete(final_text="bail")
            return
        arg_variants = [
            '{"a": 1, "b": 2}',
            '{"b": 2, "a": 1}',
            '{ "a" : 1 , "b" : 2 }',
        ]
        yield ToolCallDelta(
            index=0,
            call_id=f"c{calls}",
            name="spawn",
            arguments_delta=arg_variants[(calls - 1) % 3],
        )

    mock_model = MagicMock()
    mock_model.query_stream = reordered_stream
    tools = ToolRegistry()
    tools.register(
        "spawn",
        {"type": "function", "function": {"name": "spawn"}},
        lambda a: ToolResult("started"),
    )

    agent = VoiceAgent(
        model=mock_model,
        tools=tools,
        config=VoiceConfig(tools=["spawn"], max_calls_per_tool_per_turn={}),
    )
    events = [e async for e in agent.step_stream("go")]
    tc = [e for e in events if isinstance(e, TurnComplete)][-1]
    assert tc.exit_reason == "repeated_tool_call"
    assert calls == 3


