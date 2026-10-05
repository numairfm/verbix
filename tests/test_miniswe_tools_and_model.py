from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import litellm.exceptions
import pytest

from verbix.core.events import (
    TextDelta,
    ThoughtDelta,
    ToolCallDelta,
    TurnComplete,
)
from verbix.minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from verbix.minisweagent.tools.registry import ToolRegistry


def test_tool_registry():
    reg = ToolRegistry()
    reg.register(
        "greet",
        {"type": "function", "function": {"name": "greet"}},
        lambda args: "hello " + args["name"],
    )
    defs = reg.get_definitions(["greet"])
    assert len(defs) == 1
    assert defs[0]["function"]["name"] == "greet"
    assert reg.execute("greet", {"name": "world"}) == "hello world"


def test_tool_registry_get_all_definitions():
    reg = ToolRegistry()
    reg.register("t1", {"type": "function", "function": {"name": "t1"}}, lambda args: "1")
    reg.register("t2", {"type": "function", "function": {"name": "t2"}}, lambda args: "2")

    defs = reg.get_definitions()
    assert len(defs) == 2
    names = [d["function"]["name"] for d in defs]
    assert "t1" in names and "t2" in names


def test_tool_registry_not_found():
    reg = ToolRegistry()
    with pytest.raises(KeyError, match="not registered"):
        reg.execute("missing", {})


@pytest.mark.asyncio
async def test_tool_registry_async_handler():
    reg = ToolRegistry()

    async def async_add(args: dict[str, Any]) -> int:
        return args["a"] + args["b"]

    reg.register(
        "add",
        {"type": "function", "function": {"name": "add"}},
        async_add,
    )

    result = await reg.aexecute("add", {"a": 2, "b": 3})
    assert result == 5


def test_tool_registry_does_not_swallow_internal_type_error():
    reg = ToolRegistry()
    call_count = 0

    def bad_handler(args: dict[str, Any]) -> str:
        nonlocal call_count
        call_count += 1
        raise TypeError("internal handler bug")

    reg.register(
        "bad",
        {"type": "function", "function": {"name": "bad"}},
        bad_handler,
    )

    with pytest.raises(TypeError, match="internal handler bug"):
        reg.execute("bad", {"x": 1}, extra="unused")

    assert call_count == 1


@pytest.mark.asyncio
async def test_tool_registry_sync_execute_async_in_running_loop_raises():
    reg = ToolRegistry()

    async def async_fn(args: dict[str, Any]) -> str:
        return "async done"

    reg.register(
        "async_tool",
        {"type": "function", "function": {"name": "async_tool"}},
        async_fn,
    )

    with pytest.raises(RuntimeError, match="Cannot execute async tool 'async_tool' synchronously"):
        reg.execute("async_tool", {})


class MockStreamChunk:
    def __init__(
        self,
        content: str | None = None,
        reasoning_content: str | None = None,
        tool_calls: list[Any] | None = None,
        finish_reason: str | None = None,
    ) -> None:
        self.choices = [
            MockChoice(
                content=content,
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
        ]


class MockChoice:
    def __init__(
        self,
        content: str | None = None,
        reasoning_content: str | None = None,
        tool_calls: list[Any] | None = None,
        finish_reason: str | None = None,
    ) -> None:
        self.delta = MockDelta(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
        )
        self.finish_reason = finish_reason


class MockDelta:
    def __init__(
        self,
        content: str | None = None,
        reasoning_content: str | None = None,
        tool_calls: list[Any] | None = None,
    ) -> None:
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls


class MockToolCall:
    def __init__(self, index: int, call_id: str | None, name: str | None, arguments: str) -> None:
        self.index = index
        self.id = call_id
        self.function = MockFunction(name=name, arguments=arguments)


class MockFunction:
    def __init__(self, name: str | None, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


async def async_generator(items: list[Any]):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_litellm_model_query_stream_text_and_thought():
    chunks = [
        MockStreamChunk(reasoning_content="Let's think "),
        MockStreamChunk(reasoning_content="carefully."),
        MockStreamChunk(content="Hello "),
        MockStreamChunk(content="world!", finish_reason="stop"),
    ]

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = async_generator(chunks)

        model = LitellmModel(model_name="test-model")
        events = [
            e
            async for e in model.query_stream(
                [{"role": "user", "content": "hi"}],
                tools=None,
            )
        ]

        # Verify call to litellm.acompletion
        mock_acompletion.assert_called_once()
        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["stream"] is True

        # Check yielded events
        thought_events = [e for e in events if isinstance(e, ThoughtDelta)]
        assert len(thought_events) == 2
        assert thought_events[0].thought == "Let's think "
        assert thought_events[1].thought == "carefully."

        text_events = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_events) == 2
        assert text_events[0].text == "Hello "
        assert text_events[1].text == "world!"

        turn_complete = [e for e in events if isinstance(e, TurnComplete)]
        assert len(turn_complete) == 1
        assert turn_complete[0].final_text == "Hello world!"


@pytest.mark.asyncio
async def test_litellm_model_query_stream_tool_calls():
    chunks = [
        MockStreamChunk(
            tool_calls=[
                MockToolCall(
                    index=0,
                    call_id="call_123",
                    name="bash",
                    arguments='{"command": "echo ',
                )
            ]
        ),
        MockStreamChunk(
            tool_calls=[
                MockToolCall(
                    index=0,
                    call_id=None,
                    name=None,
                    arguments='hi"}',
                )
            ],
            finish_reason="tool_calls",
        ),
    ]

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = async_generator(chunks)

        model = LitellmModel(model_name="test-model")
        tools = [{"type": "function", "function": {"name": "bash"}}]
        events = [
            e
            async for e in model.query_stream(
                [{"role": "user", "content": "run echo"}],
                tools=tools,
            )
        ]

        mock_acompletion.assert_called_once()
        assert mock_acompletion.call_args.kwargs["tools"] == tools

        tool_deltas = [e for e in events if isinstance(e, ToolCallDelta)]
        assert len(tool_deltas) == 2
        assert tool_deltas[0].call_id == "call_123"
        assert tool_deltas[0].name == "bash"
        assert tool_deltas[0].arguments_delta == '{"command": "echo '
        assert tool_deltas[1].arguments_delta == 'hi"}'

        # No TurnComplete because finish_reason is "tool_calls"
        assert not any(isinstance(e, TurnComplete) for e in events)


@pytest.mark.asyncio
async def test_litellm_model_query_stream_authentication_error():
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = litellm.exceptions.AuthenticationError(
            message="Invalid API Key",
            model="test-model",
            llm_provider="openai",
        )

        model = LitellmModel(model_name="test-model")
        with pytest.raises(litellm.exceptions.AuthenticationError) as exc_info:
            _ = [
                e
                async for e in model.query_stream(
                    [{"role": "user", "content": "hi"}]
                )
            ]
        assert "You can set your API key in environment variables or provider config." in exc_info.value.message


def test_tool_call_delta_reexported_in_core():
    from verbix.core import ToolCallDelta as CoreToolCallDelta
    assert CoreToolCallDelta is ToolCallDelta


def test_litellm_model_config():
    config = LitellmModelConfig(model_name="anthropic/claude-3-5-sonnet", model_kwargs={"temperature": 0.2})
    assert config.model_name == "anthropic/claude-3-5-sonnet"
    assert config.model_kwargs["temperature"] == 0.2
    model = LitellmModel(config_class=lambda **kw: config)
    assert model.config.model_name == "anthropic/claude-3-5-sonnet"
