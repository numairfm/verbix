import json
from pathlib import Path

import pytest
from httpx2 import ASGITransport, AsyncClient

from verbix.core.agent import AgentSession
from verbix.core.events import (
    TextDelta,
    ThoughtDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnComplete,
)
from verbix.provider.client import ProviderClient
from verbix.tools.filesystem import FileReadTool, FileWriteTool


# Mock SSE server that simulates multi-step agent dialogues
async def mock_agent_sse_app(scope, receive, send):
    assert scope["type"] == "http"

    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body", False):
            break

    payload = json.loads(body.decode("utf-8")) if body else {}
    messages = payload.get("messages", [])
    last_msg = messages[-1] if messages else {}

    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")],
        }
    )

    # Case 1: Simple conversational turn
    if last_msg.get("content") == "Say hello":
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "User asked to say hello."}}]},
            {"choices": [{"delta": {"content": "Hello! How can "}}]},
            {"choices": [{"delta": {"content": "I help you today?"}}]},
            {"choices": [{"finish_reason": "stop"}]},
        ]
    # Case 1b: Reasoning only, empty text and tool calls
    elif last_msg.get("content") == "Thought only":
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "Internal reflection without speech."}}]},
            {"choices": [{"finish_reason": "stop"}]},
        ]
    # Case 1c: Completely blank response
    elif last_msg.get("content") == "Blank response":
        chunks = [
            {"choices": [{"finish_reason": "stop"}]},
        ]
    # Case 2: Tool call requested
    elif last_msg.get("content") == "Create test.txt":
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_w1",
                                    "function": {
                                        "name": "file_write",
                                        "arguments": '{"path": "test.txt", "content": "hello agent"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"finish_reason": "tool_calls"}]},
        ]
    # Case 3: Following tool observation
    elif last_msg.get("role") == "tool" and last_msg.get("tool_call_id") == "call_w1":
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "content": "I have created test.txt with 'hello agent'."
                        }
                    }
                ]
            },
            {"choices": [{"finish_reason": "stop"}]},
        ]
    # Case 4: Repeating loop for cycle detection testing
    elif "looping" in last_msg.get("content", "") or last_msg.get("role") == "tool":
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_loop",
                                    "function": {
                                        "name": "file_read",
                                        "arguments": '{"path": "loop.txt"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"finish_reason": "tool_calls"}]},
        ]
    else:
        chunks = [
            {"choices": [{"delta": {"content": "Default response."}}]},
            {"choices": [{"finish_reason": "stop"}]},
        ]

    for chunk in chunks:
        sse_line = f"data: {json.dumps(chunk)}\n\n"
        await send(
            {
                "type": "http.response.body",
                "body": sse_line.encode("utf-8"),
                "more_body": True,
            }
        )

    await send(
        {"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False}
    )


@pytest.mark.asyncio
async def test_agent_single_conversational_turn(tmp_path: Path):
    transport = ASGITransport(app=mock_agent_sse_app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        provider = ProviderClient(
            base_url="http://test/v1", model="test-model", http_client=http_client
        )
        session = AgentSession(
            session_id="sess_1",
            system_prompt="You are Verbix.",
            provider=provider,
            workspace=tmp_path,
        )

        events = []
        async for event in session.step(user_input="Say hello"):
            events.append(event)

        thought_events = [e for e in events if isinstance(e, ThoughtDelta)]
        text_events = [e for e in events if isinstance(e, TextDelta)]
        turn_complete = [e for e in events if isinstance(e, TurnComplete)]

        assert len(thought_events) == 1
        assert thought_events[0].thought == "User asked to say hello."
        assert (
            "".join(e.text for e in text_events) == "Hello! How can I help you today?"
        )
        assert len(turn_complete) == 1
        assert turn_complete[0].final_text == "Hello! How can I help you today?"

        # Verify history state
        assert len(session.history.messages) == 2
        assert session.history.messages[0].role == "user"
        assert session.history.messages[0].content == "Say hello"
        assert session.history.messages[1].role == "assistant"
        assert session.history.messages[1].content == "Hello! How can I help you today?"
        assert session.history.messages[1].thought == "User asked to say hello."


@pytest.mark.asyncio
async def test_agent_tool_calling_turn(tmp_path: Path):
    transport = ASGITransport(app=mock_agent_sse_app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        provider = ProviderClient(
            base_url="http://test/v1", model="test-model", http_client=http_client
        )
        session = AgentSession(
            session_id="sess_tools",
            system_prompt="You are Verbix.",
            provider=provider,
            tools=[FileWriteTool(), FileReadTool()],
            workspace=tmp_path,
        )

        events = []
        async for event in session.step(user_input="Create test.txt"):
            events.append(event)

        tool_starts = [e for e in events if isinstance(e, ToolCallStarted)]
        tool_finishes = [e for e in events if isinstance(e, ToolCallFinished)]
        turn_completes = [e for e in events if isinstance(e, TurnComplete)]

        assert len(tool_starts) == 1
        assert tool_starts[0].tool_name == "file_write"
        assert len(tool_finishes) == 1
        assert tool_finishes[0].is_error is False
        assert len(turn_completes) == 1
        assert "created test.txt" in turn_completes[0].final_text

        # File was actually written to workspace
        assert (tmp_path / "test.txt").read_text() == "hello agent"

        # History contains: user -> assistant(tool_calls) -> tool -> assistant(final)
        assert len(session.history.messages) == 4
        assert session.history.messages[0].role == "user"
        assert session.history.messages[1].role == "assistant"
        assert session.history.messages[1].tool_calls is not None
        assert session.history.messages[2].role == "tool"
        assert session.history.messages[3].role == "assistant"


@pytest.mark.asyncio
async def test_agent_cycle_detection(tmp_path: Path):
    transport = ASGITransport(app=mock_agent_sse_app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        provider = ProviderClient(
            base_url="http://test/v1", model="test-model", http_client=http_client
        )
        session = AgentSession(
            session_id="sess_cycle",
            system_prompt="You are Verbix.",
            provider=provider,
            tools=[FileReadTool()],
            workspace=tmp_path,
            max_turns=10,
        )

        events = []
        async for event in session.step(user_input="start looping"):
            events.append(event)

        # Agent should detect repeated file_read loop and stop
        turn_completes = [e for e in events if isinstance(e, TurnComplete)]
        assert len(turn_completes) == 1
        assert (
            "loop detected" in turn_completes[0].final_text.lower()
            or "stopped" in turn_completes[0].final_text.lower()
        )


@pytest.mark.asyncio
async def test_agent_empty_content_with_thought_fallback(tmp_path: Path):
    transport = ASGITransport(app=mock_agent_sse_app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        provider = ProviderClient(
            base_url="http://test/v1", model="test-model", http_client=http_client
        )
        session = AgentSession(
            session_id="sess_thought_fallback",
            system_prompt="You are Verbix.",
            provider=provider,
            workspace=tmp_path,
        )

        events = []
        async for event in session.step(user_input="Thought only"):
            events.append(event)

        turn_completes = [e for e in events if isinstance(e, TurnComplete)]
        assert len(turn_completes) == 1
        assert turn_completes[0].final_text == "Give me a moment."


@pytest.mark.asyncio
async def test_agent_empty_content_without_thought_fallback(tmp_path: Path):
    transport = ASGITransport(app=mock_agent_sse_app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        provider = ProviderClient(
            base_url="http://test/v1", model="test-model", http_client=http_client
        )
        session = AgentSession(
            session_id="sess_blank_fallback",
            system_prompt="You are Verbix.",
            provider=provider,
            workspace=tmp_path,
        )

        events = []
        async for event in session.step(user_input="Blank response"):
            events.append(event)

        turn_completes = [e for e in events if isinstance(e, TurnComplete)]
        assert len(turn_completes) == 1
        assert turn_completes[0].final_text == "Sorry, something went wrong. Could you repeat that?"

