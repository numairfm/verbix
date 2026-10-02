import json

import pytest
from httpx2 import ASGITransport, AsyncClient

from verbix.core.models import CanonicalMessage
from verbix.provider.client import ProviderClient
from verbix.provider.errors import AuthenticationError, RateLimitError
from verbix.provider.models import TextDelta, ThoughtDelta, ToolCallDelta, Usage


# A minimal ASGI app simulating an OpenAI-compatible SSE endpoint
async def mock_sse_app(scope, receive, send):
    assert scope["type"] == "http"

    # Read body
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body", False):
            break

    payload = json.loads(body.decode("utf-8")) if body else {}

    # Check for simulated error scenarios
    auth_header = (
        dict(scope.get("headers", [])).get(b"authorization", b"").decode("utf-8")
    )
    if "invalid-key" in auth_header:
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": json.dumps({"error": {"message": "Invalid API key"}}).encode(
                    "utf-8"
                ),
            }
        )
        return

    if "rate-limit" in auth_header:
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": json.dumps(
                    {"error": {"message": "Rate limit exceeded"}}
                ).encode("utf-8"),
            }
        )
        return

    # Normal streaming SSE response
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")],
        }
    )

    # Scenario: reasoning + text
    if payload.get("stream_reasoning"):
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "Thinking about math..."}}]},
            {"choices": [{"delta": {"content": "2 + 2 is "}}]},
            {"choices": [{"delta": {"content": "4."}}]},
            {"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        ]
    elif payload.get("stream_tool"):
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "bash", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"command":'}}
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": ' "ls -la"}'}}
                            ]
                        }
                    }
                ]
            },
        ]
    else:
        chunks = [
            {"choices": [{"delta": {"content": "Hello "}}]},
            {"choices": [{"delta": {"content": "world!"}}]},
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
async def test_streaming_text():
    transport = ASGITransport(app=mock_sse_app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        client = ProviderClient(
            base_url="http://test/v1",
            api_key="valid-key",
            model="test-model",
            http_client=http_client,
        )
        messages = [CanonicalMessage(role="user", content="Hi")]

        events = []
        async for event in client.stream(messages):
            events.append(event)

        text_deltas = [e.text for e in events if isinstance(e, TextDelta)]
        assert "".join(text_deltas) == "Hello world!"


@pytest.mark.asyncio
async def test_streaming_reasoning_and_text():
    transport = ASGITransport(app=mock_sse_app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        client = ProviderClient(
            base_url="http://test/v1",
            api_key="valid-key",
            model="test-model",
            http_client=http_client,
        )
        # Send flag via extra kwargs or payload
        messages = [CanonicalMessage(role="user", content="Calculate")]

        events = []
        async for event in client.stream(
            messages, extra_body={"stream_reasoning": True}
        ):
            events.append(event)

        thought_deltas = [e.thought for e in events if isinstance(e, ThoughtDelta)]
        text_deltas = [e.text for e in events if isinstance(e, TextDelta)]
        usage_events = [e for e in events if isinstance(e, Usage)]

        assert "".join(thought_deltas) == "Thinking about math..."
        assert "".join(text_deltas) == "2 + 2 is 4."
        assert len(usage_events) == 1
        assert usage_events[0].prompt_tokens == 10
        assert usage_events[0].completion_tokens == 5


@pytest.mark.asyncio
async def test_streaming_tool_calls():
    transport = ASGITransport(app=mock_sse_app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        client = ProviderClient(
            base_url="http://test/v1",
            api_key="valid-key",
            model="test-model",
            http_client=http_client,
        )
        messages = [CanonicalMessage(role="user", content="List files")]

        events = []
        async for event in client.stream(messages, extra_body={"stream_tool": True}):
            events.append(event)

        tool_deltas = [e for e in events if isinstance(e, ToolCallDelta)]
        assert len(tool_deltas) == 3
        assert tool_deltas[0].call_id == "call_1"
        assert tool_deltas[0].name == "bash"
        combined_args = "".join(e.arguments_delta for e in tool_deltas)
        assert combined_args == '{"command": "ls -la"}'


@pytest.mark.asyncio
async def test_auth_error_handling():
    transport = ASGITransport(app=mock_sse_app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        client = ProviderClient(
            base_url="http://test/v1",
            api_key="invalid-key",
            model="test-model",
            http_client=http_client,
        )
        messages = [CanonicalMessage(role="user", content="Hi")]

        with pytest.raises(AuthenticationError):
            async for _ in client.stream(messages):
                pass


@pytest.mark.asyncio
async def test_rate_limit_error_handling():
    transport = ASGITransport(app=mock_sse_app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        client = ProviderClient(
            base_url="http://test/v1",
            api_key="rate-limit",
            model="test-model",
            http_client=http_client,
        )
        messages = [CanonicalMessage(role="user", content="Hi")]

        with pytest.raises(RateLimitError):
            async for _ in client.stream(messages):
                pass
