from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx2

from verbix.core.models import CanonicalMessage
from verbix.provider.errors import (
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
    ServerError,
)
from verbix.provider.models import (
    Finish,
    StreamEvent,
    TextDelta,
    ThoughtDelta,
    ToolCallDelta,
    ToolDefinition,
    Usage,
)


class ProviderClient:
    """Async streaming client for OpenAI-compatible and OpenRouter endpoints using httpx2."""

    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "google/gemini-3.5-flash-lite",
        api_key: str | None = None,
        timeout: float = 60.0,
        max_tokens: int | None = 2048,
        http_client: httpx2.AsyncClient | None = None,
        default_extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._external_client = http_client
        self._client: httpx2.AsyncClient | None = http_client
        self.default_extra_body = default_extra_body

    def _get_client(self) -> httpx2.AsyncClient:
        if self._client is None:
            self._client = httpx2.AsyncClient(timeout=self.timeout)
        return self._client

    async def warm(self) -> None:
        """Warms the network connection by establishing a pooled TLS keep-alive session."""
        try:
            client = self._get_client()
            headers = (
                {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )
            await client.get(f"{self.base_url}/models", headers=headers)
        except (httpx2.HTTPError, OSError, RuntimeError):
            pass

    async def aclose(self) -> None:
        """Closes the underlying persistent HTTP client."""
        if self._client is not None and self._external_client is None:
            await self._client.aclose()
            self._client = None

    def _serialize_messages(
        self, messages: list[CanonicalMessage]
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for msg in messages:
            item: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                item["tool_call_id"] = msg.tool_call_id
            if msg.name:
                item["name"] = msg.name
            serialized.append(item)
        return serialized

    async def stream(
        self,
        messages: list[CanonicalMessage],
        tools: list[ToolDefinition] | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streams events, retrying once on a stale pooled connection if nothing was yielded yet."""
        yielded = False
        for attempt in range(2):
            try:
                async for event in self._stream_once(
                    messages, tools, reasoning_effort, max_tokens, extra_body
                ):
                    yielded = True
                    yield event
                return
            except (httpx2.ReadError, httpx2.RemoteProtocolError, httpx2.ConnectError):
                if yielded or attempt == 1:
                    raise
                # Drop the dead pooled connection and retry on a fresh one.
                if self._external_client is None and self._client is not None:
                    await self._client.aclose()
                    self._client = None

    async def _stream_once(
        self,
        messages: list[CanonicalMessage],
        tools: list[ToolDefinition] | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streams completion events from the provider endpoint via Server-Sent Events (SSE)."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._serialize_messages(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        if effective_max_tokens is not None:
            payload["max_tokens"] = effective_max_tokens
        if tools:
            payload["tools"] = [t.model_dump() for t in tools]
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        if self.default_extra_body:
            payload.update(self.default_extra_body)
        if extra_body:
            payload.update(extra_body)

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"

        client = self._get_client()

        async with client.stream(
            "POST", url, json=payload, headers=headers
        ) as response:
            if response.status_code == 401:
                err_body = await response.aread()
                raise AuthenticationError(
                    f"Authentication failed: {err_body.decode('utf-8', errors='replace')}",
                    status_code=401,
                )
            if response.status_code == 429:
                err_body = await response.aread()
                raise RateLimitError(
                    f"Rate limit exceeded: {err_body.decode('utf-8', errors='replace')}",
                    status_code=429,
                )
            if response.status_code == 400:
                err_body = await response.aread()
                raise InvalidRequestError(
                    f"Invalid request: {err_body.decode('utf-8', errors='replace')}",
                    status_code=400,
                )
            if response.status_code >= 500:
                err_body = await response.aread()
                raise ServerError(
                    f"Server error {response.status_code}: {err_body.decode('utf-8', errors='replace')}",
                    status_code=response.status_code,
                )
            if response.status_code != 200:
                err_body = await response.aread()
                raise ProviderError(
                    f"HTTP error {response.status_code}: {err_body.decode('utf-8', errors='replace')}",
                    status_code=response.status_code,
                )

            async for line in response.aiter_lines():
                trimmed = line.strip()
                if not trimmed or not trimmed.startswith("data:"):
                    continue

                data_str = trimmed[5:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})

                    # 1. Structured reasoning/thought delta (OpenRouter / DeepSeek / vLLM style)
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        yield ThoughtDelta(thought=reasoning)

                    # 2. Assistant text content delta
                    content = delta.get("content")
                    if content:
                        yield TextDelta(text=content)

                    # 3. Tool call streaming deltas
                    tool_calls = delta.get("tool_calls", [])
                    for tc in tool_calls:
                        idx = tc.get("index", 0)
                        call_id = tc.get("id")
                        fn = tc.get("function", {})
                        fn_name = fn.get("name")
                        args_delta = fn.get("arguments", "")
                        yield ToolCallDelta(
                            index=idx,
                            call_id=call_id,
                            name=fn_name,
                            arguments_delta=args_delta,
                        )

                    finish_reason = choices[0].get("finish_reason")
                    if finish_reason:
                        yield Finish(reason=finish_reason)

                # Usage metadata
                usage = chunk.get("usage")
                if usage:
                    reasoning_tokens = 0
                    comp_details = usage.get("completion_tokens_details")
                    if isinstance(comp_details, dict):
                        reasoning_tokens = comp_details.get("reasoning_tokens", 0)
                    elif "reasoning_tokens" in usage:
                        reasoning_tokens = usage.get("reasoning_tokens", 0)

                    yield Usage(
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        reasoning_tokens=reasoning_tokens,
                    )
