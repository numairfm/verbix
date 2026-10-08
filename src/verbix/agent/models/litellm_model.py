from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import litellm
from pydantic import BaseModel, Field

litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

from verbix.agent.exceptions import FormatError
from verbix.agent.models import GLOBAL_MODEL_STATS
from verbix.agent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from verbix.agent.models.utils.anthropic_utils import (
    _reorder_anthropic_thinking_blocks,
)
from verbix.agent.models.utils.cache_control import set_cache_control
from verbix.agent.models.utils.openai_multimodal import (
    DEFAULT_MULTIMODAL_REGEX,
    expand_multimodal_content,
)
from verbix.agent.models.utils.retry import retry
from verbix.core.events import (
    AgentEvent,
    TextDelta,
    ThoughtDelta,
    ToolCallDelta,
    TurnComplete,
)
from verbix.core.telemetry import record_latency

logger = logging.getLogger("litellm_model")

# Transient mid-stream failures worth one clean restart each - but only while
# nothing has been yielded yet. Re-requesting after speech started would
# repeat what the user already heard; that path belongs to the turn boundary.
_TRANSIENT_STREAM_ERRORS = (
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.Timeout,
    litellm.exceptions.RateLimitError,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.MidStreamFallbackError,
)
_MAX_STREAM_ATTEMPTS = 3


class LitellmModelConfig(BaseModel):
    model_name: str
    """Model name. Recommended to include provider, e.g. `anthropic/claude-3-5-sonnet`."""
    model_kwargs: dict[str, Any] = Field(default_factory=lambda: {"num_retries": 3})
    """Additional arguments passed to the API."""
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    """Model registry for cost tracking and model metadata."""
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models."""
    cost_tracking: Literal["default", "ignore_errors"] = cast(
        Literal["default", "ignore_errors"],
        os.getenv("MSWEA_COST_TRACKING", "default"),
    )
    """Cost tracking mode for this model. Can be 'default' or 'ignore_errors'."""
    format_error_template: str = "{{ error }}"
    """Template used when the LM's output is not in the expected format."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = DEFAULT_MULTIMODAL_REGEX
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class LitellmModel:
    abort_exceptions: ClassVar[list[type[BaseException]]] = [
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(self, *, config_class: Callable[..., Any] = LitellmModelConfig, **kwargs: Any) -> None:
        self.config = config_class(**kwargs)
        self.last_stream_cost = 0.0
        self._last_warm = 0.0
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    async def warm(self) -> None:
        """Pre-warm provider connection; rate-limited, never raises."""
        now = time.monotonic()
        if now - self._last_warm < 60.0:
            return
        self._last_warm = now
        try:
            await litellm.acompletion(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "warm"}],
                max_tokens=1,
            )
            record_latency("provider_warm")
        except Exception:
            logger.debug("Provider warm-up failed", exc_info=True)

    def _query(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        kwargs_dict = dict(self.config.model_kwargs)
        kwargs_dict.update(kwargs)
        kwargs_dict.setdefault("num_retries", 3)
        tools = kwargs_dict.pop("tools", [BASH_TOOL])
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=tools,
                **kwargs_dict,
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can set your API key in environment variables or provider config."
            raise

    def _prepare_messages_for_api(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        return set_cache_control(prepared, mode=self.config.set_cache_control)

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        response: Any = None
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        assert response is not None
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        try:
            actions = self._parse_actions(response, kwargs.get("tools"))
        except FormatError as e:
            e.messages[0]["extra"].update(cost_output)
            try:
                e.messages[0]["extra"]["response"] = response.model_dump(mode="json")
            except Exception:  # noqa: BLE001
                e.messages[0]["extra"]["response"] = repr(response)
            raise
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": actions,
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    async def query_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentEvent]:
        """Stream model completion chunks as unified AgentEvents."""
        prepared_messages = self._prepare_messages_for_api(messages)
        call_kwargs = dict(self.config.model_kwargs)
        call_kwargs.update(kwargs)
        call_kwargs.setdefault("num_retries", 3)
        call_kwargs.setdefault("stream_options", {"include_usage": True})
        if tools is not None:
            call_kwargs["tools"] = tools

        attempts = 0
        while True:
            accumulated_text: list[str] = []
            finish_reason: str | None = None
            chunks: list[Any] = []
            saw_usage = False
            yielded_any = False
            try:
                response = cast(
                    Any,
                    await litellm.acompletion(
                        model=self.config.model_name,
                        messages=prepared_messages,
                        stream=True,
                        **call_kwargs,
                    ),
                )
                async for chunk in response:
                    yielded_any = True
                    chunks.append(chunk)
                    if getattr(chunk, "usage", None):
                        saw_usage = True
                    choices = getattr(chunk, "choices", None)
                    if choices is None and isinstance(chunk, dict):
                        choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = getattr(choice, "delta", None)
                    if delta is None and isinstance(choice, dict):
                        delta = choice.get("delta", {})

                    fr = getattr(choice, "finish_reason", None)
                    if fr is None and isinstance(choice, dict):
                        fr = choice.get("finish_reason")
                    if fr:
                        finish_reason = fr

                    # 1. Thought / Reasoning
                    reasoning = (
                        getattr(delta, "reasoning_content", None)
                        or getattr(delta, "reasoning", None)
                        or getattr(delta, "thought", None)
                    )
                    if reasoning is None and isinstance(delta, dict):
                        reasoning = (
                            delta.get("reasoning_content")
                            or delta.get("reasoning")
                            or delta.get("thought")
                        )
                    if reasoning:
                        yield ThoughtDelta(thought=reasoning)

                    # 2. Text Content
                    content = getattr(delta, "content", None)
                    if content is None and isinstance(delta, dict):
                        content = delta.get("content")
                    if content:
                        accumulated_text.append(content)
                        yield TextDelta(text=content)

                    # 3. Tool Calls
                    tool_calls = getattr(delta, "tool_calls", None)
                    if tool_calls is None and isinstance(delta, dict):
                        tool_calls = delta.get("tool_calls", [])
                    if tool_calls:
                        for tc in tool_calls:
                            idx = getattr(tc, "index", 0)
                            call_id = getattr(tc, "id", None)
                            fn = getattr(tc, "function", None)
                            if isinstance(tc, dict):
                                idx = tc.get("index", 0)
                                call_id = tc.get("id")
                                fn = tc.get("function", {})
                            fn_name = getattr(fn, "name", None) if fn else None
                            args_delta = getattr(fn, "arguments", "") if fn else ""
                            if isinstance(fn, dict):
                                fn_name = fn.get("name")
                                args_delta = fn.get("arguments", "")
                            yield ToolCallDelta(
                                index=idx,
                                call_id=call_id,
                                name=fn_name,
                                arguments_delta=args_delta or "",
                            )
            except litellm.exceptions.AuthenticationError as e:
                e.message += " You can set your API key in environment variables or provider config."
                logger.error(f"Authentication failed during stream for model {self.config.model_name}: {e}")
                raise
            except _TRANSIENT_STREAM_ERRORS as e:
                if yielded_any or attempts >= _MAX_STREAM_ATTEMPTS - 1:
                    raise
                attempts += 1
                logger.warning(
                    f"Transient stream error for model {self.config.model_name} "
                    f"(attempt {attempts + 1}/{_MAX_STREAM_ATTEMPTS}): {e}",
                )
                await asyncio.sleep(min(8.0, 2.0 ** (attempts - 1)))
                continue
            break

        full_text = "".join(accumulated_text)
        if saw_usage:
            self._track_stream_cost(chunks, prepared_messages)
        if full_text and finish_reason != "tool_calls":
            yield TurnComplete(final_text=full_text)

    def _track_stream_cost(self, chunks: list[Any], messages: list[dict[str, Any]]) -> None:
        """Rebuild the full response from stream chunks and record its cost (never raises)."""
        try:
            full_response = litellm.stream_chunk_builder(chunks, messages=messages)
            cost = litellm.cost_calculator.completion_cost(full_response, model=self.config.model_name)
            if cost > 0:
                self.last_stream_cost += cost
                GLOBAL_MODEL_STATS.add(cost)
        except Exception:  # noqa: BLE001
            logger.debug("Streaming cost calculation failed", exc_info=True)

    def _calculate_cost(self, response: Any) -> dict[str, float]:
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors'."
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        return {"cost": cost}

    def _parse_actions(self, response: Any, tool_defs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        tool_calls = response.choices[0].message.tool_calls or []
        allowed: set[str] = set()
        for definition in tool_defs or []:
            name = definition.get("function", {}).get("name") if isinstance(definition, dict) else None
            if name:
                allowed.add(name)
        return parse_toolcall_actions(
            tool_calls,
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": response.choices[0].finish_reason},
            allowed_tools=allowed or None,
        )

    def format_message(self, **kwargs: Any) -> dict[str, Any]:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
