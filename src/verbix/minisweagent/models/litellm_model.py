from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import litellm
from pydantic import BaseModel

from verbix.core.events import (
    AgentEvent,
    TextDelta,
    ThoughtDelta,
    ToolCallDelta,
    TurnComplete,
)
from verbix.minisweagent.exceptions import FormatError
from verbix.minisweagent.models import GLOBAL_MODEL_STATS
from verbix.minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from verbix.minisweagent.models.utils.anthropic_utils import (
    _reorder_anthropic_thinking_blocks,
)
from verbix.minisweagent.models.utils.cache_control import set_cache_control
from verbix.minisweagent.models.utils.openai_multimodal import expand_multimodal_content
from verbix.minisweagent.models.utils.retry import retry

logger = logging.getLogger("litellm_model")


class LitellmModelConfig(BaseModel):
    model_name: str
    """Model name. Recommended to include provider, e.g. `anthropic/claude-3-5-sonnet`."""
    model_kwargs: dict[str, Any] = {}
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
    multimodal_regex: str = ""
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
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    def _query(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        kwargs_dict = dict(self.config.model_kwargs)
        kwargs_dict.update(kwargs)
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
            actions = self._parse_actions(response)
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
        if tools is not None:
            call_kwargs["tools"] = tools

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
        except litellm.exceptions.AuthenticationError as e:
            logger.error(f"Authentication failed for model {self.config.model_name}: {e}")
            raise

        accumulated_text: list[str] = []
        finish_reason: str | None = None

        try:
            async for chunk in response:
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
            logger.error(f"Authentication failed during stream for model {self.config.model_name}: {e}")
            raise

        full_text = "".join(accumulated_text)
        if full_text and finish_reason != "tool_calls":
            yield TurnComplete(final_text=full_text)

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

    def _parse_actions(self, response: Any) -> list[dict[str, Any]]:
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_toolcall_actions(
            tool_calls,
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": response.choices[0].finish_reason},
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
