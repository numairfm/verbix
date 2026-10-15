"""VoiceAgent: Conversational streaming voice agent with speech segmentation and tool execution."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import platform
import time
from collections.abc import AsyncIterator
from typing import Any

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel, Field, model_validator

from verbix.agent.tools.registry import ToolRegistry
from verbix.agent.utils.compaction import compact_messages
from verbix.agent.utils.history import (
    CANCELLED_TOOL_CONTENT,
    validate_and_repair_history,
)
from verbix.core.events import (
    AgentEvent,
    ExitReason,
    TextDelta,
    ThoughtDelta,
    ToolCallDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnComplete,
)
from verbix.core.telemetry import record_latency
from verbix.tools.base import FileAttachment, ToolResult, parse_lenient_json
from verbix.voice.segments import split_chunks

_REPEAT_LIMIT = 3


def tool_message_content(
    observation: str, attachments: tuple[FileAttachment, ...] = ()
) -> str | list[dict[str, Any]]:
    """Format tool message observation and optional multimodal attachments."""
    if not attachments:
        return observation
    parts: list[dict[str, Any]] = [{"type": "text", "text": observation}]
    for a in attachments:
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{a.mime};base64,{base64.b64encode(a.data).decode()}"},
            }
        )
    return parts


class VoiceConfig(BaseModel):
    """Configuration for VoiceAgent."""

    system_template: str = ""
    instance_template: str = "{{task}}"
    tools: list[str] = Field(default_factory=list)
    step_limit: int = 0
    cost_limit: float = 3.0
    wall_time_limit_seconds: int = 0
    turn_scoped_limits: bool = True
    max_context_tokens: int | None = 16000
    tool_choice: str | None = None
    system_prompt: str = ""
    spoken_error_fallback: str = "Sorry — I ran into a problem. Could you say that again?"
    max_calls_per_tool_per_turn: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "tools_list" in data and "tools" not in data:
                data["tools"] = data["tools_list"]
            if "system_prompt" in data and not data.get("system_template"):
                data["system_template"] = data["system_prompt"]
            elif "system_template" in data and not data.get("system_prompt"):
                data["system_prompt"] = data["system_template"]
        return data


class VoiceAgent:
    """Conversational voice agent with streaming, speech segmentation, and tool execution."""

    def __init__(
        self,
        model: Any,
        tools: ToolRegistry | None = None,
        config: VoiceConfig | BaseModel | dict[str, Any] | None = None,
        env: Any = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.tools = tools
        self.env = env

        if config is None:
            self.config = VoiceConfig(**kwargs)
        elif isinstance(config, dict):
            self.config = VoiceConfig(**config)
        elif isinstance(config, VoiceConfig):
            self.config = config
        elif isinstance(config, BaseModel):
            self.config = VoiceConfig(**config.model_dump())
        else:
            self.config = VoiceConfig(**kwargs)

        self.messages: list[dict[str, Any]] = []
        self.cost: float = 0.0
        self.turn_cost: float = 0.0
        self.n_calls: int = 0
        self.turn_calls: int = 0
        self._start_time: float = time.time()
        self._delivered_text: str = ""
        self.extra_template_vars: dict[str, Any] = {}
        self.logger = logging.getLogger("verbix.agent.voice")

    @property
    def delivered_text(self) -> str:
        return self._delivered_text

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        env_vars = (
            self.env.get_template_vars()
            if self.env is not None and hasattr(self.env, "get_template_vars")
            else {}
        )
        model_vars = (
            self.model.get_template_vars()
            if self.model is not None and hasattr(self.model, "get_template_vars")
            else {}
        )
        platform_vars = platform.uname()._asdict()
        return {
            **platform_vars,
            **self.config.model_dump(),
            **env_vars,
            **model_vars,
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
            "elapsed_seconds": int(time.time() - self._start_time),
            **self.extra_template_vars,
            **kwargs,
        }

    def _render_template(self, template_str: str, **kwargs: Any) -> str:
        if not template_str:
            return ""
        context = self.get_template_vars(**kwargs)
        try:
            return Template(template_str, undefined=StrictUndefined).render(**context)
        except Exception:  # noqa: BLE001
            try:
                return Template(template_str).render(**context)
            except Exception:  # noqa: BLE001
                return template_str

    def add_messages(self, *messages: dict[str, Any]) -> list[dict[str, Any]]:
        self.messages.extend(messages)
        return list(messages)

    def get_tool_definitions(
        self, excluded: set[str] | None = None
    ) -> list[dict[str, Any]] | None:
        """Return tool schemas configured for this agent from the registry."""
        if self.tools is None or self.config.tool_choice == "none":
            return None
        excl = excluded or set()
        if self.config.tools:
            allowed = [n for n in self.config.tools if n not in excl]
            if not allowed:
                return None
            defs = self.tools.get_definitions(allowed)
            return defs if defs else None
        all_defs = self.tools.get_definitions()
        if excl and all_defs:
            all_defs = [
                d
                for d in all_defs
                if d.get("function", {}).get("name") not in excl
            ]
        return all_defs if all_defs else None

    def _model_stream_cost(self) -> float:
        cost = getattr(self.model, "last_stream_cost", 0.0)
        return cost if isinstance(cost, (int, float)) else 0.0

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        if self.tools is None:
            return ToolResult(f"Error: Tool '{name}' is not registered or supported.", is_error=True)
        try:
            res = await self.tools.aexecute(name, args)
            if isinstance(res, ToolResult):
                return res
            elif isinstance(res, str):
                return ToolResult(res, is_error=False, is_terminal=False)
            else:
                return ToolResult(str(res), is_error=False, is_terminal=False)
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"Error executing tool '{name}': {e}", is_error=True)

    def _abort_turn(self, exit_reason: ExitReason) -> TurnComplete:
        """Repairs history without 'exit' role and returns a user-safe TurnComplete."""
        self.messages[:] = validate_and_repair_history(
            self.messages, delivered_text=self._delivered_text
        )
        if self._delivered_text.strip():
            final = self._delivered_text
        else:
            final = self.config.spoken_error_fallback
            if final:
                self.messages.append({"role": "assistant", "content": final})
        return TurnComplete(final_text=final, exit_reason=exit_reason, is_error=True)

    async def step_stream(self, user_input: str | None = None) -> AsyncIterator[AgentEvent]:
        """Stream an agent turn asynchronously, yielding AgentEvents and executing tools."""
        if self.config.turn_scoped_limits:
            self.turn_calls = 0
            self.turn_cost = 0.0

        is_initial_turn = not any(m.get("role") == "user" for m in self.messages)

        if not self.messages and self.config.system_template:
            sys_content = self._render_template(self.config.system_template)
            if sys_content:
                self.messages.append({"role": "system", "content": sys_content})

        if user_input is not None:
            if is_initial_turn and self.config.instance_template:
                user_content = self._render_template(self.config.instance_template, task=user_input)
            else:
                user_content = user_input
            self.messages.append({"role": "user", "content": user_content})

        repeat_history: list[tuple[str, str]] = []
        per_turn_counts: dict[str, int] = {}
        self._delivered_text = ""

        try:
            while True:
                calls = self.turn_calls if self.config.turn_scoped_limits else self.n_calls
                spent = self.turn_cost if self.config.turn_scoped_limits else self.cost
                if 0 < self.config.step_limit <= calls or 0 < self.config.cost_limit <= spent:
                    yield self._abort_turn("limits_exceeded")
                    return

                if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
                    yield self._abort_turn("time_exceeded")
                    return

                self.n_calls += 1
                self.turn_calls += 1
                exhausted = {
                    k
                    for k, v in per_turn_counts.items()
                    if v >= self.config.max_calls_per_tool_per_turn.get(k, 999999)
                }
                tool_defs = self.get_tool_definitions(excluded=exhausted)
                prev_stream_cost = self._model_stream_cost()

                accumulated_tool_calls: dict[int, dict[str, Any]] = {}
                turn_complete_event: TurnComplete | None = None
                buffered_text_events: list[TextDelta] = []
                has_tool_calls = False
                pending_text = ""
                released_text = ""
                step_delivered_text = ""
                first_chunk_marked = False

                self.messages[:] = compact_messages(
                    self.messages, max_tokens=self.config.max_context_tokens or 16000
                )
                stream_kwargs: dict[str, Any] = {}
                if self.config.tool_choice is not None:
                    stream_kwargs["tool_choice"] = self.config.tool_choice
                elif tool_defs is None:
                    stream_kwargs["tool_choice"] = "none"

                stream = self.model.query_stream(self.messages, tools=tool_defs, **stream_kwargs)
                async for event in stream:
                    if isinstance(event, TextDelta):
                        if has_tool_calls:
                            continue
                        buffered_text_events.append(event)
                        pending_text += event.text
                        if len(split_chunks(pending_text)) > 1:
                            for buffered in buffered_text_events:
                                self._delivered_text += buffered.text
                                step_delivered_text += buffered.text
                                yield buffered
                            released_text += pending_text
                            pending_text = ""
                            buffered_text_events.clear()
                            if not first_chunk_marked:
                                record_latency("first_speakable_chunk")
                                first_chunk_marked = True
                    elif isinstance(event, ThoughtDelta):
                        yield event
                    elif isinstance(event, ToolCallDelta):
                        has_tool_calls = True
                        buffered_text_events.clear()
                        pending_text = ""
                        idx = event.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "call_id": event.call_id or f"call_{idx}",
                                "name": event.name or "",
                                "arguments": "",
                            }
                        tc = accumulated_tool_calls[idx]
                        if event.call_id:
                            tc["call_id"] = event.call_id
                        if event.name:
                            tc["name"] = event.name
                        if event.arguments_delta:
                            tc["arguments"] += event.arguments_delta
                    elif isinstance(event, ToolCallStarted):
                        has_tool_calls = True
                        buffered_text_events.clear()
                        pending_text = ""
                        idx = len(accumulated_tool_calls)
                        accumulated_tool_calls[idx] = {
                            "call_id": event.call_id,
                            "name": event.tool_name,
                            "arguments": json.dumps(event.arguments),
                            "parsed_arguments": event.arguments,
                        }
                    elif isinstance(event, TurnComplete):
                        turn_complete_event = event
                    else:
                        yield event

                stream_cost_delta = self._model_stream_cost() - prev_stream_cost
                if stream_cost_delta:
                    self.cost += stream_cost_delta
                    self.turn_cost += stream_cost_delta

                if not has_tool_calls:
                    for buffered in buffered_text_events:
                        self._delivered_text += buffered.text
                        step_delivered_text += buffered.text
                        yield buffered
                    buffered_text_events.clear()

                if accumulated_tool_calls:
                    tool_calls_to_execute: list[dict[str, Any]] = []
                    for idx in sorted(accumulated_tool_calls.keys()):
                        tc_info = accumulated_tool_calls[idx]
                        args_str = tc_info["arguments"]
                        if "parsed_arguments" in tc_info:
                            args_dict = tc_info["parsed_arguments"]
                        else:
                            try:
                                args_dict = json.loads(args_str) if args_str else {}
                            except json.JSONDecodeError:
                                try:
                                    args_dict = parse_lenient_json(args_str)
                                except Exception:  # noqa: BLE001
                                    args_dict = {}
                        tool_calls_to_execute.append(
                            {
                                "call_id": tc_info["call_id"],
                                "name": tc_info["name"],
                                "args": args_dict,
                                "args_str": args_str,
                            }
                        )

                    assistant_msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": released_text or None,
                        "tool_calls": [
                            {
                                "id": tc["call_id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["args_str"],
                                },
                            }
                            for tc in tool_calls_to_execute
                        ],
                    }
                    self.messages.append(assistant_msg)

                    for tc in tool_calls_to_execute:
                        call_id = tc["call_id"]
                        name = tc["name"]
                        args = tc["args"]

                        yield ToolCallStarted(call_id=call_id, tool_name=name, arguments=args)

                        cap = self.config.max_calls_per_tool_per_turn.get(name)
                        if cap is not None and per_turn_counts.get(name, 0) >= cap:
                            res = ToolResult(
                                f"Tool '{name}' was already called in this turn. Respond to the user directly using the previous result.",
                                is_error=True,
                            )
                        elif (
                            self.tools is not None
                            and hasattr(self.tools, "is_side_effecting")
                            and self.tools.is_side_effecting(name)
                        ):
                            per_turn_counts[name] = per_turn_counts.get(name, 0) + 1
                            pending = asyncio.ensure_future(self._execute_tool(name, args))
                            try:
                                res = await asyncio.shield(pending)
                            except asyncio.CancelledError:
                                res = await pending
                                self.messages[:] = [
                                    m
                                    for m in self.messages
                                    if not (
                                        m.get("role") == "tool"
                                        and m.get("tool_call_id") == call_id
                                        and m.get("content") == CANCELLED_TOOL_CONTENT
                                    )
                                ]
                                self.messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": call_id,
                                        "name": name,
                                        "content": tool_message_content(
                                            res.observation, getattr(res, "attachments", ())
                                        ),
                                    }
                                )
                                raise
                        else:
                            per_turn_counts[name] = per_turn_counts.get(name, 0) + 1
                            res = await self._execute_tool(name, args)

                        yield ToolCallFinished(
                            call_id=call_id, observation=res.observation, is_error=res.is_error
                        )

                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": name,
                                "content": tool_message_content(
                                    res.observation, getattr(res, "attachments", ())
                                ),
                            }
                        )

                        try:
                            sig = (name, json.dumps(args, sort_keys=True))
                        except (TypeError, ValueError):
                            sig = (name, (tc["args_str"] or "").strip())
                        repeat_history.append(sig)
                        if (
                            len(repeat_history) >= _REPEAT_LIMIT
                            and len(set(repeat_history[-_REPEAT_LIMIT:])) == 1
                        ):
                            yield self._abort_turn("repeated_tool_call")
                            return

                        if res.is_terminal:
                            yield TurnComplete(
                                final_text=res.observation,
                                exit_reason="terminal_tool",
                                is_error=res.is_error,
                            )
                            return

                    continue

                has_final = bool(
                    step_delivered_text
                    or self._delivered_text
                    or (turn_complete_event is not None and turn_complete_event.final_text)
                )
                if not has_final and self.turn_calls <= 1:
                    yield self._abort_turn("empty_response")
                    return

                content_to_record = step_delivered_text or (
                    turn_complete_event.final_text if turn_complete_event is not None else ""
                )
                if content_to_record:
                    self.messages.append({"role": "assistant", "content": content_to_record})

                if turn_complete_event is not None:
                    yield turn_complete_event
                else:
                    yield TurnComplete(final_text=self._delivered_text)
                return

        except asyncio.CancelledError:
            self.messages[:] = validate_and_repair_history(
                self.messages, delivered_text=""
            )
            raise
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Voice agent turn failed: %s", e)
            yield self._abort_turn("exception")
            return
