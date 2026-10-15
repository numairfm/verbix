from __future__ import annotations

import asyncio
import base64
import inspect
import json
import platform
import time
from typing import Any

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel, Field, model_validator

from verbix.agent.exceptions import FormatError
from verbix.agent.tools.registry import ToolRegistry
from verbix.agent.utils.compaction import compact_messages
from verbix.core.events import TextDelta, ToolCallFinished, ToolCallStarted
from verbix.tools.base import FileAttachment, ToolResult, parse_lenient_json


def tool_message_content(
    observation: str, attachments: tuple[FileAttachment, ...] = ()
) -> str | list[dict[str, Any]]:
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


class WorkerConfig(BaseModel):
    """Configuration for autonomous WorkerAgent."""

    system_template: str = ""
    instance_template: str = ""
    tools: list[str] = Field(default_factory=list)
    step_limit: int = 0
    cost_limit: float = 3.0
    wall_time_limit_seconds: int = 0
    max_consecutive_format_errors: int = 5
    tool_output: dict[str, Any] | None = None
    max_context_tokens: int | None = 16000

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "tools_list" in data and "tools" not in data:
                data["tools"] = data["tools_list"]
            if "system_prompt" in data and not data.get("system_template"):
                data["system_template"] = data["system_prompt"]
        return data


class WorkerAgent:
    """Autonomous single-loop worker agent channeling mini-swe-agent."""

    def __init__(
        self,
        model: Any,
        tools: ToolRegistry | None = None,
        config: WorkerConfig | BaseModel | dict[str, Any] | None = None,
        env: Any = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.tools = tools
        self.env = env

        if config is None:
            self.config = WorkerConfig(**kwargs)
        elif isinstance(config, dict):
            self.config = WorkerConfig(**config)
        elif isinstance(config, WorkerConfig):
            self.config = config
        elif isinstance(config, BaseModel):
            # Compatibility with AgentConfig from default.py
            self.config = WorkerConfig(**config.model_dump())
        else:
            self.config = WorkerConfig(**kwargs)

        self.messages: list[dict[str, Any]] = []
        self.cost: float = 0.0
        self.n_calls: int = 0
        self.n_consecutive_format_errors: int = 0
        self._start_time: float = time.time()
        self.extra_template_vars: dict[str, Any] = {}
        self.on_event: Any = None
        self._last_tool_call: tuple[str, str] = ("", "")
        self._consecutive_tool_repeats: int = 0
        self._tools_executed: int = 0

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        env_vars = (
            self.env.get_template_vars()
            if self.env is not None and hasattr(self.env, "get_template_vars")
            else {}
        )
        platform_vars = platform.uname()._asdict()
        return {
            **platform_vars,
            **self.config.model_dump(),
            **env_vars,
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

    def get_tool_definitions(self) -> list[dict[str, Any]] | None:
        if self.tools is None:
            return None
        if self.config.tools:
            return self.tools.get_definitions(self.config.tools)
        return self.tools.get_definitions()

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        self._tools_executed += 1
        if self.tools is None:
            return ToolResult(f"Error: Tool '{name}' is not registered or supported.", is_error=True)
        try:
            res = await self.tools.aexecute(name, args)
            if isinstance(res, ToolResult):
                return res
            return ToolResult(str(res), is_error=False, is_terminal=False)
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"Error executing tool '{name}': {e}", is_error=True)

    async def step(self) -> ToolResult:
        """Executes a single autonomous step: queries model, enforces tool calling, and executes tool."""
        # 1. Enforce limits
        if 0 < self.config.step_limit <= self.n_calls:
            return ToolResult("LimitsExceeded: step limit reached", is_error=True, is_terminal=True)
        if 0 < self.config.cost_limit <= self.cost:
            return ToolResult("LimitsExceeded: cost limit reached", is_error=True, is_terminal=True)
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            return ToolResult("TimeExceeded: wall time limit reached", is_error=True, is_terminal=True)
        if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
            return ToolResult("RepeatedFormatError: maximum format errors reached", is_error=True, is_terminal=True)

        self.n_calls += 1

        # 2. Query model non-streaming
        if self.config.max_context_tokens:
            self.messages[:] = compact_messages(self.messages, max_tokens=self.config.max_context_tokens)

        tool_defs = self.get_tool_definitions()
        query_kwargs: dict[str, Any] = {}
        if tool_defs is not None:
            query_kwargs["tools"] = tool_defs

        try:
            query_fn = self.model.query
            if inspect.iscoroutinefunction(query_fn):
                response = await query_fn(self.messages, **query_kwargs)
            else:
                response = await asyncio.to_thread(query_fn, self.messages, **query_kwargs)
                if inspect.iscoroutine(response):
                    response = await response
        except FormatError as e:
            if e.messages:
                for m in e.messages:
                    if isinstance(m, dict):
                        self.cost += m.get("extra", {}).get("cost", 0.0)
            if self._tools_executed > 0 and e.messages:
                first_msg = e.messages[0]
                if isinstance(first_msg, dict) and first_msg.get("role") == "assistant":
                    raw_content = first_msg.get("content")
                    content = raw_content.strip() if isinstance(raw_content, str) else ""
                    if content:
                        self.messages.append(first_msg)
                        self.n_consecutive_format_errors = 0
                        return ToolResult(content, is_error=False, is_terminal=True)
            if 0 < self.config.cost_limit <= self.cost:
                return ToolResult("LimitsExceeded: cost limit reached", is_error=True, is_terminal=True)
            self.n_consecutive_format_errors += 1
            if self._tools_executed == 0:
                err_content = "No tool calls found. You have not executed any tools yet. Execute a tool to perform work."
            else:
                err_content = "No tool calls or response found. Execute a tool if you need to take an action, or output your direct factual conclusion in text if you are done."
            if e.messages:
                self.messages.extend(e.messages)
                err_content = e.messages[-1].get("content", err_content)
            else:
                self.messages.append({"role": "user", "content": err_content})
            if self.on_event is not None:
                try:
                    self.on_event(TextDelta(text=f"! format error: {err_content}"))
                except Exception:  # noqa: BLE001, S110
                    pass
            if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                return ToolResult("RepeatedFormatError: maximum format errors reached", is_error=True, is_terminal=True)
            return ToolResult(err_content, is_error=True, is_terminal=False)
        except Exception as e:
            if self.on_event is not None:
                try:
                    self.on_event(TextDelta(text=f"! query error: {e}"))
                except Exception:  # noqa: BLE001, S110
                    pass
            raise

        # 3. Process cost & response
        cost = response.get("extra", {}).get("cost", 0.0) if isinstance(response, dict) else 0.0
        self.cost += cost

        assistant_msg = response if isinstance(response, dict) else {"role": "assistant", "content": str(response)}
        self.messages.append(assistant_msg)

        tool_calls = assistant_msg.get("tool_calls") or []

        # 4. Strict tool-calling guard
        if not tool_calls:
            raw_content = assistant_msg.get("content")
            content = raw_content.strip() if isinstance(raw_content, str) else ""
            if self._tools_executed > 0 and content:
                self.n_consecutive_format_errors = 0
                return ToolResult(content, is_error=False, is_terminal=True)

            self.n_consecutive_format_errors += 1
            if self._tools_executed == 0:
                err_msg = "No tool calls found. You have not executed any tools yet. Execute a tool to perform work."
            else:
                err_msg = "No tool calls or response found. Execute a tool if you need to take an action, or output your direct factual conclusion in text if you are done."
            self.messages.append({"role": "user", "content": err_msg})
            if self.on_event is not None:
                try:
                    self.on_event(TextDelta(text=f"! format error: {err_msg}"))
                except Exception:  # noqa: BLE001, S110
                    pass
            if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                return ToolResult("RepeatedFormatError: maximum format errors reached", is_error=True, is_terminal=True)
            return ToolResult(err_msg, is_error=True, is_terminal=False)

        self.n_consecutive_format_errors = 0

        # 5. Dispatch tool calls
        last_result = ToolResult("", is_error=False)
        for i, tc in enumerate(tool_calls):
            call_id = (tc.get("id") if isinstance(tc, dict) else None) or f"call_{i}"
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = fn.get("name") or (tc.get("name") if isinstance(tc, dict) else "") or ""
            raw_args = (
                fn.get("arguments")
                if "arguments" in fn
                else (tc.get("arguments", {}) if isinstance(tc, dict) else {})
            )

            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    args = parse_lenient_json(raw_args)
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}

            if self.on_event is not None:
                try:
                    self.on_event(ToolCallStarted(call_id=call_id, tool_name=name, arguments=args))
                except Exception:  # noqa: BLE001, S110
                    pass

            try:
                call_sig = (name, json.dumps(args, sort_keys=True))
            except (TypeError, ValueError):
                call_sig = (name, str(args))

            if call_sig == self._last_tool_call:
                self._consecutive_tool_repeats += 1
                if self._consecutive_tool_repeats >= 3:
                    result = ToolResult(
                        f"Error: Tool '{name}' was called 3 times consecutively with identical arguments without progress. "
                        "You must try a different approach or output your direct conclusion.",
                        is_error=True,
                    )
                else:
                    result = await self._execute_tool(name, args)
            else:
                self._last_tool_call = call_sig
                self._consecutive_tool_repeats = 1
                result = await self._execute_tool(name, args)

            last_result = result

            if self.on_event is not None:
                try:
                    self.on_event(
                        ToolCallFinished(
                            call_id=call_id,
                            observation=result.observation,
                            is_error=result.is_error,
                        )
                    )
                except Exception:  # noqa: BLE001, S110
                    pass

            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": tool_message_content(result.observation, getattr(result, "attachments", ())),
                }
            )

            if result.is_terminal:
                return result

        return last_result

    async def run(self, task: str = "", **kwargs: Any) -> dict[str, Any]:
        """Runs autonomous loop until completion, limits exceeded, or failure."""
        self.extra_template_vars = {**self.extra_template_vars, "task": task, **kwargs}
        self.messages = []
        self.cost = 0.0
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        self._last_tool_call = ("", "")
        self._consecutive_tool_repeats = 0
        self._tools_executed = 0
        self._start_time = time.time()

        if self.config.system_template:
            sys_content = self._render_template(self.config.system_template)
            if sys_content:
                self.messages.append({"role": "system", "content": sys_content})

        user_content = task
        if self.config.instance_template:
            rendered = self._render_template(self.config.instance_template, task=task)
            if rendered:
                user_content = rendered
        if user_content:
            self.messages.append({"role": "user", "content": user_content})

        terminal_result: ToolResult | None = None
        while True:
            res = await self.step()
            if res.is_terminal:
                terminal_result = res
                break

        obs = terminal_result.observation if terminal_result else ""
        prefix = "Task completed: "
        summary = obs.removeprefix(prefix)

        status = "failed" if (terminal_result and terminal_result.is_error) else "completed"
        if status == "failed" and self.on_event is not None:
            try:
                self.on_event(TextDelta(text=f"! task failed: {summary}"))
            except Exception:  # noqa: BLE001, S110
                pass
        return {
            "status": status,
            "summary": summary,
            "cost": self.cost,
            "n_calls": self.n_calls,
        }
