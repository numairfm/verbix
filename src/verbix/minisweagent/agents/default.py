"""Default agent class ported from Mini-SWE-Agent with async event streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import time
import traceback
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel, model_validator

from verbix.core.events import (
    AgentEvent,
    TextDelta,
    ThoughtDelta,
    ToolCallDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnComplete,
)
from verbix.minisweagent.exceptions import (
    FormatError,
    InterruptAgentFlow,
    LimitsExceeded,
    Submitted,
    TimeExceeded,
)
from verbix.minisweagent.models.utils.actions_toolcall import BASH_TOOL
from verbix.minisweagent.tools.registry import ToolRegistry
from verbix.minisweagent.utils.history import validate_and_repair_history
from verbix.minisweagent.utils.serialize import recursive_merge


class AgentConfig(BaseModel):
    """Configuration for DefaultAgent."""

    system_template: str = ""
    """Template for the system message."""
    instance_template: str = ""
    """Template for the user message specifying the task."""
    tools: list[str] = []
    """List of allowed/configured tool names."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding this cost."""
    wall_time_limit_seconds: int = 0
    """Stop agent after this many seconds of wall-clock time (0 = no limit)."""
    max_consecutive_format_errors: int = 3
    """Exit after this many format errors in a row (0 = no limit)."""
    output_path: Path | None = None
    """Save the trajectory to this path."""
    system_prompt: str = ""
    """Alternative alias for system_template."""

    @model_validator(mode="before")
    @classmethod
    def _normalize_config_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "tools_list" in data and "tools" not in data:
                data["tools"] = data["tools_list"]
            if "system_prompt" in data and not data.get("system_template"):
                data["system_template"] = data["system_prompt"]
            elif "system_template" in data and not data.get("system_prompt"):
                data["system_prompt"] = data["system_template"]
        return data


class DefaultAgent:
    """Unified agent supporting both synchronous execution and async streaming."""

    def __init__(
        self,
        model: Any,
        env: Any = None,
        tools: Any = None,
        config: AgentConfig | None = None,
        *,
        config_class: type[AgentConfig] = AgentConfig,
        **kwargs: Any,
    ) -> None:
        registry: ToolRegistry | None = None
        if isinstance(tools, ToolRegistry) or (tools is not None and not isinstance(tools, (list, tuple))):
            registry = tools
        elif isinstance(tools, (list, tuple)) and "tools" not in kwargs:
            kwargs["tools"] = list(tools)

        if config is not None:
            self.config = config
        elif "config" in kwargs and isinstance(kwargs["config"], AgentConfig):
            self.config = kwargs.pop("config")
        else:
            self.config = config_class(**kwargs)


        self.messages: list[dict[str, Any]] = []
        self.history: Any = None
        self.model = model
        self.env = env
        self.tools = registry
        self.extra_template_vars: dict[str, Any] = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        self._start_time = time.time()
        self._delivered_text: str = ""

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
        return recursive_merge(
            platform_vars,
            self.config.model_dump(),
            env_vars,
            model_vars,
            {
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "elapsed_seconds": int(time.time() - self._start_time),
            },
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str, **kwargs: Any) -> str:
        if not template:
            return ""
        try:
            return Template(template, undefined=StrictUndefined).render(**self.get_template_vars(**kwargs))
        except Exception:  # noqa: BLE001
            return Template(template).render(**self.get_template_vars(**kwargs))

    def add_messages(self, *messages: dict[str, Any]) -> list[dict[str, Any]]:
        self.logger.debug(messages)
        self.messages.extend(messages)
        return list(messages)

    def get_tool_definitions(self) -> list[dict[str, Any]] | None:
        """Return tool schemas configured for this agent, including registry and bash."""
        definitions: list[dict[str, Any]] = []
        registered_names: set[str] = set()

        if self.tools is not None:
            defs = (
                self.tools.get_definitions(self.config.tools)
                if self.config.tools
                else self.tools.get_definitions()
            )
            for d in defs:
                definitions.append(d)
                fn_name = d.get("function", {}).get("name") or d.get("name")
                if fn_name:
                    registered_names.add(fn_name)

        if self.config.tools:
            for tool_name in self.config.tools:
                if tool_name not in registered_names and tool_name == "bash":
                    definitions.append(BASH_TOOL)
                    registered_names.add("bash")
        elif not definitions and self.env is not None:
            definitions.append(BASH_TOOL)

        return definitions if definitions else None

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Execute a tool via ToolRegistry or environment, returning (observation, is_error)."""
        # 1. Check ToolRegistry first if available
        if self.tools is not None and name in self.tools._handlers:
            try:
                res = await self.tools.aexecute(name, args)
                obs = res if isinstance(res, str) else json.dumps(res)
                return obs, False
            except Submitted:
                raise
            except Exception as e:  # noqa: BLE001
                return f"Error executing tool '{name}': {e}", True

        # 2. Check environment for bash commands
        if name == "bash":
            if self.env is None:
                return "Error: Cannot execute bash command because no environment is configured.", True

            action = args if isinstance(args, dict) else {"command": str(args)}
            try:
                res = await asyncio.to_thread(self.env.execute, action)
                obs = res.get("output", "") if isinstance(res, dict) else str(res)
                returncode = res.get("returncode", 0) if isinstance(res, dict) else 0
                is_err = returncode != 0
                if is_err and not obs and isinstance(res, dict) and res.get("exception_info"):
                    obs = res["exception_info"]
                if (
                    is_err
                    and f"returncode: {returncode}" not in obs.lower()
                    and f"exit code: {returncode}" not in obs.lower()
                ):
                    obs = f"[Returncode: {returncode}]\n{obs}".strip()
                return obs, is_err
            except Submitted:
                raise
            except Exception as e:  # noqa: BLE001
                return f"Error executing bash action: {e}", True

        return f"Error: Tool '{name}' is not registered or supported.", True

    def _format_message(self, role: str, content: str, **kwargs: Any) -> dict[str, Any]:
        if hasattr(self.model, "format_message"):
            try:
                res = self.model.format_message(role=role, content=content, **kwargs)
                if isinstance(res, dict) and "role" in res:
                    return res
            except Exception:  # noqa: BLE001, S110
                pass
        return {"role": role, "content": content, **kwargs}

    async def step_stream(self, user_input: str | None = None) -> AsyncIterator[AgentEvent]:
        """Stream an agent turn asynchronously, yielding AgentEvents and executing tools."""
        is_initial_turn = not any(m.get("role") == "user" for m in self.messages)
        # Initialize conversation messages if starting fresh
        if not self.messages and self.config.system_template:
            sys_content = self._render_template(self.config.system_template)
            self.add_messages(self._format_message(role="system", content=sys_content))

        if user_input is not None:
            if is_initial_turn and self.config.instance_template:
                user_content = self._render_template(self.config.instance_template, task=user_input)
            else:
                user_content = user_input
            self.add_messages(self._format_message(role="user", content=user_content))

        try:
            while True:
                # Check limits
                if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
                    exc = LimitsExceeded(
                        {
                            "role": "exit",
                            "content": "LimitsExceeded",
                            "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                        }
                    )
                    self.add_messages(*exc.messages)
                    yield TurnComplete(final_text="LimitsExceeded")
                    return

                if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
                    time_exc = TimeExceeded(
                        {
                            "role": "exit",
                            "content": "TimeExceeded",
                            "extra": {"exit_status": "TimeExceeded", "submission": ""},
                        }
                    )
                    self.add_messages(*time_exc.messages)
                    yield TurnComplete(final_text="TimeExceeded")
                    return

                self.n_calls += 1
                tool_defs = self.get_tool_definitions()
                self._delivered_text = ""

                accumulated_tool_calls: dict[int, dict[str, Any]] = {}
                turn_complete_event: TurnComplete | None = None
                buffered_text_events: list[TextDelta] = []
                has_tool_calls = False

                stream = self.model.query_stream(self.messages, tools=tool_defs)
                async for event in stream:
                    if isinstance(event, TextDelta):
                        if has_tool_calls:
                            continue
                        buffered_text_events.append(event)
                    elif isinstance(event, ThoughtDelta):
                        yield event
                    elif isinstance(event, ToolCallDelta):
                        has_tool_calls = True
                        buffered_text_events.clear()
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

                # Pure speech: emit buffered text now that no tool call preempted it.
                # Tool turn: discard the buffer so pre-tool chatter is never spoken nor stored.
                if not has_tool_calls:
                    for buffered in buffered_text_events:
                        self._delivered_text += buffered.text
                        yield buffered
                    buffered_text_events.clear()

                # If tool calls were accumulated, execute them
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
                                args_dict = {"command": args_str}
                        tool_calls_to_execute.append(
                            {
                                "call_id": tc_info["call_id"],
                                "name": tc_info["name"],
                                "args": args_dict,
                                "args_str": args_str,
                            }
                        )

                    # Record assistant response message (suppressed pre-tool chatter is not stored)
                    assistant_msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": None,
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
                    self.add_messages(assistant_msg)

                    # Execute each tool call and yield events
                    for tc in tool_calls_to_execute:
                        call_id = tc["call_id"]
                        name = tc["name"]
                        args = tc["args"]

                        yield ToolCallStarted(call_id=call_id, tool_name=name, arguments=args)

                        if name == "finish":
                            summary = str(args.get("summary", "")).strip() if isinstance(args, dict) else str(args).strip()
                            obs = f"Task completed: {summary}" if summary else "Task completed."
                            if self.tools is not None and "finish" in self.tools._handlers:
                                try:
                                    res = await self.tools.aexecute("finish", args)
                                    obs = res if isinstance(res, str) else str(res)
                                except Exception:  # noqa: BLE001, S110
                                    pass
                            tool_msg: dict[str, Any] = {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": name,
                                "content": obs,
                            }
                            self.add_messages(tool_msg)
                            yield ToolCallFinished(call_id=call_id, observation=obs, is_error=False)
                            yield TurnComplete(final_text=summary)
                            return

                        try:
                            if self.tools is not None and self.tools.is_side_effecting(name):
                                pending = asyncio.ensure_future(self._execute_tool(name, args))
                                try:
                                    obs, is_err = await asyncio.shield(pending)
                                except asyncio.CancelledError:
                                    # Started side-effecting op runs to completion and is
                                    # recorded; cancellation then unwinds to repair below.
                                    obs, is_err = await pending
                                    self.add_messages(
                                        {
                                            "role": "tool",
                                            "tool_call_id": call_id,
                                            "name": name,
                                            "content": obs,
                                        }
                                    )
                                    raise
                            else:
                                obs, is_err = await self._execute_tool(name, args)
                        except Submitted as e:
                            self.add_messages(*e.messages)
                            submission = ""
                            if e.messages:
                                submission = (
                                    e.messages[-1].get("extra", {}).get("submission")
                                    or e.messages[-1].get("content", "")
                                )
                            yield ToolCallFinished(call_id=call_id, observation=submission, is_error=False)
                            yield TurnComplete(final_text=submission)
                            return

                        yield ToolCallFinished(call_id=call_id, observation=obs, is_error=is_err)

                        tool_msg: dict[str, Any] = {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": obs,
                        }
                        self.add_messages(tool_msg)

                    # Continue turn loop to let model produce response to observations
                    continue

                # No tool calls were made: finish the turn
                full_text = self._delivered_text
                assistant_msg = {
                    "role": "assistant",
                    "content": full_text,
                }
                self.add_messages(assistant_msg)

                if turn_complete_event is not None:
                    yield turn_complete_event
                else:
                    yield TurnComplete(final_text=full_text)
                return
        except asyncio.CancelledError:
            self.messages[:] = validate_and_repair_history(
                self.messages, delivered_text=self._delivered_text
            )
            raise

    def handle_uncaught_exception(self, e: Exception) -> list[dict[str, Any]]:
        msg = self._format_message(
            role="exit",
            content=str(e),
            extra={
                "exit_status": type(e).__name__,
                "submission": "",
                "exception_str": str(e),
                "traceback": traceback.format_exc(),
            },
        )
        return self.add_messages(msg)

    def run(self, task: str = "", **kwargs: Any) -> dict[str, Any]:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        if self.config.system_template:
            sys_content = self._render_template(self.config.system_template)
            self.add_messages(self._format_message(role="system", content=sys_content))
        if self.config.instance_template:
            inst_content = self._render_template(self.config.instance_template)
            self.add_messages(self._format_message(role="user", content=inst_content))


        while True:
            try:
                self.step()
                self.n_consecutive_format_errors = 0
            except FormatError as e:
                self.cost += e.messages[0].get("extra", {}).get("cost", 0.0)
                self.n_consecutive_format_errors += 1
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict[str, Any]]:
        """Query the LM, execute actions."""
        return self.execute_actions(self.query())

    def query(self) -> dict[str, Any]:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        tool_defs = self.get_tool_definitions()
        kwargs: dict[str, Any] = {}
        if tool_defs is not None:
            kwargs["tools"] = tool_defs
        message = self.model.query(self.messages, **kwargs)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def execute_actions(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute actions in message, add observation messages, return them."""
        actions = message.get("extra", {}).get("actions", [])
        outputs: list[dict[str, Any]] = []
        for action in actions:
            if self.env is not None:
                outputs.append(self.env.execute(action))
            elif self.tools is not None:
                tool_name = action.get("name") or action.get("tool_name", "bash")
                args = action.get("arguments", action)
                try:
                    res = self.tools.execute(tool_name, args)
                    outputs.append({"output": str(res), "returncode": 0, "exception_info": ""})
                except Exception as e:  # noqa: BLE001
                    outputs.append({"output": "", "returncode": -1, "exception_info": str(e)})

            else:
                outputs.append({"output": "No environment or tools configured", "returncode": -1, "exception_info": ""})
        if hasattr(self.model, "format_observation_messages"):
            return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))
        observations = []
        for action, output in zip(actions, outputs):
            observations.append(
                {
                    "role": "tool",
                    "tool_call_id": action.get("tool_call_id", ""),
                    "content": output.get("output", "") or output.get("exception_info", ""),
                }
            )
        return self.add_messages(*observations)

    def serialize(self, *extra_dicts: dict[str, Any]) -> dict[str, Any]:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": "2.4.6",
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        model_ser = self.model.serialize() if self.model and hasattr(self.model, "serialize") else {}
        env_ser = self.env.serialize() if self.env and hasattr(self.env, "serialize") else {}
        return recursive_merge(agent_data, model_ser, env_ser, *extra_dicts)

    def save(self, path: Path | None, *extra_dicts: dict[str, Any]) -> dict[str, Any]:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data."""
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
