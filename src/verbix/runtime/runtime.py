from __future__ import annotations

import asyncio
import logging
import platform
from collections.abc import AsyncIterator, Callable, Coroutine
from pathlib import Path
from typing import Any
from unittest.mock import Mock, NonCallableMock

import yaml

logger = logging.getLogger("verbix.runtime")

from verbix.core.events import (
    AgentEvent,
)
from verbix.core.events import (
    TextDelta as CoreTextDelta,
)
from verbix.core.events import (
    ThoughtDelta as CoreThoughtDelta,
)
from verbix.core.events import (
    ToolCallDelta as CoreToolCallDelta,
)
from verbix.core.events import (
    TurnComplete as CoreTurnComplete,
)
from verbix.core.models import CanonicalMessage, CanonicalToolCall
from verbix.minisweagent.agents.default import AgentConfig, DefaultAgent
from verbix.minisweagent.environments.local import LocalEnvironment
from verbix.minisweagent.models.litellm_model import LitellmModel
from verbix.minisweagent.tools.registry import ToolRegistry
from verbix.provider.models import (
    Finish as ProviderFinish,
)
from verbix.provider.models import (
    TextDelta as ProviderTextDelta,
)
from verbix.provider.models import (
    ThoughtDelta as ProviderThoughtDelta,
)
from verbix.provider.models import (
    ToolCallDelta as ProviderToolCallDelta,
)
from verbix.provider.models import (
    ToolDefinition,
    ToolFunction,
)
from verbix.runtime.buffer import CompletionBuffer
from verbix.runtime.tools import CancelTool, SpawnTool, StatusTool
from verbix.runtime.worker import WorkerTask
from verbix.tools.base import ToolContext

DEFAULT_VERBIX_PROMPT = """You are Verbix, a voice-operated desktop assistant and calm, capable peer. You say "I," speak plainly, admit limits plainly, no flattery.
- Your answers are spoken aloud. Write for listening: lead with the answer and speak in plain, continuous conversational prose. Never use markdown syntax, asterisks, bullet points (* or -), backticks, code blocks, tables, headings, or numbered lists.
- Be concise. Give the shortest response that completely satisfies the user's request; when they ask for detail, a story, or a report, their request sets the length. Speak only the answer or status asked for; internal execution artifacts and unrequested instructions stay unspoken.
- When a turn calls tools, begin with one short spoken sentence stating what you are starting.
- When spawn accepts work, confirm starting in one brief sentence and call no further tools; background workers announce terminal results upon completion.
- Inspect active background work with status only when the user explicitly asks how work is going.
- Stop background work with cancel.
- Speak answers and messages directly to the user. By default, describe past tool operations in the past tense."""

DEFAULT_WORKER_PROMPT = """You are an autonomous background worker executing tasks for Verbix.
Use your filesystem and shell tools directly to inspect, write, test, and complete the requested task.
When complete, provide a clear, concise summary of the actions taken and final results."""


class DefaultAgentHistoryProxy:
    """Provides backwards-compatible .messages and .append_assistant() on top of DefaultAgent.messages."""

    def __init__(self, agent: DefaultAgent) -> None:
        self._agent = agent

    @property
    def messages(self) -> list[CanonicalMessage]:
        result: list[CanonicalMessage] = []
        for m in self._agent.messages:
            if isinstance(m, CanonicalMessage):
                result.append(m)
            elif isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content") or ""
                tc_list: list[CanonicalToolCall] | None = None
                raw_tcs = m.get("tool_calls")
                if raw_tcs:
                    tc_list = []
                    for tc in raw_tcs:
                        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                        tc_list.append(
                            CanonicalToolCall(
                                id=tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", ""),
                                name=fn.get("name", tc.get("name", "")) if isinstance(tc, dict) else getattr(tc, "name", ""),
                                arguments=fn.get("arguments", tc.get("arguments", "")) if isinstance(tc, dict) else getattr(tc, "arguments", ""),
                            )
                        )
                result.append(
                    CanonicalMessage(
                        role=role if role in ("system", "user", "assistant", "tool") else "user",
                        content=content,
                        tool_calls=tc_list,
                        tool_call_id=m.get("tool_call_id"),
                        name=m.get("name"),
                    )
                )
        return result

    def append_assistant(
        self,
        content: str,
        thought: str | None = None,
        tool_calls: list[Any] | None = None,
    ) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if thought:
            msg["thought"] = thought
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id if hasattr(tc, "id") else tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.name if hasattr(tc, "name") else tc.get("name", ""),
                        "arguments": tc.arguments if hasattr(tc, "arguments") else tc.get("arguments", ""),
                    },
                }
                for tc in tool_calls
            ]
        self._agent.messages.append(msg)


class ProviderStreamAdapter:
    """Adapts a legacy ProviderClient or stream-based provider to MiniSWEAgent's query_stream."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        if hasattr(provider, "config"):
            self.config = provider.config

    def get_template_vars(self) -> dict[str, Any]:
        if hasattr(self.provider, "get_template_vars") and callable(self.provider.get_template_vars):
            res = self.provider.get_template_vars()
            if isinstance(res, dict):
                return res
        return {}

    def format_message(self, role: str, content: str, **kwargs: Any) -> dict[str, Any]:
        if hasattr(self.provider, "format_message") and callable(self.provider.format_message):
            res = self.provider.format_message(role=role, content=content, **kwargs)
            if isinstance(res, dict):
                return res
        return {"role": role, "content": content, **kwargs}

    async def query_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentEvent]:
        converted_messages: list[CanonicalMessage] = []
        for m in messages:
            if isinstance(m, CanonicalMessage):
                converted_messages.append(m)
            elif isinstance(m, dict):
                role = m.get("role", "user")
                if role not in ("system", "user", "assistant", "tool"):
                    role = "user"
                tc_list: list[CanonicalToolCall] | None = None
                raw_tcs = m.get("tool_calls")
                if raw_tcs:
                    tc_list = []
                    for tc in raw_tcs:
                        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                        tc_list.append(
                            CanonicalToolCall(
                                id=tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", ""),
                                name=fn.get("name", tc.get("name", "")) if isinstance(tc, dict) else getattr(tc, "name", ""),
                                arguments=fn.get("arguments", tc.get("arguments", "")) if isinstance(tc, dict) else getattr(tc, "arguments", ""),
                            )
                        )
                converted_messages.append(
                    CanonicalMessage(
                        role=role,
                        content=m.get("content") or "",
                        tool_calls=tc_list,
                        tool_call_id=m.get("tool_call_id"),
                        name=m.get("name"),
                    )
                )
            else:
                converted_messages.append(m)

        converted_tools: list[ToolDefinition] | None = None
        if tools:
            converted_tools = []
            for t in tools:
                if isinstance(t, ToolDefinition):
                    converted_tools.append(t)
                elif isinstance(t, dict):
                    fn = t.get("function", {})
                    name = fn.get("name", t.get("name", ""))
                    desc = fn.get("description", t.get("description", ""))
                    params = fn.get("parameters", t.get("parameters", {}))
                    converted_tools.append(
                        ToolDefinition(
                            type=t.get("type", "function"),
                            function=ToolFunction(
                                name=name,
                                description=desc,
                                parameters=params,
                            ),
                        )
                    )

        accumulated_text: list[str] = []
        finish_reason: str | None = None
        has_tool_calls = False

        try:
            stream_gen = self.provider.stream(
                converted_messages,
                tools=converted_tools,
                **kwargs,
            )
        except TypeError:
            stream_gen = self.provider.stream(converted_messages)

        async for event in stream_gen:
            if isinstance(event, ProviderTextDelta):
                accumulated_text.append(event.text)
                yield CoreTextDelta(text=event.text)
            elif isinstance(event, CoreTextDelta):
                accumulated_text.append(event.text)
                yield event
            elif isinstance(event, ProviderThoughtDelta):
                yield CoreThoughtDelta(thought=event.thought)
            elif isinstance(event, CoreThoughtDelta):
                yield event
            elif isinstance(event, ProviderToolCallDelta):
                has_tool_calls = True
                yield CoreToolCallDelta(
                    index=event.index,
                    call_id=event.call_id,
                    name=event.name,
                    arguments_delta=event.arguments_delta,
                )
            elif isinstance(event, CoreToolCallDelta):
                has_tool_calls = True
                yield event
            elif isinstance(event, ProviderFinish):
                finish_reason = event.reason
            elif isinstance(event, CoreTurnComplete):
                yield event
                return
            elif isinstance(event, AgentEvent):
                yield event

        full_text = "".join(accumulated_text)
        if full_text and finish_reason != "tool_calls" and not has_tool_calls:
            yield CoreTurnComplete(final_text=full_text)


def adapt_provider(provider: Any) -> Any:
    if isinstance(provider, (LitellmModel, ProviderStreamAdapter)):
        return provider
    if "query_stream" in getattr(provider, "__dict__", {}):
        return provider
    if hasattr(type(provider), "query_stream") and not isinstance(provider, (Mock, NonCallableMock)):
        return provider
    if hasattr(provider, "stream"):
        return ProviderStreamAdapter(provider)
    return provider


class Runtime:
    """Core Verbix runtime orchestrating the foreground agent and independent background workers."""

    def __init__(
        self,
        workspace: Path,
        provider: Any,
        system_prompt: str | None = None,
        worker_system_prompt: str | None = None,
        buffer_window: float = 0.3,
        on_wake: Callable[[str], Coroutine[Any, Any, None] | None] | None = None,
        worker_model: Any = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.provider = provider
        self.worker_model = worker_model

        # Load voice.yaml configuration
        config_path = (
            Path(__file__).resolve().parent.parent
            / "minisweagent"
            / "config"
            / "voice.yaml"
        )
        raw_voice_cfg: dict[str, Any] = {}
        if config_path.is_file():
            raw_voice_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        agent_cfg_dict = raw_voice_cfg.get("agent", {})

        base_prompt = (
            system_prompt
            or agent_cfg_dict.get("system_template")
            or DEFAULT_VERBIX_PROMPT
        )
        grounding = (
            f"\n- Host operating system: {platform.system()}. Shell: bash."
            f"\n- The working directory is {self.workspace}."
        )
        self.system_prompt = (
            base_prompt
            if "Host operating system:" in base_prompt
            else base_prompt + grounding
        )
        self.worker_system_prompt = worker_system_prompt or DEFAULT_WORKER_PROMPT
        self.on_wake = on_wake
        self.wake_output: Any = None
        self.user_active: bool = False

        self.tasks: dict[str, WorkerTask] = {}
        self._task_counter: int = 0
        self._turn_lock = asyncio.Lock()
        self._active_turn_task: asyncio.Task[Any] | None = None
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

        # Completion buffer for batching finished worker results
        self.buffer = CompletionBuffer(
            window_seconds=buffer_window,
            on_flush=self._on_buffer_flush,
        )

        # Foreground agent with strictly spawn, status, cancel tools
        self.foreground_tools = [
            SpawnTool(self),
            StatusTool(self),
            CancelTool(self),
        ]
        registry = ToolRegistry()
        for tool in self.foreground_tools:
            definition = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema,
                },
            }

            def _make_handler(t: Any) -> Callable[..., Any]:
                async def _handler(args: dict[str, Any], **kwargs: Any) -> str:
                    ctx = ToolContext(workspace=self.workspace)
                    res = await t.execute(args, ctx)
                    return res.observation

                return _handler

            registry.register(tool.name, definition, _make_handler(tool))

        agent_cfg_dict["system_template"] = self.system_prompt
        agent_cfg_dict["system_prompt"] = self.system_prompt
        fg_config = AgentConfig(**agent_cfg_dict)

        self.foreground = DefaultAgent(
            model=adapt_provider(self.provider),
            env=None,
            tools=registry,
            config=fg_config,
        )
        self.foreground.history = DefaultAgentHistoryProxy(self.foreground)

    @property
    def is_idle(self) -> bool:
        """Returns True if no foreground turn is actively running."""
        return not self._turn_lock.locked()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Subscribes an event queue to receive runtime events."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        """Unsubscribes an event queue."""
        self._subscribers.discard(q)

    def _broadcast_event(self, event_type: str, payload: dict[str, Any]) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait({"type": event_type, "payload": payload})
            except asyncio.QueueFull:
                pass

    async def step(self, user_text: str | None = None) -> AsyncIterator[AgentEvent]:
        """Executes a foreground conversation turn with Verbix."""
        async with self._turn_lock:
            self.foreground.model = adapt_provider(self.provider)
            async for event in self.foreground.step_stream(user_text):
                yield event

    async def submit(self, user_text: str, output: Any = None) -> str:
        """Runs a complete foreground turn, streaming events to output and returning final text."""
        final_text = ""
        self._active_turn_task = asyncio.current_task()
        try:
            async for event in self.step(user_text):
                if output and hasattr(output, "emit"):
                    res = output.emit(event)
                    if asyncio.iscoroutine(res):
                        await res
                if isinstance(event, CoreTurnComplete) and event.final_text:
                    final_text = event.final_text
        finally:
            self._active_turn_task = None
        return final_text

    async def cancel_turn(self) -> None:
        """Cancels an active foreground turn (e.g. on barge-in)."""
        if self._active_turn_task and not self._active_turn_task.done():
            self._active_turn_task.cancel()

    async def spawn(self, task: str) -> str:
        """Spawns an independent background worker with full workspace tools."""
        self._task_counter += 1
        task_id = f"task-{self._task_counter}"

        swe_worker_path = (
            Path(__file__).resolve().parent.parent
            / "minisweagent"
            / "config"
            / "swe_worker.yaml"
        )
        raw_worker_cfg: dict[str, Any] = {}
        if swe_worker_path.is_file():
            raw_worker_cfg = (
                yaml.safe_load(swe_worker_path.read_text(encoding="utf-8")) or {}
            )

        agent_cfg_dict = raw_worker_cfg.get("agent", {})
        if self.worker_system_prompt != DEFAULT_WORKER_PROMPT:
            agent_cfg_dict["system_template"] = self.worker_system_prompt
            agent_cfg_dict["system_prompt"] = self.worker_system_prompt

        worker_config = AgentConfig(**agent_cfg_dict)

        env_cfg_dict = raw_worker_cfg.get("environment", {})
        env_extra = env_cfg_dict.get("env", {})
        env = LocalEnvironment(cwd=str(self.workspace), env=env_extra)

        model_to_use = adapt_provider(self.worker_model or self.provider)
        worker_agent = DefaultAgent(
            model=model_to_use,
            env=env,
            tools=None,
            config=worker_config,
        )
        worker_agent.history = DefaultAgentHistoryProxy(worker_agent)

        worker_task = WorkerTask(
            id=task_id,
            description=task,
            session=worker_agent,
        )

        def _on_finish(t: WorkerTask) -> None:
            self.buffer.push(t.id, t.result or "Task completed.")

        worker_task.on_completed(_on_finish)
        self.tasks[task_id] = worker_task
        worker_task.start()
        return task_id

    def list_active_tasks(self) -> list[WorkerTask]:
        return [t for t in self.tasks.values() if t.status == "running"]

    def list_all_tasks(self) -> list[WorkerTask]:
        return list(self.tasks.values())

    def status(self, task_id: str | None = None, full: bool = False) -> str:
        """Inspects background task status from in-memory histories."""
        if task_id:
            task = self.tasks.get(task_id)
            if not task:
                return f"No task found with ID '{task_id}'."
            return task.get_status(full=full)

        active = self.list_active_tasks()
        recent_done = [t for t in self.tasks.values() if t.status != "running"][-3:]

        if not active and not recent_done:
            return "No active background tasks."

        lines = []
        if active:
            lines.append("Active tasks:")
            for t in active:
                lines.append(f"- {t.id}: '{t.description}' (running)")
        if recent_done:
            lines.append("Recently completed:")
            for t in recent_done:
                res_suffix = f": {t.result}" if t.result else ""
                lines.append(f"- {t.id}: '{t.description}' ({t.status}{res_suffix})")

        return "\n".join(lines)

    async def cancel(self, task_id: str | None = None) -> str:
        """Cancels running tasks immediately."""
        active = self.list_active_tasks()

        if task_id == "all":
            if not active:
                return "No active tasks to cancel."
            for t in active:
                await t.cancel()
            return f"Cancelled all {len(active)} active tasks."

        if task_id:
            task = self.tasks.get(task_id)
            if not task:
                return f"No task found with ID '{task_id}'."
            if task.status != "running":
                return f"Task '{task_id}' is not running (status: {task.status})."
            await task.cancel()
            return f"Cancelled {task.id} ('{task.description}')."

        # task_id omitted
        if not active:
            return "No active tasks to cancel."
        if len(active) == 1:
            t = active[0]
            await t.cancel()
            return f"Cancelled {t.id} ('{t.description}')."

        task_lines = "\n".join(f"- {t.id}: '{t.description}'" for t in active)
        return (
            "Multiple tasks are running. Specify which task ID to cancel, or pass 'all':\n"
            f"{task_lines}"
        )

    async def _on_buffer_flush(self, batch: list[tuple[str, str]]) -> None:
        """Synthesizes background worker results via foreground companion and delivers to output."""
        items = [f"- {task_id}: {result.strip()}" for task_id, result in batch if result.strip()]
        if not items:
            return
        wake_prompt = (
            "[Background task completion update. Announce the completed results directly to the user in spoken conversational prose:]\n"
            + "\n".join(items)
        )
        async with self._turn_lock:
            final_text = ""
            try:
                self.foreground.model = adapt_provider(self.provider)
                async for event in self.foreground.step_stream(wake_prompt):
                    if self.wake_output and hasattr(self.wake_output, "emit"):
                        res = self.wake_output.emit(event)
                        if asyncio.iscoroutine(res):
                            await res
                    if isinstance(event, CoreTurnComplete) and event.final_text:
                        final_text = event.final_text
            except Exception:
                logger.exception("Failed to synthesize background task completion announcement")
                return

            if self.on_wake and final_text:
                wake_res = self.on_wake(final_text)
                if asyncio.iscoroutine(wake_res):
                    await wake_res
