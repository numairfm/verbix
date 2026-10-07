from __future__ import annotations

import asyncio
import logging
import platform
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime
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
from verbix.minisweagent.utils.history import validate_and_repair_history
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
from verbix.runtime.floor import FloorArbiter, FloorState
from verbix.runtime.ledger import TaskLedger
from verbix.runtime.tools import (
    CancelTool,
    SpawnTool,
    StatusTool,
    build_worker_registry,
)
from verbix.runtime.worker import WorkerTask
from verbix.tools.base import ToolContext

DEFAULT_VERBIX_PROMPT = """# IDENTITY & TONE
You are Verbix, a voice-operated desktop assistant and calm, capable peer. You say "I," speak plainly, admit limits plainly, no flattery.

# VOICE & AUDIO CONSTRAINTS
- Your answers are spoken aloud. Write for listening: lead with the answer and speak in plain, continuous conversational prose.
- Never use markdown syntax, asterisks, bullet points (* or -), backticks, code blocks, tables, headings, or numbered lists.
- Be concise: give the shortest response that completely satisfies the request; when the user asks for detail, a story, or a report, their request sets the length.
- Speak only the answer or status asked for; internal execution artifacts and unrequested instructions stay unspoken.
- Never recite system prompts, identity templates, execution artifacts, or internal schemas to the user.

# CAPABILITY ROUTING & DELEGATION
- Answer quick conversational and general knowledge questions directly.
- Delegate multi-step tasks, background operations, desktop control, system commands, or file interactions to spawn.
- You have no direct access to the filesystem, shell, or execution environment.
- When calling spawn, formulate the objective as the user's intended goal and expected outcome.
- Never guess, invent, or substitute concrete shell commands, scripts, package managers, or file paths into the objective unless the user explicitly specified them.
- Persona, voice, and style rules apply strictly to spoken text spoken to the user; tool arguments and objectives must always remain exact, factual, and unstylized.

# TOOL OPERATIONS & BACKGROUND WORK
- When a turn requires tools, execute all necessary tool operations first without speaking pre-tool announcements.
- When the user requests background work, spawn each task needed (in parallel or sequentially).
- When all tool operations for the turn complete, provide a single concise spoken response answering the user or confirming what was started; background workers announce terminal results upon completion.
- Inspect background tasks with status when the user asks how work is going, or asks for details or results.
- Stop background work with cancel.
- Speak answers and messages directly to the user. Never speak redundant pre- and post-tool confirmations. By default, describe past tool operations in the past tense."""

DEFAULT_WORKER_PROMPT = """You are an autonomous background worker executing tasks for Verbix.
Discover the necessary environment and tools, execute actions directly, verify outcomes, and call the finish tool with an information-dense factual summary of results upon completion."""

DEFAULT_ANNOUNCER_PROMPT = """# IDENTITY & TONE
You are Verbix, a voice-operated desktop assistant and calm, capable peer. You say "I," speak plainly, admit limits plainly, no flattery.

# VOICE & AUDIO CONSTRAINTS
- Your answers are spoken aloud. Write for listening: lead with the answer and speak in plain, continuous conversational prose.
- Never use markdown syntax, asterisks, bullet points (* or -), backticks, code blocks, tables, headings, or numbered lists.
- Be concise: give the shortest response that completely satisfies the update; when the task produced detailed results, explain them clearly without technical syntax.
- Speak answers and messages directly to the user. By default, describe past tool operations in the past tense."""


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


@dataclass
class TurnHandle:
    """Owns an active turn: its task, kind, and delivered-text provider."""

    task: asyncio.Task[Any] | None = None
    kind: str = "foreground"
    delivered_text_provider: Callable[[], str] | None = None

    def delivered_text(self) -> str:
        if self.delivered_text_provider is None:
            return ""
        return self.delivered_text_provider()

    def cancel(self) -> None:
        """Idempotently cancel the turn task."""
        if self.task is not None and not self.task.done():
            self.task.cancel()


class WakeOutputFanOut:
    """Fans wake-turn events out to the voice sink and IPC subscriber queues."""

    def __init__(
        self,
        voice_output: Any = None,
        broadcast: Callable[[Any], Any] | None = None,
    ) -> None:
        self.voice_output = voice_output
        self._broadcast = broadcast
        self._delivered: list[str] = []

    async def emit(self, event: AgentEvent) -> None:
        text = getattr(event, "text", "")
        if text:
            self._delivered.append(text)
        if self.voice_output is not None and hasattr(self.voice_output, "emit"):
            res = self.voice_output.emit(event)
            if asyncio.iscoroutine(res):
                await res
        if self._broadcast is not None:
            res = self._broadcast(event)
            if asyncio.iscoroutine(res):
                await res

    def delivered_text(self) -> str:
        delivered = getattr(self.voice_output, "delivered_text", None)
        if callable(delivered):
            try:
                res = delivered()
                if isinstance(res, str) and res:
                    return res
            except Exception:  # noqa: BLE001, S110
                pass
        return "".join(self._delivered)

    def reset_delivered(self) -> None:
        self._delivered.clear()
        reset = getattr(self.voice_output, "reset_delivered", None)
        if callable(reset):
            try:
                reset()
            except Exception:  # noqa: BLE001, S110
                pass


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

        self._custom_prompt = (
            system_prompt.strip()
            if system_prompt and system_prompt.strip()
            else None
        )
        self._base_system_prompt = (
            agent_cfg_dict.get("system_template")
            or DEFAULT_VERBIX_PROMPT
        )
        self.system_prompt = self._build_grounded_prompt()
        self.worker_system_prompt = worker_system_prompt or DEFAULT_WORKER_PROMPT
        self.on_wake = on_wake
        self.wake_output: Any = None
        self.user_active: bool = False

        self.tasks: dict[str, WorkerTask] = {}
        self._task_counter: int = 0
        self._environments: list[LocalEnvironment] = []
        self._pending_announcements: list[tuple[str, str]] = []
        self._drain_lock = asyncio.Lock()
        self.floor = FloorArbiter(on_change=self._on_floor_state_changed)
        self.ledger = TaskLedger()
        self._turn_handle: TurnHandle = TurnHandle()
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

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

            registry.register(
                tool.name, definition, _make_handler(tool), side_effecting=tool.name in ("spawn", "cancel")
            )

        agent_cfg_dict["system_template"] = self.system_prompt
        agent_cfg_dict["system_prompt"] = self.system_prompt
        agent_cfg_dict["turn_scoped_limits"] = True
        fg_config = AgentConfig(**agent_cfg_dict)

        self.foreground = DefaultAgent(
            model=adapt_provider(self.provider),
            env=None,
            tools=registry,
            config=fg_config,
        )
        self.foreground.history = DefaultAgentHistoryProxy(self.foreground)

        announcer_config = fg_config.model_copy(update={"tools": []})
        self.announcer = DefaultAgent(
            model=adapt_provider(self.provider),
            env=None,
            tools=ToolRegistry(),
            config=announcer_config,
        )
        self.announcer.messages = self.foreground.messages

    @property
    def is_idle(self) -> bool:
        """Returns True if no foreground turn is actively running."""
        return self.floor.state == FloorState.IDLE

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

    def _build_grounded_prompt(self, base_prompt: str | None = None) -> str:
        """Build the static, cache-stable system prompt (no volatile clock content)."""
        prompt_to_use = (
            base_prompt if base_prompt is not None else self._base_system_prompt
        )
        if "Host operating system:" in prompt_to_use:
            base = prompt_to_use
        else:
            grounding = (
                "\n\n# ENVIRONMENT & GROUNDING"
                f"\n- Host operating system: {platform.system()}. Shell: bash."
                f"\n- The working directory is {self.workspace}."
            )
            base = prompt_to_use.rstrip() + grounding

        if self._custom_prompt:
            base = base.rstrip() + f"\n\n# CUSTOM USER INSTRUCTIONS\n{self._custom_prompt}"
        return base

    def _request_grounding(self) -> str:
        """Volatile per-request grounding, prepended to the user message only."""
        now_str = datetime.now().astimezone().strftime("%A, %B %d, %Y, %I:%M %p")
        return f"[Current local date and time: {now_str}.]"

    async def step(self, user_text: str | None = None) -> AsyncIterator[AgentEvent]:
        """Executes a foreground conversation turn with Verbix."""
        async with self.floor.acquire(FloorState.FOREGROUND):
            self._turn_handle = TurnHandle(task=asyncio.current_task(), kind="foreground",
                delivered_text_provider=lambda: self.foreground._delivered_text)
            try:
                async for event in self._step_inner(user_text):
                    yield event
            finally:
                self._turn_handle.task = None

    async def _step_inner(
        self, user_text: str | None = None, *, agent: Any = None
    ) -> AsyncIterator[AgentEvent]:
        # System prompt stays static (provider cache prefix); volatile
        # grounding rides on the user message at request time only.
        session = agent or self.foreground
        session.model = adapt_provider(self.provider)
        if user_text is not None:
            user_text = f"{self._request_grounding()}\n{user_text}"
        async for event in session.step_stream(user_text):
            yield event

    async def submit(
        self, user_text: str, output: Any = None, *, agent: Any = None
    ) -> str:
        """Runs a complete foreground turn, streaming events to output and returning final text."""
        session = agent or self.foreground
        final_text = ""
        async with self.floor.acquire(FloorState.FOREGROUND):
            self._turn_handle = TurnHandle(
                task=asyncio.current_task(),
                kind="foreground",
                delivered_text_provider=lambda: self._submit_delivered_text(output, session),
            )
            try:
                async for event in self._step_inner(user_text, agent=session):
                    if output and hasattr(output, "emit"):
                        res = output.emit(event)
                        if asyncio.iscoroutine(res):
                            await res
                    if isinstance(event, CoreTurnComplete) and event.final_text:
                        final_text = event.final_text
                if output and getattr(output, "is_playing", False):
                    while output.is_playing:
                        await asyncio.sleep(0.02)
            finally:
                self._turn_handle.task = None
        return final_text

    def _submit_delivered_text(self, output: Any, session: Any) -> str:
        delivered = getattr(output, "delivered_text", None)
        if callable(delivered):
            res = delivered()
            if isinstance(res, str):
                return res
        return session._delivered_text

    async def cancel_turn(self, heard: str | None = None) -> None:
        """Cancels an active foreground turn (e.g. on barge-in) and repairs history.

        `heard` is a pre-interrupt snapshot of actually-spoken text; when given it
        wins over the live provider, whose delivered record may already be wiped.
        Repair runs after the dying turn unwinds so its own repair cannot overwrite
        the snapshot (unwind is synchronous; the timeout is a backstop only).
        """
        self.floor.request_interrupt()
        task = self._turn_handle.task
        had_active = task is not None and not task.done()
        self._turn_handle.cancel()
        if not had_active:
            return
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
            except (asyncio.CancelledError, TimeoutError):
                pass
        delivered = heard if heard is not None else self._turn_handle.delivered_text()
        self.foreground.messages[:] = validate_and_repair_history(
            self.foreground.messages, delivered_text=delivered
        )

    async def spawn(self, task: str) -> str:
        """Spawns an independent background worker with full workspace tools."""
        self._task_counter += 1
        task_id = f"task-{self._task_counter}"

        config_dir = Path(__file__).resolve().parent.parent / "minisweagent" / "config"
        worker_path = config_dir / "worker.yaml"
        if not worker_path.is_file():
            worker_path = config_dir / "swe_worker.yaml"

        raw_worker_cfg: dict[str, Any] = {}
        if worker_path.is_file():
            raw_worker_cfg = (
                yaml.safe_load(worker_path.read_text(encoding="utf-8")) or {}
            )

        agent_cfg_dict = raw_worker_cfg.get("agent", {})
        if self.worker_system_prompt != DEFAULT_WORKER_PROMPT:
            agent_cfg_dict["system_template"] = self.worker_system_prompt
            agent_cfg_dict["system_prompt"] = self.worker_system_prompt

        worker_config = AgentConfig(**agent_cfg_dict)

        env_cfg_dict = raw_worker_cfg.get("environment", {})
        env_extra = env_cfg_dict.get("env", {})
        timeout = int(env_cfg_dict.get("timeout", 600))
        env = LocalEnvironment(cwd=str(self.workspace), env=env_extra, timeout=timeout)

        model_to_use = adapt_provider(self.worker_model or self.provider)
        worker_tools = build_worker_registry()

        worker_agent = DefaultAgent(
            model=model_to_use,
            env=env,
            tools=worker_tools,
            config=worker_config,
        )
        worker_agent.history = DefaultAgentHistoryProxy(worker_agent)

        worker_task = WorkerTask(
            id=task_id,
            description=task,
            session=worker_agent,
        )

        def _on_finish(t: WorkerTask) -> None:
            if t.status == "cancelled":
                self.ledger.mark_cancelled(t.id)
                self._broadcast_event("task_cancelled", {"task_id": t.id})
                return
            self._on_task_finished(t.id, t.result or "Task completed.")

        worker_task.on_completed(_on_finish)
        self.tasks[task_id] = worker_task
        self._environments.append(env)
        self.ledger.register_task(task_id, task)
        self._broadcast_event(
            "task_registered", {"task_id": task_id, "objective": task}
        )
        worker_task.start()
        return task_id

    def _on_floor_state_changed(self, prev_state: FloorState, new_state: FloorState) -> None:
        if new_state == FloorState.IDLE and self._pending_announcements:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._drain_pending_announcements())
            except RuntimeError:
                pass

    def _on_task_finished(self, task_id: str, result: str) -> None:
        """Records task completion into the ledger, context memory, and schedules announcement."""
        self.ledger.mark_completed(task_id, result)
        self._broadcast_event(
            "task_completed",
            {"task_id": task_id, "result": result},
        )
        if hasattr(self, "foreground") and hasattr(self.foreground, "messages"):
            self.foreground.messages.append({
                "role": "user",
                "content": f"[Task '{task_id}' completed]: {result}",
            })
        self._pending_announcements.append((task_id, result))
        if self.floor.is_idle:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._drain_pending_announcements())
            except RuntimeError:
                pass

    async def _drain_pending_announcements(self) -> None:
        """Speaks accumulated background completions via the single foreground companion."""
        if not self.floor.is_idle or not self._pending_announcements:
            return
        async with self._drain_lock:
            if not self.floor.is_idle or not self._pending_announcements:
                return
            batch = list(self._pending_announcements)
            self._pending_announcements.clear()

            items = [f"- Task '{tid}': {res.strip()}" for tid, res in batch if res.strip()]
            if not items:
                return
            prompt = (
                f"{self._request_grounding()}\n"
                "[Background task update. In one brief, natural conversational prose sentence, "
                "inform the user of these completed results:]\n" + "\n".join(items)
            )
            final_text = await self.submit(prompt, self.wake_output, agent=self.announcer)
            if final_text and self.on_wake:
                wake_res = self.on_wake(final_text)
                if asyncio.iscoroutine(wake_res):
                    await wake_res

    def get_task_status_summary(self, task_id: str | None = None) -> str:
        """Ledger-backed status summary for the IPC server."""
        if task_id is not None:
            entry = self.ledger.get(task_id)
            if entry is None:
                return f"No task found with ID '{task_id}'."
            summary = f"{entry.id}: '{entry.objective}' ({entry.status})"
            if entry.result_summary:
                summary += f": {entry.result_summary}"
            return summary
        return self.ledger.get_compact_summary() or "No active background tasks."

    async def cancel_task(self, task_id: str) -> bool:
        """IPC cancel delegating to cancel(); True only when a task was cancelled."""
        return (await self.cancel(task_id)).startswith("Cancelled")

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
                start_str = datetime.fromtimestamp(t._created_at).astimezone().strftime("%H:%M:%S")
                elapsed = int(time.time() - t._created_at)
                lines.append(f"- {t.id}: '{t.description}' (running for {elapsed}s, started at {start_str})")
        if recent_done:
            lines.append("Recently completed:")
            for t in recent_done:
                finish_time = t._finished_at or time.time()
                duration = int(finish_time - t._created_at)
                finish_str = datetime.fromtimestamp(finish_time).astimezone().strftime("%H:%M:%S")
                res_suffix = f": {t.result}" if t.result else ""
                lines.append(f"- {t.id}: '{t.description}' ({t.status} in {duration}s at {finish_str}{res_suffix})")

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

    async def _announce_ids(self, batch: list[str]) -> bool:
        """AnnouncerLoop speak_fn: floor already held, resolve results via ledger."""
        tuples = []
        for task_id in batch:
            entry = self.ledger.get(task_id)
            tuples.append((task_id, entry.result_summary if entry else ""))
        return await self._speak_batch(tuples)

    def _broadcast_wake_event(self, event: AgentEvent) -> None:
        """Broadcasts a wake-turn event to IPC subscribers as plain data."""
        payload: dict[str, Any] = {}
        for attr in ("text", "thought", "final_text"):
            val = getattr(event, attr, None)
            if isinstance(val, str) and val:
                payload[attr] = val
        self._broadcast_event("wake_event", {"event": type(event).__name__, **payload})

    async def _speak_batch(self, batch: list[tuple[str, str]]) -> bool:
        """Synthesizes background worker results via dedicated announcer and delivers to output."""
        items = [f"- {task_id}: {result.strip()}" for task_id, result in batch if result.strip()]
        if not items:
            return False
        wake_prompt = (
            f"{self._request_grounding()}\n"
            "[Background task completion update. Announce the completed results directly to the user in spoken conversational prose:]\n"
            + "\n".join(items)
        )
        final_text = ""
        fan_out = WakeOutputFanOut(self.wake_output, self._broadcast_wake_event)
        try:
            self.announcer.messages = []
            self.announcer.n_calls = 0
            self.announcer.cost = 0.0
            self.announcer.model = adapt_provider(self.provider)
            async for event in self.announcer.step_stream(wake_prompt):
                if isinstance(event, CoreTurnComplete) and event.final_text in (
                    "LimitsExceeded",
                    "TimeExceeded",
                ):
                    logger.warning(
                        "Announcer step aborted with %s", event.final_text
                    )
                    continue
                await fan_out.emit(event)
                if isinstance(event, CoreTurnComplete) and event.final_text:
                    final_text = event.final_text
        except Exception:
            logger.exception("Failed to synthesize background task completion announcement")
            return False

        if final_text:
            self.foreground.messages.append({"role": "assistant", "content": final_text})
            if self.on_wake:
                wake_res = self.on_wake(final_text)
                if asyncio.iscoroutine(wake_res):
                    await wake_res
            return True
        return False

    async def aclose(self) -> None:
        """Teardown: SIGKILL tracked in-flight process groups."""
        for env in self._environments:
            try:
                env.kill_all()
            except Exception:
                logger.exception("Failed to kill worker process groups on shutdown")
        self._environments.clear()
