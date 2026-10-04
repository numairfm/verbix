from __future__ import annotations

import asyncio
import platform
from collections.abc import AsyncIterator, Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any

from verbix.core.agent import AgentSession
from verbix.core.events import AgentEvent, TurnComplete
from verbix.runtime.buffer import CompletionBuffer
from verbix.runtime.tools import CancelTool, SpawnTool, StatusTool
from verbix.runtime.worker import WorkerTask
from verbix.tools.bash import BashTool
from verbix.tools.filesystem import FileEditTool, FileReadTool, FileWriteTool

if TYPE_CHECKING:
    from verbix.provider.client import ProviderClient

DEFAULT_VERBIX_PROMPT = """You are Verbix, a voice-operated assistant and calm, capable peer. You say "I," speak plainly, admit limits honestly, and avoid flattery.

- Your answers are spoken aloud. Write for listening: lead with the answer and speak in plain, continuous conversational prose. Never use markdown, bullet points, asterisks, backticks, or code syntax.
- Be concise. Give the direct answer without narrating internal steps or tool mechanics.
- Answer conversational and general knowledge questions directly.
- Delegate tasks requiring shell commands, file tools, or multi-step execution with spawn; base work tools live on workers only.
- When a turn calls tools, begin with one short spoken sentence stating what you are starting.
- When spawn accepts work, confirm starting in one brief sentence; background workers announce terminal results upon completion. Preserve all clauses and requested phrasing in the objective.
- Inspect active background work with status only when the user explicitly asks how work is going.
- Stop background work with cancel."""

DEFAULT_WORKER_PROMPT = """You are an autonomous background worker executing tasks for Verbix.
Use your filesystem and shell tools directly to inspect, write, test, and complete the requested task.
When complete, provide a clear, concise summary of the actions taken and final results."""


class Runtime:
    """Core Verbix runtime orchestrating the foreground agent and independent background workers."""

    def __init__(
        self,
        workspace: Path,
        provider: ProviderClient,
        system_prompt: str | None = None,
        worker_system_prompt: str | None = None,
        buffer_window: float = 0.3,
        on_wake: Callable[[str], Coroutine[Any, Any, None] | None] | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.provider = provider
        base_prompt = system_prompt or DEFAULT_VERBIX_PROMPT
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
        self.foreground = AgentSession(
            session_id="verbix-foreground",
            system_prompt=self.system_prompt,
            provider=self.provider,
            workspace=self.workspace,
            tools=self.foreground_tools,
        )

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
            async for event in self.foreground.step(user_text):
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
                if isinstance(event, TurnComplete) and event.final_text:
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

        worker_tools = [
            FileReadTool(),
            FileWriteTool(),
            FileEditTool(),
            BashTool(),
        ]
        worker_session = AgentSession(
            session_id=f"worker-{task_id}",
            system_prompt=self.worker_system_prompt,
            provider=self.provider,
            workspace=self.workspace,
            tools=worker_tools,
        )

        worker_task = WorkerTask(
            id=task_id,
            description=task,
            session=worker_session,
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
        """Directly announces completed worker results and records them in history."""
        results = [result.strip() for _, result in batch if result.strip()]
        if not results:
            return
        announcement = " ".join(results)

        # 1. Commit into foreground conversation history as assistant speech
        self.foreground.history.append_assistant(content=announcement)

        # 2. Directly deliver to wake_output (TTS player & CLI printer)
        if self.wake_output and hasattr(self.wake_output, "emit"):
            res = self.wake_output.emit(TurnComplete(final_text=announcement))
            if asyncio.iscoroutine(res):
                await res

        # 3. Notify any registered on_wake callback
        if self.on_wake:
            wake_res = self.on_wake(announcement)
            if asyncio.iscoroutine(wake_res):
                await wake_res
