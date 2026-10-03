from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from verbix.core.agent import AgentSession
from verbix.core.events import (
    AgentEvent,
    ToolCallFinished,
    TurnComplete,
)
from verbix.core.history import ConversationHistory
from verbix.core.models import TaskState, WorkerResult
from verbix.provider.client import ProviderClient
from verbix.runtime.orchestration import (
    TaskCancelTool,
    TaskSpawnTool,
    TaskStatusTool,
    TaskSteerTool,
    WorkerFinishTool,
)
from verbix.runtime.worktree import WorktreeManager
from verbix.tools.bash import BashTool
from verbix.tools.filesystem import FileEditTool, FileReadTool, FileWriteTool

if TYPE_CHECKING:
    from verbix.tools.base import Tool

logger = logging.getLogger(__name__)


class TaskRuntime:
    """Manages the background execution of autonomous worker sessions, structured concurrency,

    announcement serialization, and turn-boundary completion injection.
    """

    def __init__(
        self,
        workspace: Path,
        provider: ProviderClient,
        max_worker_turns: int = 30,
    ) -> None:
        self.workspace = workspace.resolve()
        self.provider = provider
        self.max_worker_turns = max_worker_turns
        self.worktree_manager = WorktreeManager(self.workspace)

        # In-memory task registry (plain dict)
        self.tasks: dict[str, TaskState] = {}
        self._task_counter: int = 0

        # Concurrency and steering queues
        self._worker_tasks: dict[str, asyncio.Task[None]] = {}
        self._steer_queues: dict[str, asyncio.Queue[str]] = {}

        # Announcement queue for spoken completions: (task_id, WorkerResult)
        self.announcement_q: asyncio.Queue[tuple[str, WorkerResult]] = asyncio.Queue()

        # Subscriber queues for broadcasting events to interfaces/frontends
        self.subscribers: list[asyncio.Queue[dict[str, Any]]] = []

        # Turn boundary synchronization with main session
        self.main_session: AgentSession | None = None
        self.is_main_turn_active: bool = False
        self._pending_completions: list[tuple[str, str, WorkerResult]] = []

        # Optional audio player for runtime-spoken acknowledgements
        self.player: Any = None

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Registers a subscriber queue to receive runtime events."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        """Removes a subscriber queue."""
        if q in self.subscribers:
            self.subscribers.remove(q)

    def _broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        """Broadcasts an event to all active subscriber queues without blocking."""
        payload = {"event": event_type, **data}
        for q in list(self.subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    def get_receptionist_tools(self) -> list[Tool]:
        """Returns the strictly orchestration-only tools for the main receptionist agent."""
        return [
            TaskSpawnTool(self),
            TaskCancelTool(self),
            TaskSteerTool(self),
            TaskStatusTool(self),
        ]

    def create_receptionist(
        self,
        system_prompt: str | None = None,
        history: ConversationHistory | None = None,
    ) -> AgentSession:
        """Initializes and returns the main receptionist AgentSession."""
        default_prompt = (
            "You are Verbix Receptionist, an orchestration and voice assistant. "
            "Your role is to converse with the user and orchestrate background tasks using "
            "task_spawn, task_cancel, task_steer, and task_status.\n"
            "When the user asks for background coding, building, searching, or execution, "
            "spawn a task immediately with a clear title and instructions.\n"
            "When background tasks complete, their status and results appear as system notifications in the conversation history. "
            "Use these completed task results directly to inform the user when they ask about progress or results, without needing to call task_status.\n"
            "Keep your spoken responses concise, natural, and conversational."
        )
        tools = self.get_receptionist_tools()
        session = AgentSession(
            session_id="receptionist-main",
            system_prompt=system_prompt or default_prompt,
            provider=self.provider,
            workspace=self.workspace,
            tools=tools,
            history=history,
            reasoning_effort="low",
        )
        self.main_session = session
        return session

    async def step_receptionist(
        self,
        user_input: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Runs one conversational turn on the main receptionist agent with automatic

        ephemeral running-task annotations and turn-boundary completion synchronization.
        """
        if self.main_session is None:
            self.create_receptionist()
        assert self.main_session is not None

        self.on_main_turn_start()
        try:
            async for event in self.main_session.step(
                user_input=user_input,
                running_tasks=self.list_active_tasks,
            ):
                yield event
        finally:
            self.on_main_turn_end()

    def list_active_tasks(self) -> list[TaskState]:
        """Returns all currently running background tasks."""
        return [t for t in self.tasks.values() if t.is_running]

    def list_all_tasks(self) -> list[TaskState]:
        """Returns all registered tasks."""
        return list(self.tasks.values())

    def get_task(self, task_id: str) -> TaskState | None:
        """Returns the state of a specific task."""
        return self.tasks.get(task_id)

    def get_task_status_summary(self, task_id: str | None = None) -> str:
        """Generates a text summary of task status for the receptionist tool."""
        if task_id:
            task = self.tasks.get(task_id)
            if not task:
                return f"Task '{task_id}' not found."
            lines = [
                f"Task '{task.title}' ({task.id}):",
                f"  Status: {task.status}",
                f"  Step count: {task.step_count}",
            ]
            if task.result:
                outcome = "Succeeded" if task.result.success else "Failed"
                lines.append(f"  Result: {outcome}")
                lines.append(f"  Summary: {task.result.speech_summary}")
                lines.append(f"  Details: {task.result.details}")
            return "\n".join(lines)

        active = self.list_active_tasks()
        if not active:
            recent = [t for t in self.tasks.values() if not t.is_running][-3:]
            if not recent:
                return "No active background tasks. No recently completed tasks."
            recent_lines = [
                f"- {t.id} '{t.title}': {t.status} ({t.result.speech_summary if t.result else 'done'})"
                for t in recent
            ]
            return "No active background tasks.\nRecent tasks:\n" + "\n".join(
                recent_lines
            )

        active_lines = [
            f"- {t.id} '{t.title}' (running, step {t.step_count})" for t in active
        ]
        return f"{len(active)} active background task(s):\n" + "\n".join(active_lines)

    async def spawn_task(
        self,
        title: str,
        prompt: str,
        isolate_worktree: bool = False,
        workspace: Path | str | None = None,
    ) -> str:
        """Spawns an independent background worker session in structured concurrency."""
        self._task_counter += 1
        task_id = f"task-{self._task_counter}"
        target_ws = Path(workspace).resolve() if workspace else self.workspace

        task = TaskState(
            id=task_id,
            title=title,
            status="running",
            step_count=0,
            workspace=target_ws,
        )
        self.tasks[task_id] = task
        self._steer_queues[task_id] = asyncio.Queue()

        self._broadcast("task_spawned", {"task_id": task_id, "title": title})

        # Immediate runtime-spoken acknowledgement
        if self.player is not None:
            asyncio.create_task(
                self.player.speak_foreground_sentences([f"Starting task: {title}."])
            )

        # Launch background worker task
        worker_coro = self._run_worker(
            task_id=task_id,
            title=title,
            prompt=prompt,
            isolate_worktree=isolate_worktree,
            workspace=target_ws,
        )
        bg_task = asyncio.create_task(worker_coro, name=f"worker-{task_id}")
        self._worker_tasks[task_id] = bg_task

        return task_id

    async def cancel_task(self, task_id: str) -> bool:
        """Cancels an active background task cleanly."""
        task = self.tasks.get(task_id)
        if not task or not task.is_running:
            return False

        bg_task = self._worker_tasks.get(task_id)
        if bg_task and not bg_task.done():
            bg_task.cancel()

        result = WorkerResult(
            success=False,
            speech_summary=f"Task '{task.title}' was cancelled by the user.",
            details="Task cancelled mid-execution via task_cancel.",
        )
        self._finish_task(task_id, result, status="cancelled")
        self._broadcast("task_cancelled", {"task_id": task_id, "title": task.title})
        return True

    async def steer_task(self, task_id: str, instruction: str) -> bool:
        """Delivers a steering instruction to an active background worker."""
        task = self.tasks.get(task_id)
        if not task or not task.is_running:
            return False

        q = self._steer_queues.get(task_id)
        if q is not None:
            await q.put(instruction)
            self._broadcast(
                "task_steered",
                {"task_id": task_id, "instruction": instruction},
            )
            return True
        return False

    def on_main_turn_start(self) -> None:
        """Marks that the main receptionist session has started an active turn and drains any pending completions."""
        self.drain_pending_completions()
        self.is_main_turn_active = True

    def on_main_turn_end(self) -> None:
        """Marks that the main receptionist turn has completed and drains any pending completions."""
        self.is_main_turn_active = False
        self.drain_pending_completions()

    def drain_pending_completions(self) -> None:
        """Flushes any buffered completions into main session history at the turn boundary."""
        if not self.main_session:
            return

        while self._pending_completions:
            task_id, title, result = self._pending_completions.pop(0)
            self.main_session.history.inject_task_completion(task_id, title, result)

    def _finish_task(
        self,
        task_id: str,
        result: WorkerResult,
        status: str = "completed",
    ) -> None:
        """Records task completion, enqueues spoken announcement, and performs turn-boundary injection."""
        task = self.tasks.get(task_id)
        if not task:
            return

        # Idempotency guard: avoid duplicate announcements or duplicate history entries
        if not task.is_running and task.status == status:
            return

        task.status = status  # type: ignore
        task.result = result

        # Enqueue for serialized audio speech playback
        self.announcement_q.put_nowait((task_id, result))

        # Turn boundary injection:
        # If main session is idle, inject immediately; if turn is active, buffer until turn ends.
        if not self.is_main_turn_active and self.main_session:
            self.main_session.history.inject_task_completion(
                task_id, task.title, result
            )
        else:
            self._pending_completions.append((task_id, task.title, result))

        self._broadcast(
            "task_finished",
            {
                "task_id": task_id,
                "title": task.title,
                "success": result.success,
                "speech_summary": result.speech_summary,
                "details": result.details,
            },
        )

    async def _run_worker(
        self,
        task_id: str,
        title: str,
        prompt: str,
        isolate_worktree: bool,
        workspace: Path | None = None,
    ) -> None:
        """Executes the worker agent loop until finish or cancellation."""
        task = self.tasks[task_id]
        base_dir = workspace or self.workspace
        worker_dir = base_dir

        if isolate_worktree:
            try:
                worker_dir = await self.worktree_manager.create_worktree(task_id)
            except (OSError, RuntimeError) as e:
                logger.warning(
                    "Worktree creation failed for %s, falling back to root workspace: %s",
                    task_id,
                    e,
                )
                worker_dir = base_dir

        finish_tool = WorkerFinishTool(
            on_finish=lambda res: self._finish_task(task_id, res, status="completed")
        )
        worker_tools: list[Tool] = [
            FileReadTool(),
            FileWriteTool(),
            FileEditTool(),
            BashTool(),
            finish_tool,
        ]

        system_prompt = (
            f"You are a Verbix autonomous worker agent executing task '{title}'.\n"
            f"Workspace directory: {worker_dir}\n"
            "You have access to filesystem and bash tools to inspect, modify, and test code.\n"
            "When you have accomplished your goal or if you encounter an unrecoverable error, "
            "you MUST invoke the 'worker_finish' tool with:\n"
            "- success: boolean\n"
            "- speech_summary: a concise, natural summary suitable for spoken audio playback (max 25-30 words, no markdown)\n"
            "- details: complete technical details, modified files, diffs, test logs, or error traces."
        )

        session = AgentSession(
            session_id=f"worker-{task_id}",
            system_prompt=system_prompt,
            provider=self.provider,
            workspace=worker_dir,
            tools=worker_tools,
            max_turns=self.max_worker_turns,
        )

        last_text: str = ""
        current_user_input: str | None = prompt

        try:
            while task.status == "running" and task.step_count < self.max_worker_turns:
                # Check for incoming steering instructions
                steer_q = self._steer_queues.get(task_id)
                if steer_q and not steer_q.empty():
                    steer_msgs = []
                    while not steer_q.empty():
                        steer_msgs.append(steer_q.get_nowait())
                    steering_prompt = "\n".join(
                        f"[Steering instruction from user: {m}]" for m in steer_msgs
                    )
                    if current_user_input:
                        current_user_input += "\n" + steering_prompt
                    else:
                        current_user_input = steering_prompt

                task.step_count += 1

                # Execute one step of the agent session
                async for event in session.step(user_input=current_user_input):
                    current_user_input = None  # Consumed by step

                    if isinstance(event, TurnComplete):
                        last_text = event.final_text
                    elif isinstance(event, ToolCallFinished):
                        self._broadcast(
                            "worker_tool_call",
                            {
                                "task_id": task_id,
                                "call_id": event.call_id,
                                "observation": event.observation[:200],
                                "is_error": event.is_error,
                            },
                        )

                # Check if worker called worker_finish (if not already handled via on_finish callback)
                if finish_tool.captured_result is not None:
                    if task.is_running:
                        self._finish_task(
                            task_id, finish_tool.captured_result, status="completed"
                        )
                    return

                # If no tools were called and model produced a final text without calling worker_finish
                if (
                    session.history.messages
                    and session.history.messages[-1].role == "assistant"
                ):
                    last_msg = session.history.messages[-1]
                    if not last_msg.tool_calls:
                        # Model stopped speaking without calling worker_finish
                        speech = (
                            last_text[:180]
                            if last_text
                            else f"Task '{title}' completed."
                        )
                        fallback_result = WorkerResult(
                            success=True,
                            speech_summary=speech,
                            details=last_text or "Task ended without tool calls.",
                        )
                        self._finish_task(task_id, fallback_result, status="completed")
                        return

            # If max turns reached without finish
            timeout_result = WorkerResult(
                success=False,
                speech_summary=f"Task '{title}' stopped after reaching maximum turn limit.",
                details=f"Worker reached step limit of {self.max_worker_turns} turns without calling worker_finish.",
            )
            self._finish_task(task_id, timeout_result, status="failed")

        except asyncio.CancelledError:
            # Cancellation was handled in cancel_task
            pass
        except Exception as e:
            logger.exception("Unexpected error in worker task %s", task_id)
            err_result = WorkerResult(
                success=False,
                speech_summary=f"Task '{title}' failed due to an internal error.",
                details=f"Exception: {type(e).__name__}: {e}",
            )
            self._finish_task(task_id, err_result, status="failed")
