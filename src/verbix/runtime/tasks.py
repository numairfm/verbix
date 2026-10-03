from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from verbix.core.agent import AgentSession
from verbix.core.events import (
    AgentEvent,
    TextDelta,
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


class TurnOutput(Protocol):
    """Protocol for receiving events and reporting delivered text for an active turn."""

    async def emit(self, event: AgentEvent) -> None: ...
    def delivered_text(self) -> str: ...
    def reset_delivered(self) -> None: ...


class DefaultTurnOutput:
    """Standard in-memory turn output tracking events and delivered text."""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []
        self._delivered: list[str] = []

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event)
        if isinstance(event, TextDelta):
            self._delivered.append(event.text)

    def delivered_text(self) -> str:
        return "".join(self._delivered)

    def reset_delivered(self) -> None:
        self._delivered.clear()


class TaskRuntime:
    """Manages foreground conversational turns, background worker tasks,

    structured concurrency, wake-up turns, and worktree/sandbox isolation.
    """

    def __init__(
        self,
        workspace: Path,
        provider: ProviderClient,
        max_worker_turns: int = 30,
        receptionist_model: str | None = None,
        worker_model: str | None = None,
        receptionist_reasoning_effort: str | None = None,
        isolate_worktrees: bool = False,
        allow_network: bool = False,
    ) -> None:
        self.workspace = workspace.resolve()
        self.provider = provider
        self.max_worker_turns = max_worker_turns
        self.receptionist_model = receptionist_model or provider.model
        self.worker_model = worker_model or provider.model
        self.receptionist_reasoning_effort = receptionist_reasoning_effort
        self.isolate_worktrees = isolate_worktrees
        self.allow_network = allow_network
        self.worktree_manager = WorktreeManager(self.workspace)

        # In-memory task registry
        self.tasks: dict[str, TaskState] = {}

        # Concurrency and steering queues
        self._worker_tasks: dict[str, asyncio.Task[None]] = {}
        self._steer_queues: dict[str, asyncio.Queue[str]] = {}

        # Subscriber queues for broadcasting events to interfaces/frontends
        self.subscribers: list[asyncio.Queue[dict[str, Any]]] = []

        # Foreground turn serialization
        self.main_session: AgentSession | None = None
        self._turn_task: asyncio.Task[str | None] | None = None
        self._wake_chain: int = 0
        self.user_active: bool = False
        self.wake_output: TurnOutput | None = None

    @property
    def is_idle(self) -> bool:
        """Returns True if no foreground turn is running and user is not recording."""
        return (
            self._turn_task is None or self._turn_task.done()
        ) and not self.user_active

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
        """Returns orchestration-only tools for the main receptionist agent."""
        return [
            TaskSpawnTool(self),
            TaskCancelTool(self),
            TaskSteerTool(self),
            TaskStatusTool(self),
        ]

    def _get_provider_for_model(self, model: str) -> ProviderClient:
        """Returns a ProviderClient targeting the given model, reusing base config if identical."""
        if model == self.provider.model:
            return self.provider
        return ProviderClient(
            base_url=self.provider.base_url,
            model=model,
            api_key=self.provider.api_key,
            timeout=self.provider.timeout,
            max_tokens=self.provider.max_tokens,
            default_extra_body=self.provider.default_extra_body,
        )

    def create_receptionist(
        self,
        system_prompt: str | None = None,
        history: ConversationHistory | None = None,
        reasoning_effort: str | None = None,
    ) -> AgentSession:
        """Initializes and returns the main receptionist AgentSession."""
        default_prompt = (
            "You are Verbix Receptionist, an orchestration and voice assistant. "
            "Your role is to converse with the user and orchestrate background tasks using "
            "task_spawn, task_cancel, task_steer, and task_status.\n"
            "When the user asks for background coding, building, searching, or execution, "
            "spawn a task immediately with a clear title and instructions.\n"
            "When background tasks complete, their status and results appear as notices in the conversation history. "
            "Use these completed task results directly to inform the user when they ask about progress or results, without needing to call task_status.\n"
            "Keep your spoken responses concise, natural, and conversational."
        )
        tools = self.get_receptionist_tools()
        provider = self._get_provider_for_model(self.receptionist_model)
        eff = reasoning_effort or self.receptionist_reasoning_effort
        if eff is None:
            eff = "none" if "gemma" in self.receptionist_model.lower() else "minimal"
        session = AgentSession(
            session_id="receptionist-main",
            system_prompt=system_prompt or default_prompt,
            provider=provider,
            workspace=self.workspace,
            tools=tools,
            history=history,
            reasoning_effort=eff,
        )
        self.main_session = session
        return session

    def start_turn(
        self,
        user_input: str | None,
        output: TurnOutput,
        kind: str = "user",
    ) -> asyncio.Task[str | None]:
        """Initiates a foreground turn. User input cancels active turn; wake turns queue behind it."""
        prev_task = self._turn_task
        if prev_task is not None and not prev_task.done() and kind == "user":
            prev_task.cancel()

        task = asyncio.create_task(
            self._run_turn(user_input, output, kind=kind, after=prev_task),
            name=f"turn-{kind}",
        )
        self._turn_task = task
        return task

    async def submit(self, user_input: str, output: TurnOutput) -> str | None:
        """Submits user input and waits for the turn to complete."""
        task = self.start_turn(user_input, output, kind="user")
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def cancel_turn(self) -> bool:
        """Cancels the currently active foreground turn and awaits its clean termination."""
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
            try:
                await self._turn_task
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001
                logger.debug("Prior turn cancelled with exception: %s", e)
            return True
        return False

    async def _run_turn(
        self,
        user_input: str | None,
        output: TurnOutput,
        kind: str,
        after: asyncio.Task[Any] | None = None,
    ) -> str | None:
        """Internal execution loop for a foreground turn."""
        if after is not None:
            await asyncio.wait({after})

        if kind == "wake":
            unannounced = [
                t for t in self.tasks.values() if not t.is_running and not t.announced
            ]
            if not unannounced:
                return None

        if self.main_session is None:
            self.create_receptionist()
        assert self.main_session is not None

        # 1. Inject unrecorded terminal completions into history
        unrecorded = [
            t for t in self.tasks.values() if not t.is_running and not t.recorded
        ]
        for t in unrecorded:
            if t.result is not None:
                self.main_session.history.inject_task_completion(
                    t.id, t.title, t.result
                )
            t.recorded = True

        # Track which tasks are pending announcement
        tasks_to_announce = [
            t for t in self.tasks.values() if not t.is_running and not t.announced
        ]

        output.reset_delivered()
        final_text: str | None = None
        try:
            async for event in self.main_session.step(
                user_input=user_input,
                running_tasks=self.list_active_tasks,
            ):
                if isinstance(event, TurnComplete):
                    final_text = event.final_text
                await output.emit(event)
        except asyncio.CancelledError:
            self.main_session.history.close_aborted_turn(output.delivered_text())
            raise
        except Exception as e:
            logger.exception("Error in receptionist turn")
            self.main_session.history.close_aborted_turn(output.delivered_text())
            if kind == "user":
                err_str = str(e)
                if "401" in err_str:
                    err_msg = "Provider error: unauthorized (check API key)."
                elif "429" in err_str:
                    err_msg = "Provider error: rate limit exceeded."
                elif "503" in err_str or "502" in err_str:
                    err_msg = "Provider error: service unavailable."
                elif (
                    "ConnectError" in type(e).__name__ or "timed out" in err_str.lower()
                ):
                    err_msg = "Provider connection failed. Please check network."
                else:
                    err_msg = f"Sorry, request failed: {e}"
                self.main_session.history.append_assistant(content=err_msg)
                await output.emit(TurnComplete(final_text=err_msg))
                final_text = err_msg
            else:
                # Wake turn fallback: speak summaries directly
                summaries = [
                    t.result.speech_summary for t in tasks_to_announce if t.result
                ]
                fallback = " ".join(summaries) if summaries else f"Task completed: {e}"
                self.main_session.history.append_assistant(content=fallback)
                await output.emit(TurnComplete(final_text=fallback))
                final_text = fallback
                for t in tasks_to_announce:
                    t.announced = True
        else:
            # Normal completion
            for t in tasks_to_announce:
                t.announced = True
            if kind == "user":
                self._wake_chain = 0
            else:
                self._wake_chain += 1
        finally:
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon(self._maybe_wake)
            except RuntimeError:
                pass

        return final_text

    def _maybe_wake(self) -> None:
        """Queues a foreground wake-up turn if runtime is idle and unannounced completions exist."""
        if not self.is_idle or self.wake_output is None:
            return

        unannounced = [
            t for t in self.tasks.values() if not t.is_running and not t.announced
        ]
        if not unannounced:
            return

        if self._wake_chain >= 3:
            # Bound wake chains: deliver summaries directly
            summaries = [t.result.speech_summary for t in unannounced if t.result]
            msg = " ".join(summaries)
            for t in unannounced:
                t.announced = True
            if msg and self.main_session:
                self.main_session.history.append_assistant(content=msg)
                asyncio.create_task(self.wake_output.emit(TurnComplete(final_text=msg)))
            return

        last_is_notice = False
        if self.main_session and self.main_session.history.messages:
            last_msg = self.main_session.history.messages[-1]
            last_is_notice = last_msg.role == "user" and "[Task completed:" in (
                last_msg.content or ""
            )
        prompt = (
            None
            if last_is_notice
            else "[Verbix runtime notice — please inform the user about the completed task(s) above in a concise sentence.]"
        )
        self.start_turn(prompt, self.wake_output, kind="wake")

    async def step_receptionist(
        self,
        user_input: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Compatibility wrapper: streams events from a submit call over an async iterator."""
        q: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

        class QueueOutput:
            def __init__(self) -> None:
                self._delivered: list[str] = []

            async def emit(self, event: AgentEvent) -> None:
                if isinstance(event, TextDelta):
                    self._delivered.append(event.text)
                await q.put(event)

            def delivered_text(self) -> str:
                return "".join(self._delivered)

            def reset_delivered(self) -> None:
                self._delivered.clear()

        out = QueueOutput()
        task = self.start_turn(user_input, out, kind="user")

        async def _waiter():
            try:
                await task
            finally:
                await q.put(None)

        waiter_task = asyncio.create_task(_waiter())
        try:
            while True:
                ev = await q.get()
                if ev is None:
                    break
                yield ev
        finally:
            if not task.done():
                task.cancel()
            waiter_task.cancel()

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
            f"- {t.id} '{t.title}' (running, {t.step_count} tool calls)" for t in active
        ]
        return f"{len(active)} active background task(s):\n" + "\n".join(active_lines)

    async def spawn_task(
        self,
        title: str,
        prompt: str,
        workspace: Path | str | None = None,
    ) -> str:
        """Spawns an independent background worker session in structured concurrency."""
        task_id = f"task-{secrets.token_hex(3)}"
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

        worker_coro = self._run_worker(
            task_id=task_id,
            title=title,
            prompt=prompt,
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
        task.recorded = True
        task.announced = True
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

    def _drain_steer(self, task_id: str) -> list[str]:
        """Drains any queued steering instructions for the given worker."""
        q = self._steer_queues.get(task_id)
        if not q or q.empty():
            return []
        msgs: list[str] = []
        while not q.empty():
            try:
                msgs.append(q.get_nowait())
            except asyncio.QueueEmpty:
                break
        return msgs

    def _finish_task(
        self,
        task_id: str,
        result: WorkerResult,
        status: str = "completed",
    ) -> None:
        """Records task completion and triggers wake turn processing."""
        task = self.tasks.get(task_id)
        if not task:
            return

        if not task.is_running and task.status == status:
            return

        task.status = status  # type: ignore
        task.result = result

        if self.main_session is not None and not task.recorded:
            self.main_session.history.inject_task_completion(
                task.id, task.title, result
            )
            task.recorded = True

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
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon(self._maybe_wake)
        except RuntimeError:
            pass

    async def _run_worker(
        self,
        task_id: str,
        title: str,
        prompt: str,
        workspace: Path | None = None,
    ) -> None:
        """Executes the worker agent loop until finish or cancellation."""
        task = self.tasks[task_id]
        base_dir = workspace or self.workspace
        worker_dir = base_dir
        extra_binds: tuple[tuple[Path, Path, bool], ...] = ()

        if self.isolate_worktrees:
            try:
                worker_dir = await self.worktree_manager.create_worktree(task_id)
                main_git = self.workspace / ".git"
                if main_git.exists():
                    extra_binds = ((main_git, main_git, True),)
            except (OSError, RuntimeError, FileExistsError) as e:
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
            BashTool(allow_network=self.allow_network, extra_binds=extra_binds),
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

        worker_provider = self._get_provider_for_model(self.worker_model)
        session = AgentSession(
            session_id=f"worker-{task_id}",
            system_prompt=system_prompt,
            provider=worker_provider,
            workspace=worker_dir,
            tools=worker_tools,
            max_turns=self.max_worker_turns,
            steer_callback=lambda: self._drain_steer(task_id),
        )

        last_text: str = ""
        try:
            async for event in session.step(user_input=prompt):
                if isinstance(event, TurnComplete):
                    last_text = event.final_text
                elif isinstance(event, ToolCallFinished):
                    task.step_count += 1
                    self._broadcast(
                        "worker_tool_call",
                        {
                            "task_id": task_id,
                            "call_id": event.call_id,
                            "observation": event.observation[:200],
                            "is_error": event.is_error,
                        },
                    )

            if finish_tool.captured_result is not None:
                if task.is_running:
                    self._finish_task(
                        task_id, finish_tool.captured_result, status="completed"
                    )
                return

            # Model stopped without calling worker_finish: re-prompt once
            reprompt = (
                "You stopped without calling worker_finish. "
                "You MUST call the 'worker_finish' tool now with success, speech_summary, and details."
            )
            session.max_turns = 3
            async for event in session.step(user_input=reprompt):
                if isinstance(event, TurnComplete):
                    last_text = event.final_text
                elif isinstance(event, ToolCallFinished):
                    task.step_count += 1

            if finish_tool.captured_result is not None:
                if task.is_running:
                    self._finish_task(
                        task_id, finish_tool.captured_result, status="completed"
                    )
                return

            # Still did not finish: mark failed! Never fabricate success
            failed_res = WorkerResult(
                success=False,
                speech_summary=f"Task '{title}' failed: stopped without calling worker_finish.",
                details=last_text or "Worker ended without calling worker_finish.",
            )
            self._finish_task(task_id, failed_res, status="failed")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("Unexpected error in worker task %s", task_id)
            err_result = WorkerResult(
                success=False,
                speech_summary=f"Task '{title}' failed due to an internal error.",
                details=f"Exception: {type(e).__name__}: {e}",
            )
            self._finish_task(task_id, err_result, status="failed")
        finally:
            self._steer_queues.pop(task_id, None)
