from __future__ import annotations

import asyncio
import logging
import platform
import re
import sys
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("verbix.runtime")

_KEY_ASSIGN_RE = re.compile(r"([A-Za-z0-9_]*KEY\s*=\s*)\S+")
_API_KEY_JSON_RE = re.compile(r"""(['"]api_key['"]\s*:\s*['"])[^'"]*(['"])""")


def _redact_secrets(text: str) -> str:
    text = _KEY_ASSIGN_RE.sub(r"\1[REDACTED]", text)
    return _API_KEY_JSON_RE.sub(r"\1[REDACTED]\2", text)

from verbix.agent.utils.history import validate_and_repair_history
from verbix.core.events import (
    AgentEvent,
)
from verbix.core.events import (
    TurnComplete as CoreTurnComplete,
)
from verbix.runtime.announcer import (
    WakeOutputFanOut,
    build_announcer,
    build_announcer_agent,
    drain_pending_announcements,
)
from verbix.runtime.floor import FloorState
from verbix.runtime.foreground import (
    DEFAULT_VERBIX_PROMPT,
    DEFAULT_WORKER_PROMPT,
    DefaultAgentHistoryProxy,
    ProviderStreamAdapter,
    adapt_provider,
    build_foreground,
    build_foreground_agent,
)
from verbix.runtime.worker import WorkerTask
from verbix.runtime.workers import finish_runtime_task, spawn_worker

__all__ = [
    "DEFAULT_VERBIX_PROMPT",
    "DEFAULT_WORKER_PROMPT",
    "DefaultAgentHistoryProxy",
    "ProviderStreamAdapter",
    "Runtime",
    "TurnHandle",
    "WakeOutputFanOut",
    "adapt_provider",
    "build_announcer",
    "build_announcer_agent",
    "build_foreground",
    "build_foreground_agent",
    "drain_pending_announcements",
    "spawn_worker",
]


@dataclass
class TurnHandle:
    """Owns an active turn: its task, kind, and delivered-text provider."""

    task: asyncio.Task[Any] | None = None
    kind: str = "foreground"
    delivered_text_provider: Callable[[], str] | None = None
    turn_id: int = 0
    repaired: bool = False
    completed: bool = False

    def delivered_text(self) -> str:
        if self.delivered_text_provider is None:
            return ""
        return self.delivered_text_provider()

    def cancel(self) -> None:
        """Idempotently cancel the turn task."""
        if self.task is not None and not self.task.done():
            self.task.cancel()


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
        worker_echo: bool = False,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.provider = provider
        self.worker_model = worker_model
        self.worker_echo = worker_echo

        fg_config = build_foreground(
            self,
            system_prompt=system_prompt,
            worker_system_prompt=worker_system_prompt,
            on_wake=on_wake,
        )
        build_announcer(self, fg_config)

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

    def _echo_worker_event(self, task_id: str, event: Any) -> None:
        """Prints one tagged background-turn line for --show-worker-turns."""
        from verbix.runtime.worker import format_worker_event

        line = format_worker_event(task_id, event)
        if line:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def _build_grounded_prompt(self, base_prompt: str | None = None) -> str:
        """Build the static, cache-stable system prompt (no volatile clock content)."""
        prompt_to_use = (
            base_prompt if base_prompt is not None else self._base_system_prompt
        )
        if "Host operating system:" in prompt_to_use:
            base = prompt_to_use
        else:
            from verbix.tools.sandbox import probe_sandbox

            shell_part = " Shell: bash." if probe_sandbox().available else ""
            caps = getattr(self, "worker_capabilities", "")
            caps_line = (
                f"\n- Background worker capabilities (available via spawn): {caps}."
                if caps
                else ""
            )
            grounding = (
                "\n\n# ENVIRONMENT & GROUNDING"
                f"\n- Host operating system: {platform.system()}.{shell_part}"
                f"\n- The working directory is {self.workspace}."
                f"{caps_line}"
            )
            base = prompt_to_use.rstrip() + grounding

        if self._custom_prompt:
            base = base.rstrip() + f"\n\n# CUSTOM USER INSTRUCTIONS\n{self._custom_prompt}"
        return base

    def _request_grounding(self, interruption: Any = None) -> str:
        """Volatile per-request grounding, prepended to the user message only."""
        now_str = datetime.now().astimezone().strftime("%A, %B %d, %Y, %I:%M %p")
        envelope = [f"[Current local date and time: {now_str}.]"]
        if interruption is not None:
            confirmed_heard = (
                getattr(interruption, "confirmed_heard", "")
                or getattr(interruption, "heard_text", "")
            )
            interrupted_sentence = getattr(interruption, "interrupted_sentence", "")
            unplayed_text = (
                getattr(interruption, "unplayed_text", "")
                or getattr(interruption, "unspoken_draft", "")
            )

            envelope.append("[Floor context: Your previous turn was interrupted while speaking.]")
            envelope.append(f'- Text confirmed as fully played: "{confirmed_heard or "(None)"}"')
            envelope.append(
                f'- Sentence interrupted mid-playback (partially audible before cut-off): "{interrupted_sentence or "(None)"}"'
            )
            envelope.append(
                f'- Subsequent text that had not yet started playing (discarded unless the user explicitly requests continuation): "{unplayed_text or "(None)"}"'
            )
        return "\n".join(envelope)

    def _mark_task_heard(self, task_id: str) -> None:
        if hasattr(self, "ledger"):
            self.ledger.mark_heard(task_id)
        if hasattr(self, "_pending_announcements"):
            self._pending_announcements[:] = [
                (tid, r) for tid, r in self._pending_announcements if tid != task_id
            ]

    def _foreground_grounding(self, interruption: Any = None) -> tuple[str, list[str]]:
        """Builds per-turn foreground grounding including active and newly completed tasks."""
        envelope = [self._request_grounding(interruption)]
        active = self.list_active_tasks()
        if active:
            active_parts = [
                f"{t.id} ('{t.description}', running for {max(0, int(time.time() - t._created_at))}s)"
                for t in active
            ]
            envelope.append(f"[Active background tasks: {'; '.join(active_parts)}]")

        folded_ids: list[str] = []
        pending_lines: list[str] = []
        if hasattr(self, "ledger"):
            for entry in self.ledger.pending_deliveries():
                folded_ids.append(entry.id)
                pending_lines.append(
                    f"[Task '{entry.id}' {entry.status}]: {entry.result_summary}"
                )
        if hasattr(self, "_pending_announcements"):
            for tid, res in self._pending_announcements:
                if tid not in folded_ids and res.strip():
                    entry = self.ledger.get(tid) if hasattr(self, "ledger") else None
                    if entry is None or entry.delivery == "pending":
                        folded_ids.append(tid)
                        pending_lines.append(f"[Task '{tid}' completed]: {res.strip()}")
        if pending_lines:
            envelope.append(
                "[Newly completed background tasks (report briefly before answering):]\n"
                + "\n".join(pending_lines)
            )
        return "\n".join(envelope), folded_ids

    async def step(
        self, user_text: str | None = None, *, interruption: Any = None
    ) -> AsyncIterator[AgentEvent]:
        """Executes a foreground conversation turn with Verbix."""
        async with self.floor.acquire(FloorState.FOREGROUND):
            self._turn_counter = getattr(self, "_turn_counter", 0) + 1
            handle = TurnHandle(
                task=asyncio.current_task(),
                kind="foreground",
                turn_id=self._turn_counter,
                delivered_text_provider=lambda: self.foreground._delivered_text,
            )
            self._turn_handle = handle
            self._last_turn_handle = handle
            folded_ids: list[str] = []
            turn_ok = False
            try:
                async for event in self._step_inner(
                    user_text, interruption=interruption, folded_ids_out=folded_ids
                ):
                    if isinstance(event, CoreTurnComplete) and not getattr(
                        event, "is_error", False
                    ):
                        turn_ok = True
                    yield event
            finally:
                if not self.floor.interrupt_requested:
                    handle.completed = True
                    if turn_ok:
                        for tid in folded_ids:
                            self._mark_task_heard(tid)
                handle.task = None

    async def _step_inner(
        self,
        user_text: str | None = None,
        *,
        agent: Any = None,
        interruption: Any = None,
        folded_ids_out: list[str] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        # System prompt stays static (provider cache prefix); volatile
        # grounding rides on the user message at request time only.
        session = agent or self.foreground
        session.model = adapt_provider(self.provider)
        if user_text is not None:
            if session is self.foreground:
                grounding, folded = self._foreground_grounding(interruption)
                if folded_ids_out is not None:
                    folded_ids_out.extend(folded)
                user_text = f"{grounding}\n{user_text}"
            else:
                user_text = f"{self._request_grounding(interruption)}\n{user_text}"
        async for event in session.step_stream(user_text):
            yield event

    async def submit(
        self,
        user_text: str,
        output: Any = None,
        *,
        agent: Any = None,
        interruption: Any = None,
    ) -> str:
        """Runs a complete foreground turn, streaming events to output and returning final text."""
        session = agent or self.foreground
        final_text = ""
        async with self.floor.acquire(FloorState.FOREGROUND):
            self._turn_counter = getattr(self, "_turn_counter", 0) + 1
            handle = TurnHandle(
                task=asyncio.current_task(),
                kind="foreground",
                turn_id=self._turn_counter,
                delivered_text_provider=lambda: self._submit_delivered_text(output, session),
            )
            self._turn_handle = handle
            self._last_turn_handle = handle
            folded_ids: list[str] = []
            turn_ok = False
            try:
                async for event in self._step_inner(
                    user_text,
                    agent=session,
                    interruption=interruption,
                    folded_ids_out=folded_ids,
                ):
                    if output and hasattr(output, "emit"):
                        res = output.emit(event)
                        if asyncio.iscoroutine(res):
                            await res
                    if isinstance(event, CoreTurnComplete):
                        if not getattr(event, "is_error", False):
                            turn_ok = True
                        if event.final_text:
                            final_text = event.final_text
                if output and getattr(output, "is_playing", False):
                    while output.is_playing:
                        await asyncio.sleep(0.02)
                    await asyncio.sleep(0.15)
            finally:
                was_interrupted = (
                    self.floor.interrupt_requested
                    or (output is not None and getattr(output, "last_interruption", None) is not None)
                )
                if not was_interrupted:
                    handle.completed = True
                    if turn_ok:
                        for tid in folded_ids:
                            self._mark_task_heard(tid)
                handle.task = None
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
        target = (
            self._turn_handle
            if self._turn_handle.task is not None
            else getattr(self, "_last_turn_handle", None)
        )
        if target is None:
            return

        task = target.task
        had_active = task is not None and not task.done()
        target.cancel()
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
            except (asyncio.CancelledError, TimeoutError):
                pass

        if target.completed or target.repaired:
            return

        if not had_active and heard is None:
            return

        delivered = heard if heard is not None else target.delivered_text()
        self.foreground.messages[:] = validate_and_repair_history(
            self.foreground.messages, delivered_text=delivered
        )
        target.repaired = True

    async def spawn(self, task: str) -> str:
        """Spawns an independent background worker with full workspace tools."""
        return await spawn_worker(self, task)

    def _on_floor_state_changed(self, prev_state: FloorState, new_state: FloorState) -> None:
        if self.floor.can_announce() and self._pending_announcements:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._drain_pending_announcements())
            except RuntimeError:
                pass

    def _on_task_finished(self, task_id: str, result: str) -> None:
        """Records task completion into the ledger, context memory, and schedules announcement."""
        logger.debug("Worker %s finished: %s", task_id, _redact_secrets(result[:2000]))
        finish_runtime_task(self, task_id, result)

    async def _drain_pending_announcements(self) -> None:
        """Speaks accumulated background completions via the single foreground companion."""
        await drain_pending_announcements(self)

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
            if task.status in ("completed", "failed"):
                self._mark_task_heard(task.id)
            return task.get_status(full=full)

        active = self.list_active_tasks()
        recent_done = [t for t in self.tasks.values() if t.status != "running"][-3:]

        if not active and not recent_done:
            return "No active background tasks."

        lines = []
        if active:
            lines.append("Active tasks:")
            for t in active:
                lines.append(f"- {t.get_status(full=full)}")
        if recent_done:
            lines.append("Recently completed:")
            for t in recent_done:
                if t.status in ("completed", "failed"):
                    self._mark_task_heard(t.id)
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

    def _broadcast_wake_event(self, event: AgentEvent) -> None:
        """Broadcasts a wake-turn event to IPC subscribers as plain data."""
        payload: dict[str, Any] = {}
        for attr in ("text", "thought", "final_text"):
            val = getattr(event, attr, None)
            if isinstance(val, str) and val:
                payload[attr] = val
        self._broadcast_event("wake_event", {"event": type(event).__name__, **payload})

    async def aclose(self) -> None:
        """Teardown: SIGKILL tracked in-flight process groups."""
        from verbix.tools.sandbox import kill_active_processes

        kill_active_processes()
