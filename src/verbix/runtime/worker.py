from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from verbix.core.events import TextDelta, TurnComplete

if TYPE_CHECKING:
    from verbix.core.agent import AgentSession

TaskStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


class WorkerTask:
    """Manages an autonomous background worker running an AgentSession in an asyncio.Task."""

    def __init__(
        self,
        id: str,
        description: str,
        session: AgentSession,
    ) -> None:
        self.id = id
        self.description = description
        self.session = session

        self._status: TaskStatus = "pending"
        self._result: str | None = None
        self._error: str | None = None
        self._created_at = time.time()
        self._finished_at: float | None = None

        self._asyncio_task: asyncio.Task[None] | None = None
        self._completion_callbacks: list[Callable[[WorkerTask], None]] = []

    @property
    def status(self) -> TaskStatus:
        return self._status

    @property
    def result(self) -> str | None:
        return self._result

    @property
    def error(self) -> str | None:
        return self._error

    def on_completed(self, callback: Callable[[WorkerTask], None]) -> None:
        self._completion_callbacks.append(callback)

    def start(self) -> None:
        """Starts the background worker execution loop."""
        if self._asyncio_task and not self._asyncio_task.done():
            return
        self._status = "running"
        self._asyncio_task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        accumulated_text: list[str] = []
        try:
            async for event in self.session.step(self.description):
                if isinstance(event, TextDelta):
                    accumulated_text.append(event.text)
                elif isinstance(event, TurnComplete) and event.final_text:
                    self._result = event.final_text
            if not self._result and accumulated_text:
                self._result = "".join(accumulated_text).strip()
            if not self._result:
                self._result = "Task finished without output."
            self._status = "completed"
        except asyncio.CancelledError:
            self._status = "cancelled"
            self._result = "Task cancelled by user."
            raise
        except Exception as e:  # noqa: BLE001
            self._status = "failed"
            self._error = str(e)
            self._result = f"Task failed: {e}"
        finally:
            self._finished_at = time.time()
            for cb in self._completion_callbacks:
                try:
                    cb(self)
                except Exception:  # noqa: BLE001, S110
                    pass

    async def wait(self) -> None:
        """Waits for the background task to complete."""
        if self._asyncio_task:
            try:
                await self._asyncio_task
            except asyncio.CancelledError:
                pass

    async def cancel(self) -> None:
        """Immediately halts the worker task."""
        if self._asyncio_task and not self._asyncio_task.done():
            self._asyncio_task.cancel()
            try:
                await self._asyncio_task
            except asyncio.CancelledError:
                pass
        self._status = "cancelled"

    def get_status(self, full: bool = False) -> str:
        """Reads worker progress directly from session history in memory."""
        msgs = getattr(self.session.history, "messages", [])
        elapsed = int(time.time() - self._created_at)

        header = f"{self.id}: '{self.description}' ({self.status}, running for {elapsed}s)"
        if self.result:
            header += f"\nResult: {self.result}"
        if not msgs:
            return header

        if full:
            lines = [header, "Full history:"]
            for m in msgs:
                if m.tool_calls:
                    calls = ", ".join(f"{tc.name}({tc.arguments})" for tc in m.tool_calls)
                    lines.append(f"- Assistant called: {calls}")
                elif m.role == "tool":
                    obs = m.content[:200] + ("..." if len(m.content) > 200 else "")
                    lines.append(f"- Tool result: {obs}")
                elif m.content:
                    lines.append(f"- {m.role.capitalize()}: {m.content}")
            return "\n".join(lines)

        # Concise: extract last 2-3 action lines
        action_lines = []
        for m in msgs[-6:]:
            if m.tool_calls:
                names = ", ".join(tc.name for tc in m.tool_calls)
                action_lines.append(f"Called {names}")
            elif m.role == "tool":
                snippet = m.content[:100].replace("\n", " ").strip()
                action_lines.append(f"Result: {snippet}")

        if action_lines:
            return f"{header}\nRecent actions:\n" + "\n".join(f"- {a}" for a in action_lines[-3:])
        return header
