from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import Mock, NonCallableMock

from verbix.core.events import TextDelta, TurnComplete

if TYPE_CHECKING:
    from verbix.core.agent import AgentSession
    from verbix.minisweagent.agents.default import DefaultAgent

TaskStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


class WorkerTask:
    """Manages an autonomous background worker running an AgentSession or DefaultAgent in an asyncio.Task."""

    def __init__(
        self,
        id: str,
        description: str,
        session: AgentSession | DefaultAgent | Any,
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
        session: Any = self.session
        try:
            has_step_stream = "step_stream" in getattr(session, "__dict__", {}) or (
                hasattr(type(session), "step_stream")
                and not isinstance(session, (Mock, NonCallableMock))
            )
            if has_step_stream:
                async for event in session.step_stream(self.description):
                    if isinstance(event, TextDelta):
                        accumulated_text.append(event.text)
                    elif isinstance(event, TurnComplete) and event.final_text:
                        self._result = event.final_text
            elif hasattr(session, "step"):
                step_res = session.step(self.description)
                if hasattr(step_res, "__aiter__"):
                    async for event in step_res:
                        if isinstance(event, TextDelta):
                            accumulated_text.append(event.text)
                        elif isinstance(event, TurnComplete) and event.final_text:
                            self._result = event.final_text
                else:
                    res = await asyncio.to_thread(session.step)
                    if isinstance(res, list) and res:
                        last = res[-1]
                        if isinstance(last, dict):
                            self._result = last.get("content")
            if not self._result:
                msgs = getattr(session, "messages", None)
                if not msgs and hasattr(session, "history"):
                    msgs = getattr(session.history, "messages", None)
                if msgs:
                    for m in reversed(msgs):
                        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                        tool_calls = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
                        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
                        if role == "assistant" and not tool_calls and content and content.strip():
                            self._result = content.strip()
                            break

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
        msgs: list[Any] = []
        session: Any = self.session
        if hasattr(session, "history") and hasattr(session.history, "messages"):
            msgs = session.history.messages
        elif hasattr(session, "messages"):
            msgs = session.messages

        elapsed = int(time.time() - self._created_at)

        header = f"{self.id}: '{self.description}' ({self.status}, running for {elapsed}s)"
        if self.result:
            header += f"\nResult: {self.result}"
        if not msgs:
            return header

        def _get_field(obj: Any, key: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        if full:
            lines = [header, "Full history:"]
            for m in msgs:
                role = _get_field(m, "role", "")
                content = _get_field(m, "content", "")
                tool_calls = _get_field(m, "tool_calls", None)
                if tool_calls:
                    calls_list: list[str] = []
                    for tc in tool_calls:
                        name = _get_field(tc, "name") or _get_field(_get_field(tc, "function", {}), "name", "")
                        args = _get_field(tc, "arguments") or _get_field(_get_field(tc, "function", {}), "arguments", "")
                        calls_list.append(f"{name}({args})")
                    calls = ", ".join(calls_list)
                    lines.append(f"- Assistant called: {calls}")
                elif role == "tool":
                    obs = content[:200] + ("..." if len(content) > 200 else "")
                    lines.append(f"- Tool result: {obs}")
                elif content:
                    lines.append(f"- {role.capitalize()}: {content}")
            return "\n".join(lines)

        # Concise: extract last 2-3 action lines
        action_lines = []
        for m in msgs[-6:]:
            role = _get_field(m, "role", "")
            content = _get_field(m, "content", "")
            tool_calls = _get_field(m, "tool_calls", None)
            if tool_calls:
                names = ", ".join(
                    _get_field(tc, "name") or _get_field(_get_field(tc, "function", {}), "name", "")
                    for tc in tool_calls
                )
                action_lines.append(f"Called {names}")
            elif role == "tool" and content:
                snippet = content[:100].replace("\n", " ").strip()
                action_lines.append(f"Result: {snippet}")

        if action_lines:
            return f"{header}\nRecent actions:\n" + "\n".join(f"- {a}" for a in action_lines[-3:])
        return header

