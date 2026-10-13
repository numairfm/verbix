from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import Mock, NonCallableMock

from verbix.core.events import TextDelta, ToolCallDelta, ToolCallStarted, TurnComplete

if TYPE_CHECKING:
    from verbix.agent.agents.default import DefaultAgent
    from verbix.agent.worker import WorkerAgent

TaskStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

# Backstop for zero-cost models, where the spending cap never trips and a
# talk-only loop would otherwise run forever. Cost-limited runs end sooner
# via LimitsExceeded. A worker turn ends only on a finish tool call.
_MAX_TURNS_WITHOUT_FINISH = 30


def _is_finish_tool_message(m: Any) -> bool:
    role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
    if role != "tool":
        return False
    name = m.get("name") if isinstance(m, dict) else getattr(m, "name", None)
    return name == "finish"



def _content_text(content: Any) -> str:
    """Renders tool content as display text, eliding image parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "image_url":
                    chunks.append("[image elided]")
                elif part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
                else:
                    chunks.append(str(part))
            else:
                chunks.append(str(part))
        return "".join(chunks)
    return str(content) if content is not None else ""


def format_worker_event(task_id: str, event: Any) -> str | None:
    """One tagged CLI line per background event; None for events with no display."""
    if isinstance(event, TextDelta):
        return f"[{task_id}] {event.text}"
    if isinstance(event, ToolCallDelta) and event.name:
        return f"[{task_id}] $ {event.name}({event.arguments_delta})"
    if isinstance(event, ToolCallStarted):
        return f"[{task_id}] $ {event.tool_name}({json.dumps(event.arguments)})"
    return None


class WorkerTask:
    """Manages an autonomous background worker running a DefaultAgent in an asyncio.Task."""

    def __init__(
        self,
        id: str,
        description: str,
        session: WorkerAgent | DefaultAgent | Any,
        on_event: Callable[[str, Any], None] | None = None,
    ) -> None:
        self.id = id
        self.description = description
        self.session = session
        self.on_event = on_event

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

    def _emit(self, event: Any) -> None:
        """Forwards one session event to the watch callback; never breaks the turn."""
        if self.on_event is None:
            return
        try:
            self.on_event(self.id, event)
        except Exception:  # noqa: BLE001, S110
            pass

    async def _run(self) -> None:
        session: Any = self.session
        try:
            has_run = "run" in getattr(session, "__dict__", {}) or (
                hasattr(type(session), "run")
                and not isinstance(session, (Mock, NonCallableMock))
            )
            if has_run and callable(getattr(session, "run", None)):
                if hasattr(session, "on_event"):
                    session.on_event = lambda ev: self._emit(ev)
                res = session.run(self.description)
                if inspect.isawaitable(res):
                    res = await res
                if isinstance(res, dict):
                    self._status = res.get("status", "completed")
                    self._result = res.get("summary", "Task completed.")
                else:
                    self._status = "completed"
                    self._result = str(res) if res is not None else "Task completed."
                if self._status == "failed" and not self._error:
                    self._error = self._result
            else:
                has_step_stream = "step_stream" in getattr(session, "__dict__", {}) or (
                    hasattr(type(session), "step_stream")
                    and not isinstance(session, (Mock, NonCallableMock))
                )
                rounds = 0
                while True:
                    rounds += 1
                    if has_step_stream:
                        async for event in session.step_stream(self.description):
                            self._emit(event)
                            if isinstance(event, TurnComplete) and event.final_text:
                                self._result = event.final_text
                    elif hasattr(session, "step"):
                        step_res = session.step(self.description)
                        if hasattr(step_res, "__aiter__"):
                            async for event in step_res:
                                self._emit(event)
                                if isinstance(event, TurnComplete) and event.final_text:
                                    self._result = event.final_text
                        else:
                            res = await asyncio.to_thread(session.step)
                            if isinstance(res, list) and res:
                                last = res[-1]
                                if isinstance(last, dict):
                                    self._result = last.get("content")
                    else:
                        self._status = "failed"
                        self._result = "Task session has no runnable interface."
                        self._error = self._result
                        break
                    msgs = self._session_messages(session)
                    if any(_is_finish_tool_message(m) for m in msgs):
                        if not self._result:
                            for m in reversed(msgs):
                                if _is_finish_tool_message(m):
                                    content = (
                                        m.get("content")
                                        if isinstance(m, dict)
                                        else getattr(m, "content", None)
                                    )
                                    text = _content_text(content).strip()
                                    if text:
                                        self._result = text
                                        break
                        if not self._result:
                            self._result = "Task completed."
                        self._status = "completed"
                        break
                    if self._result in ("LimitsExceeded", "TimeExceeded"):
                        self._status = "failed"
                        self._result = (
                            "Task ended without calling finish; no verified result."
                        )
                        self._error = self._result
                        break
                    if rounds >= _MAX_TURNS_WITHOUT_FINISH:
                        self._status = "failed"
                        self._result = (
                            "Task ended without calling finish; no verified result."
                        )
                        self._error = self._result
                        break
                    self._result = None
        except asyncio.CancelledError:
            self._status = "cancelled"
            self._result = "Task cancelled by user."
            raise
        except Exception as e:  # noqa: BLE001
            self._status = "failed"
            self._error = str(e)
            self._result = f"Task failed: {e}"
            self._emit(TextDelta(text=f"! task failed: {e}"))
        finally:
            self._finished_at = time.time()
            for cb in self._completion_callbacks:
                try:
                    cb(self)
                except Exception:  # noqa: BLE001, S110
                    pass

    @staticmethod
    def _session_messages(session: Any) -> list[Any]:
        msgs = getattr(session, "messages", None)
        if not msgs and hasattr(session, "history"):
            msgs = getattr(session.history, "messages", None)
        return list(msgs) if isinstance(msgs, list) else []

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

        start_str = datetime.fromtimestamp(self._created_at).astimezone().strftime("%H:%M:%S")
        if self.status == "running":
            elapsed = int(time.time() - self._created_at)
            status_desc = f"running for {elapsed}s, started at {start_str}"
        else:
            finish_time = self._finished_at or time.time()
            duration = int(finish_time - self._created_at)
            finish_str = datetime.fromtimestamp(finish_time).astimezone().strftime("%H:%M:%S")
            status_desc = f"{self.status} in {duration}s, started at {start_str}, finished at {finish_str}"

        header = f"{self.id}: '{self.description}' ({status_desc})"
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
                if role in ("system", "user"):
                    continue
                if tool_calls:
                    calls_list: list[str] = []
                    for tc in tool_calls:
                        name = _get_field(tc, "name") or _get_field(_get_field(tc, "function", {}), "name", "")
                        args = _get_field(tc, "arguments") or _get_field(_get_field(tc, "function", {}), "arguments", "")
                        calls_list.append(f"{name}({args})")
                    calls = ", ".join(calls_list)
                    lines.append(f"- Assistant called: {calls}")
                elif role == "tool":
                    text = _content_text(content)
                    obs = text[:200] + ("..." if len(text) > 200 else "")
                    lines.append(f"- Tool result: {obs}")
                elif content:
                    lines.append(f"- Assistant: {content}")
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
                snippet = _content_text(content)[:100].replace("\n", " ").strip()
                action_lines.append(f"Result: {snippet}")

        if action_lines:
            return f"{header}\nRecent actions:\n" + "\n".join(f"- {a}" for a in action_lines[-3:])
        return header

