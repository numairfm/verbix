from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from verbix.tools.base import ToolContext, ToolResult

_FALLBACK_TIMERS: dict[str, _TimerEntry] = {}
_FALLBACK_COUNTER: int = 0


@dataclass
class _TimerEntry:
    id: str
    label: str
    duration: float
    fire_at: float
    task: asyncio.Task[None]


def _fmt_seconds(seconds: float) -> str:
    s = max(0, round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {sec}s"
    if m > 0:
        return f"{m}m {sec}s"
    return f"{sec}s"


class TimerTool:
    """Sets, lists, and cancels non-blocking timers and reminders that announce via Verbix's voice."""

    name: str = "timer"
    side_effecting: bool = True
    description: str = (
        "Sets, lists, or cancels non-blocking countdown timers and reminders. "
        "When a timer finishes, Verbix automatically announces the reminder aloud."
    )

    def __init__(self, runtime: Any = None) -> None:
        self.runtime = runtime

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["set", "list", "cancel"],
                    "description": "Timer action: 'set', 'list', or 'cancel'.",
                },
                "seconds": {
                    "type": "number",
                    "description": "Countdown duration in seconds when action='set' (e.g. 300 for 5 minutes).",
                },
                "label": {
                    "type": "string",
                    "description": "Optional reminder label or message when action='set' (e.g. 'check the oven').",
                },
                "timer_id": {
                    "type": "string",
                    "description": "Optional timer ID to cancel (or 'all' to cancel all active timers) when action='cancel'.",
                },
            },
            "required": ["action"],
        }

    def _store(self) -> dict[str, _TimerEntry]:
        if self.runtime is not None:
            store = getattr(self.runtime, "_timers", None)
            if store is None:
                store = {}
                self.runtime._timers = store
            return store
        return _FALLBACK_TIMERS

    def _next_id(self) -> str:
        global _FALLBACK_COUNTER
        if self.runtime is not None:
            c = int(getattr(self.runtime, "_timer_counter", 0)) + 1
            self.runtime._timer_counter = c
            return f"timer-{c}"
        _FALLBACK_COUNTER += 1
        return f"timer-{_FALLBACK_COUNTER}"

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        action = str(args.get("action") or "").strip().lower()
        if action not in ("set", "list", "cancel"):
            return ToolResult(
                f"error: invalid action '{action}'; expected 'set', 'list', or 'cancel'",
                is_error=True,
            )

        store = self._store()

        if action == "list":
            now = time.time()
            active = [e for e in store.values() if not e.task.done()]
            if not active:
                return ToolResult("No active timers.")
            lines = [
                f"- {e.id}: '{e.label}' ({_fmt_seconds(e.fire_at - now)} remaining of {_fmt_seconds(e.duration)})"
                for e in active
            ]
            return ToolResult("Active timers:\n" + "\n".join(lines))

        if action == "cancel":
            target_id = str(args.get("timer_id") or "").strip()
            active = [e for e in store.values() if not e.task.done()]
            if not active:
                return ToolResult("No active timers to cancel.")
            if target_id == "all" or (not target_id and len(active) == 1):
                to_cancel = active if target_id == "all" else [active[0]]
                for e in to_cancel:
                    e.task.cancel()
                    store.pop(e.id, None)
                if len(to_cancel) == 1:
                    e = to_cancel[0]
                    return ToolResult(f"Cancelled {e.id} ('{e.label}').")
                return ToolResult(f"Cancelled all {len(to_cancel)} active timers.")
            if not target_id:
                ids = ", ".join(f"{e.id} ('{e.label}')" for e in active)
                return ToolResult(
                    f"error: multiple timers active ({ids}); specify 'timer_id' or 'all'",
                    is_error=True,
                )
            entry = store.pop(target_id, None)
            if entry is None or entry.task.done():
                return ToolResult(f"No active timer found with ID '{target_id}'.")
            entry.task.cancel()
            return ToolResult(f"Cancelled {entry.id} ('{entry.label}').")

        # action == "set"
        raw_seconds = args.get("seconds")
        if raw_seconds is None or isinstance(raw_seconds, bool):
            return ToolResult(
                "error: positive 'seconds' is required when action='set'",
                is_error=True,
            )
        try:
            seconds = float(raw_seconds)
        except (TypeError, ValueError):
            return ToolResult(
                "error: 'seconds' must be a positive number",
                is_error=True,
            )
        if seconds <= 0:
            return ToolResult(
                "error: 'seconds' must be greater than 0",
                is_error=True,
            )

        label = str(args.get("label") or "timer").strip() or "timer"
        timer_id = self._next_id()
        dur_str = _fmt_seconds(seconds)
        runtime_ref = self.runtime

        async def _run_timer() -> None:
            try:
                await asyncio.sleep(seconds)
            except asyncio.CancelledError:
                return
            finally:
                store.pop(timer_id, None)

            msg = f"Timer '{label}' ({dur_str}) finished."
            if runtime_ref is not None:
                on_done = getattr(runtime_ref, "_on_task_finished", None)
                if callable(on_done):
                    on_done(timer_id, msg)
                    return
                from verbix.runtime.workers import finish_runtime_task

                finish_runtime_task(runtime_ref, timer_id, msg)

        task = asyncio.create_task(_run_timer())
        store[timer_id] = _TimerEntry(
            id=timer_id,
            label=label,
            duration=seconds,
            fire_at=time.time() + seconds,
            task=task,
        )
        return ToolResult(
            f"Set {timer_id} ('{label}') for {dur_str}. Verbix will announce automatically when time is up."
        )
