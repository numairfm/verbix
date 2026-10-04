from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class CompletionBuffer:
    """Debounces worker task completions over a sliding window before flushing."""

    def __init__(
        self,
        window_seconds: float = 0.3,
        on_flush: Callable[[list[tuple[str, str]]], Coroutine[Any, Any, None] | None] | None = None,
    ) -> None:
        self.window_seconds = window_seconds
        self.on_flush = on_flush
        self._pending: list[tuple[str, str]] = []
        self._flush_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    def push(self, task_id: str, result: str) -> None:
        """Pushes a completed task result into the pending buffer and resets the window."""
        self._pending.append((task_id, result))
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        self._flush_task = asyncio.create_task(self._delayed_flush())

    async def _delayed_flush(self) -> None:
        try:
            await asyncio.sleep(self.window_seconds)
            await self.flush_now()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Error in completion buffer flush")

    async def flush_now(self) -> None:
        """Immediately flushes any pending completions to the registered callback."""
        async with self._lock:
            if not self._pending:
                return
            batch = list(self._pending)
            self._pending.clear()

        if self.on_flush:
            try:
                res = self.on_flush(batch)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                logger.exception("Error in on_flush handler")
