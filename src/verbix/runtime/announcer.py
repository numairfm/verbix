from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from verbix.runtime.floor import FloorArbiter, FloorState
from verbix.runtime.ledger import TaskLedger

logger = logging.getLogger(__name__)


class AnnouncerLoop:
    """Non-cancelling persistent queue loop replacing CompletionBuffer.

    push() only enqueues; in-flight announcements are never cancelled,
    so late arrivals wait safely for the next idle window.
    """

    def __init__(
        self,
        floor: FloorArbiter,
        ledger: TaskLedger,
        speak_fn: Callable[[list[str]], Any],
        batch_window: float = 0.3,
    ) -> None:
        self.floor = floor
        self.ledger = ledger
        self.speak_fn = speak_fn
        self.batch_window = batch_window
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    def push(self, task_id: str) -> None:
        self._queue.put_nowait(task_id)
        self.start()  # No-op without a running loop; retried on next push.

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # Sync context; push() starts it once a loop runs.
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                first = await self._queue.get()
            except asyncio.CancelledError:
                break
            batch = [first]
            try:
                await asyncio.sleep(self.batch_window)
                while not self._queue.empty():
                    batch.append(self._queue.get_nowait())
                while self.floor.state != FloorState.IDLE:
                    await asyncio.sleep(0.01)
                async with self.floor.acquire(FloorState.ANNOUNCING):
                    try:
                        res = self.speak_fn(batch)
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception:
                        logger.exception("Announcer speak_fn failed")
                        continue
                    for task_id in batch:
                        self.ledger.mark_delivered(task_id)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in announcer loop")
