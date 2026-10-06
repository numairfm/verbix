from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from enum import Enum


class FloorState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    FOREGROUND = "foreground"
    ANNOUNCING = "announcing"


class FloorArbiter:
    """Single floor owner (F1); synchronous interrupt flag (F3)."""

    def __init__(self) -> None:
        self._state = FloorState.IDLE
        self._lock = asyncio.Lock()
        self.interrupt_requested = False

    @property
    def state(self) -> FloorState:
        return self._state

    def request_interrupt(self) -> None:
        self.interrupt_requested = True

    @asynccontextmanager
    async def acquire(self, state: FloorState):
        async with self._lock:
            self._state = state
            self.interrupt_requested = False
            try:
                yield self
            finally:
                self._state = FloorState.IDLE
