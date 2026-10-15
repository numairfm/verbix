from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from enum import Enum


class FloorState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    SPEAKING = "SPEAKING"

    # Backward compatibility aliases
    FOREGROUND = "SPEAKING"  # noqa: PIE796 - intentional alias for backward compatibility
    ANNOUNCING = "SPEAKING"  # noqa: PIE796 - intentional alias for backward compatibility


class FloorArbiter:
    """Three-state floor arbiter with synchronous interrupt and observable transitions."""

    def __init__(
        self,
        on_change: Callable[[FloorState, FloorState], None] | None = None,
    ) -> None:
        self._state = FloorState.IDLE
        self._lock = asyncio.Lock()
        self.interrupt_requested = False
        self.user_speaking = False
        self.continuous_mode = False
        self.on_change = on_change

    @property
    def state(self) -> FloorState:
        return self._state

    @property
    def is_idle(self) -> bool:
        return self._state == FloorState.IDLE

    @property
    def is_listening(self) -> bool:
        return self._state == FloorState.LISTENING

    @property
    def is_speaking(self) -> bool:
        return self._state == FloorState.SPEAKING

    def _normalize_state(self, state: FloorState | str) -> FloorState:
        if isinstance(state, FloorState):
            return state
        s = str(state).strip().upper()
        if s in ("FOREGROUND", "ANNOUNCING"):
            return FloorState.SPEAKING
        return FloorState(s)

    def set_state(self, state: FloorState | str) -> None:
        target = self._normalize_state(state)
        if target != self._state:
            prev = self._state
            self._state = target
            if self.on_change is not None:
                self.on_change(prev, target)

    def request_interrupt(self) -> None:
        self.interrupt_requested = True

    def set_user_speaking(self, active: bool) -> None:
        prev = self.user_speaking
        self.user_speaking = bool(active)
        if not self.user_speaking:
            self.interrupt_requested = False
        if prev and not self.user_speaking and self.on_change is not None:
            self.on_change(self._state, self._state)

    def can_announce(self) -> bool:
        if self.user_speaking or self.interrupt_requested:
            return False
        return self._state in (FloorState.IDLE, FloorState.LISTENING)

    @asynccontextmanager
    async def acquire(self, state: FloorState | str):
        target = self._normalize_state(state)
        async with self._lock:
            prev = self._state
            self.set_state(target)
            self.interrupt_requested = False
            try:
                yield self
            finally:
                if (
                    prev == FloorState.LISTENING or self._state == FloorState.LISTENING
                ) and not self.user_speaking:
                    self.set_state(FloorState.LISTENING)
                else:
                    self.set_state(FloorState.IDLE)
