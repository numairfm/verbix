"""Models package for Verbix agents."""

from __future__ import annotations

import threading


class GlobalModelStats:
    """Global model call and cost statistics tracker."""

    def __init__(self) -> None:
        self._cost = 0.0
        self._n_calls = 0
        self._lock = threading.Lock()

    def add(self, cost: float) -> None:
        """Record a model call with its cost."""
        with self._lock:
            self._cost += cost
            self._n_calls += 1

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def n_calls(self) -> int:
        return self._n_calls


GLOBAL_MODEL_STATS = GlobalModelStats()

from verbix.agent.models.litellm_model import (
    LitellmModel,
    LitellmModelConfig,
)

__all__ = [
    "GLOBAL_MODEL_STATS",
    "GlobalModelStats",
    "LitellmModel",
    "LitellmModelConfig",
]
