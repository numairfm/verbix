from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verbix.core.models import WorkerResult


@dataclass(frozen=True)
class AgentEvent:
    pass


@dataclass(frozen=True)
class TextDelta(AgentEvent):
    text: str


@dataclass(frozen=True)
class ThoughtDelta(AgentEvent):
    thought: str


@dataclass(frozen=True)
class ToolCallStarted(AgentEvent):
    call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolCallFinished(AgentEvent):
    call_id: str
    observation: str
    is_error: bool = False


@dataclass(frozen=True)
class TurnComplete(AgentEvent):
    final_text: str


@dataclass(frozen=True)
class TaskSpawnedEvent(AgentEvent):
    task_id: str
    title: str


@dataclass(frozen=True)
class TaskProgressEvent(AgentEvent):
    task_id: str
    step: int
    message: str


@dataclass(frozen=True)
class TaskFinishedEvent(AgentEvent):
    task_id: str
    result: WorkerResult
