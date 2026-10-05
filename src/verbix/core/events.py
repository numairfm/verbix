from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
class ToolCallDelta(AgentEvent):
    index: int = 0
    call_id: str | None = None
    name: str | None = None
    arguments_delta: str = ""


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
