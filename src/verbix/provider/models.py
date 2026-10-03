from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class StreamEvent:
    pass


@dataclass(frozen=True)
class TextDelta(StreamEvent):
    text: str


@dataclass(frozen=True)
class ThoughtDelta(StreamEvent):
    thought: str


@dataclass(frozen=True)
class ToolCallDelta(StreamEvent):
    index: int
    call_id: str | None = None
    name: str | None = None
    arguments_delta: str = ""


@dataclass(frozen=True)
class Usage(StreamEvent):
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class Finish(StreamEvent):
    reason: str


class ToolFunction(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    type: str = "function"
    function: ToolFunction
