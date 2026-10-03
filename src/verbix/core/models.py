from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]
TaskStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


class CanonicalToolCall(BaseModel):
    id: str
    name: str
    arguments: str

    def parsed_arguments(self) -> dict[str, Any]:
        """Parses arguments JSON string into a Python dictionary."""
        try:
            val = json.loads(self.arguments)
            if isinstance(val, dict):
                return val
            return {"_value": val}
        except (json.JSONDecodeError, TypeError):
            return {"_raw": self.arguments}


class CanonicalMessage(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[CanonicalToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    thought: str | None = None


class WorkerResult(BaseModel):
    success: bool
    speech_summary: str = Field(
        ...,
        max_length=180,
        description="Terse, natural spoken summary for TTS playback. Maximum 25-30 words. No markdown, no fences.",
    )
    details: str = Field(
        ...,
        description="Comprehensive technical details, changed files, test output, diffs, or error logs for text/CLI display.",
    )


class TaskState(BaseModel):
    id: str
    title: str
    status: TaskStatus = "pending"
    step_count: int = 0
    recorded: bool = False
    announced: bool = False
    result: WorkerResult | None = None
    workspace: Any = None

    @property
    def is_running(self) -> bool:
        return self.status == "running"
