from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field


class IPCRequest(BaseModel):
    id: str = Field(default_factory=lambda: "req-0")
    action: str
    prompt: str | None = None
    task_id: str | None = None
    instruction: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class IPCResponse(BaseModel):
    id: str | None = None
    status: str = "ok"
    error: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    def to_jsonl(self) -> bytes:
        return (json.dumps(self.model_dump(exclude_none=True)) + "\n").encode("utf-8")


class IPCEvent(BaseModel):
    type: str
    request_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_jsonl(self) -> bytes:
        return (json.dumps(self.model_dump(exclude_none=True)) + "\n").encode("utf-8")
