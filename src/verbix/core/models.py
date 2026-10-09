from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel

Role = Literal["system", "user", "assistant", "tool"]


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
    content: str | list[dict[str, Any]] = ""
    tool_calls: list[CanonicalToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    thought: str | None = None
