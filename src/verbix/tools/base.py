from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolResult:
    observation: str
    is_error: bool = False
    is_terminal: bool = False


@dataclass(frozen=True)
class ToolContext:
    """Workspace plus the per-session file-version log used to detect stale writes.

    ``file_versions`` maps an absolute resolved path to the sha256 of the bytes
    this session last observed for it (via ``read`` or its own writes).
    A session-scoped context lets ``write``/``edit`` refuse to clobber
    a change another worker made after this session last saw the file.
    """

    workspace: Path
    file_versions: dict[str, str] = field(
        default_factory=dict, compare=False, hash=False
    )


class Tool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters_schema(self) -> dict[str, Any]: ...

    async def execute(
        self, args: dict[str, Any], context: ToolContext
    ) -> ToolResult: ...


def parse_lenient_json(raw: str) -> dict[str, Any]:
    """Leniently parses JSON from raw model strings, stripping markdown code fences

    and surrounding prose to recover valid function arguments from weak models.
    """
    trimmed = raw.strip()

    # 1. Direct JSON parse
    try:
        val = json.loads(trimmed)
        if isinstance(val, dict):
            return val
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown code fences ```json ... ```
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", trimmed, re.DOTALL)
    if fence_match:
        try:
            val = json.loads(fence_match.group(1))
            if isinstance(val, dict):
                return val
        except json.JSONDecodeError:
            pass

    # 3. Find outer braces
    first_brace = trimmed.find("{")
    last_brace = trimmed.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = trimmed[first_brace : last_brace + 1]
        try:
            val = json.loads(candidate)
            if isinstance(val, dict):
                return val
        except json.JSONDecodeError:
            pass

    return {"_raw": raw}
