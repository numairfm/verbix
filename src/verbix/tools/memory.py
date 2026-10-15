from __future__ import annotations

import asyncio
import re
from typing import Any

from verbix.config import _GUIDANCE_BLOCK, _MEMORY_TEMPLATE, memory_file
from verbix.tools.base import ToolContext, ToolResult

_SECRET_RE = re.compile(
    r"([A-Za-z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)\s*=\s*\S+|\bsk-[A-Za-z0-9_-]{8,}|['\"]api_key['\"]\s*:)",
    re.IGNORECASE,
)


def _load_bullets() -> list[str]:
    path = memory_file()
    if path.parent.is_symlink() or path.is_symlink():
        raise ValueError("refusing to access symlinked MEMORY.md")
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8", errors="replace")
    body = _GUIDANCE_BLOCK.sub("", raw, count=1)
    bullets: list[str] = []
    for line in body.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(("- ", "* ")):
            s = s[2:].strip()
        if s:
            bullets.append(s)
    return bullets


def _save_bullets(bullets: list[str]) -> None:
    path = memory_file()
    if path.parent.is_symlink() or path.is_symlink():
        raise ValueError("refusing to access symlinked MEMORY.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"- {b}" for b in bullets)
    content = _MEMORY_TEMPLATE + (body + "\n" if body else "")
    path.write_text(content, encoding="utf-8")


class MemoryTool:
    """Saves, removes, and lists durable user facts and preferences in ~/.verbix/MEMORY.md."""

    name: str = "memory"
    side_effecting: bool = True
    description: str = (
        "Saves, removes, or lists durable user facts and preferences in ~/.verbix/MEMORY.md "
        "(actions: 'remember', 'forget', 'list'). Only call when the user asks to remember, update, or forget something."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["remember", "forget", "list"],
                    "description": "Memory action: 'remember' to save a fact, 'forget' to remove matching facts, or 'list' to view saved facts.",
                },
                "text": {
                    "type": "string",
                    "description": "Fact or preference to save (when action='remember') or substring to remove (when action='forget').",
                },
            },
            "required": ["action"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        return await asyncio.to_thread(self._execute_sync, args)

    def _execute_sync(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "").strip().lower()
        if action not in ("remember", "forget", "list"):
            return ToolResult(
                f"error: invalid action '{action}'; expected 'remember', 'forget', or 'list'",
                is_error=True,
            )

        try:
            bullets = _load_bullets()
        except (OSError, ValueError) as e:
            return ToolResult(f"error: {e}", is_error=True)

        if action == "list":
            if not bullets:
                return ToolResult("(memory is empty)")
            return ToolResult("\n".join(f"- {b}" for b in bullets))

        raw_text = str(args.get("text") or "").strip()
        if not raw_text:
            return ToolResult(
                f"error: 'text' is required when action='{action}'",
                is_error=True,
            )

        if action == "remember":
            cleaned = " ".join(raw_text.split()).lstrip("-*#[ ").strip()
            if not cleaned:
                return ToolResult("error: memory entry cannot be empty", is_error=True)
            if len(cleaned) > 500:
                return ToolResult(
                    "error: memory entry exceeds 500 characters", is_error=True
                )
            if _SECRET_RE.search(cleaned):
                return ToolResult(
                    "error: refusing to store API keys or secrets in MEMORY.md",
                    is_error=True,
                )
            if any(cleaned.lower() == b.lower() for b in bullets):
                return ToolResult(f"Already in memory: '{cleaned}'.")
            bullets.append(cleaned)
            try:
                _save_bullets(bullets)
            except (OSError, ValueError) as e:
                return ToolResult(f"error: {e}", is_error=True)
            return ToolResult(f"Remembered: '{cleaned}'.")

        # action == "forget"
        needle = " ".join(raw_text.split()).lower()
        kept = [b for b in bullets if needle not in b.lower()]
        removed = [b for b in bullets if needle in b.lower()]
        if not removed:
            return ToolResult(f"No memory entry matching '{raw_text}' found.")
        try:
            _save_bullets(kept)
        except (OSError, ValueError) as e:
            return ToolResult(f"error: {e}", is_error=True)
        return ToolResult(
            f"Forgot {len(removed)} entry(s): {', '.join(repr(r) for r in removed)}."
        )
