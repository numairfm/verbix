from __future__ import annotations

import asyncio
from typing import Any

import pyperclip

from verbix.tools.base import ToolContext, ToolResult


class ClipboardTool:
    """Reads or writes the system clipboard via pyperclip."""

    name: str = "clipboard"
    side_effecting: bool = True
    description: str = (
        "Reads from or writes text to the system clipboard. "
        "Use action='read' to inspect copied text or action='write' with 'text' to copy text."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write"],
                    "description": "Clipboard operation: 'read' or 'write'.",
                },
                "text": {
                    "type": "string",
                    "description": "Text to copy to the clipboard when action='write'.",
                },
            },
            "required": ["action"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        action = (args.get("action") or "").strip().lower()
        if action not in ("read", "write"):
            return ToolResult(
                f"error: invalid action '{action}'; expected 'read' or 'write'",
                is_error=True,
            )
        if action == "write":
            if "text" not in args or args.get("text") is None:
                return ToolResult(
                    "error: 'text' is required when action='write'",
                    is_error=True,
                )
            text = str(args["text"])
            try:
                await asyncio.to_thread(pyperclip.copy, text)
                return ToolResult(f"Copied {len(text)} characters to clipboard.")
            except (pyperclip.PyperclipException, OSError, RuntimeError) as e:
                return ToolResult(f"error: clipboard write failed: {e}", is_error=True)

        try:
            content = await asyncio.to_thread(pyperclip.paste)
            return ToolResult(content if content else "(clipboard is empty)")
        except (pyperclip.PyperclipException, OSError, RuntimeError) as e:
            return ToolResult(f"error: clipboard read failed: {e}", is_error=True)
