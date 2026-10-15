from __future__ import annotations

import asyncio
import io
from typing import Any

import pyperclip
from PIL import Image, ImageGrab

from verbix.tools.base import FileAttachment, ToolContext, ToolResult


def _grab_clipboard_image() -> ToolResult | None:
    try:
        grabbed = ImageGrab.grabclipboard()
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(grabbed, Image.Image):
        return None
    w, h = grabbed.size
    rgba = grabbed.convert("RGBA")
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    bg.thumbnail((1568, 1568), Image.LANCZOS)
    buf = io.BytesIO()
    bg.save(buf, format="JPEG", quality=82)
    return ToolResult(
        f"[clipboard image attached: {w}x{h}]",
        attachments=(
            FileAttachment(
                data=buf.getvalue(),
                mime="image/jpeg",
                name="clipboard.jpg",
            ),
        ),
    )


class ClipboardTool:
    """Reads or writes text (and reads copied images) from the system clipboard."""

    name: str = "clipboard"
    side_effecting: bool = True
    description: str = (
        "Reads from or writes text to the system clipboard (and attaches copied images when reading). "
        "Use action='read' to inspect copied text or images, or action='write' with 'text' to copy text."
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
            if content:
                return ToolResult(content)
            img_res = await asyncio.to_thread(_grab_clipboard_image)
            if img_res is not None:
                return img_res
            return ToolResult("(clipboard is empty)")
        except (pyperclip.PyperclipException, OSError, RuntimeError) as e:
            img_res = await asyncio.to_thread(_grab_clipboard_image)
            if img_res is not None:
                return img_res
            return ToolResult(f"error: clipboard read failed: {e}", is_error=True)
