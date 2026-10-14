from __future__ import annotations

import asyncio
import io
from typing import Any

from PIL import Image, ImageGrab

from verbix.tools.base import FileAttachment, ToolContext, ToolResult


class ScreenshotTool:
    """Captures the user's desktop screen and attaches it as a compressed JPEG."""

    name: str = "screenshot"
    side_effecting: bool = False
    description: str = (
        "Captures a screenshot of the user's desktop screen and attaches it as an image. "
        "Use this to inspect visual output, UI layouts, browser rendering, or desktop errors."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    async def execute(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        return await asyncio.to_thread(self._capture)

    def _capture(self) -> ToolResult:
        try:
            im = ImageGrab.grab()
            orig_w, orig_h = im.size
            rgba = im.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[3])
            bg.thumbnail((1568, 1568), Image.LANCZOS)
            buf = io.BytesIO()
            bg.save(buf, format="JPEG", quality=82)
            return ToolResult(
                f"[screenshot captured: {orig_w}x{orig_h}]",
                attachments=(
                    FileAttachment(
                        data=buf.getvalue(),
                        mime="image/jpeg",
                        name="screenshot.jpg",
                    ),
                ),
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"error: could not capture screenshot: {e}", is_error=True)
