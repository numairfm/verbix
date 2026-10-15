from __future__ import annotations

import asyncio
import io
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageGrab

from verbix.tools.base import FileAttachment, ToolContext, ToolResult


def _prune_screenshots(cache_dir: Path, max_age_days: float = 14) -> None:
    """Silently prune cached screenshot files older than max_age_days."""
    try:
        cutoff = time.time() - max_age_days * 86400
        for p in cache_dir.glob("screenshot_*.png"):
            try:
                if p.is_file() and p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
            except OSError:
                continue
    except OSError:
        pass


class ScreenshotTool:
    """Captures the user's desktop screen and attaches it directly as a compressed JPEG."""

    name: str = "screenshot"
    side_effecting: bool = False
    description: str = (
        "Captures a screenshot of the user's desktop screen and attaches the image directly to this tool result "
        "(no 'read' call needed). Use this to inspect visual output, UI layouts, browser rendering, or desktop state."
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
            ts = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S")
            im = ImageGrab.grab()
            orig_w, orig_h = im.size

            cache_dir = Path.home() / ".verbix" / "screenshots"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached_file = cache_dir / f"screenshot_{ts}.png"
            im.save(cached_file, format="PNG")
            _prune_screenshots(cache_dir, max_age_days=14)

            observation = f"[screenshot captured: {orig_w}x{orig_h} (cached)]"

            rgba = im.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[3])
            bg.thumbnail((1568, 1568), Image.LANCZOS)
            buf = io.BytesIO()
            bg.save(buf, format="JPEG", quality=82)
            return ToolResult(
                observation,
                attachments=(
                    FileAttachment(
                        data=buf.getvalue(),
                        mime="image/jpeg",
                        name="screenshot.jpg",
                    ),
                ),
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(
                f"error: could not capture screenshot: {e}", is_error=True
            )
