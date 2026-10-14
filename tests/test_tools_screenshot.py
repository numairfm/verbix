from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from verbix.tools.base import ToolContext
from verbix.tools.screenshot import ScreenshotTool


@pytest.fixture
def tool_context(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


def test_screenshot_tool_metadata() -> None:
    tool = ScreenshotTool()
    assert tool.name == "screenshot"
    assert tool.side_effecting is False
    assert tool.parameters_schema == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    assert "screenshot" in tool.description.lower()


@pytest.mark.asyncio
async def test_screenshot_success(tool_context: ToolContext) -> None:
    synthetic_im = Image.new("RGB", (1920, 1080), color="blue")
    with patch("PIL.ImageGrab.grab", return_value=synthetic_im):
        tool = ScreenshotTool()
        result = await tool.execute({}, tool_context)

    assert not result.is_error
    assert "[screenshot captured: 1920x1080]" in result.observation
    assert len(result.attachments) == 1
    att = result.attachments[0]
    assert att.mime == "image/jpeg"
    assert att.name == "screenshot.jpg"

    # Verify attachment is a valid readable JPEG
    with Image.open(io.BytesIO(att.data)) as im:
        assert im.format == "JPEG"
        assert im.size == (1568, 882)


@pytest.mark.asyncio
async def test_screenshot_resizes_4k_display(tool_context: ToolContext) -> None:
    synthetic_4k = Image.new("RGB", (3840, 2160), color="red")
    with patch("PIL.ImageGrab.grab", return_value=synthetic_4k):
        tool = ScreenshotTool()
        result = await tool.execute({}, tool_context)

    assert not result.is_error
    assert "[screenshot captured: 3840x2160]" in result.observation
    att = result.attachments[0]
    with Image.open(io.BytesIO(att.data)) as im:
        assert im.size == (1568, 882)


@pytest.mark.asyncio
async def test_screenshot_handles_rgba_transparency(tool_context: ToolContext) -> None:
    synthetic_rgba = Image.new("RGBA", (800, 600), color=(0, 255, 0, 128))
    with patch("PIL.ImageGrab.grab", return_value=synthetic_rgba):
        tool = ScreenshotTool()
        result = await tool.execute({}, tool_context)

    assert not result.is_error
    att = result.attachments[0]
    with Image.open(io.BytesIO(att.data)) as im:
        assert im.format == "JPEG"
        assert im.size == (800, 600)


@pytest.mark.asyncio
async def test_screenshot_handles_headless_oserror(tool_context: ToolContext) -> None:
    with patch("PIL.ImageGrab.grab", side_effect=OSError("no display found")):
        tool = ScreenshotTool()
        result = await tool.execute({}, tool_context)

    assert result.is_error
    assert "error: could not capture screenshot: no display found" in result.observation
    assert len(result.attachments) == 0
