from __future__ import annotations

import io
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from verbix.tools.base import ToolContext
from verbix.tools.screenshot import ScreenshotTool


@pytest.fixture(autouse=True)
def isolate_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home_dir = tmp_path / "mock_home"
    home_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: home_dir)
    return home_dir


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
    assert "[screenshot captured: 1920x1080 (cached)]" in result.observation
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
    assert "[screenshot captured: 3840x2160 (cached)]" in result.observation
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


@pytest.mark.asyncio
async def test_screenshot_saves_to_global_cache(
    tool_context: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    synthetic_im = Image.new("RGB", (800, 600), color="green")
    with patch("PIL.ImageGrab.grab", return_value=synthetic_im):
        tool = ScreenshotTool()
        result = await tool.execute({}, tool_context)

    assert not result.is_error
    assert "[screenshot captured: 800x600 (cached)]" in result.observation
    cache_dir = fake_home / ".verbix" / "screenshots"
    assert cache_dir.is_dir()
    saved_files = list(cache_dir.glob("screenshot_*.png"))
    assert len(saved_files) == 1
    with Image.open(saved_files[0]) as im:
        assert im.format == "PNG"
        assert im.size == (800, 600)


@pytest.mark.asyncio
async def test_screenshot_prunes_old_cache_files(
    tool_context: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    cache_dir = fake_home / ".verbix" / "screenshots"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    old_file = cache_dir / "screenshot_20200101_000000.png"
    old_file.write_bytes(b"dummy")
    fifteen_days_ago = time.time() - (15 * 86400)
    os.utime(old_file, (fifteen_days_ago, fifteen_days_ago))

    synthetic_im = Image.new("RGB", (800, 600), color="blue")
    with patch("PIL.ImageGrab.grab", return_value=synthetic_im):
        tool = ScreenshotTool()
        result = await tool.execute({}, tool_context)

    assert not result.is_error
    assert not old_file.exists()
    new_files = list(cache_dir.glob("screenshot_*.png"))
    assert len(new_files) == 1
    assert new_files[0] != old_file


@pytest.mark.asyncio
async def test_screenshot_never_writes_into_workspace(
    tool_context: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    synthetic_im = Image.new("RGB", (640, 480), color="yellow")
    with patch("PIL.ImageGrab.grab", return_value=synthetic_im):
        tool = ScreenshotTool()
        result = await tool.execute({}, tool_context)

    assert not result.is_error
    assert list(tmp_path.glob("*.png")) == []


@pytest.mark.asyncio
async def test_screenshot_handles_file_save_oserror(
    tool_context: ToolContext,
) -> None:
    synthetic_im = Image.new("RGB", (100, 100), color="white")
    with patch("PIL.ImageGrab.grab", return_value=synthetic_im), patch.object(
        Image.Image, "save", side_effect=OSError("disk full")
    ):
        tool = ScreenshotTool()
        result = await tool.execute({}, tool_context)

    assert result.is_error
    assert "could not capture screenshot: disk full" in result.observation
