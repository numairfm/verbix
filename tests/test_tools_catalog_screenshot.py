from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from verbix.runtime.workers import build_worker_registry
from verbix.tools import TOOL_CATALOG, build_registry
from verbix.tools.base import ToolResult
from verbix.tools.screenshot import ScreenshotTool


def test_screenshot_in_catalog():
    import verbix.tools

    assert "ScreenshotTool" in verbix.tools.__all__
    assert "screenshot" in TOOL_CATALOG
    tool = TOOL_CATALOG["screenshot"](None)
    assert isinstance(tool, ScreenshotTool)


def test_build_registry_instantiates_screenshot(tmp_path):
    registry = build_registry(["screenshot"], workspace=tmp_path)
    definitions = registry.get_definitions()
    names = {d["function"]["name"] for d in definitions}
    assert "screenshot" in names
    assert "screenshot" in registry._handlers


def test_build_worker_registry_includes_screenshot(tmp_path):
    registry = build_worker_registry(workspace=tmp_path)
    definitions = registry.get_definitions()
    names = {d["function"]["name"] for d in definitions}
    assert "screenshot" in names


@pytest.mark.asyncio
async def test_build_registry_wraps_screenshot_with_bind_output(tmp_path, monkeypatch):
    long_output = "\n".join(f"screenshot line {i}" for i in range(100))

    async def mock_execute(self, args, ctx):
        return ToolResult(observation=long_output)

    monkeypatch.setattr(ScreenshotTool, "execute", mock_execute)

    registry = build_registry(
        ["screenshot"],
        workspace=tmp_path,
        output_limits=(5, 100),
    )
    res = await registry.aexecute("screenshot", {})
    assert "full content saved to" in res.observation


def test_worker_yaml_config_includes_screenshot():
    config_path = Path("src/verbix/agent/config/worker.yaml")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tools = data["agent"].get("tools", [])
    assert "screenshot" in tools

    prompt = data["agent"].get("instance_template", "")
    assert "screenshot" in prompt
