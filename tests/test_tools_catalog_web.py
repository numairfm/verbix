from __future__ import annotations

import pytest

from verbix.runtime.workers import build_worker_registry
from verbix.tools import TOOL_CATALOG, build_registry
from verbix.tools.base import ToolResult


def test_web_tools_in_catalog():
    assert "web_search" in TOOL_CATALOG
    assert "web_fetch" in TOOL_CATALOG


def test_build_registry_instantiates_web_tools(tmp_path):
    registry = build_registry(["web_search", "web_fetch"], workspace=tmp_path)
    definitions = registry.get_definitions()
    names = {d["function"]["name"] for d in definitions}
    assert "web_search" in names
    assert "web_fetch" in names


def test_build_worker_registry_includes_web_tools(tmp_path):
    registry = build_worker_registry(workspace=tmp_path)
    definitions = registry.get_definitions()
    names = {d["function"]["name"] for d in definitions}
    assert "web_search" in names
    assert "web_fetch" in names


@pytest.mark.asyncio
async def test_build_registry_wraps_web_tools_with_bind_output(tmp_path, monkeypatch):
    from verbix.tools.web_search import WebSearchTool

    long_output = "\n".join(f"result {i}" for i in range(100))

    async def mock_execute(self, args, ctx):
        return ToolResult(observation=long_output)

    monkeypatch.setattr(WebSearchTool, "execute", mock_execute)

    registry = build_registry(
        ["web_search"],
        workspace=tmp_path,
        output_limits=(5, 100),
    )
    res = await registry.aexecute("web_search", {"query": "test"})
    assert "full content saved to" in res.observation
