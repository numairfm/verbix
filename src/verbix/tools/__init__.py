from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from verbix.tools.app import AppTool
from verbix.tools.base import (
    Tool,
    ToolContext,
    ToolResult,
    parse_lenient_json,
)
from verbix.tools.bash import BashTool
from verbix.tools.cancel import CancelTool
from verbix.tools.clipboard import ClipboardTool
from verbix.tools.edit import FileEditTool
from verbix.tools.glob import GlobTool
from verbix.tools.grep import GrepTool
from verbix.tools.media import MediaTool
from verbix.tools.memory import MemoryTool
from verbix.tools.notify import NotifyTool
from verbix.tools.open import OpenTool
from verbix.tools.read import FileReadTool
from verbix.tools.sandbox import SandboxConfig, SandboxResult, execute_sandboxed
from verbix.tools.screenshot import ScreenshotTool
from verbix.tools.shared import bind_output, resolve_contained_path
from verbix.tools.spawn import SpawnTool
from verbix.tools.status import StatusTool
from verbix.tools.system import SystemTool
from verbix.tools.timer import TimerTool
from verbix.tools.web_fetch import WebFetchTool
from verbix.tools.web_search import WebSearchTool
from verbix.tools.write import FileWriteTool

if TYPE_CHECKING:
    from verbix.agent.tools.registry import ToolRegistry

__all__ = [
    "TOOL_CATALOG",
    "AppTool",
    "BashTool",
    "CancelTool",
    "ClipboardTool",
    "FileEditTool",
    "FileReadTool",
    "FileWriteTool",
    "GlobTool",
    "GrepTool",
    "MediaTool",
    "MemoryTool",
    "NotifyTool",
    "OpenTool",
    "SandboxConfig",
    "SandboxResult",
    "ScreenshotTool",
    "SpawnTool",
    "StatusTool",
    "SystemTool",
    "TimerTool",
    "Tool",
    "ToolContext",
    "ToolResult",
    "WebFetchTool",
    "WebSearchTool",
    "build_registry",
    "execute_sandboxed",
    "parse_lenient_json",
    "resolve_contained_path",
]

TOOL_CATALOG: dict[str, Callable[[Any], Any]] = {
    "read": lambda runtime: FileReadTool(),
    "write": lambda runtime: FileWriteTool(),
    "edit": lambda runtime: FileEditTool(),
    "bash": lambda runtime: BashTool(),
    "grep": lambda runtime: GrepTool(),
    "glob": lambda runtime: GlobTool(),
    "web_search": lambda runtime: WebSearchTool(),
    "web_fetch": lambda runtime: WebFetchTool(),
    "screenshot": lambda runtime: ScreenshotTool(),
    "clipboard": lambda runtime: ClipboardTool(),
    "media": lambda runtime: MediaTool(),
    "app": lambda runtime: AppTool(),
    "open": lambda runtime: OpenTool(),
    "system": lambda runtime: SystemTool(),
    "timer": TimerTool,
    "memory": lambda runtime: MemoryTool(),
    "notify": lambda runtime: NotifyTool(),
    "spawn": SpawnTool,
    "status": StatusTool,
    "cancel": CancelTool,
}


def build_registry(
    names: list[str],
    runtime: Any = None,
    workspace: Path | None = None,
    output_limits: tuple[int, int] | None = None,
) -> ToolRegistry:
    """Build a registry for `names` sharing one session-scoped ToolContext."""
    from dataclasses import replace

    from verbix.agent.tools.registry import ToolRegistry
    from verbix.config import default_workspace

    unknown = [name for name in names if name not in TOOL_CATALOG]
    if unknown:
        raise KeyError(f"Unknown tool(s) {unknown}; valid tools: {sorted(TOOL_CATALOG)}")
    if output_limits is None:
        output_limits = (2000, 51200)
    max_lines, max_bytes = output_limits
    ws = default_workspace() if workspace is None else Path(workspace)
    ctx = ToolContext(workspace=ws)
    spill_dir = Path.home() / ".verbix" / "logs"
    registry = ToolRegistry()

    def _make_handler(t: Any) -> Callable[..., Any]:
        async def _handler(args: dict[str, Any]) -> ToolResult:
            res = await t.execute(args, ctx)
            bound = bind_output(
                res.observation,
                max_lines=max_lines,
                max_bytes=max_bytes,
                spill_dir=spill_dir,
                rel_to=ws,
            )
            return replace(res, observation=bound)

        return _handler

    for name in names:
        tool = TOOL_CATALOG[name](runtime)
        registry.register(
            name,
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema,
                },
            },
            _make_handler(tool),
            side_effecting=getattr(tool, "side_effecting", False),
        )
    return registry
