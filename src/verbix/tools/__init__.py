from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from verbix.tools.base import (
    Tool,
    ToolContext,
    ToolResult,
    parse_lenient_json,
)
from verbix.tools.bash import BashTool
from verbix.tools.cancel import CancelTool
from verbix.tools.edit import FileEditTool
from verbix.tools.finish import FinishTool
from verbix.tools.read import FileReadTool
from verbix.tools.sandbox import SandboxConfig, SandboxResult, execute_sandboxed
from verbix.tools.shared import resolve_contained_path
from verbix.tools.spawn import SpawnTool
from verbix.tools.status import StatusTool
from verbix.tools.write import FileWriteTool

if TYPE_CHECKING:
    from verbix.agent.tools.registry import ToolRegistry

__all__ = [
    "TOOL_CATALOG",
    "BashTool",
    "CancelTool",
    "FileEditTool",
    "FileReadTool",
    "FileWriteTool",
    "FinishTool",
    "SandboxConfig",
    "SandboxResult",
    "SpawnTool",
    "StatusTool",
    "Tool",
    "ToolContext",
    "ToolResult",
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
    "spawn": SpawnTool,
    "status": StatusTool,
    "cancel": CancelTool,
    "finish": lambda runtime: FinishTool(),
}


def build_registry(
    names: list[str], runtime: Any = None, workspace: Path | None = None
) -> ToolRegistry:
    """Build a registry for `names` sharing one session-scoped ToolContext."""
    from verbix.agent.tools.registry import ToolRegistry

    unknown = [name for name in names if name not in TOOL_CATALOG]
    if unknown:
        raise KeyError(f"Unknown tool(s) {unknown}; valid tools: {sorted(TOOL_CATALOG)}")
    ctx = ToolContext(workspace=Path.cwd() if workspace is None else Path(workspace))
    registry = ToolRegistry()

    def _make_handler(t: Any) -> Callable[..., Any]:
        async def _handler(args: dict[str, Any]) -> ToolResult:
            return await t.execute(args, ctx)

        _handler.observation_cap = getattr(t, "observation_cap", 10000)  # type: ignore[attr-defined]
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
