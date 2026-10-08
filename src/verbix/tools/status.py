from __future__ import annotations

from typing import TYPE_CHECKING, Any

from verbix.tools.base import ToolContext, ToolResult

if TYPE_CHECKING:
    from verbix.runtime.runtime import Runtime


class StatusTool:
    """Inspects active or completed background tasks."""

    name: str = "status"
    observation_cap: int = 1500
    description: str = (
        "Inspects background tasks. If task_id is omitted, lists tasks. "
        "Set full=true to inspect complete history."
    )

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Optional ID of the task to inspect. Omit to list tasks.",
                },
                "full": {
                    "type": "boolean",
                    "description": "Whether to return the full transcript rather than recent actions.",
                },
            },
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        raw_id = args.get("task_id")
        task_id = str(raw_id).strip() if raw_id else None
        full = bool(args.get("full", False))

        summary = self.runtime.status(task_id=task_id, full=full)
        return ToolResult(summary)
