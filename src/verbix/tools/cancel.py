from __future__ import annotations

from typing import TYPE_CHECKING, Any

from verbix.tools.base import ToolContext, ToolResult

if TYPE_CHECKING:
    from verbix.runtime.runtime import Runtime


class CancelTool:
    """Cancels one or more background tasks."""

    name: str = "cancel"
    observation_cap: int = 1500
    description: str = (
        "Cancels background tasks. If task_id is omitted and only one task is running, "
        "cancels it. Pass 'all' to cancel all tasks."
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
                    "description": "Optional task ID to cancel, or 'all' to cancel all running tasks.",
                },
            },
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        raw_id = args.get("task_id")
        task_id = str(raw_id).strip() if raw_id else None

        result_msg = await self.runtime.cancel(task_id=task_id)
        return ToolResult(result_msg)
