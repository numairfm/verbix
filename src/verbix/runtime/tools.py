from __future__ import annotations

from typing import TYPE_CHECKING, Any

from verbix.tools.base import ToolContext, ToolResult

if TYPE_CHECKING:
    from verbix.runtime.runtime import Runtime


class SpawnTool:
    """Spawns an autonomous background worker to perform a task."""

    name: str = "spawn"
    description: str = "Starts an autonomous background worker to execute a task."

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "The complete task for the worker to execute, preserving all clauses and requested phrasing.",
                },
            },
            "required": ["objective"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        objective = str(args.get("objective", "")).strip()
        if not objective:
            return ToolResult("error: 'objective' is required to spawn work.", is_error=True)

        task_id = await self.runtime.spawn(objective)
        return ToolResult(
            f"{task_id} accepted. Confirm starting in one short sentence and call no further tools."
        )


class StatusTool:
    """Inspects active or completed background tasks."""

    name: str = "status"
    description: str = (
        "Inspects active background tasks. If task_id is omitted, lists all active tasks. "
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
                    "description": "Optional ID of the task to inspect. Omit to list active tasks.",
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


class CancelTool:
    """Cancels one or more background tasks."""

    name: str = "cancel"
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
