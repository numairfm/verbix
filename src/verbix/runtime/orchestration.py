from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from verbix.core.models import WorkerResult
from verbix.tools.base import ToolContext, ToolResult

if TYPE_CHECKING:
    from verbix.runtime.tasks import TaskRuntime


class WorkerFinishTool:
    """Tool used by autonomous workers to report structured finish results."""

    name: str = "worker_finish"
    description: str = (
        "Signals that the delegated task is complete. "
        "You MUST provide success status, a terse spoken summary for audio playback (max 180 chars, no markdown), "
        "and complete technical details."
    )

    def __init__(
        self,
        on_finish: Callable[[WorkerResult], None] | None = None,
    ) -> None:
        self.captured_result: WorkerResult | None = None
        self.on_finish = on_finish

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether the task succeeded.",
                },
                "speech_summary": {
                    "type": "string",
                    "description": "Terse, natural spoken summary for TTS playback. Maximum 25-30 words, 180 chars. No markdown, no fences.",
                },
                "details": {
                    "type": "string",
                    "description": "Comprehensive technical details, modified files, diffs, test output, or error messages for text logs.",
                },
            },
            "required": ["success", "speech_summary", "details"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        success = bool(args.get("success", True))
        raw_summary = str(args.get("speech_summary", "")).strip()
        # Enforce max length constraint gracefully
        speech_summary = (
            raw_summary[:180]
            if raw_summary
            else ("Task completed successfully." if success else "Task failed.")
        )
        details = (
            str(args.get("details", "")).strip() or "No additional details provided."
        )

        self.captured_result = WorkerResult(
            success=success,
            speech_summary=speech_summary,
            details=details,
        )
        if self.on_finish:
            self.on_finish(self.captured_result)
        return ToolResult(
            observation="Task finish reported and recorded successfully.",
            is_terminal=True,
        )


class TaskSpawnTool:
    """Orchestration tool for spawning background worker tasks."""

    name: str = "task_spawn"

    def __init__(self, runtime: TaskRuntime) -> None:
        self.runtime = runtime

    @property
    def description(self) -> str:
        return (
            f"Spawns an independent background worker agent in the active workspace ({self.runtime.workspace}) "
            "to execute long-running coding, testing, building, searching, or file inspection tasks. "
            "Use this whenever work requires running commands, editing files, or inspecting files in the workspace."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "A short descriptive title for the task (e.g. 'run tests', 'refactor parser').",
                },
                "prompt": {
                    "type": "string",
                    "description": "Detailed instructions and requirements for the worker agent.",
                },
                "isolate_worktree": {
                    "type": "boolean",
                    "description": "Whether to isolate worker in a dedicated git worktree branch (recommended for mutating code tasks).",
                    "default": False,
                },
                "workspace": {
                    "type": "string",
                    "description": f"The target workspace root directory. Defaults to the active project workspace ({self.runtime.workspace}).",
                    "default": str(self.runtime.workspace),
                },
            },
            "required": ["title", "prompt"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        title = str(args.get("title", "")).strip()
        prompt = str(args.get("prompt", "")).strip()
        isolate = bool(args.get("isolate_worktree", False))
        workspace = str(args.get("workspace", "")).strip() or None

        if not title or not prompt:
            return ToolResult(
                "error: both 'title' and 'prompt' are required to spawn a task.",
                is_error=True,
            )

        task_id = await self.runtime.spawn_task(
            title=title,
            prompt=prompt,
            isolate_worktree=isolate,
            workspace=workspace,
        )
        return ToolResult(
            observation=f"Task '{title}' spawned with id '{task_id}'. It is now running in the background."
        )


class TaskCancelTool:
    """Orchestration tool for cancelling active background tasks."""

    name: str = "task_cancel"
    description: str = "Cancels an active background task by its ID."

    def __init__(self, runtime: TaskRuntime) -> None:
        self.runtime = runtime

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The ID of the task to cancel (e.g. 'task-1').",
                }
            },
            "required": ["task_id"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        task_id = str(args.get("task_id", "")).strip()
        if not task_id:
            return ToolResult("error: 'task_id' is required.", is_error=True)

        cancelled = await self.runtime.cancel_task(task_id)
        if cancelled:
            return ToolResult(f"Task '{task_id}' was successfully cancelled.")
        return ToolResult(
            f"Task '{task_id}' could not be cancelled (not found or already finished).",
            is_error=True,
        )


class TaskSteerTool:
    """Orchestration tool for steering a running background worker mid-turn."""

    name: str = "task_steer"
    description: str = "Sends a steering instruction or guidance message to an actively running background task."

    def __init__(self, runtime: TaskRuntime) -> None:
        self.runtime = runtime

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The ID of the task to steer (e.g. 'task-1').",
                },
                "instruction": {
                    "type": "string",
                    "description": "The guidance or redirection instruction for the worker.",
                },
            },
            "required": ["task_id", "instruction"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        task_id = str(args.get("task_id", "")).strip()
        instruction = str(args.get("instruction", "")).strip()
        if not task_id or not instruction:
            return ToolResult(
                "error: both 'task_id' and 'instruction' are required.", is_error=True
            )

        steered = await self.runtime.steer_task(task_id, instruction)
        if steered:
            return ToolResult(f"Steering instruction delivered to task '{task_id}'.")
        return ToolResult(
            f"Cannot steer task '{task_id}' (not active or not found).", is_error=True
        )


class TaskStatusTool:
    """Orchestration tool for querying status and progress of tasks."""

    name: str = "task_status"

    def __init__(self, runtime: TaskRuntime) -> None:
        self.runtime = runtime

    @property
    def description(self) -> str:
        return (
            f"Queries the current status, step count, and output of background tasks in workspace '{self.runtime.workspace}', "
            "or lists all active tasks if task_id is omitted."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Optional specific task ID to inspect (e.g. 'task-1'). If omitted, summarizes all active tasks.",
                }
            },
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        task_id = args.get("task_id")
        if task_id:
            summary = self.runtime.get_task_status_summary(str(task_id).strip())
        else:
            summary = self.runtime.get_task_status_summary(None)
        return ToolResult(observation=summary)
