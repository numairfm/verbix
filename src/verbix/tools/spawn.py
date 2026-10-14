from __future__ import annotations

from typing import TYPE_CHECKING, Any

from verbix.tools.base import ToolContext, ToolResult

if TYPE_CHECKING:
    from verbix.runtime.runtime import Runtime


class SpawnTool:
    """Spawns an autonomous background worker to perform a task."""

    name: str = "spawn"
    side_effecting: bool = True
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
                    "description": (
                        "The high-level goal and verification criteria for the background worker to accomplish. "
                        "State what needs to be done. Never guess or insert specific shell commands, package managers, "
                        "file paths, or speculative log/workspace investigations unless explicitly requested by the user. "
                        "Never call spawn to figure out what an ambiguous or misheard utterance means."
                    ),
                },
            },
            "required": ["objective"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        objective = str(args.get("objective", "")).strip()
        if not objective:
            return ToolResult("error: 'objective' is required to spawn work.", is_error=True)

        task_id = await self.runtime.spawn(objective)
        return ToolResult(f"{task_id} started in background.")
