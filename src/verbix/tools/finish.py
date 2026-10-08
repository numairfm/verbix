from __future__ import annotations

from typing import Any

from verbix.tools.base import ToolContext, ToolResult


class FinishTool:
    """Signals background worker task completion with an information-dense debrief."""

    name: str = "finish"
    description: str = (
        "Call this tool when the task objective has been fully executed and verified. "
        "Provide an information-dense factual summary of actions taken, concrete numbers/metrics, "
        "and verified results."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "Information-dense factual debrief: specific commands run, exact metrics/counts, "
                        "exit codes, and verification evidence."
                    ),
                },
            },
            "required": ["summary"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        summary = str(args.get("summary", "")).strip()
        if not summary:
            return ToolResult("error: 'summary' is required to finish work.", is_error=True)
        return ToolResult(f"Task completed: {summary}")
