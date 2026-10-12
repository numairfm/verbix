from __future__ import annotations

from typing import Any

from verbix.tools.base import ToolContext, ToolResult
from verbix.tools.shared import check_fresh, record_version, resolve_contained_path


class FileEditTool:
    name: str = "edit"
    side_effecting: bool = False
    description: str = "Replaces an exact occurrence of old_str with new_str in a file."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to file"},
                "old_str": {
                    "type": "string",
                    "description": "Exact text to replace",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement text",
                },
            },
            "required": ["path", "old_str", "new_str"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        path_str = args.get("path", "")
        old_str = args.get("old_str", "")
        new_str = args.get("new_str", "")
        if not old_str:
            return ToolResult("error: old_str cannot be empty", is_error=True)
        try:
            target = resolve_contained_path(path_str, context.workspace)
            if not target.exists():
                return ToolResult(
                    f"error: file not found at '{path_str}'", is_error=True
                )

            raw = target.read_bytes()
            fresh = check_fresh(target, context)
            if fresh == "unread":
                return ToolResult(
                    f"error: refusing to edit '{path_str}' because this session "
                    "has not read it; call read first",
                    is_error=True,
                )
            if fresh == "stale":
                return ToolResult(
                    f"error: '{path_str}' changed since this session read it; "
                    "call read again and re-apply your change",
                    is_error=True,
                )

            current = raw.decode("utf-8")
            count = current.count(old_str)
            if count == 0:
                return ToolResult(
                    f"error: old_str not found in '{path_str}'", is_error=True
                )
            if count > 1:
                return ToolResult(
                    f"error: old_str occurs {count} times in '{path_str}'. Provide unique context.",
                    is_error=True,
                )

            updated = current.replace(old_str, new_str, 1)
            target.write_text(updated, encoding="utf-8")
            record_version(target, updated.encode("utf-8"), context)
            return ToolResult(f"Successfully replaced 1 occurrence in '{path_str}'")
        except ValueError as e:
            return ToolResult(f"error: {e}", is_error=True)
        except OSError as e:
            return ToolResult(
                f"error: could not edit file '{path_str}': {e}", is_error=True
            )
