from __future__ import annotations

import asyncio
from typing import Any

from verbix.tools.base import ToolContext, ToolResult
from verbix.tools.shared import check_fresh, record_version, resolve_contained_path


class FileWriteTool:
    name: str = "write"
    side_effecting: bool = True
    description: str = (
        "Writes content to a file within the workspace, creating parent directories."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to file"},
                "content": {
                    "type": "string",
                    "description": "Text content to write",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        return await asyncio.to_thread(self._execute_sync, args, context)

    def _execute_sync(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        path_str = args.get("path", "")
        content = args.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        try:
            target = resolve_contained_path(path_str, context.workspace)
            if target.exists():
                fresh = check_fresh(target, context)
                if fresh == "unread":
                    return ToolResult(
                        f"error: refusing to overwrite '{path_str}' because this "
                        "session has not read it; call read first",
                        is_error=True,
                    )
                if fresh == "stale":
                    return ToolResult(
                        f"error: '{path_str}' changed since this session read it; "
                        "call read again and re-apply your change",
                        is_error=True,
                    )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            record_version(target, content.encode("utf-8"), context)
            return ToolResult(
                f"Successfully wrote {len(content)} characters to '{path_str}'"
            )
        except ValueError as e:
            return ToolResult(f"error: {e}", is_error=True)
        except OSError as e:
            return ToolResult(
                f"error: could not write file '{path_str}': {e}", is_error=True
            )
