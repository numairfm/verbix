from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from verbix.tools.base import ToolContext, ToolResult


def resolve_contained_path(path_str: str, workspace: Path) -> Path:
    """Resolves a path and ensures it stays strictly within the workspace boundary."""
    norm_workspace = workspace.resolve()
    target = (workspace / path_str).resolve()
    try:
        target.relative_to(norm_workspace)
    except ValueError:
        raise ValueError(
            f"Path '{path_str}' escapes configured workspace '{workspace}'"
        ) from None
    return target


class FileReadTool:
    name: str = "read"
    description: str = "Reads a file within the workspace with line numbers."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to file"},
                "start_line": {
                    "type": "integer",
                    "description": "1-based starting line number",
                    "default": 1,
                },
                "end_line": {
                    "type": "integer",
                    "description": "1-based ending line number (inclusive)",
                },
            },
            "required": ["path"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        path_str = args.get("path", "")
        try:
            target = resolve_contained_path(path_str, context.workspace)
            if not target.exists():
                return ToolResult(
                    f"error: file not found at '{path_str}'", is_error=True
                )
            if not target.is_file():
                return ToolResult(
                    f"error: '{path_str}' is a directory, not a file",
                    is_error=True,
                )

            raw = target.read_bytes()
            context.file_versions[str(target)] = hashlib.sha256(raw).hexdigest()
            lines = raw.decode("utf-8", errors="replace").splitlines()
            start_line = max(1, int(args.get("start_line", 1)))
            end_line = args.get("end_line")
            if end_line is not None:
                end_line = min(len(lines), int(end_line))
            else:
                end_line = len(lines)

            selected = [f"{i}: {lines[i - 1]}" for i in range(start_line, end_line + 1)]
            return ToolResult("\n".join(selected))
        except ValueError as e:
            return ToolResult(f"error: {e}", is_error=True)
        except OSError as e:
            return ToolResult(
                f"error: could not read file '{path_str}': {e}", is_error=True
            )


class FileWriteTool:
    name: str = "write"
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
        path_str = args.get("path", "")
        content = args.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        try:
            target = resolve_contained_path(path_str, context.workspace)
            key = str(target)
            if target.exists():
                recorded = context.file_versions.get(key)
                if recorded is None:
                    return ToolResult(
                        f"error: refusing to overwrite '{path_str}' because this "
                        "session has not read it; call read first",
                        is_error=True,
                    )
                current = hashlib.sha256(target.read_bytes()).hexdigest()
                if current != recorded:
                    return ToolResult(
                        f"error: '{path_str}' changed since this session read it; "
                        "call read again and re-apply your change",
                        is_error=True,
                    )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            context.file_versions[key] = hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest()
            return ToolResult(
                f"Successfully wrote {len(content)} characters to '{path_str}'"
            )
        except ValueError as e:
            return ToolResult(f"error: {e}", is_error=True)
        except OSError as e:
            return ToolResult(
                f"error: could not write file '{path_str}': {e}", is_error=True
            )


class FileEditTool:
    name: str = "edit"
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
                "new_str": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_str", "new_str"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        path_str = args.get("path", "")
        old_str = args.get("old_str", "")
        new_str = args.get("new_str", "")
        try:
            target = resolve_contained_path(path_str, context.workspace)
            if not target.exists():
                return ToolResult(
                    f"error: file not found at '{path_str}'", is_error=True
                )

            key = str(target)
            raw = target.read_bytes()
            recorded = context.file_versions.get(key)
            if recorded is None:
                return ToolResult(
                    f"error: refusing to edit '{path_str}' because this session "
                    "has not read it; call read first",
                    is_error=True,
                )
            if hashlib.sha256(raw).hexdigest() != recorded:
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
            context.file_versions[key] = hashlib.sha256(
                updated.encode("utf-8")
            ).hexdigest()
            return ToolResult(f"Successfully replaced 1 occurrence in '{path_str}'")
        except ValueError as e:
            return ToolResult(f"error: {e}", is_error=True)
        except OSError as e:
            return ToolResult(
                f"error: could not edit file '{path_str}': {e}", is_error=True
            )
