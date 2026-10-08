from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from verbix.tools.base import ToolContext, ToolResult

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
IMAGE_MIMES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
MAX_IMAGE_BYTES = 10 * 1024 * 1024


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
    description: str = (
        "Reads a file within the workspace with line numbers, lists a directory's entries, "
        "or opens an image (jpg/png/gif/webp) as a viewable picture."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to file or directory"},
                "start_line": {
                    "type": "integer",
                    "description": "1-based starting line number (or directory entry offset)",
                    "default": 1,
                },
                "end_line": {
                    "type": "integer",
                    "description": "1-based ending line number, inclusive (or last directory entry)",
                },
            },
            "required": ["path"],
        }

    def _read_dir(self, target: Path, path_str: str, args: dict[str, Any]) -> ToolResult:
        try:
            names = sorted(e.name for e in target.iterdir())
        except OSError as e:
            return ToolResult(f"error: could not list '{path_str}': {e}", is_error=True)
        if not names:
            return ToolResult(f"'{path_str}' is an empty directory")
        rows = []
        for name in names:
            full = target / name
            rows.append(f"{name}/ (directory)" if full.is_dir() else f"{name} ({full.stat().st_size} bytes)")
        start_line = max(1, int(args.get("start_line", 1)))
        end_line = args.get("end_line")
        end_line = min(len(rows), int(end_line)) if end_line is not None else len(rows)
        return ToolResult("\n".join(rows[start_line - 1 : end_line]))

    def _read_image(self, target: Path, path_str: str) -> ToolResult:
        import base64

        try:
            raw = target.read_bytes()
        except OSError as e:
            return ToolResult(f"error: could not read file '{path_str}': {e}", is_error=True)
        if len(raw) > MAX_IMAGE_BYTES:
            return ToolResult(
                f"error: image '{path_str}' is too large "
                f"({len(raw)} bytes, max {MAX_IMAGE_BYTES})",
                is_error=True,
            )
        mime = IMAGE_MIMES[target.suffix.lower()]
        data = base64.b64encode(raw).decode("ascii")
        return ToolResult(
            "<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>"
            f"data:{mime};base64,{data}</MSWEA_MULTIMODAL_CONTENT>"
        )

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        path_str = args.get("path", "")
        try:
            target = resolve_contained_path(path_str, context.workspace)
            if not target.exists():
                return ToolResult(
                    f"error: file not found at '{path_str}'", is_error=True
                )
            if target.is_dir():
                return self._read_dir(target, path_str, args)
            if target.suffix.lower() in IMAGE_SUFFIXES:
                return self._read_image(target, path_str)
            if not target.is_file():
                return ToolResult(
                    f"error: '{path_str}' is not a file",
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
