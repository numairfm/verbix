from __future__ import annotations

from pathlib import Path
from typing import Any

from verbix.tools.base import ToolContext, ToolResult
from verbix.tools.shared import record_version, resolve_contained_path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
IMAGE_MIMES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
MAX_IMAGE_BYTES = 10 * 1024 * 1024


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
        body = "\n".join(rows[start_line - 1 : end_line])
        header = f"showing entries {start_line}-{end_line} of {len(rows)}"
        return ToolResult(f"{header}\n{body}" if body else header)

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
            record_version(target, raw, context)
            lines = raw.decode("utf-8", errors="replace").splitlines()
            start_line = max(1, int(args.get("start_line", 1)))
            end_line = args.get("end_line")
            if end_line is not None:
                end_line = min(len(lines), int(end_line))
            else:
                end_line = len(lines)

            selected = [f"{i}: {lines[i - 1]}" for i in range(start_line, end_line + 1)]
            header = f"showing lines {start_line}-{end_line} of {len(lines)}"
            body = "\n".join(selected)
            return ToolResult(f"{header}\n{body}" if body else header)
        except ValueError as e:
            return ToolResult(f"error: {e}", is_error=True)
        except OSError as e:
            return ToolResult(
                f"error: could not read file '{path_str}': {e}", is_error=True
            )
