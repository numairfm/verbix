from __future__ import annotations

import asyncio
from typing import Any

from verbix.tools.base import ToolContext, ToolResult
from verbix.tools.shared import find_ripgrep_binary, resolve_contained_path


class GrepTool:
    name: str = "grep"
    side_effecting: bool = False
    description: str = (
        "Searches for a regular expression or text pattern in files within the workspace using ripgrep. "
        "Returns matching lines with line numbers."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression or text pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Relative directory or file path to search within (default: '.')",
                    "default": ".",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional glob pattern to filter files (e.g. '*.py' or '*.html')",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case-sensitive (default: false)",
                    "default": False,
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            return ToolResult("error: 'pattern' argument is required and cannot be empty", is_error=True)

        path_str = str(args.get("path", ".")).strip() or "."
        try:
            target = resolve_contained_path(path_str, context.workspace)
        except ValueError as e:
            return ToolResult(f"error: {e}", is_error=True)

        if not target.exists():
            return ToolResult(f"error: path '{path_str}' does not exist", is_error=True)

        rg_bin = find_ripgrep_binary()
        cmd = [rg_bin, "--line-number", "--no-heading", "--color=never"]

        if not args.get("case_sensitive", False):
            cmd.append("-i")

        glob_filter = args.get("glob")
        if glob_filter and isinstance(glob_filter, str) and glob_filter.strip():
            cmd.extend(["--glob", glob_filter.strip()])

        cmd.extend(["--", pattern])

        try:
            rel_path = target.relative_to(context.workspace)
            target_arg = str(rel_path) if str(rel_path) != "." else "."
        except ValueError:
            target_arg = str(target)
        cmd.append(target_arg)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(context.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        except OSError as e:
            return ToolResult(f"error: failed to execute ripgrep: {e}", is_error=True)

        if proc.returncode == 0:
            lines = [
                line.removeprefix("./")
                for line in stdout.decode("utf-8", errors="replace").splitlines()
                if line.strip()
            ]
            if lines:
                return ToolResult("\n".join(lines))
            return ToolResult("No matches found.")
        if proc.returncode == 1:
            return ToolResult("No matches found.")
        err_text = stderr.decode("utf-8", errors="replace").strip()
        return ToolResult(f"error: {err_text}", is_error=True)
