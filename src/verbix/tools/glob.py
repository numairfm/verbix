from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from verbix.tools.base import ToolContext, ToolResult
from verbix.tools.shared import (
    _is_secret_basename,
    find_ripgrep_binary,
    iter_workspace_files,
    resolve_contained_path,
)


class GlobTool:
    name: str = "glob"
    side_effecting: bool = False
    description: str = (
        "Finds files in the workspace matching a glob pattern (e.g. '**/*.py', 'src/**', '*.json')."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match file paths against (e.g. '**/*.py', '*.html')",
                },
                "path": {
                    "type": "string",
                    "description": "Relative directory to search within (default: '.')",
                    "default": ".",
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
        if not target.is_dir():
            return ToolResult(f"error: '{path_str}' is not a directory", is_error=True)

        rg_bin = find_ripgrep_binary()
        if rg_bin is not None:
            cmd = [
                rg_bin,
                "--files",
                "--path-separator=/",
                "-g",
                pattern,
                "-g",
                "!.env",
                "-g",
                "!auth.json",
                "-g",
                "!auth.*",
            ]

            try:
                rel_path = target.relative_to(context.workspace)
                target_arg = rel_path.as_posix() if str(rel_path) != "." else "."
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
                if proc.returncode == 0:
                    lines = [
                        cleaned
                        for raw_line in stdout.decode("utf-8", errors="replace").splitlines()
                        if (cleaned := raw_line.removeprefix("./")).strip()
                        and not _is_secret_basename(Path(cleaned).name)
                    ]
                    if lines:
                        return ToolResult("\n".join(lines))
                    return ToolResult("No files matched pattern.")
                if proc.returncode == 1:
                    return ToolResult("No files matched pattern.")
                err_text = stderr.decode("utf-8", errors="replace").strip()
                return ToolResult(f"error: {err_text}", is_error=True)
            except OSError:
                pass

        try:
            ws = context.workspace.resolve()
            matches = [
                p.relative_to(ws).as_posix()
                for p in await asyncio.to_thread(
                    iter_workspace_files, target, ws, pattern
                )
            ]
            if matches:
                return ToolResult("\n".join(matches))
            return ToolResult("No files matched pattern.")
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"error: {e}", is_error=True)
