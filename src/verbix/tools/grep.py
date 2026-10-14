from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from verbix.tools.base import ToolContext, ToolResult
from verbix.tools.shared import (
    _is_secret_basename,
    find_ripgrep_binary,
    iter_workspace_files,
    resolve_contained_path,
)


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
        if rg_bin is not None:
            cmd = [rg_bin, "--line-number", "--no-heading", "--color=never", "--path-separator=/"]

            if not args.get("case_sensitive", False):
                cmd.append("-i")

            glob_filter = args.get("glob")
            if glob_filter and isinstance(glob_filter, str) and glob_filter.strip():
                cmd.extend(["--glob", glob_filter.strip()])
            cmd.extend(["--glob", "!.env", "--glob", "!auth.json", "--glob", "!auth.*"])

            cmd.extend(["--", pattern])

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
                    lines = []
                    for raw_line in stdout.decode("utf-8", errors="replace").splitlines():
                        cleaned = raw_line.removeprefix("./")
                        if not cleaned.strip():
                            continue
                        file_part = cleaned.split(":", 1)[0]
                        if _is_secret_basename(Path(file_part).name):
                            continue
                        lines.append(cleaned)
                    if lines:
                        return ToolResult("\n".join(lines))
                    return ToolResult("No matches found.")
                if proc.returncode == 1:
                    return ToolResult("No matches found.")
                err_text = stderr.decode("utf-8", errors="replace").strip()
                return ToolResult(f"error: {err_text}", is_error=True)
            except OSError:
                pass

        return await asyncio.to_thread(
            self._search_python,
            pattern,
            target,
            context.workspace.resolve(),
            bool(args.get("case_sensitive", False)),
            args.get("glob") if isinstance(args.get("glob"), str) else None,
        )

    @staticmethod
    def _search_python(
        pattern: str,
        target: Path,
        workspace: Path,
        case_sensitive: bool,
        glob_filter: str | None,
    ) -> ToolResult:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags=flags)
        except re.error as e:
            return ToolResult(f"error: invalid regex '{pattern}': {e}", is_error=True)

        matches: list[str] = []
        for fpath in iter_workspace_files(target, workspace, glob_pattern=glob_filter):
            try:
                raw = fpath.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:4096]:
                continue
            rel = fpath.relative_to(workspace).as_posix()
            for lineno, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{rel}:{lineno}:{line}")
        if matches:
            return ToolResult("\n".join(matches))
        return ToolResult("No matches found.")
