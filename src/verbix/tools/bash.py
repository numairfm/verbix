from __future__ import annotations

import re
from typing import Any

from verbix.tools.base import ToolContext, ToolResult
from verbix.tools.sandbox import SandboxConfig, execute_sandboxed


class SafeCommandChecker:
    """UX helper to identify read-only / safe commands that run without confirmation."""

    SAFE_PREFIXES = (
        "git status",
        "git log",
        "git diff",
        "ls",
        "dir",
        "cat",
        "head",
        "tail",
        "grep",
        "rg",
        "find",
        "pwd",
        "cargo check",
        "pytest",
        "uv run pytest",
        "python -m unittest",
        "go test",
    )

    DANGEROUS_PATTERNS = (
        r"\brm\b",
        r"\bkill\b",
        r"\bsudo\b",
        r"\breboot\b",
        r"\bshutdown\b",
        r"\bgit\s+reset\b",
        r"\bgit\s+clean\b",
        r"\bgit\s+checkout\s+\.",
    )

    def is_safe(self, command: str) -> bool:
        trimmed = command.strip()
        # Any dangerous pattern immediately disqualifies
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, trimmed):
                return False

        # Redirections or pipes can be dangerous if uninspected
        if "|" in trimmed or ">" in trimmed or "<" in trimmed:
            return False

        # Must match a recognized safe prefix
        for prefix in self.SAFE_PREFIXES:
            if trimmed == prefix or trimmed.startswith(prefix + " "):
                return True

        return False


class BashTool:
    name: str = "bash"
    description: str = (
        "Executes a bash command inside an isolated default-deny sandbox."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to execute"},
                "timeout_seconds": {
                    "type": "number",
                    "description": "Timeout in seconds",
                    "default": 60.0,
                },
                "allow_network": {
                    "type": "boolean",
                    "description": "Whether outbound network is permitted",
                    "default": False,
                },
            },
            "required": ["command"],
        }

    def __init__(
        self,
        max_output_length: int = 50_000,
        safe_checker: SafeCommandChecker | None = None,
    ) -> None:
        self.max_output_length = max_output_length
        self.safe_checker = safe_checker or SafeCommandChecker()

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        command = args.get("command", "").strip()
        if not command:
            return ToolResult("error: empty command", is_error=True)

        timeout = float(args.get("timeout_seconds", 60.0))
        allow_network = bool(args.get("allow_network", False))

        config = SandboxConfig(
            workspace=context.workspace,
            allow_network=allow_network,
            timeout_seconds=timeout,
        )

        try:
            res = await execute_sandboxed(command, config)
            output = res.stdout
            if res.stderr:
                if output:
                    output += "\n" + res.stderr
                else:
                    output = res.stderr

            if len(output) > self.max_output_length:
                # Tail truncation
                output = (
                    f"[Output truncated: showing last {self.max_output_length} chars]\n"
                    + output[-self.max_output_length :]
                )

            is_error = res.exit_code != 0
            if is_error and not output:
                output = f"Command exited with non-zero status code: {res.exit_code}"

            return ToolResult(observation=output, is_error=is_error)
        except (OSError, RuntimeError) as e:
            return ToolResult(f"error: command execution failed: {e}", is_error=True)
