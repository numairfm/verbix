from __future__ import annotations

from pathlib import Path
from typing import Any

from verbix.tools.base import ToolContext, ToolResult
from verbix.tools.sandbox import (
    SandboxConfig,
    execute_sandboxed,
    probe_sandbox,
)


class BashTool:
    name: str = "bash"
    side_effecting: bool = True

    def __init__(
        self,
        allow_network: bool = False,
        extra_binds: tuple[tuple[Path, Path, bool], ...] | None = None,
    ) -> None:
        self.allow_network = allow_network
        self.extra_binds = extra_binds or ()

    @property
    def description(self) -> str:
        status = probe_sandbox(allow_network=self.allow_network)
        if status.capabilities:
            return status.capabilities
        return "Executes a bash command in the workspace inside a mandatory OS sandbox."

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
            },
            "required": ["command"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        command = str(args.get("command", "")).strip()
        if not command:
            return ToolResult("error: empty command", is_error=True)

        try:
            timeout = float(args.get("timeout_seconds", 60.0))
        except (ValueError, TypeError):
            timeout = 60.0

        try:
            config = SandboxConfig(
                workspace=context.workspace,
                allow_network=self.allow_network,
                timeout_seconds=timeout,
                extra_binds=self.extra_binds,
            )
            res = await execute_sandboxed(command, config)
            output = res.stdout
            if res.stderr:
                if output:
                    output += "\n" + res.stderr
                else:
                    output = res.stderr

            is_error = res.exit_code != 0
            if is_error and not output:
                output = f"Command exited with non-zero status code: {res.exit_code}"

            return ToolResult(observation=output, is_error=is_error)
        except (OSError, RuntimeError) as e:
            return ToolResult(f"error: command execution failed: {e}", is_error=True)
