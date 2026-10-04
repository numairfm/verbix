"""Parse actions & format observations with toolcalls."""

import json
import time
from typing import Any

from jinja2 import StrictUndefined, Template

from verbix.minisweagent.exceptions import FormatError
from verbix.minisweagent.models.utils.openai_multimodal import expand_multimodal_content

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


def parse_toolcall_actions(
    tool_calls: list[Any], *, format_error_template: str, template_kwargs: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Parse tool calls from the response. Raises FormatError if unknown tool or invalid args."""
    template_kwargs = template_kwargs or {}
    if not tool_calls:
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    error="No tool calls found in the response. Every response MUST include at least one tool call.",
                    actions=[],
                    has_tool_calls=False,
                    **template_kwargs,
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )
    actions = []
    for tool_call in tool_calls:
        error_msg = ""
        args = {}
        try:
            fn = getattr(tool_call, "function", None)
            arguments_str = getattr(fn, "arguments", "{}") if fn else "{}"
            args = json.loads(arguments_str)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            error_msg = f"Error parsing tool call arguments: {e}."
        fn_name = getattr(getattr(tool_call, "function", None), "name", "")
        if fn_name != "bash":
            error_msg += f"Unknown tool '{fn_name}'."
        if not isinstance(args, dict) or "command" not in args:
            error_msg += "Missing 'command' argument in bash tool call."
        if error_msg:
            raise FormatError(
                {
                    "role": "user",
                    "content": Template(format_error_template, undefined=StrictUndefined).render(
                        actions=[], error=error_msg.strip(), has_tool_calls=True, **template_kwargs
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        call_id = getattr(tool_call, "id", None)
        actions.append({"command": args["command"], "tool_call_id": call_id})
    return actions


def format_toolcall_observation_messages(
    *,
    actions: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    observation_template: str,
    template_vars: dict[str, Any] | None = None,
    multimodal_regex: str = "",
) -> list[dict[str, Any]]:
    """Format execution outputs into tool result messages."""
    not_executed = {"output": "", "returncode": -1, "exception_info": "action was not executed"}
    padded_outputs = outputs + [not_executed] * (len(actions) - len(outputs))
    results = []
    for action, output in zip(actions, padded_outputs):
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=output, **(template_vars or {})
        )
        msg: dict[str, Any] = {
            "content": content,
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "timestamp": time.time(),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        if "tool_call_id" in action:
            msg["tool_call_id"] = action["tool_call_id"]
            msg["role"] = "tool"
        else:
            msg["role"] = "user"  # human issued commands
        if multimodal_regex:
            msg = expand_multimodal_content(msg, pattern=multimodal_regex)
        results.append(msg)
    return results
