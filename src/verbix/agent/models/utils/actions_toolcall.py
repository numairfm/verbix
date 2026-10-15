"""Parse actions from tool-call model responses."""

from __future__ import annotations

import json
from typing import Any

from jinja2 import StrictUndefined, Template

from verbix.agent.exceptions import FormatError
from verbix.tools.base import parse_lenient_json


def parse_toolcall_actions(
    tool_calls: list[Any],
    *,
    format_error_template: str,
    template_kwargs: dict[str, Any] | None = None,
    allowed_tools: set[str] | None = None,
    content: str | None = None,
) -> list[dict[str, Any]]:
    """Parse tool calls from the response. Raises FormatError if unknown tool or invalid args."""
    template_kwargs = template_kwargs or {}
    if not tool_calls:
        user_msg = {
            "role": "user",
            "content": Template(format_error_template, undefined=StrictUndefined).render(
                error="No tool calls found. Execute a tool if you need to take an action, or output your direct factual conclusion in text if you are done.",
                actions=[],
                has_tool_calls=False,
                **template_kwargs,
            ),
            "extra": {"interrupt_type": "FormatError"},
        }
        if content and content.strip():
            assistant_msg = {"role": "assistant", "content": content, "extra": {}}
            raise FormatError(assistant_msg, user_msg)
        raise FormatError(user_msg)
    actions = []
    known = allowed_tools if allowed_tools is not None else {"bash"}
    for tool_call in tool_calls:
        error_msg = ""
        args = {}
        try:
            fn = getattr(tool_call, "function", None)
            arguments_str = getattr(fn, "arguments", "{}") if fn else "{}"
            try:
                args = json.loads(arguments_str, strict=False)
            except (json.JSONDecodeError, TypeError, ValueError):
                args = parse_lenient_json(arguments_str)
                if "_raw" in args and len(args) == 1:
                    raise json.JSONDecodeError("Could not parse JSON", arguments_str, 0)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            error_msg = f"Error parsing tool call arguments: {e}."
        fn_name = getattr(getattr(tool_call, "function", None), "name", "")
        if fn_name not in known:
            error_msg += f"Unknown tool '{fn_name}'."
        elif fn_name == "bash" and (not isinstance(args, dict) or "command" not in args):
            error_msg += "Missing 'command' argument in bash tool call."
        elif fn_name != "bash" and not isinstance(args, dict):
            error_msg += f"Invalid arguments for tool '{fn_name}'."
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
        if fn_name == "bash":
            actions.append({"command": args["command"], "tool_call_id": call_id})
        else:
            actions.append({"name": fn_name, "arguments": args, "tool_call_id": call_id})
    return actions
