"""Parse actions & format observations with toolcalls."""

import json
import re
import time
from typing import Any

from jinja2 import StrictUndefined, Template

from verbix.agent.exceptions import FormatError
from verbix.agent.models.utils.openai_multimodal import expand_multimodal_content
from verbix.tools.base import parse_lenient_json

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
    tool_calls: list[Any],
    *,
    format_error_template: str,
    template_kwargs: dict[str, Any] | None = None,
    allowed_tools: set[str] | None = None,
    content: str | None = None,
) -> list[dict[str, Any]]:
    """Parse tool calls from the response. Raises FormatError if unknown tool or invalid args.

    ``allowed_tools`` names the registry tools advertised for this query
    (``bash`` is always allowed). Registry actions come back generic
    (``name``/``arguments``); bash keeps its historic ``command`` shape.
    """
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
        known = {"bash"} | (allowed_tools or set())
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
        # Pictures are atomic payloads: lift them out before the template
        # (which may truncate long text) and reattach them afterwards.
        images: list[dict[str, Any]] = []
        raw_text = output.get("output", "")
        if multimodal_regex and isinstance(raw_text, str):
            for match in re.finditer(multimodal_regex, raw_text):
                if match.group(1).strip() == "image_url":
                    images.append(
                        {"type": "image_url", "image_url": {"url": match.group(2).strip()}}
                    )
        render_from = dict(output)
        if images and isinstance(raw_text, str):
            render_from["output"] = re.sub(multimodal_regex, "[image attached]", raw_text)
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=render_from, **(template_vars or {})
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
        if images:
            msg["content"] = [{"type": "text", "text": content}, *images]
        elif multimodal_regex:
            msg = expand_multimodal_content(msg, pattern=multimodal_regex)
        results.append(msg)
    return results
