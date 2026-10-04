from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from verbix.provider.client import ProviderClient

from verbix.core.events import (
    AgentEvent,
    TextDelta,
    ThoughtDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnComplete,
)
from verbix.core.history import ConversationHistory
from verbix.core.models import (
    CanonicalMessage,
    CanonicalToolCall,
)
from verbix.provider.models import (
    Finish as ProviderFinish,
)
from verbix.provider.models import (
    TextDelta as ProviderTextDelta,
)
from verbix.provider.models import (
    ThoughtDelta as ProviderThoughtDelta,
)
from verbix.provider.models import (
    ToolCallDelta as ProviderToolCallDelta,
)
from verbix.provider.models import (
    ToolDefinition,
    ToolFunction,
)
from verbix.tools.base import Tool, ToolContext, parse_lenient_json


class CycleDetector:
    """Detects repetitive tool invocations with identical parameters to break infinite loops."""

    def __init__(self, repeat_limit: int = 3) -> None:
        self.repeat_limit = repeat_limit
        self._history: list[tuple[str, str]] = []

    def record(self, tool_name: str, arguments: str) -> bool:
        """Records a call and returns True if repeat_limit identical consecutive calls are reached."""
        self._history.append((tool_name, arguments.strip()))
        if len(self._history) < self.repeat_limit:
            return False

        last_slice = self._history[-self.repeat_limit :]
        first = last_slice[0]
        return all(item == first for item in last_slice)


class AgentSession:
    """Core autonomous agent turn loop with streaming events,

    cycle detection, and lenient tool recovery.
    """

    def __init__(
        self,
        session_id: str,
        system_prompt: str,
        provider: ProviderClient,
        workspace: Path,
        tools: list[Tool] | None = None,
        history: ConversationHistory | None = None,
        max_turns: int = 30,
        reasoning_effort: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.provider = provider
        self.workspace = workspace.resolve()
        self.tools: dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.tool_context = ToolContext(workspace=self.workspace)
        self.history = history or ConversationHistory()
        self.max_turns = max_turns
        self.reasoning_effort = reasoning_effort

    async def step(
        self,
        user_input: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Runs the agent loop until the model yields a final text response,

        or max_turns is exhausted.
        Supports clean mid-flight cancellation via standard asyncio task cancellation.
        """
        if user_input:
            self.history.append_user(user_input)

        cycle_detector = CycleDetector(repeat_limit=3)
        turn_count = 0

        # Build tool definitions for the provider
        tool_defs = [
            ToolDefinition(
                function=ToolFunction(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters_schema,
                )
            )
            for t in self.tools.values()
        ]

        while turn_count < self.max_turns:
            turn_count += 1

            messages = [
                CanonicalMessage(role="system", content=self.system_prompt),
                *self.history.messages,
            ]

            accumulated_text: list[str] = []
            accumulated_thought: list[str] = []
            tool_call_chunks: dict[int, dict[str, Any]] = {}
            finish_reason: str | None = None

            # Stream from provider
            async for event in self.provider.stream(
                messages,
                tools=tool_defs if tool_defs else None,
                reasoning_effort=self.reasoning_effort,
            ):
                if isinstance(event, ProviderTextDelta):
                    accumulated_text.append(event.text)
                    yield TextDelta(text=event.text)
                elif isinstance(event, ProviderThoughtDelta):
                    accumulated_thought.append(event.thought)
                    yield ThoughtDelta(thought=event.thought)
                elif isinstance(event, ProviderToolCallDelta):
                    entry = tool_call_chunks.setdefault(
                        event.index, {"id": "", "name": "", "args": []}
                    )
                    if event.call_id:
                        entry["id"] = event.call_id
                    if event.name:
                        entry["name"] = event.name
                    if event.arguments_delta:
                        entry["args"].append(event.arguments_delta)
                elif isinstance(event, ProviderFinish):
                    finish_reason = event.reason

            # Handle token limit truncation
            if finish_reason == "length":
                if tool_call_chunks:
                    stop_msg = "... (reply cut off at the token limit)"
                    self.history.append_assistant(content=stop_msg)
                    yield TurnComplete(final_text=stop_msg)
                    return
                cutoff_suffix = " ... (reply cut off at the token limit)"
                accumulated_text.append(cutoff_suffix)
                yield TextDelta(text=cutoff_suffix)

            # Check if model requested tool calls
            if tool_call_chunks:
                canonical_calls: list[CanonicalToolCall] = []
                for idx in sorted(tool_call_chunks.keys()):
                    c = tool_call_chunks[idx]
                    raw_args = "".join(c["args"]).strip()
                    # Support multiple concatenated JSON objects emitted in a single tool call stream
                    sub_jsons: list[str] = []
                    if raw_args.startswith("{"):
                        dec = json.JSONDecoder()
                        pos = 0
                        while pos < len(raw_args):
                            while pos < len(raw_args) and raw_args[pos].isspace():
                                pos += 1
                            if pos >= len(raw_args):
                                break
                            try:
                                _, end_pos = dec.raw_decode(raw_args, idx=pos)
                                sub_jsons.append(raw_args[pos:end_pos])
                                pos = end_pos
                            except json.JSONDecodeError:
                                break

                    if len(sub_jsons) > 1:
                        base_id = c["id"] or f"call_{idx}"
                        for sub_i, s_json in enumerate(sub_jsons):
                            canonical_calls.append(
                                CanonicalToolCall(
                                    id=f"{base_id}_{sub_i}",
                                    name=c["name"],
                                    arguments=s_json,
                                )
                            )
                    else:
                        canonical_calls.append(
                            CanonicalToolCall(
                                id=c["id"] or f"call_{idx}",
                                name=c["name"],
                                arguments=raw_args,
                            )
                        )

                # Commit assistant turn to history
                thought_str = (
                    "".join(accumulated_thought) if accumulated_thought else None
                )
                self.history.append_assistant(
                    content="".join(accumulated_text),
                    thought=thought_str,
                    tool_calls=canonical_calls,
                )

                # Cycle detection check
                cycle_triggered = False
                for tc in canonical_calls:
                    if cycle_detector.record(tc.name, tc.arguments):
                        cycle_triggered = True
                        break

                if cycle_triggered:
                    stop_msg = "Agent stopped: repetitive tool calling loop detected."
                    for tc in canonical_calls:
                        self.history.append(
                            CanonicalMessage(
                                role="tool",
                                content="error: not executed (repetitive tool calling loop detected)",
                                tool_call_id=tc.id,
                            )
                        )
                    self.history.append_assistant(content=stop_msg)
                    yield TurnComplete(final_text=stop_msg)
                    return

                # Execute tool calls against one session-scoped context so the
                # file-version log survives across iterations and tool calls.
                ctx = self.tool_context
                for tc in canonical_calls:
                    parsed_args = parse_lenient_json(tc.arguments)
                    yield ToolCallStarted(
                        call_id=tc.id,
                        tool_name=tc.name,
                        arguments=parsed_args,
                    )

                    tool = self.tools.get(tc.name)
                    if not tool:
                        obs = f"error: tool '{tc.name}' not found"
                        is_error = True
                    else:
                        try:
                            result = await tool.execute(parsed_args, ctx)
                            obs = result.observation
                            is_error = result.is_error
                        except Exception as e:  # noqa: BLE001
                            obs = f"error: tool execution failed: {e}"
                            is_error = True

                    yield ToolCallFinished(
                        call_id=tc.id, observation=obs, is_error=is_error
                    )
                    self.history.append(
                        CanonicalMessage(role="tool", content=obs, tool_call_id=tc.id)
                    )

                # Continue loop to next turn step so model can observe tool results
                continue

            # No tool calls: final text response yielded
            final_text = "".join(accumulated_text).strip()
            thought_str = "".join(accumulated_thought) if accumulated_thought else None
            if not final_text:
                final_text = (
                    "Give me a moment."
                    if thought_str
                    else "Sorry, something went wrong. Could you repeat that?"
                )
            self.history.append_assistant(content=final_text, thought=thought_str)
            yield TurnComplete(final_text=final_text)
            return

        # Max turns exhausted
        limit_msg = (
            f"Agent stopped: reached maximum step limit ({self.max_turns} turns)."
        )
        self.history.append_assistant(content=limit_msg)
        yield TurnComplete(final_text=limit_msg)
