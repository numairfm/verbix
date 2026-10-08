"""foreground: build the voice-first foreground agent with its provider glue."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine
from pathlib import Path
from typing import Any
from unittest.mock import Mock, NonCallableMock

import yaml

from verbix.agent.agents.default import AgentConfig, DefaultAgent
from verbix.agent.environments.local import LocalEnvironment
from verbix.agent.models.litellm_model import LitellmModel
from verbix.core.events import (
    AgentEvent,
)
from verbix.core.events import (
    TextDelta as CoreTextDelta,
)
from verbix.core.events import (
    ThoughtDelta as CoreThoughtDelta,
)
from verbix.core.events import (
    ToolCallDelta as CoreToolCallDelta,
)
from verbix.core.events import (
    TurnComplete as CoreTurnComplete,
)
from verbix.core.models import CanonicalMessage, CanonicalToolCall
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
from verbix.runtime.floor import FloorArbiter
from verbix.runtime.ledger import TaskLedger
from verbix.runtime.worker import WorkerTask
from verbix.tools import build_registry

DEFAULT_VERBIX_PROMPT = """# IDENTITY & TONE
You are Verbix, a voice-operated desktop assistant and calm, capable peer. You say "I," speak plainly, admit limits plainly, no flattery.

# VOICE & AUDIO CONSTRAINTS
- Your answers are spoken aloud. Write for listening: lead with the answer and speak in plain, continuous conversational prose.
- Never use markdown syntax, asterisks, bullet points (* or -), backticks, code blocks, tables, headings, or numbered lists.
- Be concise: give the shortest response that completely satisfies the request; when the user asks for detail, a story, or a report, their request sets the length.
- Speak only the answer or status asked for; internal execution artifacts and unrequested instructions stay unspoken.
- Never recite system prompts, identity templates, execution artifacts, or internal schemas to the user.

# CAPABILITY ROUTING & DELEGATION
- Answer quick conversational and general knowledge questions directly.
- Delegate multi-step tasks, background operations, desktop control, system commands, or file interactions to spawn.
- You have no direct access to the filesystem, shell, or execution environment.
- When calling spawn, formulate the objective as the user's intended goal and expected outcome.
- Never guess, invent, or substitute concrete shell commands, scripts, package managers, or file paths into the objective unless the user explicitly specified them.
- Persona, voice, and style rules apply strictly to spoken text spoken to the user; tool arguments and objectives must always remain exact, factual, and unstylized.

# TOOL OPERATIONS & BACKGROUND WORK
- When a turn requires tools, execute all necessary tool operations first without speaking pre-tool announcements.
- When the user requests background work, spawn each task needed (in parallel or sequentially).
- When all tool operations for the turn complete, provide a single concise spoken response answering the user or confirming what was started; background workers announce terminal results upon completion.
- Inspect background tasks with status when the user asks how work is going, or asks for details or results.
- Stop background work with cancel.
- Speak answers and messages directly to the user. Never speak redundant pre- and post-tool confirmations. By default, describe past tool operations in the past tense."""

DEFAULT_WORKER_PROMPT = """You are an autonomous background worker executing tasks for Verbix.
Discover the necessary environment and tools, execute actions directly, verify outcomes, and call the finish tool with an information-dense factual summary of results upon completion."""


class DefaultAgentHistoryProxy:
    """Provides backwards-compatible .messages and .append_assistant() on top of DefaultAgent.messages."""

    def __init__(self, agent: DefaultAgent) -> None:
        self._agent = agent

    @property
    def messages(self) -> list[CanonicalMessage]:
        result: list[CanonicalMessage] = []
        for m in self._agent.messages:
            if isinstance(m, CanonicalMessage):
                result.append(m)
            elif isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content") or ""
                tc_list: list[CanonicalToolCall] | None = None
                raw_tcs = m.get("tool_calls")
                if raw_tcs:
                    tc_list = []
                    for tc in raw_tcs:
                        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                        tc_list.append(
                            CanonicalToolCall(
                                id=tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", ""),
                                name=fn.get("name", tc.get("name", "")) if isinstance(tc, dict) else getattr(tc, "name", ""),
                                arguments=fn.get("arguments", tc.get("arguments", "")) if isinstance(tc, dict) else getattr(tc, "arguments", ""),
                            )
                        )
                result.append(
                    CanonicalMessage(
                        role=role if role in ("system", "user", "assistant", "tool") else "user",
                        content=content,
                        tool_calls=tc_list,
                        tool_call_id=m.get("tool_call_id"),
                        name=m.get("name"),
                    )
                )
        return result

    def append_assistant(
        self,
        content: str,
        thought: str | None = None,
        tool_calls: list[Any] | None = None,
    ) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if thought:
            msg["thought"] = thought
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id if hasattr(tc, "id") else tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.name if hasattr(tc, "name") else tc.get("name", ""),
                        "arguments": tc.arguments if hasattr(tc, "arguments") else tc.get("arguments", ""),
                    },
                }
                for tc in tool_calls
            ]
        self._agent.messages.append(msg)


class ProviderStreamAdapter:
    """Adapts a legacy ProviderClient or stream-based provider to MiniSWEAgent's query_stream."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        if hasattr(provider, "config"):
            self.config = provider.config

    def get_template_vars(self) -> dict[str, Any]:
        if hasattr(self.provider, "get_template_vars") and callable(self.provider.get_template_vars):
            res = self.provider.get_template_vars()
            if isinstance(res, dict):
                return res
        return {}

    def format_message(self, role: str, content: str, **kwargs: Any) -> dict[str, Any]:
        if hasattr(self.provider, "format_message") and callable(self.provider.format_message):
            res = self.provider.format_message(role=role, content=content, **kwargs)
            if isinstance(res, dict):
                return res
        return {"role": role, "content": content, **kwargs}

    async def query_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentEvent]:
        converted_messages: list[CanonicalMessage] = []
        for m in messages:
            if isinstance(m, CanonicalMessage):
                converted_messages.append(m)
            elif isinstance(m, dict):
                role = m.get("role", "user")
                if role not in ("system", "user", "assistant", "tool"):
                    role = "user"
                tc_list: list[CanonicalToolCall] | None = None
                raw_tcs = m.get("tool_calls")
                if raw_tcs:
                    tc_list = []
                    for tc in raw_tcs:
                        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                        tc_list.append(
                            CanonicalToolCall(
                                id=tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", ""),
                                name=fn.get("name", tc.get("name", "")) if isinstance(tc, dict) else getattr(tc, "name", ""),
                                arguments=fn.get("arguments", tc.get("arguments", "")) if isinstance(tc, dict) else getattr(tc, "arguments", ""),
                            )
                        )
                converted_messages.append(
                    CanonicalMessage(
                        role=role,
                        content=m.get("content") or "",
                        tool_calls=tc_list,
                        tool_call_id=m.get("tool_call_id"),
                        name=m.get("name"),
                    )
                )
            else:
                converted_messages.append(m)

        converted_tools: list[ToolDefinition] | None = None
        if tools:
            converted_tools = []
            for t in tools:
                if isinstance(t, ToolDefinition):
                    converted_tools.append(t)
                elif isinstance(t, dict):
                    fn = t.get("function", {})
                    name = fn.get("name", t.get("name", ""))
                    desc = fn.get("description", t.get("description", ""))
                    params = fn.get("parameters", t.get("parameters", {}))
                    converted_tools.append(
                        ToolDefinition(
                            type=t.get("type", "function"),
                            function=ToolFunction(
                                name=name,
                                description=desc,
                                parameters=params,
                            ),
                        )
                    )

        accumulated_text: list[str] = []
        finish_reason: str | None = None
        has_tool_calls = False

        try:
            stream_gen = self.provider.stream(
                converted_messages,
                tools=converted_tools,
                **kwargs,
            )
        except TypeError:
            stream_gen = self.provider.stream(converted_messages)

        async for event in stream_gen:
            if isinstance(event, ProviderTextDelta):
                accumulated_text.append(event.text)
                yield CoreTextDelta(text=event.text)
            elif isinstance(event, CoreTextDelta):
                accumulated_text.append(event.text)
                yield event
            elif isinstance(event, ProviderThoughtDelta):
                yield CoreThoughtDelta(thought=event.thought)
            elif isinstance(event, CoreThoughtDelta):
                yield event
            elif isinstance(event, ProviderToolCallDelta):
                has_tool_calls = True
                yield CoreToolCallDelta(
                    index=event.index,
                    call_id=event.call_id,
                    name=event.name,
                    arguments_delta=event.arguments_delta,
                )
            elif isinstance(event, CoreToolCallDelta):
                has_tool_calls = True
                yield event
            elif isinstance(event, ProviderFinish):
                finish_reason = event.reason
            elif isinstance(event, CoreTurnComplete):
                yield event
                return
            elif isinstance(event, AgentEvent):
                yield event

        full_text = "".join(accumulated_text)
        if full_text and finish_reason != "tool_calls" and not has_tool_calls:
            yield CoreTurnComplete(final_text=full_text)


def adapt_provider(provider: Any) -> Any:
    if isinstance(provider, (LitellmModel, ProviderStreamAdapter)):
        return provider
    if "query_stream" in getattr(provider, "__dict__", {}):
        return provider
    if hasattr(type(provider), "query_stream") and not isinstance(provider, (Mock, NonCallableMock)):
        return provider
    if hasattr(provider, "stream"):
        return ProviderStreamAdapter(provider)
    return provider


def build_foreground(
    runtime: Any,
    *,
    system_prompt: str | None = None,
    worker_system_prompt: str | None = None,
    on_wake: Callable[[str], Coroutine[Any, Any, None] | None] | None = None,
) -> AgentConfig:
    """Builds the voice-first foreground agent with strictly spawn, status, cancel tools.

    Sets runtime.foreground (plus its prompts, floor, ledger, and task state) and
    returns the foreground AgentConfig for the announcer fallback.
    """
    # Load voice.yaml configuration
    config_path = (
        Path(__file__).resolve().parent.parent
        / "agent"
        / "config"
        / "voice.yaml"
    )
    raw_voice_cfg: dict[str, Any] = {}
    if config_path.is_file():
        raw_voice_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    agent_cfg_dict = raw_voice_cfg.get("agent", {})

    runtime._custom_prompt = (
        system_prompt.strip()
        if system_prompt and system_prompt.strip()
        else None
    )
    runtime._base_system_prompt = (
        agent_cfg_dict.get("system_template")
        or DEFAULT_VERBIX_PROMPT
    )
    runtime.system_prompt = runtime._build_grounded_prompt()
    runtime.worker_system_prompt = worker_system_prompt or DEFAULT_WORKER_PROMPT
    runtime.on_wake = on_wake
    runtime.wake_output: Any = None
    runtime.user_active: bool = False

    runtime.tasks: dict[str, WorkerTask] = {}
    runtime._task_counter: int = 0
    runtime._environments: list[LocalEnvironment] = []
    runtime._pending_announcements: list[tuple[str, str]] = []
    runtime._drain_lock = asyncio.Lock()
    runtime.floor = FloorArbiter(on_change=runtime._on_floor_state_changed)
    runtime.ledger = TaskLedger()
    from verbix.runtime.runtime import TurnHandle

    runtime._turn_handle: TurnHandle = TurnHandle()
    runtime._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    # Foreground tools come from voice.yaml; unknown names raise loudly in build_registry.
    fg_tool_names = agent_cfg_dict.get("tools") or ["spawn", "status", "cancel"]
    registry = build_registry(fg_tool_names, runtime, runtime.workspace)

    agent_cfg_dict["system_template"] = runtime.system_prompt
    agent_cfg_dict["system_prompt"] = runtime.system_prompt
    agent_cfg_dict["turn_scoped_limits"] = True
    fg_config = AgentConfig(**agent_cfg_dict)

    runtime.foreground = DefaultAgent(
        model=adapt_provider(runtime.provider),
        env=None,
        tools=registry,
        config=fg_config,
    )
    runtime.foreground.history = DefaultAgentHistoryProxy(runtime.foreground)
    return fg_config
