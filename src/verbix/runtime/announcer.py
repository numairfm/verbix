"""announcer: speak background worker completions through the tool-less announcer."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from verbix.agent.agents.default import AgentConfig, DefaultAgent
from verbix.agent.tools.registry import ToolRegistry
from verbix.core.events import AgentEvent
from verbix.core.events import (
    TurnComplete as CoreTurnComplete,
)
from verbix.runtime.foreground import adapt_provider

logger = logging.getLogger("verbix.runtime")


class WakeOutputFanOut:
    """Fans wake-turn events out to the voice sink and IPC subscriber queues."""

    def __init__(
        self,
        voice_output: Any = None,
        broadcast: Callable[[Any], Any] | None = None,
    ) -> None:
        self.voice_output = voice_output
        self._broadcast = broadcast
        self._delivered: list[str] = []

    async def emit(self, event: AgentEvent) -> None:
        text = getattr(event, "text", "")
        if text:
            self._delivered.append(text)
        if self.voice_output is not None and hasattr(self.voice_output, "emit"):
            res = self.voice_output.emit(event)
            if asyncio.iscoroutine(res):
                await res
        if self._broadcast is not None:
            res = self._broadcast(event)
            if asyncio.iscoroutine(res):
                await res

    def delivered_text(self) -> str:
        delivered = getattr(self.voice_output, "delivered_text", None)
        if callable(delivered):
            try:
                res = delivered()
                if isinstance(res, str) and res:
                    return res
            except Exception:  # noqa: BLE001, S110
                pass
        return "".join(self._delivered)

    def reset_delivered(self) -> None:
        self._delivered.clear()
        reset = getattr(self.voice_output, "reset_delivered", None)
        if callable(reset):
            try:
                reset()
            except Exception:  # noqa: BLE001, S110
                pass


def build_announcer(runtime: Any, fg_config: AgentConfig) -> DefaultAgent:
    """Builds the tool-less announcer sharing the foreground voice and grounding."""
    announcer_cfg_path = (
        Path(__file__).resolve().parent.parent
        / "agent"
        / "config"
        / "announcer.yaml"
    )
    if announcer_cfg_path.is_file():
        announcer_agent_cfg = (
            yaml.safe_load(announcer_cfg_path.read_text(encoding="utf-8")) or {}
        ).get("agent", {})
        grounded = runtime._build_grounded_prompt(announcer_agent_cfg.get("system_template"))
        announcer_agent_cfg["system_template"] = grounded
        announcer_agent_cfg["system_prompt"] = grounded
        announcer_agent_cfg["turn_scoped_limits"] = True
        announcer_config = AgentConfig(**announcer_agent_cfg)
    else:
        announcer_config = fg_config.model_copy(update={"tools": []})
    runtime.announcer = DefaultAgent(
        model=adapt_provider(runtime.provider),
        env=None,
        tools=ToolRegistry(),
        config=announcer_config,
    )
    runtime.announcer.messages = []
    return runtime.announcer


async def drain_pending_announcements(runtime: Any) -> None:
    """Speaks accumulated background completions via the single foreground companion."""
    if not runtime.floor.is_idle or not runtime._pending_announcements:
        return
    async with runtime._drain_lock:
        if not runtime.floor.is_idle or not runtime._pending_announcements:
            return
        batch = list(runtime._pending_announcements)
        runtime._pending_announcements.clear()

        items = [res.strip() for _, res in batch if res.strip()]
        if not items:
            return
        prompt = (
            f"{runtime._request_grounding()}\n"
            "[Present these completed findings directly to the user. "
            "State the outcome clearly, then invite their next instruction:]\n"
            + "\n\n".join(items)
        )
        final_text = await runtime.submit(prompt, runtime.wake_output, agent=runtime.announcer)
        if final_text and runtime.on_wake:
            wake_res = runtime.on_wake(final_text)
            if asyncio.iscoroutine(wake_res):
                await wake_res


async def speak_batch(runtime: Any, batch: list[tuple[str, str]]) -> bool:
    """Synthesizes background worker results via dedicated announcer and delivers to output."""
    items = [result.strip() for _, result in batch if result.strip()]
    if not items:
        return False
    wake_prompt = (
        f"{runtime._request_grounding()}\n"
        "[Present these completed findings directly to the user in spoken conversational prose. "
        "State the outcome clearly, then invite their next instruction:]\n"
        + "\n\n".join(items)
    )
    final_text = ""
    fan_out = WakeOutputFanOut(runtime.wake_output, runtime._broadcast_wake_event)
    try:
        runtime.announcer.messages = []
        runtime.announcer.n_calls = 0
        runtime.announcer.cost = 0.0
        runtime.announcer.model = adapt_provider(runtime.provider)
        async for event in runtime.announcer.step_stream(wake_prompt):
            if isinstance(event, CoreTurnComplete) and event.final_text in (
                "LimitsExceeded",
                "TimeExceeded",
            ):
                logger.warning(
                    "Announcer step aborted with %s", event.final_text
                )
                continue
            await fan_out.emit(event)
            if isinstance(event, CoreTurnComplete) and event.final_text:
                final_text = event.final_text
    except Exception:
        logger.exception("Failed to synthesize background task completion announcement")
        return False

    if final_text:
        runtime.foreground.messages.append({"role": "assistant", "content": final_text})
        if runtime.on_wake:
            wake_res = runtime.on_wake(final_text)
            if asyncio.iscoroutine(wake_res):
                await wake_res
        return True
    return False
