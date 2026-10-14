"""announcer: speak background worker completions through the tool-less announcer."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from verbix.agent.announcer import AnnouncerAgent, AnnouncerConfig
from verbix.core.events import AgentEvent
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


def build_announcer_agent(runtime: Any, fg_config: Any) -> AnnouncerAgent:
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
        from verbix.config import fill_persona_slot, read_persona

        custom = getattr(runtime, "_custom_prompt", None)
        persona = "" if custom else read_persona(getattr(runtime, "workspace", None))
        grounded = runtime._build_grounded_prompt(
            fill_persona_slot(announcer_agent_cfg.get("system_template") or "", persona)
        )
        announcer_agent_cfg["system_template"] = grounded
        announcer_agent_cfg["system_prompt"] = grounded
        announcer_agent_cfg["turn_scoped_limits"] = True
        announcer_config = AnnouncerConfig(**announcer_agent_cfg)
    else:
        cfg_dict = fg_config.model_dump() if hasattr(fg_config, "model_dump") else {}
        cfg_dict["tools"] = []
        cfg_dict["tool_choice"] = "none"
        announcer_config = AnnouncerConfig(**cfg_dict)
    runtime.announcer = AnnouncerAgent(
        model=adapt_provider(runtime.provider),
        env=None,
        config=announcer_config,
    )
    runtime.announcer.messages = []
    return runtime.announcer


build_announcer = build_announcer_agent


def _snapshot_foreground_history(runtime: Any) -> list:
    """Copy foreground chat history for the tool-less announcer.

    Same ordered prefix (cache-friendly), fresh objects (the announcer's
    turn cannot mutate foreground entries mid-stream). Snapshot, never a
    live alias: the foreground may append while the announcer streams.
    Tool calls and tool-role messages are stripped: they are internal
    mechanics, and their syntax is what a tool-less model imitates into
    phantom tool use. User/assistant prose (including task findings)
    passes through untouched.
    """
    foreground = getattr(runtime, "foreground", None)
    messages = getattr(foreground, "messages", None) or []
    clean: list = []
    for m in messages:
        if not isinstance(m, dict):
            clean.append(m)
            continue
        if m.get("role") == "tool":
            continue
        c = dict(m)
        c.pop("tool_calls", None)
        clean.append(c)
    return clean


def _can_announce(runtime: Any) -> bool:
    floor = getattr(runtime, "floor", None)
    if floor is None:
        return False
    can_ann = getattr(floor, "can_announce", None)
    if callable(can_ann):
        return bool(can_ann())
    return bool(getattr(floor, "is_idle", False))


async def drain_pending_announcements(runtime: Any) -> None:
    """Speaks accumulated background completions via the single foreground companion."""
    if not _can_announce(runtime) or not runtime._pending_announcements:
        return
    async with runtime._drain_lock:
        if not _can_announce(runtime) or not runtime._pending_announcements:
            return
        ledger = getattr(runtime, "ledger", None)
        batch: list[tuple[str, str, str]] = []
        for tid, res in list(runtime._pending_announcements):
            entry = ledger.get(tid) if ledger is not None else None
            if entry is not None and entry.delivery != "pending":
                runtime._pending_announcements[:] = [
                    (t, r) for t, r in runtime._pending_announcements if t != tid
                ]
                continue
            status = entry.status if entry is not None else "completed"
            if res.strip():
                batch.append((tid, res.strip(), status))
            else:
                runtime._pending_announcements[:] = [
                    (t, r) for t, r in runtime._pending_announcements if t != tid
                ]
        if not batch:
            return

        batch_ids = {tid for tid, _, _ in batch}
        runtime._pending_announcements[:] = [
            (t, r) for t, r in runtime._pending_announcements if t not in batch_ids
        ]

        items = [res for _, res, _ in batch]
        runtime.announcer.messages = _snapshot_foreground_history(runtime)
        prompt = (
            f"{runtime._request_grounding()}\n"
            "[Present these completed findings directly to the user. "
            "State the outcome clearly, and only add a follow-up question if a genuine next step exists:]\n"
            + "\n\n".join(items)
        )
        announced_ok = False
        final_text = ""
        try:
            final_text = await runtime.submit(prompt, runtime.wake_output, agent=runtime.announcer)
            last_handle = getattr(runtime, "_last_turn_handle", None)
            turn_completed = (
                getattr(last_handle, "completed", True) if last_handle is not None else True
            )
            was_interrupted = (
                bool(getattr(runtime.floor, "interrupt_requested", False))
                or bool(getattr(runtime.floor, "user_speaking", False))
                or not turn_completed
            )
            if final_text and not was_interrupted:
                announced_ok = True
                if ledger is not None:
                    for task_id, _, _ in batch:
                        ledger.mark_heard(task_id)
                for task_id, res, status in batch:
                    runtime.foreground.messages.append(
                        {"role": "user", "content": f"[Task '{task_id}' {status}]: {res}"}
                    )
                runtime.foreground.messages.append({"role": "assistant", "content": final_text})
        finally:
            if not announced_ok:
                existing_ids = {t for t, _ in runtime._pending_announcements}
                for tid, res, _ in reversed(batch):
                    entry = ledger.get(tid) if ledger is not None else None
                    if (entry is None or entry.delivery == "pending") and tid not in existing_ids:
                        runtime._pending_announcements.insert(0, (tid, res))

        if announced_ok and runtime.on_wake:
            wake_res = runtime.on_wake(final_text)
            if asyncio.iscoroutine(wake_res):
                await wake_res
