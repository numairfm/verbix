"""workers: spawn and track independent background worker agents."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import yaml

from verbix.agent.environments.local import LocalEnvironment
from verbix.agent.tools.registry import ToolRegistry
from verbix.agent.worker import WorkerAgent, WorkerConfig
from verbix.runtime.announcer import drain_pending_announcements
from verbix.runtime.foreground import (
    DEFAULT_WORKER_PROMPT,
    adapt_provider,
)
from verbix.runtime.worker import WorkerTask
from verbix.tools.shared import prune_spills


def build_worker_registry(workspace: Path | None = None) -> ToolRegistry:
    """Background worker's tools: read, write, edit, bash, finish.

    File work goes through the dedicated tools, identically on every OS;
    bash is the escape hatch for what has no dedicated tool.
    """
    from verbix.tools import build_registry

    return build_registry(
        ["read", "write", "edit", "bash", "finish"], workspace=workspace
    )


def finish_runtime_task(runtime: Any, task_id: str, result: str) -> None:
    """Records task completion into the ledger, context memory, and schedules announcement."""
    if hasattr(runtime, "ledger"):
        runtime.ledger.mark_completed(task_id, result)
    if hasattr(runtime, "_broadcast_event"):
        runtime._broadcast_event("task_completed", {"task_id": task_id, "result": result})
    if hasattr(runtime, "foreground") and hasattr(runtime.foreground, "messages"):
        runtime.foreground.messages.append({
            "role": "user",
            "content": f"[Task '{task_id}' completed]: {result}",
        })
    if hasattr(runtime, "_pending_announcements"):
        runtime._pending_announcements.append((task_id, result))
    if getattr(runtime, "floor", None) and getattr(runtime.floor, "is_idle", False):
        try:
            loop = asyncio.get_running_loop()
            drain = getattr(runtime, "_drain_pending_announcements", None) or (
                lambda: drain_pending_announcements(runtime)
            )
            loop.create_task(drain())
        except RuntimeError:
            pass


def fail_runtime_task(runtime: Any, task_id: str, result: str) -> None:
    """Records task failure into the ledger, context memory, and schedules announcement."""
    if hasattr(runtime, "ledger"):
        runtime.ledger.mark_failed(task_id, result)
    if hasattr(runtime, "_broadcast_event"):
        runtime._broadcast_event("task_failed", {"task_id": task_id, "result": result})
    if hasattr(runtime, "foreground") and hasattr(runtime.foreground, "messages"):
        runtime.foreground.messages.append({
            "role": "user",
            "content": f"[Task '{task_id}' failed]: {result}",
        })
    if hasattr(runtime, "_pending_announcements"):
        runtime._pending_announcements.append((task_id, result))
    if getattr(runtime, "floor", None) and getattr(runtime.floor, "is_idle", False):
        try:
            loop = asyncio.get_running_loop()
            drain = getattr(runtime, "_drain_pending_announcements", None) or (
                lambda: drain_pending_announcements(runtime)
            )
            loop.create_task(drain())
        except RuntimeError:
            pass


async def spawn_worker(runtime: Any, objective: str) -> str:
    """Spawns an independent background worker with full workspace tools."""
    prune_spills(Path(runtime.workspace) / ".verbix" / "logs")
    runtime._task_counter += 1
    task_id = f"task-{runtime._task_counter}"
    scratch_dir = Path(runtime.workspace) / ".verbix" / "tasks" / task_id
    scratch_dir.mkdir(parents=True, exist_ok=True)

    config_dir = Path(__file__).resolve().parent.parent / "agent" / "config"
    worker_path = config_dir / "worker.yaml"
    if not worker_path.is_file():
        worker_path = config_dir / "swe_worker.yaml"

    raw_worker_cfg: dict[str, Any] = {}
    if worker_path.is_file():
        raw_worker_cfg = (
            yaml.safe_load(worker_path.read_text(encoding="utf-8")) or {}
        )

    agent_cfg_dict = raw_worker_cfg.get("agent", {})
    if runtime.worker_system_prompt != DEFAULT_WORKER_PROMPT:
        agent_cfg_dict["system_template"] = runtime.worker_system_prompt
        agent_cfg_dict["system_prompt"] = runtime.worker_system_prompt

    worker_config = WorkerConfig(**agent_cfg_dict)

    env_cfg_dict = raw_worker_cfg.get("environment", {})
    env_extra = env_cfg_dict.get("env", {})
    timeout = int(env_cfg_dict.get("timeout", 600))
    env = LocalEnvironment(cwd=str(runtime.workspace), env=env_extra, timeout=timeout)

    model_to_use = adapt_provider(runtime.worker_model or runtime.provider)
    from verbix.tools import build_registry
    from verbix.tools.shared import resolve_output_limits

    worker_tool_names = agent_cfg_dict.get("tools") or [
        "read",
        "write",
        "edit",
        "bash",
        "finish",
    ]
    worker_tools = build_registry(
        worker_tool_names,
        workspace=runtime.workspace,
        output_limits=resolve_output_limits(agent_cfg_dict.get("tool_output")),
    )

    worker_agent = WorkerAgent(
        model=model_to_use,
        env=env,
        tools=worker_tools,
        config=worker_config,
    )
    worker_agent.extra_template_vars["scratch_dir"] = str(scratch_dir)

    worker_task = WorkerTask(
        id=task_id,
        description=objective,
        session=worker_agent,
        on_event=runtime._echo_worker_event if runtime.worker_echo else None,
    )

    def _on_finish(t: WorkerTask) -> None:
        if t.status == "cancelled":
            if hasattr(runtime, "ledger"):
                runtime.ledger.mark_cancelled(t.id)
            if hasattr(runtime, "_broadcast_event"):
                runtime._broadcast_event("task_cancelled", {"task_id": t.id})
            return
        if t.status == "failed":
            result = t.result or "Task failed without a verified result."
            fail_runtime_task(runtime, t.id, result)
            return
        if hasattr(runtime, "_on_task_finished"):
            runtime._on_task_finished(t.id, t.result or "Task completed.")
        else:
            finish_runtime_task(runtime, t.id, t.result or "Task completed.")

    worker_task.on_completed(_on_finish)
    runtime.tasks[task_id] = worker_task
    runtime._environments.append(env)
    runtime.ledger.register_task(task_id, objective)
    runtime._broadcast_event(
        "task_registered", {"task_id": task_id, "objective": objective}
    )
    worker_task.start()
    return task_id
