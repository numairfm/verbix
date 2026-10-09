"""workers: spawn and track independent background worker agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from verbix.agent.agents.default import AgentConfig, DefaultAgent
from verbix.agent.environments.local import LocalEnvironment
from verbix.agent.tools.registry import ToolRegistry
from verbix.runtime.foreground import (
    DEFAULT_WORKER_PROMPT,
    DefaultAgentHistoryProxy,
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


async def spawn_worker(runtime: Any, objective: str) -> str:
    """Spawns an independent background worker with full workspace tools."""
    prune_spills(Path(runtime.workspace) / ".verbix" / "logs")
    runtime._task_counter += 1
    task_id = f"task-{runtime._task_counter}"

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

    worker_config = AgentConfig(**agent_cfg_dict)

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

    worker_agent = DefaultAgent(
        model=model_to_use,
        env=env,
        tools=worker_tools,
        config=worker_config,
    )
    worker_agent.history = DefaultAgentHistoryProxy(worker_agent)

    worker_task = WorkerTask(
        id=task_id,
        description=objective,
        session=worker_agent,
        on_event=runtime._echo_worker_event if runtime.worker_echo else None,
    )

    def _on_finish(t: WorkerTask) -> None:
        if t.status == "cancelled":
            runtime.ledger.mark_cancelled(t.id)
            runtime._broadcast_event("task_cancelled", {"task_id": t.id})
            return
        runtime._on_task_finished(t.id, t.result or "Task completed.")

    worker_task.on_completed(_on_finish)
    runtime.tasks[task_id] = worker_task
    runtime._environments.append(env)
    runtime.ledger.register_task(task_id, objective)
    runtime._broadcast_event(
        "task_registered", {"task_id": task_id, "objective": objective}
    )
    worker_task.start()
    return task_id
