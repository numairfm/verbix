from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TaskEntry:
    id: str
    objective: str
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result_summary: str = ""
    delivery: str = "pending"


class TaskLedger:
    """Single-truth record of background tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskEntry] = {}

    def register_task(self, task_id: str, objective: str) -> TaskEntry:
        entry = TaskEntry(id=task_id, objective=objective)
        self._tasks[task_id] = entry
        return entry

    def get(self, task_id: str) -> TaskEntry | None:
        return self._tasks.get(task_id)

    def mark_completed(self, task_id: str, result_summary: str = "") -> None:
        entry = self._tasks.get(task_id)
        if entry is None:
            return
        entry.status = "completed"
        entry.finished_at = time.time()
        entry.result_summary = result_summary
        entry.delivery = "pending"

    def mark_cancelled(self, task_id: str) -> None:
        entry = self._tasks.get(task_id)
        if entry is None:
            return
        entry.status = "cancelled"
        entry.finished_at = time.time()
        entry.delivery = "interrupted"

    def mark_failed(self, task_id: str, result_summary: str = "") -> None:
        entry = self._tasks.get(task_id)
        if entry is None:
            return
        entry.status = "failed"
        entry.finished_at = time.time()
        entry.result_summary = result_summary
        entry.delivery = "pending"

    def mark_heard(self, task_id: str) -> None:
        entry = self._tasks.get(task_id)
        if entry is None:
            return
        entry.delivery = "heard"

    def get_status(self, task_id: str) -> str:
        entry = self.get(task_id)
        if entry is None:
            return ""
        return f"{entry.status}: {entry.result_summary}"

    def get_compact_summary(self) -> str:
        recent = list(self._tasks.values())[-6:]
        lines = [f"- {e.id}: {e.objective[:80]} ({e.status})" for e in recent]
        return "\n".join(lines)[:600]
