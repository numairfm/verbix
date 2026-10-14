"""Worker watch: background turns echo to the CLI when enabled."""

import asyncio

from verbix.core.events import TextDelta, ToolCallDelta, TurnComplete
from verbix.runtime.worker import WorkerTask, format_worker_event


class _Session:
    def __init__(self, events):
        self._events = events
        self.messages = []
        self.on_event = None

    async def run(self, description):
        for e in self._events:
            if self.on_event is not None:
                self.on_event(e)
        return {"status": "completed", "summary": "done"}


def _run(task):
    async def go():
        task.start()
        await task.wait()

    asyncio.run(go())


def test_worker_task_emits_events_to_callback():
    seen = []
    task = WorkerTask(
        "task-1",
        "demo",
        _Session([TextDelta(text="hi"), TurnComplete(final_text="hi")]),
        on_event=lambda tid, ev: seen.append((tid, ev)),
    )
    _run(task)
    assert [t for t, _ in seen] == ["task-1", "task-1"]
    assert isinstance(seen[-1][1], TurnComplete)


def test_worker_task_without_callback_stays_silent():
    task = WorkerTask("task-1", "demo", _Session([TextDelta(text="hi")]))
    _run(task)
    assert task.status == "completed"


def test_format_worker_event_tags_turns():
    assert format_worker_event("task-2", TextDelta(text="hello")) == "[task-2] hello"
    out = format_worker_event(
        "task-2", ToolCallDelta(index=0, name="read", arguments_delta='{"path": "a"}')
    )
    assert out == '[task-2] $ read({"path": "a"})'
    assert format_worker_event("task-2", TurnComplete(final_text="done")) is None


def test_format_worker_event_covers_started_and_finished():
    from verbix.core.events import ToolCallFinished, ToolCallStarted

    out = format_worker_event(
        "task-2", ToolCallStarted(call_id="c1", tool_name="read", arguments={"path": "a"})
    )
    assert out == '[task-2] $ read({"path": "a"})'
    assert (
        format_worker_event(
            "task-2", ToolCallFinished(call_id="c1", observation="ok")
        )
        is None
    )


def test_format_worker_event_masks_format_errors():
    ev = TextDelta(text="! format error: No tool calls found.")
    assert format_worker_event("task-1", ev) == "[task-1] (thinking...)"

