import asyncio
from unittest.mock import MagicMock

import pytest

from verbix.core.events import TextDelta, TurnComplete
from verbix.runtime.buffer import CompletionBuffer
from verbix.runtime.worker import WorkerTask


@pytest.mark.asyncio
async def test_completion_buffer_batches_within_window():
    flushed = []

    async def on_flush(batch):
        flushed.append(batch)

    buf = CompletionBuffer(window_seconds=0.05, on_flush=on_flush)
    buf.push("task-1", "Done 1")
    await asyncio.sleep(0.01)
    buf.push("task-2", "Done 2")
    await asyncio.sleep(0.1)

    assert len(flushed) == 1
    assert flushed[0] == [("task-1", "Done 1"), ("task-2", "Done 2")]


@pytest.mark.asyncio
async def test_completion_buffer_flush_now():
    flushed = []

    async def on_flush(batch):
        flushed.append(batch)

    buf = CompletionBuffer(window_seconds=1.0, on_flush=on_flush)
    buf.push("task-1", "Immediate")
    await buf.flush_now()

    assert len(flushed) == 1
    assert flushed[0] == [("task-1", "Immediate")]


@pytest.mark.asyncio
async def test_worker_task_lifecycle_success():
    session = MagicMock()

    async def fake_step(prompt):
        yield TextDelta(text="Working...")
        yield TurnComplete(final_text="All tests passed successfully.")

    session.step = fake_step
    session.history.messages = []

    task = WorkerTask(id="task-1", description="run tests", session=session)
    assert task.status == "pending"

    completed_events = []
    task.on_completed(lambda t: completed_events.append(t.id))

    task.start()
    assert task.status == "running"

    await task.wait()
    assert task.status == "completed"
    assert task.result == "All tests passed successfully."
    assert completed_events == ["task-1"]


@pytest.mark.asyncio
async def test_worker_task_cancellation():
    session = MagicMock()

    async def hanging_step(prompt):
        while True:
            yield TextDelta(text="still running...")
            await asyncio.sleep(0.01)

    session.step = hanging_step
    session.history.messages = []

    task = WorkerTask(id="task-2", description="long task", session=session)
    task.start()
    await asyncio.sleep(0.02)
    assert task.status == "running"

    await task.cancel()
    assert task.status == "cancelled"


@pytest.mark.asyncio
async def test_worker_task_get_status_reading_history():
    from verbix.core.models import CanonicalMessage, CanonicalToolCall

    session = MagicMock()
    session.history.messages = [
        CanonicalMessage(role="user", content="run tests"),
        CanonicalMessage(
            role="assistant",
            tool_calls=[CanonicalToolCall(id="c1", name="bash", arguments='{"cmd": "pytest"}')],
        ),
        CanonicalMessage(role="tool", content="14 passed", tool_call_id="c1"),
        CanonicalMessage(role="assistant", content="Done."),
    ]

    task = WorkerTask(id="task-3", description="run tests", session=session)
    task._status = "running"

    # Concise status
    concise = task.get_status(full=False)
    assert "task-3" in concise
    assert "bash" in concise or "14 passed" in concise

    # Full status
    full = task.get_status(full=True)
    assert "task-3" in full
    assert "run tests" in full
    assert "14 passed" in full


@pytest.mark.asyncio
async def test_worker_task_extracts_final_result_not_intermediate_scratchpad():
    session = MagicMock()
    async def fake_step_stream(desc):
        yield TextDelta(text="I will start by analyzing the codebase...")
        yield TurnComplete(final_text="")

    session.step_stream = fake_step_stream
    session.messages = [
        {"role": "user", "content": "list dependencies"},
        {"role": "assistant", "content": "I will start by analyzing the codebase...", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "content": "pyproject.toml found"},
        {"role": "assistant", "content": "The dependencies are pytest, litellm."},
    ]

    task = WorkerTask(id="task-4", description="list dependencies", session=session)
    task.start()
    await task.wait()

    assert task.status == "completed"
    assert task.result == "The dependencies are pytest, litellm."
    assert "analyzing the codebase" not in (task.result or "")


def test_worker_task_status_timestamps():
    import re
    session = MagicMock()
    session.messages = []
    task = WorkerTask(id="task-5", description="quick task", session=session)
    task._status = "running"
    task._created_at = 1000000.0
    task._finished_at = 1000015.0

    running_status = task.get_status()
    assert re.search(r"running for \d+s", running_status)
    assert "started at " in running_status

    task._status = "completed"
    task._result = "Done"
    completed_status = task.get_status()
    assert "completed in 15s" in completed_status
    assert "started at " in completed_status
    assert "finished at " in completed_status
