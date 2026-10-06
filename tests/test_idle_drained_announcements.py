import asyncio
from unittest.mock import MagicMock
import pytest
from verbix.core.events import TextDelta
from verbix.provider.models import TextDelta as ProviderTextDelta, Finish as ProviderFinish
from verbix.runtime.runtime import Runtime
from verbix.runtime.floor import FloorState

@pytest.mark.asyncio
async def test_task_finishing_while_idle_speaks_immediately(tmp_path):
    spoken = []
    class FakeOutput:
        async def emit(self, event):
            if isinstance(event, TextDelta):
                spoken.append(event.text)
        def delivered_text(self):
            return "".join(spoken)

    provider = MagicMock()
    async def fake_stream(messages, tools=None, reasoning_effort=None):
        yield ProviderTextDelta(text="Task 1 finished successfully.")
        yield ProviderFinish(reason="stop")
    provider.stream = fake_stream

    rt = Runtime(workspace=tmp_path, provider=provider)
    rt.wake_output = FakeOutput()

    assert rt.floor.is_idle is True

    # Complete a task while floor is IDLE
    rt.ledger.register_task("task-1", "check system")
    rt._on_task_finished("task-1", "system is healthy")

    # Allow drain task to run
    for _ in range(20):
        if spoken:
            break
        await asyncio.sleep(0.02)

    assert any("Task 1 finished" in s for s in spoken)
    assert rt.floor.is_idle is True
    await rt.aclose()


@pytest.mark.asyncio
async def test_task_finishing_while_busy_holds_until_floor_returns_to_idle(tmp_path):
    spoken = []
    class FakeOutput:
        async def emit(self, event):
            if isinstance(event, TextDelta):
                spoken.append(event.text)
        def delivered_text(self):
            return "".join(spoken)

    provider = MagicMock()
    turn_count = 0
    async def fake_stream(messages, tools=None, reasoning_effort=None):
        nonlocal turn_count
        turn_count += 1
        last_content = str(messages[-1].content) if messages else ""
        if "joke" in last_content:
            yield ProviderTextDelta(text="Here is a funny joke.")
        else:
            yield ProviderTextDelta(text="Background update: task 2 finished.")
        yield ProviderFinish(reason="stop")
    provider.stream = fake_stream

    rt = Runtime(workspace=tmp_path, provider=provider)
    rt.wake_output = FakeOutput()

    # 1. Simulate an active foreground turn (SPEAKING)
    async with rt.floor.acquire(FloorState.SPEAKING):
        assert rt.floor.is_speaking is True

        # Task completes DURING the active turn
        rt.ledger.register_task("task-2", "run build")
        rt._on_task_finished("task-2", "build succeeded")

        await asyncio.sleep(0.05)
        # MUST NOT have spoken the announcement yet while floor is busy!
        assert len(spoken) == 0

    # 2. Floor exited acquire block and returned to IDLE
    assert rt.floor.is_idle is True

    # Wait for pending announcement to drain on IDLE
    for _ in range(20):
        if spoken:
            break
        await asyncio.sleep(0.02)

    assert any("task 2 finished" in s for s in spoken)
    assert rt.floor.is_idle is True
    await rt.aclose()
