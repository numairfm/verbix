import asyncio
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from verbix.runtime.runtime import Runtime
from verbix.runtime.floor import FloorState

@pytest.mark.asyncio
async def test_worker_completion_updates_ledger_without_speech(tmp_path):
    spoken = []
    class FakeOutput:
        async def emit(self, event):
            spoken.append(event)
        def delivered_text(self):
            return ""

    rt = Runtime(workspace=tmp_path, provider=MagicMock())
    rt.wake_output = FakeOutput()

    # Register and complete multiple tasks in rapid succession
    rt.ledger.register_task("task-10", "check git")
    rt.ledger.register_task("task-11", "run tests")

    rt._on_task_finished("task-10", "git clean")
    rt._on_task_finished("task-11", "tests passed")

    await asyncio.sleep(0.05)

    # Ledger must have both results
    assert "git clean" in rt.ledger.get_status("task-10")
    assert "tests passed" in rt.ledger.get_status("task-11")

    # MUST NOT have triggered unsolicited speech
    assert len(spoken) == 0
    # Floor must remain IDLE
    assert rt.floor.state == FloorState.IDLE
    await rt.aclose()
