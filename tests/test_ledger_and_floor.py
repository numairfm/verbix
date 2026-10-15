import asyncio

import pytest

from verbix.runtime.floor import FloorArbiter, FloorState
from verbix.runtime.ledger import TaskLedger


def test_ledger_state_transitions():
    ledger = TaskLedger()
    ledger.register_task("task-1", "calculate metrics")
    entry = ledger.get("task-1")
    assert entry.status == "running"
    ledger.mark_completed("task-1", "metric result = 42")
    assert entry.status == "completed"
    assert entry.delivery == "pending"


def test_ledger_compact_summary_bounds():
    ledger = TaskLedger()
    for i in range(8):
        ledger.register_task(f"task-{i}", f"objective number {i} with some extra words")
        ledger.mark_completed(f"task-{i}", f"result {i}")
    summary = ledger.get_compact_summary()
    assert len(summary) <= 600
    assert "task-0" not in summary
    assert "task-7" in summary


@pytest.mark.asyncio
async def test_floor_arbiter_transitions_and_interrupt():
    floor = FloorArbiter()
    assert floor.state == FloorState.IDLE
    async with floor.acquire(FloorState.FOREGROUND):
        assert floor.state == FloorState.FOREGROUND
    assert floor.state == FloorState.IDLE


@pytest.mark.asyncio
async def test_floor_single_owner_and_sync_interrupt():
    floor = FloorArbiter()
    entered = []
    async with floor.acquire(FloorState.FOREGROUND):
        entered.append(floor.state)

        async def contender():
            async with floor.acquire(FloorState.ANNOUNCING):
                entered.append(floor.state)

        t = asyncio.create_task(contender())
        await asyncio.sleep(0.02)
        assert entered == [FloorState.FOREGROUND]  # contender waits: single owner
        floor.request_interrupt()  # synchronous flag, no await needed
        assert floor.interrupt_requested
    await t
    assert entered == [FloorState.FOREGROUND, FloorState.ANNOUNCING]
    assert floor.state == FloorState.IDLE


@pytest.mark.asyncio
async def test_runtime_wires_floor_and_ledger(tmp_path):
    from unittest.mock import MagicMock

    from verbix.provider.models import Finish as ProviderFinish
    from verbix.provider.models import TextDelta as ProviderTextDelta
    from verbix.runtime.runtime import Runtime

    provider = MagicMock()

    async def fake_stream(messages, tools=None, **kw):
        yield ProviderTextDelta(text="done")
        yield ProviderFinish(reason="stop")

    provider.stream = fake_stream
    rt = Runtime(workspace=tmp_path, provider=provider)
    assert isinstance(rt.floor, FloorArbiter)
    assert isinstance(rt.ledger, TaskLedger)
    assert rt.is_idle
    assert rt.floor.state == FloorState.IDLE
    await rt.submit("hello")
    assert rt.floor.state == FloorState.IDLE
    assert rt.is_idle


def test_floor_can_announce_only_when_idle_and_not_interrupted():
    floor = FloorArbiter()
    assert floor.can_announce() is True

    floor.set_state(FloorState.LISTENING)
    assert floor.can_announce() is False

    floor.set_state(FloorState.SPEAKING)
    assert floor.can_announce() is False

    floor.set_state(FloorState.IDLE)
    floor.request_interrupt()
    assert floor.can_announce() is False
    assert not hasattr(floor, "user_speaking")

