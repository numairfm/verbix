import asyncio

import pytest

from verbix.runtime.announcer import AnnouncerLoop
from verbix.runtime.floor import FloorArbiter, FloorState
from verbix.runtime.ledger import TaskLedger


@pytest.mark.asyncio
async def test_announcer_loop_batches_without_dropping_items():
    floor = FloorArbiter()
    ledger = TaskLedger()
    announced = []

    async def speak_fn(batch):
        announced.append(batch)
        await asyncio.sleep(0.05)

    announcer = AnnouncerLoop(floor=floor, ledger=ledger, speak_fn=speak_fn, batch_window=0.03)
    announcer.start()

    ledger.register_task("t1", "obj 1")
    ledger.mark_completed("t1", "res 1")
    announcer.push("t1")

    await asyncio.sleep(0.04)  # Announcement in progress
    ledger.register_task("t2", "obj 2")
    ledger.mark_completed("t2", "res 2")
    announcer.push("t2")       # Second push during active announcement

    await asyncio.sleep(0.15)
    await announcer.stop()

    flattened = [item for batch in announced for item in batch]
    assert "t1" in flattened
    assert "t2" in flattened


@pytest.mark.asyncio
async def test_announcer_loop_merges_burst_within_window():
    floor = FloorArbiter()
    ledger = TaskLedger()
    announced = []

    async def speak_fn(batch):
        announced.append(list(batch))

    announcer = AnnouncerLoop(floor=floor, ledger=ledger, speak_fn=speak_fn, batch_window=0.05)
    announcer.start()

    for tid in ("t1", "t2"):
        ledger.register_task(tid, f"obj {tid}")
        ledger.mark_completed(tid, f"res {tid}")
        announcer.push(tid)  # Both pushes land inside one batch window

    await asyncio.sleep(0.15)
    await announcer.stop()

    assert len(announced) == 1
    assert set(announced[0]) == {"t1", "t2"}
    assert ledger.get("t1").delivery == "heard"
    assert ledger.get("t2").delivery == "heard"
