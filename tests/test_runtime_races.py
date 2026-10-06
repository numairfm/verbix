import asyncio
import os
import subprocess
import pytest
from verbix.runtime.floor import FloorArbiter
from verbix.runtime.ledger import TaskLedger
from verbix.minisweagent.agents.default import DefaultAgent, AgentConfig
from verbix.minisweagent.tools.registry import ToolRegistry
from verbix.minisweagent.environments.local import LocalEnvironment
from verbix.core.events import ToolCallDelta

@pytest.mark.asyncio
async def test_regression_completion_buffer_drops_concurrent_finish():
    # Announcer collapsed; concurrent completions record directly into TaskLedger without dropping.
    ledger = TaskLedger()

    for task_id, res in (("task-1", "res-1"), ("task-2", "res-2")):
        ledger.register_task(task_id, task_id)
        ledger.mark_completed(task_id, res)

    assert "res-1" in ledger.get_status("task-1")
    assert "res-2" in ledger.get_status("task-2")

@pytest.mark.asyncio
async def test_regression_cancel_mid_tool_call_orphans_state():
    class SlowToolModel:
        async def query_stream(self, messages, tools=None, **kw):
            yield ToolCallDelta(index=0, call_id="c1", name="slow", arguments_delta="{}")

    reg = ToolRegistry()
    async def slow(args, **kw):
        await asyncio.sleep(5)
        return "done"
    reg.register("slow", {"type": "function", "function": {"name": "slow", "parameters": {}}}, slow)

    agent = DefaultAgent(model=SlowToolModel(), tools=reg, config=AgentConfig(system_template="sys"))
    async def run():
        async for _ in agent.step_stream("hi"):
            pass

    t = asyncio.create_task(run())
    await asyncio.sleep(0.05)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

    # In current codebase, agent.messages ends with assistant message containing tool_calls without tool response
    last = agent.messages[-1]
    assert last.get("role") != "assistant" or not last.get("tool_calls"), "Dangling tool_calls message left in history!"

@pytest.mark.asyncio
async def test_regression_worker_cancel_kills_process_group():
    env = LocalEnvironment(cwd="/tmp", timeout=10)
    marker = f"verbix_test_proc_{os.getpid()}_{int(asyncio.get_event_loop().time() * 1000)}"

    async def run():
        await env.aexecute({"command": f"exec -a {marker} sleep 10"})

    t = asyncio.create_task(run())
    await asyncio.sleep(0.1)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

    await asyncio.sleep(0.1)
    alive = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout.strip()
    assert not alive, f"Process {alive} survived task cancellation!"
