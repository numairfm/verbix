import asyncio
import os
import subprocess
import pytest
from verbix.runtime.buffer import CompletionBuffer
from verbix.minisweagent.agents.default import DefaultAgent, AgentConfig
from verbix.minisweagent.tools.registry import ToolRegistry
from verbix.minisweagent.environments.local import LocalEnvironment
from verbix.core.events import ToolCallDelta

@pytest.mark.asyncio
async def test_regression_completion_buffer_drops_concurrent_finish():
    flushed = []
    completed = []

    async def on_flush(batch):
        flushed.append(batch)
        await asyncio.sleep(0.1)
        completed.append(batch)

    buf = CompletionBuffer(window_seconds=0.02, on_flush=on_flush)
    buf.push("task-1", "res-1")
    await asyncio.sleep(0.04)  # Flush task-1 begins
    buf.push("task-2", "res-2")  # Second push arrives while flush is in-flight
    await asyncio.sleep(0.2)

    # In current buggy buffer, task-1's flush was cancelled mid-flight, so completed only has task-2
    all_completed_ids = [t[0] for batch in completed for t in batch]
    assert "task-1" in all_completed_ids
    assert "task-2" in all_completed_ids

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
        await asyncio.to_thread(env.execute, {"command": f"exec -a {marker} sleep 10"})

    t = asyncio.create_task(run())
    await asyncio.sleep(0.1)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

    await asyncio.sleep(0.1)
    alive = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout.strip()
    assert not alive, f"Process {alive} survived task cancellation!"
