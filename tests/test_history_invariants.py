from verbix.agent.utils.history import validate_and_repair_history


def test_repair_unanswered_tool_calls():
    msgs = [
        {"role": "user", "content": "run task"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}]},
    ]
    repaired = validate_and_repair_history(msgs)
    assert len(repaired) == 3
    assert repaired[-1]["role"] == "tool"
    assert repaired[-1]["tool_call_id"] == "call_1"
    assert "cancelled" in repaired[-1]["content"]


def test_repair_truncates_to_delivered_speech():
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "This is a very long response that was interrupted halfway."},
    ]
    repaired = validate_and_repair_history(msgs, delivered_text="This is a very")
    assert repaired[-1]["content"] == "This is a very"


import asyncio

import pytest

from verbix.agent.agents.default import AgentConfig, DefaultAgent
from verbix.agent.tools.registry import ToolRegistry
from verbix.core.events import ToolCallDelta


@pytest.mark.asyncio
async def test_shielded_cancel_records_single_tool_response():
    started = asyncio.Event()
    release = asyncio.Event()

    async def spawn(args, **kw):
        started.set()
        await release.wait()
        return "task-1 started"

    reg = ToolRegistry()
    reg.register(
        "spawn",
        {"type": "function", "function": {"name": "spawn", "parameters": {}}},
        spawn,
        side_effecting=True,
    )

    class SlowSpawnModel:
        async def query_stream(self, messages, tools=None, **kw):
            yield ToolCallDelta(index=0, call_id="c1", name="spawn", arguments_delta="{}")

    agent = DefaultAgent(model=SlowSpawnModel(), tools=reg, config=AgentConfig(system_template="sys"))

    async def run():
        async for _ in agent.step_stream("hi"):
            pass

    t = asyncio.create_task(run())
    await asyncio.wait_for(started.wait(), timeout=10)
    # Simulate cancel_turn(): task cancel + synchronous repair, no await between.
    t.cancel()
    agent.messages[:] = validate_and_repair_history(agent.messages)
    assert sum(1 for m in agent.messages if m.get("tool_call_id") == "c1") == 1
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await t

    tool_msgs = [m for m in agent.messages if m.get("tool_call_id") == "c1"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "task-1 started"


def test_double_repair_is_idempotent():
    msgs = [
        {"role": "user", "content": "run task"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}]},
    ]
    once = validate_and_repair_history(msgs)
    twice = validate_and_repair_history(once)
    assert once == twice
    assert sum(1 for m in twice if m.get("tool_call_id") == "call_1") == 1


def test_repair_truncates_speech_even_with_tool_calls():
    msgs = [
        {"role": "user", "content": "check status"},
        {
            "role": "assistant",
            "content": "I will check that right now",
            "tool_calls": [{"id": "c1", "function": {"name": "status", "arguments": "{}"}}],
        },
    ]
    repaired = validate_and_repair_history(msgs, delivered_text="I will check")
    asst = next(m for m in repaired if m.get("role") == "assistant")
    assert asst["content"] == "I will check"
    tool_resp = next(m for m in repaired if m.get("role") == "tool")
    assert tool_resp["tool_call_id"] == "c1"


def test_repair_truncates_final_speech_after_completed_tool_calls():
    msgs = [
        {"role": "user", "content": "check status"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "status", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {
            "role": "assistant",
            "content": "Status is ok, everything is working properly.",
        },
    ]
    repaired = validate_and_repair_history(msgs, delivered_text="Status is ok")
    assts = [m for m in repaired if m.get("role") == "assistant"]
    assert assts[-1]["content"] == "Status is ok"


def test_repair_removes_current_turn_assistant_not_identical_prior_turn():
    """Cancelling a turn with 0 delivered speech must remove the current turn's assistant,
    not an earlier turn's identical message."""
    msgs = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "acknowledged"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "acknowledged"},
    ]
    repaired = validate_and_repair_history(msgs, delivered_text="")
    assert len(repaired) == 3
    assert repaired[0]["content"] == "turn 1"
    assert repaired[1]["content"] == "acknowledged"
    assert repaired[2]["content"] == "turn 2"


