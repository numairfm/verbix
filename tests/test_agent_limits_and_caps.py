import asyncio

import pytest

from verbix.agent.tools.registry import ToolRegistry
from verbix.agent.voice import VoiceAgent, VoiceConfig
from verbix.core.events import TextDelta, ToolCallDelta, TurnComplete


class _CostStreamModel:
    """Stub model exposing cumulative streaming cost like LitellmModel.last_stream_cost."""

    def __init__(self) -> None:
        self.last_stream_cost = 0.0
        self.calls = 0

    async def query_stream(self, messages, tools=None, **kw):
        self.calls += 1
        self.last_stream_cost += 0.05 if self.calls == 1 else 0.07
        yield TextDelta(text="done")
        yield TurnComplete(final_text="done")


@pytest.mark.asyncio
async def test_foreground_limits_reset_per_turn():
    agent = VoiceAgent(
        model=_CostStreamModel(),
        config=VoiceConfig(turn_scoped_limits=True, system_template="sys"),
    )
    await _consume(agent, "one")
    assert agent.turn_calls == 1
    assert agent.turn_cost == pytest.approx(0.05)
    assert agent.cost == pytest.approx(0.05)
    assert agent.n_calls == 1

    # Second turn must reset turn-scoped counters while globals accumulate.
    await _consume(agent, "two")
    assert agent.turn_calls == 1
    assert agent.turn_cost == pytest.approx(0.07)
    assert agent.cost == pytest.approx(0.12)
    assert agent.n_calls == 2

    # Scoped step limit applies per turn: a third turn still proceeds.
    agent.config.step_limit = 1
    await _consume(agent, "three")
    assert agent.turn_calls == 1


@pytest.mark.asyncio
async def test_global_limits_without_turn_scope():
    agent = VoiceAgent(
        model=_CostStreamModel(),
        config=VoiceConfig(step_limit=1, turn_scoped_limits=False, system_template="sys"),
    )
    await _consume(agent, "one")
    assert agent.turn_calls == 1
    events = await _consume(agent, "two")
    assert any(
        isinstance(e, TurnComplete) and e.exit_reason == "limits_exceeded"
        for e in events
    )


async def _consume(agent, text):
    return [e async for e in agent.step_stream(text)]


class _RepeatToolModel:
    """Stub model repeating one identical tool call, then giving up."""

    def __init__(self) -> None:
        self.last_stream_cost = 0.0
        self.calls = 0

    async def query_stream(self, messages, tools=None, **kw):
        self.calls += 1
        if self.calls > 10:
            yield TextDelta(text="done")
            yield TurnComplete(final_text="done")
            return
        yield ToolCallDelta(
            index=0, call_id=f"c{self.calls}", name="status", arguments_delta="{}"
        )


def _counting_registry(counter: list) -> ToolRegistry:
    reg = ToolRegistry()

    async def _handler(args):
        counter.append(args)
        return "ok"

    reg.register(
        "status",
        {"type": "function", "function": {"name": "status", "description": "s"}},
        _handler,
    )
    return reg


@pytest.mark.asyncio
async def test_repeated_identical_tool_call_stops_turn():
    executed: list = []
    agent = VoiceAgent(
        model=_RepeatToolModel(),
        tools=_counting_registry(executed),
        config=VoiceConfig(system_template="sys", max_calls_per_tool_per_turn={}),
    )
    events = await asyncio.wait_for(_consume(agent, "hi"), timeout=30)
    assert len(executed) == 3
    assert any(
        isinstance(e, TurnComplete) and e.exit_reason == "repeated_tool_call"
        for e in events
    )


@pytest.mark.asyncio
async def test_varying_tool_calls_do_not_trip_repeat_guard():
    class _VaryingModel:
        last_stream_cost = 0.0

        def __init__(self) -> None:
            self.calls = 0

        async def query_stream(self, messages, tools=None, **kw):
            self.calls += 1
            if self.calls > 3:
                yield TextDelta(text="done")
                yield TurnComplete(final_text="done")
                return
            yield ToolCallDelta(
                index=0,
                call_id=f"c{self.calls}",
                name="status",
                arguments_delta=f'{{"n": {self.calls}}}',
            )

    executed: list = []
    agent = VoiceAgent(
        model=_VaryingModel(),
        tools=_counting_registry(executed),
        config=VoiceConfig(system_template="sys", max_calls_per_tool_per_turn={}),
    )
    events = await asyncio.wait_for(_consume(agent, "hi"), timeout=30)
    assert len(executed) == 3
    assert not any(
        isinstance(e, TurnComplete) and e.exit_reason == "repeated_tool_call"
        for e in events
    )
