import pytest

from verbix.core.events import TextDelta, TurnComplete
from verbix.minisweagent.agents.default import AgentConfig, DefaultAgent, cap_observation
from verbix.minisweagent.tools.registry import ToolRegistry


def test_cap_observation_bounds_large_outputs():
    huge = "a" * 50000
    capped = cap_observation(huge, max_chars=10000)
    assert len(capped) <= 10200
    assert "elided" in capped or "..." in capped
    assert capped.startswith("aaaa")
    assert capped.endswith("aaaa")


def test_cap_observation_scales_with_tool_scope():
    tools = ToolRegistry()
    tools.register(
        "status",
        {"type": "function", "function": {"name": "status"}},
        lambda args: "s" * 5000,
    )
    tools.register(
        "mytool",
        {"type": "function", "function": {"name": "mytool"}},
        lambda args: "w" * 5000,
    )
    agent = DefaultAgent(model=None, tools=tools, system_prompt="sys")
    import asyncio

    fg_obs, _ = asyncio.run(agent._execute_tool("status", {}))
    worker_obs, _ = asyncio.run(agent._execute_tool("mytool", {}))
    assert len(fg_obs) <= 1700
    assert len(worker_obs) <= 10200
    assert len(worker_obs) > len(fg_obs)


class _CostStreamModel:
    """Stub model exposing cumulative streaming cost like LitellmModel.last_stream_cost."""

    def __init__(self) -> None:
        self.last_stream_cost = 0.0
        self.calls = 0

    async def query_stream(self, messages, tools=None):
        self.calls += 1
        self.last_stream_cost += 0.05 if self.calls == 1 else 0.07
        yield TextDelta(text="done")
        yield TurnComplete(final_text="done")


@pytest.mark.asyncio
async def test_foreground_limits_reset_per_turn():
    agent = DefaultAgent(
        model=_CostStreamModel(),
        config=AgentConfig(turn_scoped_limits=True, system_template="sys"),
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
    agent = DefaultAgent(
        model=_CostStreamModel(),
        config=AgentConfig(step_limit=1, system_template="sys"),
    )
    await _consume(agent, "one")
    assert agent.turn_calls == 1
    events = await _consume(agent, "two")
    assert any(isinstance(e, TurnComplete) and e.final_text == "LimitsExceeded" for e in events)


async def _consume(agent, text):
    return [e async for e in agent.step_stream(text)]
