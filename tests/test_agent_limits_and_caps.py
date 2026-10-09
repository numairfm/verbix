import pytest

from verbix.agent.agents.default import AgentConfig, DefaultAgent
from verbix.core.events import TextDelta, TurnComplete


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
