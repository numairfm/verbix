"""Turn-boundary tests: unexpected turn failures end as repaired, spoken
turns — the session always survives."""

import pytest

from tests.conftest import assert_history_valid
from verbix.core.events import TextDelta, ToolCallDelta, TurnComplete
from verbix.minisweagent.agents.default import AgentConfig, DefaultAgent


class BlipModel:
    """First turn: tool round then mid-stream death. Later turns: fine."""

    def __init__(self):
        self.calls = 0

    def get_template_vars(self):
        return {}

    async def query_stream(self, messages, tools=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            yield TextDelta(text="Working on it. ")
            yield ToolCallDelta(
                index=0, call_id="c1", name="spawn", arguments_delta='{"objective": "x"}'
            )
            return
        if self.calls == 2:
            yield TextDelta(text="half ")
            raise RuntimeError("Connection closed.")
        yield TurnComplete(final_text="second turn ok")


def _agent():
    return DefaultAgent(model=BlipModel(), env=None, tools=None, config=AgentConfig())


@pytest.mark.asyncio
async def test_midstream_blip_becomes_spoken_apology_with_valid_history():
    agent = _agent()
    events = [e async for e in agent.step_stream("do thing")]

    assert isinstance(events[-1], TurnComplete)
    assert events[-1].final_text.strip(), "apology must be spoken, not empty"
    assert_history_valid(agent.messages)
    assert any(
        m.get("role") == "assistant" and (m.get("content") or "").strip()
        for m in agent.messages
    ), "failure must be recorded in history"


@pytest.mark.asyncio
async def test_session_usable_after_blip():
    agent = _agent()
    [e async for e in agent.step_stream("do thing")]
    events = [e async for e in agent.step_stream("try again")]
    assert isinstance(events[-1], TurnComplete)
    assert events[-1].final_text == "second turn ok"
    assert_history_valid(agent.messages)
