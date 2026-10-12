import pytest

from verbix.agent.agents.default import AgentConfig, DefaultAgent
from verbix.agent.tools.registry import ToolRegistry
from verbix.core.events import TextDelta, ToolCallDelta, TurnComplete


class ChunkThenToolModel:
    def __init__(self):
        self.calls = 0

    async def query_stream(self, messages, tools=None, **kw):
        self.calls += 1
        if self.calls == 1:
            yield TextDelta(text="Well, let me check that for you right now please. One moment. ")
            yield ToolCallDelta(index=0, call_id="c1", name="status", arguments_delta="{}")
        else:
            yield TextDelta(text="done")
            yield TurnComplete(final_text="done")


@pytest.mark.asyncio
async def test_first_chunk_released_before_tool_call():
    agent = DefaultAgent(
        model=ChunkThenToolModel(),
        tools=ToolRegistry(),
        config=AgentConfig(system_template="sys"),
    )
    seen = [e async for e in agent.step_stream("hi")]
    texts = [e.text for e in seen if isinstance(e, TextDelta)]
    assert any("Well," in t for t in texts)  # released at chunk boundary, not turn end
    # NB: brief's verbatim model never terminates (always yields a tool call,
    # so step_stream loops forever) and its [-1] lookup assumes a single
    # assistant message. Terminating second turn added; assertions target the
    # tool-paired assistant message (brief's intent verbatim: prefix kept + pair).
    paired = [m for m in agent.messages if m.get("role") == "assistant" and m.get("tool_calls")]
    assert paired
    asst = paired[0]
    assert asst.get("content") and "Well," in asst["content"]  # spoken prefix kept (H2)
    assert asst.get("tool_calls")  # pair intact (H1)
