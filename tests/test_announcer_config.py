"""Announcer prompt lives in config/announcer.yaml, not in runtime code."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _announcer_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "src"
        / "verbix"
        / "agent"
        / "config"
        / "announcer.yaml"
    )


def test_announcer_config_file_exists_with_constraints():
    import yaml

    cfg = yaml.safe_load(_announcer_path().read_text(encoding="utf-8"))
    template = cfg["agent"]["system_template"]
    assert "# VOICE & AUDIO CONSTRAINTS" in template
    assert "Prefer plain words over symbols" in template
    assert "Never recite links, paths, or code" in template


def test_runtime_announcer_uses_config_file(tmp_path):
    from verbix.runtime.runtime import Runtime

    rt = Runtime(workspace=tmp_path, provider=MagicMock())
    prompt = rt.announcer.config.system_template
    assert "Prefer plain words over symbols" in prompt
    assert "Never recite links, paths, or code" in prompt


def test_announcer_history_isolated_from_foreground(tmp_path):
    from unittest.mock import MagicMock

    from verbix.runtime.runtime import Runtime

    rt = Runtime(workspace=tmp_path, provider=MagicMock())
    rt.foreground.messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "spawn", "arguments": "{}"}],
        }
    )
    assert rt.announcer.messages is not rt.foreground.messages
    assert not any(m.get("tool_calls") for m in rt.announcer.messages)


@pytest.mark.asyncio
async def test_tool_choice_none_forwarded_when_configured():
    from verbix.agent.agents.default import AgentConfig, DefaultAgent
    from verbix.agent.tools.registry import ToolRegistry
    from verbix.core.events import TurnComplete

    seen = {}

    class CapModel:
        async def query_stream(self, messages, **kwargs):
            seen.update(kwargs)
            yield TurnComplete(final_text="done")

    agent = DefaultAgent(
        model=CapModel(), env=None, tools=ToolRegistry(), config=AgentConfig(tool_choice="none"),
    )
    [e async for e in agent.step_stream("hi")]
    assert seen.get("tool_choice") == "none"


@pytest.mark.asyncio
async def test_tool_choice_absent_by_default():
    from verbix.agent.agents.default import AgentConfig, DefaultAgent
    from verbix.agent.tools.registry import ToolRegistry
    from verbix.core.events import TurnComplete

    seen = {}

    class CapModel:
        async def query_stream(self, messages, **kwargs):
            seen.update(kwargs)
            yield TurnComplete(final_text="done")

    agent = DefaultAgent(
        model=CapModel(), env=None, tools=ToolRegistry(), config=AgentConfig(),
    )
    [e async for e in agent.step_stream("hi")]
    assert "tool_choice" not in seen
