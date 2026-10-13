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


@pytest.mark.asyncio
async def test_drain_shares_foreground_context_snapshot():
    import asyncio
    from types import SimpleNamespace

    from verbix.runtime.announcer import drain_pending_announcements

    foreground_messages = [
        {"role": "user", "content": "check the last commit"},
        {"role": "assistant", "content": "Sure."},
    ]
    seen = {}

    async def fake_submit(prompt, output, agent=None):
        seen["prompt"] = prompt
        seen["agent"] = agent
        return "There is no git repo here yet."

    runtime = SimpleNamespace(
        floor=SimpleNamespace(is_idle=True),
        _drain_lock=asyncio.Lock(),
        _pending_announcements=[("task-1", "no git repository here")],
        _request_grounding=lambda: "[grounding]",
        announcer=SimpleNamespace(messages=[{"role": "user", "content": "stale"}]),
        foreground=SimpleNamespace(messages=foreground_messages),
        wake_output=None,
        on_wake=None,
        submit=fake_submit,
    )
    await drain_pending_announcements(runtime)
    assert runtime._pending_announcements == []
    assert "Task 'task-1'" not in seen["prompt"]
    assert "no git repository here" in seen["prompt"]
    msgs = runtime.announcer.messages
    assert msgs is not foreground_messages
    assert [m.get("content") for m in msgs] == ["check the last commit", "Sure."]
    assert all(a is not b for a, b in zip(msgs, foreground_messages))
    assert foreground_messages[-1] == {
        "role": "assistant",
        "content": "There is no git repo here yet.",
    }


@pytest.mark.asyncio
async def test_drain_blank_result_speaks_nothing():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from verbix.runtime.announcer import drain_pending_announcements

    runtime = SimpleNamespace(
        floor=SimpleNamespace(is_idle=True),
        _drain_lock=asyncio.Lock(),
        _pending_announcements=[("task-1", "   ")],
        _request_grounding=lambda: "[grounding]",
        announcer=SimpleNamespace(messages=[]),
        foreground=SimpleNamespace(messages=[]),
        wake_output=None,
        on_wake=None,
        submit=AsyncMock(),
    )
    await drain_pending_announcements(runtime)
    assert runtime._pending_announcements == []
    runtime.submit.assert_not_called()


@pytest.mark.asyncio
async def test_speak_batch_shares_snapshot_and_writes_back():
    from types import SimpleNamespace

    from verbix.core.events import TurnComplete
    from verbix.runtime.announcer import speak_batch

    async def fake_step_stream(prompt):
        yield TurnComplete(final_text="spoken answer")

    announcer = SimpleNamespace(
        messages=[{"role": "user", "content": "stale"}],
        n_calls=7,
        cost=1.0,
        model=None,
        step_stream=fake_step_stream,
    )
    foreground_messages = [{"role": "user", "content": "original question"}]
    runtime = SimpleNamespace(
        _request_grounding=lambda: "[grounding]",
        wake_output=None,
        _broadcast_wake_event=None,
        provider=object(),
        announcer=announcer,
        foreground=SimpleNamespace(messages=foreground_messages),
        on_wake=None,
    )
    assert await speak_batch(runtime, [("task-9", "result text")]) is True
    assert [m.get("content") for m in announcer.messages] == ["original question"]
    assert foreground_messages[-1] == {"role": "assistant", "content": "spoken answer"}
