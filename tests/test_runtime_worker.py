import asyncio
from unittest.mock import MagicMock

import pytest

from verbix.runtime.worker import WorkerTask


@pytest.mark.asyncio
async def test_worker_without_tool_call_does_not_complete_on_chatter():
    """A worker that only talks without executing any tools must fail on RepeatedFormatError."""
    from verbix.agent.tools.registry import ToolRegistry
    from verbix.agent.worker import WorkerAgent, WorkerConfig

    mock_model = MagicMock()
    mock_model.query.return_value = {
        "role": "assistant",
        "content": "I built the game, all done!",
        "tool_calls": [],
    }
    session = WorkerAgent(
        model=mock_model,
        tools=ToolRegistry(),
        config=WorkerConfig(max_consecutive_format_errors=2),
    )

    task = WorkerTask(id="task-9", description="build a flappy game", session=session)
    task.start()
    await task.wait()

    assert task.status == "failed"
    assert "RepeatedFormatError" in (task.result or "")
    assert "I built the game" not in (task.result or "")
    assert mock_model.query.call_count == 2


@pytest.mark.asyncio
async def test_worker_with_tool_then_speech_completes_with_result():
    """A worker executing a tool and concluding in plain text completes with its summary."""
    from verbix.agent.tools.registry import ToolRegistry
    from verbix.agent.worker import WorkerAgent

    mock_model = MagicMock()
    mock_model.query.side_effect = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "write", "arguments": '{"path": "index.html"}'}}],
        },
        {
            "role": "assistant",
            "content": "wrote index.html (412 bytes, verified).",
            "tool_calls": None,
        },
    ]
    reg = ToolRegistry()
    reg.register("write", {"type": "function", "function": {"name": "write"}}, lambda a: "ok")
    session = WorkerAgent(model=mock_model, tools=reg)

    task = WorkerTask(id="task-10", description="write index.html", session=session)
    task.start()
    await task.wait()

    assert task.status == "completed"
    assert "412 bytes" in (task.result or "")


@pytest.mark.asyncio
async def test_worker_task_lifecycle_success():
    session = MagicMock()

    async def fake_run(prompt):
        return {"status": "completed", "summary": "All tests passed successfully."}

    session.run = fake_run
    session.messages = []

    task = WorkerTask(id="task-1", description="run tests", session=session)
    assert task.status == "pending"

    completed_events = []
    task.on_completed(lambda t: completed_events.append(t.id))

    task.start()
    assert task.status == "running"

    await task.wait()
    assert task.status == "completed"
    assert task.result == "All tests passed successfully."
    assert completed_events == ["task-1"]


@pytest.mark.asyncio
async def test_worker_task_cancellation():
    session = MagicMock()

    async def hanging_run(prompt):
        await asyncio.sleep(10.0)
        return {"status": "completed", "summary": "done"}

    session.run = hanging_run
    session.messages = []

    task = WorkerTask(id="task-2", description="long task", session=session)
    task.start()
    await asyncio.sleep(0.02)
    assert task.status == "running"

    await task.cancel()
    assert task.status == "cancelled"


@pytest.mark.asyncio
async def test_worker_task_get_status_reading_history():
    from verbix.core.models import CanonicalMessage, CanonicalToolCall

    session = MagicMock()
    session.history.messages = [
        CanonicalMessage(role="user", content="run tests"),
        CanonicalMessage(
            role="assistant",
            tool_calls=[CanonicalToolCall(id="c1", name="bash", arguments='{"cmd": "pytest"}')],
        ),
        CanonicalMessage(role="tool", content="14 passed", tool_call_id="c1"),
        CanonicalMessage(role="assistant", content="Done."),
    ]

    task = WorkerTask(id="task-3", description="run tests", session=session)
    task._status = "running"

    # Concise status
    concise = task.get_status(full=False)
    assert "task-3" in concise
    assert "bash" in concise or "14 passed" in concise

    # Full status
    full = task.get_status(full=True)
    assert "task-3" in full
    assert "run tests" in full
    assert "14 passed" in full


@pytest.mark.asyncio
async def test_worker_task_extracts_final_result_not_intermediate_scratchpad():
    from verbix.agent.tools.registry import ToolRegistry
    from verbix.agent.worker import WorkerAgent

    mock_model = MagicMock()
    mock_model.query.side_effect = [
        {
            "role": "assistant",
            "content": "I will start by analyzing the codebase...",
            "tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": '{"path": "pyproject.toml"}'}}],
        },
        {
            "role": "assistant",
            "content": "The dependencies are pytest, litellm.",
            "tool_calls": None,
        },
    ]
    reg = ToolRegistry()
    reg.register("read", {"type": "function", "function": {"name": "read"}}, lambda a: "pyproject.toml found")
    session = WorkerAgent(model=mock_model, tools=reg)

    task = WorkerTask(id="task-4", description="list dependencies", session=session)
    task.start()
    await task.wait()

    assert task.status == "completed"
    assert task.result == "The dependencies are pytest, litellm."
    assert "analyzing the codebase" not in (task.result or "")


def test_worker_task_status_timestamps():
    import re
    session = MagicMock()
    session.messages = []
    task = WorkerTask(id="task-5", description="quick task", session=session)
    task._status = "running"
    task._created_at = 1000000.0
    task._finished_at = 1000015.0

    running_status = task.get_status()
    assert re.search(r"running for \d+s", running_status)
    assert "started at " in running_status

    task._status = "completed"
    task._result = "Done"
    completed_status = task.get_status()
    assert "completed in 15s" in completed_status
    assert "started at " in completed_status
    assert "finished at " in completed_status


def test_worker_task_get_status_excludes_system_prompt_and_scaffolding():
    session = MagicMock(spec=["messages"])
    session.messages = [
        {"role": "system", "content": "You are an autonomous background worker executing tasks for Verbix."},
        {"role": "user", "content": "Please execute this task: Run tests.\n\n## Execution Invariants:\n1. Discover..."},
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": '{"command": "pytest"}'}}]},
        {"role": "tool", "content": "86 passed in 7.0s"},
        {"role": "assistant", "content": "Ran pytest. All 86 passed."},
    ]

    task = WorkerTask(id="task-6", description="Run tests", session=session)
    task._status = "completed"
    task._result = "Ran pytest. All 86 passed."

    full = task.get_status(full=True)

    # Invariants: system prompt and template scaffolding MUST NOT leak into status
    assert "You are an autonomous background worker" not in full
    assert "## Execution Invariants" not in full
    assert "- System:" not in full
    assert "- User:" not in full

    # Execution history MUST be preserved
    assert 'bash({"command": "pytest"})' in full
    assert "86 passed in 7.0s" in full
    assert "Ran pytest. All 86 passed." in full


@pytest.mark.asyncio
async def test_spawn_creates_private_scratch_dir(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from verbix.runtime.foreground import DEFAULT_WORKER_PROMPT
    from verbix.runtime.ledger import TaskLedger
    from verbix.runtime.worker import WorkerTask
    from verbix.runtime.workers import spawn_worker

    started = []
    monkeypatch.setattr(WorkerTask, "start", lambda self: started.append(self.id))
    broadcasted = []
    runtime = SimpleNamespace(
        workspace=tmp_path,
        _task_counter=0,
        worker_system_prompt=DEFAULT_WORKER_PROMPT,
        worker_model=None,
        provider=MagicMock(),
        worker_echo=False,
        tasks={},
        _environments=[],
        ledger=TaskLedger(),
        _broadcast_event=lambda *a: broadcasted.append(a),
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    task_id = await spawn_worker(runtime, "do stuff")
    assert task_id == "task-1"
    scratch = tmp_path / "home" / ".verbix" / "tasks" / "task-1"
    assert scratch.is_dir()
    assert not (tmp_path / ".verbix").exists()
    assert started == ["task-1"]
    assert ("task_registered", {"task_id": "task-1", "objective": "do stuff"}) in [
        (a[0], a[1]) for a in broadcasted
    ]


def test_worker_instance_template_renders_scratch_dir(tmp_path):
    import yaml

    from verbix.agent.worker import WorkerAgent, WorkerConfig

    cfg_path = (
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "src"
        / "verbix"
        / "agent"
        / "config"
        / "worker.yaml"
    )
    agent_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["agent"]
    agent = WorkerAgent(
        model=None, env=None, tools=None, config=WorkerConfig(**agent_cfg)
    )
    agent.extra_template_vars["scratch_dir"] = str(tmp_path)
    out = agent._render_template(agent_cfg["instance_template"], task="do stuff")
    assert "do stuff" in out
    assert str(tmp_path) in out


@pytest.mark.asyncio
async def test_worker_finish_completes_and_reports_summary():
    mock_agent = MagicMock()

    async def fake_run(task):
        return {"status": "completed", "summary": "Wrote flappy.html (2400 bytes, verified)."}

    mock_agent.run = fake_run
    mock_agent.messages = []

    task = WorkerTask(id="task-1", description="create flappy bird", session=mock_agent)
    task.start()
    await task.wait()

    assert task.status == "completed"
    assert "Wrote flappy.html" in (task.result or "")


@pytest.mark.asyncio
async def test_worker_finish_handoff_to_announcer_queue(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from verbix.agent.worker import WorkerAgent
    from verbix.runtime.foreground import DEFAULT_WORKER_PROMPT
    from verbix.runtime.ledger import TaskLedger
    from verbix.runtime.workers import spawn_worker

    mock_drain = AsyncMock()
    runtime = SimpleNamespace(
        workspace=tmp_path,
        _task_counter=0,
        worker_system_prompt=DEFAULT_WORKER_PROMPT,
        worker_model=None,
        provider=MagicMock(),
        worker_echo=False,
        tasks={},
        _environments=[],
        ledger=TaskLedger(),
        _pending_announcements=[],
        _broadcast_event=lambda *a: None,
        floor=SimpleNamespace(is_idle=True),
        _drain_pending_announcements=mock_drain,
    )

    async def fake_run(self, objective):
        return {"status": "completed", "summary": "Created game asset."}

    monkeypatch.setattr(WorkerAgent, "run", fake_run)
    task_id = await spawn_worker(runtime, "create asset")
    task = runtime.tasks[task_id]
    await task.wait()

    assert task.status == "completed"
    assert task.result == "Created game asset."
    assert (task_id, "Created game asset.") in runtime._pending_announcements
    assert mock_drain.called


@pytest.mark.asyncio
async def test_worker_task_emits_event_on_task_failure():
    from unittest.mock import MagicMock

    from verbix.core.events import TextDelta
    from verbix.runtime.worker import WorkerTask

    mock_agent = MagicMock()

    async def fake_run(task):
        raise RuntimeError("API error: out of credits")

    mock_agent.run = fake_run
    mock_agent.messages = []

    events = []
    task = WorkerTask(
        id="task-fail",
        description="failing task",
        session=mock_agent,
        on_event=lambda tid, ev: events.append((tid, ev)),
    )
    task.start()
    await task.wait()

    assert task.status == "failed"
    text_events = [ev for _, ev in events if isinstance(ev, TextDelta)]
    assert any("! task failed:" in e.text for e in text_events)


