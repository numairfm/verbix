from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from verbix.core.events import (
    TextDelta,
    ThoughtDelta,
    ToolCallDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnComplete,
)
from verbix.minisweagent.agents.default import AgentConfig, DefaultAgent
from verbix.minisweagent.config import get_config_from_spec
from verbix.minisweagent.exceptions import Submitted
from verbix.minisweagent.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_agent_step_stream_yields_text():
    mock_model = MagicMock()

    async def mock_stream(messages, tools=None):
        yield TextDelta(text="Hello from MiniSWE!")
        yield TurnComplete(final_text="Hello from MiniSWE!")

    mock_model.query_stream = mock_stream

    agent = DefaultAgent(model=mock_model, system_prompt="You are Verbix.")
    events = [e async for e in agent.step_stream("hi")]

    assert any(isinstance(e, TextDelta) and "Hello from MiniSWE!" in e.text for e in events)
    assert any(isinstance(e, TurnComplete) and "Hello from MiniSWE!" in e.final_text for e in events)
    assert len(agent.messages) >= 2
    assert agent.messages[0]["role"] == "system"
    assert agent.messages[0]["content"] == "You are Verbix."
    assert agent.messages[1]["role"] == "user"
    assert agent.messages[1]["content"] == "hi"


@pytest.mark.asyncio
async def test_agent_step_stream_executes_tool_registry_tool():
    tools = ToolRegistry()
    tools.register(
        "spawn",
        {"type": "function", "function": {"name": "spawn"}},
        lambda args: f"task-{args.get('task_name')}-started",
    )

    mock_model = MagicMock()
    call_count = 0

    async def mock_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield ThoughtDelta(thought="Spawning worker task")
            yield ToolCallDelta(
                index=0,
                call_id="call_spawn_1",
                name="spawn",
                arguments_delta='{"task_name": "fix-bug"}',
            )
        else:
            yield TextDelta(text="Worker task spawned successfully.")
            yield TurnComplete(final_text="Worker task spawned successfully.")

    mock_model.query_stream = mock_stream

    agent = DefaultAgent(
        model=mock_model,
        tools=tools,
        tools_list=["spawn"],
        system_prompt="You are Verbix.",
    )
    events = [e async for e in agent.step_stream("spawn fix-bug")]

    # Verify tool events yielded
    started_events = [e for e in events if isinstance(e, ToolCallStarted)]
    assert len(started_events) == 1
    assert started_events[0].call_id == "call_spawn_1"
    assert started_events[0].tool_name == "spawn"
    assert started_events[0].arguments == {"task_name": "fix-bug"}

    finished_events = [e for e in events if isinstance(e, ToolCallFinished)]
    assert len(finished_events) == 1
    assert finished_events[0].call_id == "call_spawn_1"
    assert finished_events[0].observation == "task-fix-bug-started"
    assert finished_events[0].is_error is False

    # Verify final response
    turn_complete = [e for e in events if isinstance(e, TurnComplete)]
    assert len(turn_complete) == 1
    assert "Worker task spawned successfully." in turn_complete[0].final_text

    # Verify message sequence in agent
    assert any(m.get("role") == "tool" and m.get("content") == "task-fix-bug-started" for m in agent.messages)


@pytest.mark.asyncio
async def test_agent_step_stream_executes_parallel_tool_calls():
    tools = ToolRegistry()
    tools.register(
        "spawn",
        {"type": "function", "function": {"name": "spawn"}},
        lambda args: f"task-{args.get('task_name')}-started",
    )

    mock_model = MagicMock()
    call_count = 0

    async def mock_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Emit two tool calls in the same streaming turn (parallel tool calling)
            yield ToolCallDelta(
                index=0,
                call_id="call_spawn_1",
                name="spawn",
                arguments_delta='{"task_name": "task-a"}',
            )
            yield ToolCallDelta(
                index=1,
                call_id="call_spawn_2",
                name="spawn",
                arguments_delta='{"task_name": "task-b"}',
            )
        else:
            yield TextDelta(text="Both tasks spawned.")
            yield TurnComplete(final_text="Both tasks spawned.")

    mock_model.query_stream = mock_stream

    agent = DefaultAgent(
        model=mock_model,
        tools=tools,
        tools_list=["spawn"],
        system_prompt="You are Verbix.",
    )
    events = [e async for e in agent.step_stream("start task-a and task-b")]

    started_events = [e for e in events if isinstance(e, ToolCallStarted)]
    assert len(started_events) == 2
    assert started_events[0].call_id == "call_spawn_1"
    assert started_events[0].arguments == {"task_name": "task-a"}
    assert started_events[1].call_id == "call_spawn_2"
    assert started_events[1].arguments == {"task_name": "task-b"}

    finished_events = [e for e in events if isinstance(e, ToolCallFinished)]
    assert len(finished_events) == 2
    assert finished_events[0].observation == "task-task-a-started"
    assert finished_events[1].observation == "task-task-b-started"

    # Both tool observations are in message history
    assert any(m.get("content") == "task-task-a-started" for m in agent.messages)
    assert any(m.get("content") == "task-task-b-started" for m in agent.messages)



@pytest.mark.asyncio
async def test_agent_step_stream_executes_env_tool():
    mock_env = MagicMock()
    mock_env.execute.return_value = {
        "output": "total 0\n-rw-r--r-- 1 test test 0 test.py",
        "returncode": 0,
        "exception_info": "",
    }

    mock_model = MagicMock()
    call_count = 0

    async def mock_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield ToolCallDelta(
                index=0,
                call_id="call_bash_1",
                name="bash",
                arguments_delta='{"command": "ls -l"}',
            )
        else:
            yield TextDelta(text="Here are the files.")
            yield TurnComplete(final_text="Here are the files.")

    mock_model.query_stream = mock_stream

    agent = DefaultAgent(
        model=mock_model,
        env=mock_env,
        tools_list=["bash"],
        system_prompt="You are an SWE assistant.",
    )
    events = [e async for e in agent.step_stream("list files")]

    mock_env.execute.assert_called_once_with({"command": "ls -l"})
    started = [e for e in events if isinstance(e, ToolCallStarted)]
    assert len(started) == 1
    assert started[0].tool_name == "bash"

    finished = [e for e in events if isinstance(e, ToolCallFinished)]
    assert len(finished) == 1
    assert "test.py" in finished[0].observation


@pytest.mark.asyncio
async def test_agent_step_stream_catches_submitted_cleanly():
    mock_env = MagicMock()
    mock_env.execute.side_effect = Submitted(
        {
            "role": "exit",
            "content": "Bug fix implemented and verified.",
            "extra": {
                "exit_status": "Submitted",
                "submission": "Bug fix implemented and verified.",
            },
        }
    )

    mock_model = MagicMock()

    async def mock_stream(messages, tools=None):
        yield ToolCallDelta(
            index=0,
            call_id="call_submit_1",
            name="bash",
            arguments_delta='{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nBug fix implemented and verified."}',
        )

    mock_model.query_stream = mock_stream

    agent = DefaultAgent(
        model=mock_model,
        env=mock_env,
        tools_list=["bash"],
        system_prompt="Solve issue.",
    )
    events = [e async for e in agent.step_stream("fix issue")]

    # Finished cleanly with TurnComplete containing the submission text
    turn_completes = [e for e in events if isinstance(e, TurnComplete)]
    assert len(turn_completes) == 1
    assert turn_completes[0].final_text == "Bug fix implemented and verified."


def test_agent_yaml_configs():
    voice_cfg_dict = get_config_from_spec("voice")
    assert "agent" in voice_cfg_dict
    voice_agent = AgentConfig(**voice_cfg_dict["agent"])
    assert "spawn" in voice_agent.tools
    assert "status" in voice_agent.tools
    assert "cancel" in voice_agent.tools
    assert "Verbix" in voice_agent.system_template

    # Worker config
    worker_cfg_dict = get_config_from_spec("worker")
    assert "agent" in worker_cfg_dict
    worker_agent = AgentConfig(**worker_cfg_dict["agent"])
    assert "bash" in worker_agent.tools
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in worker_agent.instance_template

    # Legacy swe_worker alias
    legacy_cfg_dict = get_config_from_spec("swe_worker")
    assert "agent" in legacy_cfg_dict
    legacy_agent = AgentConfig(**legacy_cfg_dict["agent"])
    assert "bash" in legacy_agent.tools


@pytest.mark.asyncio
async def test_agent_multi_turn_history():
    mock_model = MagicMock()
    turn_counter = 0

    async def mock_stream(messages, tools=None):
        nonlocal turn_counter
        turn_counter += 1
        yield TextDelta(text=f"Response {turn_counter}")
        yield TurnComplete(final_text=f"Response {turn_counter}")

    mock_model.query_stream = mock_stream
    agent = DefaultAgent(model=mock_model, system_prompt="You are Verbix.")

    # First turn
    events1 = [e async for e in agent.step_stream("hello")]
    assert any(isinstance(e, TurnComplete) and e.final_text == "Response 1" for e in events1)

    # Second turn
    events2 = [e async for e in agent.step_stream("how are you?")]
    assert any(isinstance(e, TurnComplete) and e.final_text == "Response 2" for e in events2)

    # Verify message sequence: system, user1, assistant1, user2, assistant2
    roles = [m["role"] for m in agent.messages]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    assert agent.messages[1]["content"] == "hello"
    assert agent.messages[2]["content"] == "Response 1"
    assert agent.messages[3]["content"] == "how are you?"
    assert agent.messages[4]["content"] == "Response 2"


@pytest.mark.asyncio
async def test_agent_step_stream_limits_exceeded():
    mock_model = MagicMock()

    async def mock_stream(messages, tools=None):
        yield TextDelta(text="First call")
        yield TurnComplete(final_text="First call")

    mock_model.query_stream = mock_stream
    agent = DefaultAgent(model=mock_model, step_limit=1, system_prompt="Test agent")

    # First step should succeed (reaches step_limit=1)
    events1 = [e async for e in agent.step_stream("step 1")]
    assert any(isinstance(e, TurnComplete) and e.final_text == "First call" for e in events1)

    # Second step should yield LimitsExceeded
    events2 = [e async for e in agent.step_stream("step 2")]
    assert any(isinstance(e, TurnComplete) and "LimitsExceeded" in e.final_text for e in events2)


@pytest.mark.asyncio
async def test_agent_step_stream_unknown_tool():
    mock_model = MagicMock()
    call_count = 0

    async def mock_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield ToolCallDelta(
                index=0,
                call_id="call_unk_1",
                name="unknown_tool",
                arguments_delta="{}",
            )
        else:
            yield TextDelta(text="Handled unknown tool.")
            yield TurnComplete(final_text="Handled unknown tool.")

    mock_model.query_stream = mock_stream
    agent = DefaultAgent(model=mock_model, system_prompt="Test")
    events = [e async for e in agent.step_stream("trigger unknown")]

    finished = [e for e in events if isinstance(e, ToolCallFinished)]
    assert len(finished) == 1
    assert finished[0].is_error is True
    assert "not registered" in finished[0].observation.lower()


def test_agent_sync_run_and_save(tmp_path):
    mock_model = MagicMock()
    mock_model.query.return_value = {
        "role": "assistant",
        "content": "Done task",
        "extra": {"actions": [{"command": "echo done"}]},
    }
    mock_model.serialize.return_value = {"model_info": "mock"}
    mock_env = MagicMock()
    mock_env.execute.side_effect = Submitted(
        {
            "role": "exit",
            "content": "task completed",
            "extra": {"exit_status": "Submitted", "submission": "task completed"},
        }
    )
    mock_env.serialize.return_value = {"env_info": "mock"}

    agent = DefaultAgent(model=mock_model, env=mock_env, system_prompt="Sync agent")
    res = agent.run(task="sample task")
    assert res.get("exit_status") == "Submitted"
    assert res.get("submission") == "task completed"

    out_file = tmp_path / "trajectory.json"
    data = agent.save(out_file)
    assert out_file.exists()
    assert "info" in data
    assert "messages" in data


def test_default_agent_positional_config():
    mock_model = MagicMock()
    mock_env = MagicMock()
    tools = ToolRegistry()
    config = AgentConfig(system_template="System", instance_template="Task: {{task}}", tools=["spawn"])

    agent = DefaultAgent(mock_model, mock_env, tools, config)
    assert agent.model is mock_model
    assert agent.env is mock_env
    assert agent.tools is tools
    assert agent.config is config
    assert agent.config.tools == ["spawn"]


@pytest.mark.asyncio
async def test_step_stream_renders_instance_template_initial_turn():
    mock_model = MagicMock()
    turn_count = 0

    async def mock_stream(messages, tools=None):
        nonlocal turn_count
        turn_count += 1
        yield TextDelta(text=f"Response {turn_count}")
        yield TurnComplete(final_text=f"Response {turn_count}")

    mock_model.query_stream = mock_stream
    config = AgentConfig(
        system_template="You are a helper.",
        instance_template="Please solve this issue: {{task}}\nWorkflow: step-by-step.",
    )
    agent = DefaultAgent(mock_model, config=config)

    # Initial turn: user message rendered via instance_template
    _ = [e async for e in agent.step_stream("fix issue 42")]
    assert len(agent.messages) >= 2
    assert "Please solve this issue: fix issue 42" in agent.messages[1]["content"]
    assert "Workflow: step-by-step." in agent.messages[1]["content"]

    # Second turn: user message used directly
    _ = [e async for e in agent.step_stream("what is the status?")]
    assert agent.messages[3]["content"] == "what is the status?"


@pytest.mark.asyncio
async def test_unknown_tool_with_env_attached_returns_tool_error():
    mock_model = MagicMock()
    mock_env = MagicMock()
    call_count = 0

    async def mock_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield ToolCallDelta(
                index=0,
                call_id="call_custom_1",
                name="nonexistent_tool",
                arguments_delta='{"arg": 1}',
            )
        else:
            yield TextDelta(text="Done.")
            yield TurnComplete(final_text="Done.")

    mock_model.query_stream = mock_stream
    agent = DefaultAgent(mock_model, env=mock_env, tools_list=["bash"])
    events = [e async for e in agent.step_stream("run something")]

    # Crucial assertion: mock_env.execute was NOT called for unknown tool
    mock_env.execute.assert_not_called()

    finished = [e for e in events if isinstance(e, ToolCallFinished)]
    assert len(finished) == 1
    assert finished[0].call_id == "call_custom_1"
    assert finished[0].is_error is True
    assert "not registered or supported" in finished[0].observation


@pytest.mark.asyncio
async def test_bash_observation_includes_returncode_on_failure():
    mock_model = MagicMock()
    mock_env = MagicMock()
    mock_env.execute.return_value = {
        "output": "cat: missing.txt: No such file or directory",
        "returncode": 1,
        "exception_info": "",
    }
    call_count = 0

    async def mock_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield ToolCallDelta(
                index=0,
                call_id="call_bash_err",
                name="bash",
                arguments_delta='{"command": "cat missing.txt"}',
            )
        else:
            yield TextDelta(text="File was missing.")
            yield TurnComplete(final_text="File was missing.")

    mock_model.query_stream = mock_stream
    agent = DefaultAgent(mock_model, env=mock_env, tools_list=["bash"])
    events = [e async for e in agent.step_stream("read file")]

    finished = [e for e in events if isinstance(e, ToolCallFinished)]
    assert len(finished) == 1
    assert finished[0].is_error is True
    assert "[Returncode: 1]" in finished[0].observation



