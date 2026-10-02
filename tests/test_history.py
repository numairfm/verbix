import pytest

from verbix.core.history import ConversationHistory
from verbix.core.models import (
    CanonicalMessage,
    CanonicalToolCall,
    TaskState,
    WorkerResult,
)


def test_history_append_and_is_turn_boundary():
    history = ConversationHistory()
    assert history.is_turn_boundary() is True
    assert len(history.messages) == 0

    history.append_user("Build the project")
    assert history.is_turn_boundary() is False
    assert len(history.messages) == 1

    tc = CanonicalToolCall(id="c1", name="task_spawn", arguments='{"title": "build"}')
    history.append_assistant("Starting build", tool_calls=[tc])
    # Expecting tool result, not at turn boundary yet
    assert history.is_turn_boundary() is False

    history.append(
        CanonicalMessage(role="tool", content="task-1 spawned", tool_call_id="c1")
    )
    # Tool result provided, now ready for next turn step or completion
    assert history.is_turn_boundary() is True


def test_runtime_turn_parity():
    history = ConversationHistory()
    history.append_runtime_turn(
        user_input="status", runtime_reply="Task 1 is running at step 5."
    )

    assert len(history.messages) == 2
    assert history.messages[0].role == "user"
    assert history.messages[0].content == "status"
    assert history.messages[1].role == "assistant"
    assert history.messages[1].content == "Task 1 is running at step 5."
    assert history.is_turn_boundary() is True


def test_inject_task_completion():
    history = ConversationHistory()
    history.append_user("Run the tests in background")
    history.append_assistant("Spawned test worker.")

    res = WorkerResult(
        success=True,
        speech_summary="All 14 unit tests passed.",
        details="test_auth: ok\ntest_db: ok",
    )
    history.inject_task_completion(task_id="task-42", title="run tests", result=res)

    # Injected as an immutable event message
    assert len(history.messages) == 3
    last = history.messages[-1]
    assert last.role == "system"
    assert "[Task completed: 'run tests' (task-42)" in last.content
    assert "All 14 unit tests passed." in last.content
    assert history.is_turn_boundary() is True


def test_barge_in_pruning_applies_strictly_to_foreground_assistant():
    history = ConversationHistory()
    history.append_user("Explain quantum computing")
    # Assistant started speaking 3 sentences, but user interrupted after sentence 1
    full_text = "Quantum computing uses qubits. Qubits can exist in superposition. This enables exponential speedup."
    history.append_assistant(content=full_text)

    played_text = "Quantum computing uses qubits."
    history.prune_foreground_barge_in(played_text)

    # Last assistant message pruned to what was actually heard
    assert history.messages[-1].role == "assistant"
    assert history.messages[-1].content == "Quantum computing uses qubits."

    # Background task completion added after is NOT pruned
    res = WorkerResult(success=True, speech_summary="Download done.", details="Saved.")
    history.inject_task_completion("task-1", "download", res)
    # Another prune call shouldn't touch task completions
    with pytest.raises(ValueError, match="Cannot prune non-assistant message"):
        history.prune_foreground_barge_in("anything")


def test_render_turn_prompt_ephemeral_running_task_line():
    history = ConversationHistory()
    history.append_user("How much memory is left?")

    # Active running task
    running_task = TaskState(
        id="task-1", title="compile kernel", status="running", step_count=12
    )
    # Finished task (must NOT be rendered in ephemeral line!)
    finished_task = TaskState(
        id="task-2",
        title="download dataset",
        status="completed",
        result=WorkerResult(success=True, speech_summary="Done.", details="OK"),
    )

    rendered = history.render_turn_prompt(running_tasks=[running_task, finished_task])
    assert len(rendered) == 1
    # Check that running task was prepended
    assert "[Running: task-1 'compile kernel' (step 12)]" in rendered[0].content
    assert "How much memory is left?" in rendered[0].content
    # Finished task must NOT be in the rendered prompt
    assert "download dataset" not in rendered[0].content

    # Invariant: Underlying history must NOT be mutated!
    assert "[Running:" not in history.messages[0].content
    assert history.messages[0].content == "How much memory is left?"
