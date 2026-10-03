from tests.test_regressions import assert_valid_history
from verbix.core.history import ConversationHistory
from verbix.core.models import (
    CanonicalMessage,
    CanonicalToolCall,
    TaskState,
    WorkerResult,
)


def test_history_append_and_messages():
    history = ConversationHistory()
    assert len(history.messages) == 0

    history.append_user("Build the project")
    assert len(history.messages) == 1
    assert history.messages[0].role == "user"

    tc = CanonicalToolCall(id="c1", name="task_spawn", arguments='{"title": "build"}')
    history.append_assistant("Starting build", tool_calls=[tc])
    assert len(history.messages) == 2

    history.append(
        CanonicalMessage(role="tool", content="task-1 spawned", tool_call_id="c1")
    )
    assert len(history.messages) == 3
    assert_valid_history(history.messages)


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

    assert len(history.messages) == 3
    last = history.messages[-1]
    assert last.role == "user"
    assert "[Verbix runtime notice — not said by the user]" in last.content
    assert (
        "[Task completed: 'run tests' (task-42) -> Succeeded: All 14 unit tests passed. Details: test_auth: ok\ntest_db: ok]"
        in last.content
    )
    assert_valid_history(history.messages)


def test_inject_task_completion_capped_details():
    history = ConversationHistory()
    long_details = "D" * 800
    res = WorkerResult(
        success=False,
        speech_summary="Build failed.",
        details=long_details,
    )
    history.inject_task_completion(task_id="task-99", title="compile", result=res)

    last = history.messages[-1]
    assert "... (call task_status for full details)" in last.content
    assert len(last.content) < 800


def test_close_aborted_turn_with_tool_calls():
    history = ConversationHistory()
    history.append_user("Do something")
    tc1 = CanonicalToolCall(id="c1", name="tool_a", arguments="{}")
    tc2 = CanonicalToolCall(id="c2", name="tool_b", arguments="{}")
    history.append_assistant("Calling tools", tool_calls=[tc1, tc2])

    # Abort before tool execution finishes
    history.close_aborted_turn("")

    # Must append synthetic tool responses so tool calls are not dangling
    assert len(history.messages) == 4
    assert history.messages[2].role == "tool"
    assert history.messages[2].tool_call_id == "c1"
    assert history.messages[3].role == "tool"
    assert history.messages[3].tool_call_id == "c2"
    assert_valid_history(history.messages)


def test_close_aborted_turn_with_partial_speech():
    history = ConversationHistory()
    history.append_user("Explain gravity")
    history.append_assistant(
        "Gravity is a fundamental interaction. It curves spacetime."
    )

    # User barged in after hearing first sentence
    history.close_aborted_turn("Gravity is a fundamental interaction.")

    assert len(history.messages) == 2
    assert history.messages[1].role == "assistant"
    assert history.messages[1].content == "Gravity is a fundamental interaction."
    assert_valid_history(history.messages)


def test_close_aborted_turn_with_no_speech():
    history = ConversationHistory()
    history.append_user("Explain gravity")
    history.append_assistant("Gravity is a fundamental interaction.")

    # User barged in before anything was spoken
    history.close_aborted_turn("")

    # Assistant message should be popped
    assert len(history.messages) == 1
    assert history.messages[0].role == "user"
    assert_valid_history(history.messages)


def test_render_turn_prompt_ephemeral_running_task_line():
    history = ConversationHistory()
    history.append_user("How much memory is left?")

    running_task = TaskState(
        id="task-1", title="compile kernel", status="running", step_count=12
    )
    finished_task = TaskState(
        id="task-2",
        title="download dataset",
        status="completed",
        result=WorkerResult(success=True, speech_summary="Done.", details="OK"),
    )

    rendered = history.render_turn_prompt(running_tasks=[running_task, finished_task])
    assert len(rendered) == 1
    assert "[Running: task-1 'compile kernel' (12 tool calls)]" in rendered[0].content
    assert "How much memory is left?" in rendered[0].content
    assert "download dataset" not in rendered[0].content

    # Invariant: Underlying history must NOT be mutated!
    assert "[Running:" not in history.messages[0].content
    assert history.messages[0].content == "How much memory is left?"
