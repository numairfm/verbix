from verbix.core.events import (
    TextDelta,
    ThoughtDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnComplete,
)


def test_text_and_thought_deltas():
    t_delta = TextDelta(text="hello ")
    assert t_delta.text == "hello "

    th_delta = ThoughtDelta(thought="thinking through problem")
    assert th_delta.thought == "thinking through problem"


def test_tool_call_events():
    started = ToolCallStarted(
        call_id="call-1", tool_name="task_spawn", arguments={"title": "build"}
    )
    assert started.call_id == "call-1"
    assert started.tool_name == "task_spawn"
    assert started.arguments == {"title": "build"}

    finished = ToolCallFinished(
        call_id="call-1", observation="spawned task-1", is_error=False
    )
    assert finished.call_id == "call-1"
    assert finished.observation == "spawned task-1"
    assert finished.is_error is False


def test_turn_complete():
    turn = TurnComplete(final_text="I spawned the build task.")
    assert turn.final_text == "I spawned the build task."
    assert turn.exit_reason == "ok"
    assert turn.is_error is False

    err_turn = TurnComplete(
        final_text="", exit_reason="repeated_tool_call", is_error=True
    )
    assert err_turn.exit_reason == "repeated_tool_call"
    assert err_turn.is_error is True

