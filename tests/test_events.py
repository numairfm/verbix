from verbix.core.events import (
    TaskFinishedEvent,
    TextDelta,
    ThoughtDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnComplete,
)
from verbix.core.models import WorkerResult


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


def test_turn_complete_and_task_finished():
    turn = TurnComplete(final_text="I spawned the build task.")
    assert turn.final_text == "I spawned the build task."

    res = WorkerResult(success=True, speech_summary="Done.", details="Output: ok")
    t_fin = TaskFinishedEvent(task_id="task-1", result=res)
    assert t_fin.task_id == "task-1"
    assert t_fin.result.speech_summary == "Done."
