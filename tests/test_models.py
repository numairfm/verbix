import pytest
from pydantic import ValidationError

from verbix.core.models import (
    CanonicalMessage,
    CanonicalToolCall,
    TaskState,
    WorkerResult,
)


def test_canonical_message_creation():
    msg = CanonicalMessage(role="user", content="Hello Verbix")
    assert msg.role == "user"
    assert msg.content == "Hello Verbix"
    assert msg.tool_calls is None
    assert msg.tool_call_id is None


def test_canonical_message_with_tool_calls():
    tc = CanonicalToolCall(
        id="call_123", name="task_spawn", arguments='{"instructions": "test"}'
    )
    msg = CanonicalMessage(role="assistant", content="", tool_calls=[tc])
    assert msg.role == "assistant"
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "task_spawn"
    assert msg.tool_calls[0].parsed_arguments() == {"instructions": "test"}


def test_canonical_message_invalid_role():
    with pytest.raises(ValidationError):
        CanonicalMessage(role="invalid_role", content="oops")  # type: ignore


def test_worker_result_speech_summary_length_cap():
    valid = WorkerResult(
        success=True,
        speech_summary="Saved 4 release binaries to dist.",
        details="Built with cargo --release in 14s. Files: dist/bin1, dist/bin2.",
    )
    assert valid.success is True
    assert len(valid.speech_summary) <= 180

    long_summary = "A" * 181
    with pytest.raises(ValidationError):
        WorkerResult(
            success=True,
            speech_summary=long_summary,
            details="Some details",
        )


def test_task_state_defaults_and_announced_tracking():
    task = TaskState(id="task-1", title="compile kernel")
    assert task.status == "pending"
    assert task.step_count == 0
    assert task.announced is False
    assert task.result is None

    # Update announced status separately from completion
    task.status = "completed"
    task.result = WorkerResult(
        success=True,
        speech_summary="Kernel build finished.",
        details="bzImage generated.",
    )
    assert task.announced is False
    task.announced = True
    assert task.announced is True
