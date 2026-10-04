import pytest
from pydantic import ValidationError

from verbix.core.models import (
    CanonicalMessage,
    CanonicalToolCall,
)


def test_canonical_message_creation():
    msg = CanonicalMessage(role="user", content="Hello Verbix")
    assert msg.role == "user"
    assert msg.content == "Hello Verbix"
    assert msg.tool_calls is None
    assert msg.tool_call_id is None


def test_canonical_message_with_tool_calls():
    tc = CanonicalToolCall(
        id="call_123", name="spawn", arguments='{"task": "test"}'
    )
    msg = CanonicalMessage(role="assistant", content="", tool_calls=[tc])
    assert msg.role == "assistant"
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "spawn"
    assert msg.tool_calls[0].parsed_arguments() == {"task": "test"}


def test_canonical_message_invalid_role():
    with pytest.raises(ValidationError):
        CanonicalMessage(role="invalid_role", content="oops")  # type: ignore
