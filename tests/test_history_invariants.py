from verbix.minisweagent.utils.history import validate_and_repair_history


def test_repair_unanswered_tool_calls():
    msgs = [
        {"role": "user", "content": "run task"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}]},
    ]
    repaired = validate_and_repair_history(msgs)
    assert len(repaired) == 3
    assert repaired[-1]["role"] == "tool"
    assert repaired[-1]["tool_call_id"] == "call_1"
    assert "cancelled" in repaired[-1]["content"]


def test_repair_truncates_to_delivered_speech():
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "This is a very long response that was interrupted halfway."},
    ]
    repaired = validate_and_repair_history(msgs, delivered_text="This is a very")
    assert repaired[-1]["content"] == "This is a very"
