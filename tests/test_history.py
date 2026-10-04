from verbix.core.history import ConversationHistory
from verbix.core.models import (
    CanonicalMessage,
    CanonicalToolCall,
)


def assert_valid_history(msgs: list[CanonicalMessage]) -> None:
    """Every assistant tool_calls message must be immediately followed by results for all its ids,

    and there must be no orphan tool messages.
    """
    i = 0
    n = len(msgs)
    while i < n:
        msg = msgs[i]
        if msg.role == "tool":
            raise AssertionError(
                f"Orphan tool message at index {i} with id {msg.tool_call_id}"
            )
        if msg.role == "assistant" and msg.tool_calls:
            call_ids = [tc.id for tc in msg.tool_calls]
            num_calls = len(call_ids)
            tool_res_msgs = msgs[i + 1 : i + 1 + num_calls]
            if len(tool_res_msgs) != num_calls:
                raise AssertionError(
                    f"Assistant message at {i} has {num_calls} tool calls ({call_ids}), but {len(tool_res_msgs)} messages follow"
                )
            for j, t_msg in enumerate(tool_res_msgs):
                if t_msg.role != "tool":
                    raise AssertionError(
                        f"Expected tool message at index {i + 1 + j} for call {call_ids[j]}, found role {t_msg.role}"
                    )
            i += 1 + num_calls
            continue
        i += 1


def test_history_append_and_messages():
    history = ConversationHistory()
    assert len(history.messages) == 0

    history.append_user("Build the project")
    assert len(history.messages) == 1
    assert history.messages[0].role == "user"

    tc = CanonicalToolCall(id="c1", name="spawn", arguments='{"task": "build"}')
    history.append_assistant("Starting build", tool_calls=[tc])
    assert len(history.messages) == 2

    history.append(
        CanonicalMessage(role="tool", content="task-1 spawned", tool_call_id="c1")
    )
    assert len(history.messages) == 3
    assert_valid_history(history.messages)


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
