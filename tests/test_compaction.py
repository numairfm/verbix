"""Tier 1 deterministic context compaction tests (Task 4)."""

import copy

from verbix.minisweagent.utils.compaction import compact_messages


def test_compact_messages_preserves_recent_turns():
    msgs = [
        {"role": "system", "content": "You are Verbix"},
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Resp 1"},
        {"role": "user", "content": "Turn 2"},
        {"role": "assistant", "content": "Resp 2"},
    ]
    compacted = compact_messages(msgs, max_tokens=1000)
    assert len(compacted) == 5


def test_compact_messages_elides_old_tool_observations_without_breaking_pairs():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old task"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "status", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "x" * 5000},
        {"role": "assistant", "content": "old task done"},
        {"role": "user", "content": "new task"},
        {"role": "assistant", "content": "new resp"},
    ]
    compacted = compact_messages(msgs, max_tokens=500)
    # The tool result for c1 must still exist, but content elided
    tool_msg = next(m for m in compacted if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == "c1"
    assert len(tool_msg["content"]) < 200


def test_compact_messages_never_drops_system_task_or_splits_pairs():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task 0"}]
    for i in range(1, 10):
        msgs += [
            {"role": "user", "content": f"task {i}"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "y" * 2000},
            {"role": "assistant", "content": f"done {i}"},
        ]
    snapshot = copy.deepcopy(msgs)
    compacted = compact_messages(msgs, max_tokens=300)
    assert msgs == snapshot  # input never mutated
    assert compacted[0] == {"role": "system", "content": "sys"}
    assert {"role": "user", "content": "task 0"} in compacted
    call_ids = {tc["id"] for m in compacted if m.get("tool_calls") for tc in m["tool_calls"]}
    result_ids = {m.get("tool_call_id") for m in compacted if m.get("role") == "tool"}
    assert result_ids <= call_ids  # every surviving result still has its call


def test_compact_messages_worker_truncates_old_steps():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(12):
        msgs += [
            {"role": "assistant", "content": None, "tool_calls": [{"id": f"w{i}", "function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"w{i}", "content": "z" * 2000},
        ]
    compacted = compact_messages(msgs, max_tokens=300, is_worker=True)
    call_ids = {tc["id"] for m in compacted if m.get("tool_calls") for tc in m["tool_calls"]}
    result_ids = {m.get("tool_call_id") for m in compacted if m.get("role") == "tool"}
    assert result_ids <= call_ids
    assert compacted[0]["role"] == "system"
