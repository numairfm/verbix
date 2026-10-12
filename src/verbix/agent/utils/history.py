"""Validation and repair of dict-message histories after turn cancellation."""

from __future__ import annotations

from typing import Any

CANCELLED_TOOL_CONTENT = "cancelled: interrupted by user before completion"


def _tool_call_ids(msg: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict) and tc.get("id"):
            ids.append(tc["id"])
    return ids


def validate_and_repair_history(
    messages: list[dict[str, Any]], delivered_text: str = ""
) -> list[dict[str, Any]]:
    """Repair a dict-message history left inconsistent by turn cancellation.

    Port of ``ConversationHistory.close_aborted_turn`` (core/history.py) for
    standard message dicts. Returns a new list; the input is not mutated.
    """
    msgs = [dict(m) for m in messages]

    if not msgs:
        if delivered_text:
            msgs.append({"role": "assistant", "content": delivered_text})
        return msgs

    # 1. Synthesize responses for any dangling tool calls in the turn
    turn_start = 0
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            turn_start = i
            break

    for i in range(turn_start, len(msgs)):
        msg = msgs[i]
        if msg.get("role") == "assistant" and _tool_call_ids(msg):
            expected = _tool_call_ids(msg)
            provided = {
                m.get("tool_call_id")
                for m in msgs[i + 1 :]
                if m.get("role") == "tool" and m.get("tool_call_id") is not None
            }
            for cid in expected:
                if cid not in provided:
                    msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": cid,
                            "content": CANCELLED_TOOL_CONTENT,
                        }
                    )

    # 2. Reconcile spoken text with delivered_text
    turn_assistants = [
        m for m in msgs[turn_start:] if m.get("role") == "assistant"
    ]

    if turn_assistants:
        last_asst = turn_assistants[-1]
        if not last_asst.get("tool_calls"):
            if delivered_text:
                last_asst["content"] = delivered_text
            else:
                for idx in range(len(msgs) - 1, -1, -1):
                    if msgs[idx] is last_asst:
                        del msgs[idx]
                        break
        else:
            if delivered_text:
                last_asst["content"] = delivered_text
            else:
                last_asst["content"] = None
    elif delivered_text:
        msgs.append({"role": "assistant", "content": delivered_text})

    return _merge_assistant_text(msgs)


def _merge_assistant_text(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive assistant text messages for API compatibility."""
    merged: list[dict[str, Any]] = []
    for msg in msgs:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.get("role") == "assistant"
            and msg.get("role") == "assistant"
            and not prev.get("tool_calls")
            and not msg.get("tool_calls")
        ):
            prev["content"] = (prev.get("content") or "") + (msg.get("content") or "")
        else:
            merged.append(msg)
    return merged
