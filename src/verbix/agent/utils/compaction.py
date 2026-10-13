"""Tier 1 deterministic context compaction with pairing-safe pruning.

Foreground histories are grouped into turns (split on user messages); worker
histories into steps (one atomic unit per step). An assistant message bearing
tool_calls and its following tool results always travel as one atomic unit, so
pruning can never split a call from its results (H5).
"""

from __future__ import annotations

import copy
import json
from typing import Any

ELIDED_OBSERVATION = "[status output elided]"
MAX_TOOL_ARG_CHARS = 500
FOREGROUND_KEEP_TURNS = 3
RESIDUAL_TRUNCATE_CHARS = 1000


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Best-effort token count via litellm, falling back to chars // 4."""
    try:
        from litellm import token_counter

        return int(token_counter(messages=messages))
    except Exception:  # noqa: BLE001
        return sum(len(json.dumps(m, default=str)) for m in messages) // 4


def _elide_large_values(obj: Any, threshold: int) -> Any:
    if isinstance(obj, str):
        return "[truncated]" if len(obj) > threshold else obj
    if isinstance(obj, dict):
        return {k: _elide_large_values(v, threshold) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_elide_large_values(v, threshold) for v in obj]
    return obj


def _truncate_args(tool_calls: list[dict[str, Any]]) -> None:
    for tc in tool_calls:
        fn = tc.get("function") if isinstance(tc, dict) else None
        args = fn.get("arguments") if isinstance(fn, dict) else None
        if isinstance(args, str) and len(args) > MAX_TOOL_ARG_CHARS:
            try:
                parsed = json.loads(args)
                dumped = json.dumps(_elide_large_values(parsed, MAX_TOOL_ARG_CHARS))
                if len(dumped) > MAX_TOOL_ARG_CHARS:
                    dumped = json.dumps(_elide_large_values(parsed, 100))
                fn["arguments"] = dumped
            except (json.JSONDecodeError, TypeError):
                fn["arguments"] = args[:MAX_TOOL_ARG_CHARS]



def _atomic_units(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group messages so an assistant tool-call block and its tool results stay together."""
    units: list[list[dict[str, Any]]] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                j += 1
            units.append(messages[i:j])
            i = j
        else:
            units.append([m])
            i += 1
    return units


def compact_messages(messages: list[dict[str, Any]], max_tokens: int) -> list[dict[str, Any]]:
    """Deterministically fit messages into max_tokens without breaking tool-call pairs."""
    if estimate_tokens(messages) <= max_tokens:
        return messages

    working = copy.deepcopy(messages)

    # Pin system message(s) and the initial task (first user message); never dropped.
    header_end = 0
    while header_end < len(working) and working[header_end].get("role") == "system":
        header_end += 1
    for k in range(header_end, len(working)):
        if working[k].get("role") == "user":
            header_end = k + 1
            break
    header, body = working[:header_end], working[header_end:]

    units = _atomic_units(body)
    # NOTE: groups are atomic message units (an assistant tool-call block plus its
    # results is one unit), not user-delimited turns: with user-delimited turns a
    # short history (e.g. 2 turns) has nothing "older than last 3", so Stage 1
    # could never fire and Stage 3 would drop pairs instead of eliding them.
    # Unit granularity keeps every stage meaningful while never splitting pairs.
    groups = [list(u) for u in units]
    keep = FOREGROUND_KEEP_TURNS

    def assembled() -> list[dict[str, Any]]:
        return header + [m for g in groups for m in g]

    # Stage 1: elide tool observations older than the last 3 turns and truncate
    # tool arguments > 500 chars outside the recent window.
    for g in groups[:-keep] if len(groups) > keep else []:
        for m in g:
            if m.get("role") == "tool" and isinstance(m.get("content"), str):
                m["content"] = ELIDED_OBSERVATION
            elif m.get("role") == "tool" and isinstance(m.get("content"), list):
                m["content"] = [
                    {"type": "text", "text": ELIDED_OBSERVATION}
                    if isinstance(p, dict) and p.get("type") == "image_url"
                    else p
                    for p in m["content"]
                ]
            if m.get("role") == "assistant" and m.get("tool_calls"):
                _truncate_args(m["tool_calls"])
    if estimate_tokens(assembled()) <= max_tokens:
        return assembled()

    # Stage 2: drop oldest groups entirely; header and pairing stay intact.
    while len(groups) > 1 and estimate_tokens(assembled()) > max_tokens:
        groups.pop(0)

    # Stage 3: a single over-budget group is capped in place so we never return over budget.
    if estimate_tokens(assembled()) > max_tokens:
        for g in groups:
            for m in g:
                if m.get("role") == "tool" and isinstance(m.get("content"), str):
                    m["content"] = ELIDED_OBSERVATION
                elif m.get("role") == "tool" and isinstance(m.get("content"), list):
                    m["content"] = [
                        {"type": "text", "text": ELIDED_OBSERVATION}
                        if isinstance(p, dict) and p.get("type") == "image_url"
                        else p
                        for p in m["content"]
                    ]
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    _truncate_args(m["tool_calls"])
                content = m.get("content")
                if isinstance(content, str) and len(content) > RESIDUAL_TRUNCATE_CHARS:
                    m["content"] = content[:RESIDUAL_TRUNCATE_CHARS]
    return assembled()
