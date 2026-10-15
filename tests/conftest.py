from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _isolate_home_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isolates Path.home() per test so pytest never touches the user's real ~/.verbix."""
    isolated_home = tmp_path_factory.mktemp("verbix_home")
    monkeypatch.setattr(Path, "home", lambda: isolated_home)


def assert_history_valid(messages: list[dict[str, Any]]) -> None:
    """H1: every assistant tool_call id has a matching tool message (no dangling calls)."""
    expected = [tc.get("id") for m in messages for tc in (m.get("tool_calls") or [])]
    provided = {
        m.get("tool_call_id") for m in messages if m.get("role") == "tool" and m.get("tool_call_id")
    }
    dangling = [cid for cid in expected if cid not in provided]
    assert not dangling, f"Dangling tool_calls without tool response: {dangling}"
    unknown = sorted(c for c in provided if c not in set(expected))
    assert not unknown, f"Tool messages without matching tool_call: {unknown}"
