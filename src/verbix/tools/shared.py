from __future__ import annotations

import hashlib
from pathlib import Path

from verbix.tools.base import ToolContext


def resolve_contained_path(path_str: str, workspace: Path) -> Path:
    """Resolves a path and ensures it stays strictly within the workspace boundary."""
    norm_workspace = workspace.resolve()
    target = (workspace / path_str).resolve()
    try:
        target.relative_to(norm_workspace)
    except ValueError:
        raise ValueError(
            f"Path '{path_str}' escapes configured workspace '{workspace}'"
        ) from None
    return target


def record_version(target: Path, raw: bytes, context: ToolContext) -> None:
    context.file_versions[str(target)] = hashlib.sha256(raw).hexdigest()


def check_fresh(target: Path, context: ToolContext) -> str | None:
    """Returns None when fresh, else "unread" (never seen) or "stale" (changed)."""
    recorded = context.file_versions.get(str(target))
    if recorded is None:
        return "unread"
    if hashlib.sha256(target.read_bytes()).hexdigest() != recorded:
        return "stale"
    return None
