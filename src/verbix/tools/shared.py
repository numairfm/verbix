from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path

from verbix.tools.base import ToolContext

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 51200


def _cap_text(text: str, max_lines: int, max_bytes: int) -> str:
    lines = text.splitlines()
    if len(lines) > max_lines:
        head_n = max_lines // 2
        tail_n = max_lines - head_n
        lines = lines[:head_n] + lines[-tail_n:] if tail_n else lines[:head_n]
    capped = "\n".join(lines)
    total = sum(len(c.encode()) for c in capped)
    if total <= max_bytes:
        return capped
    head_budget = max_bytes // 2
    tail_budget = max_bytes - head_budget
    head_chars: list[str] = []
    used = 0
    for c in capped:
        n = len(c.encode())
        if used + n > head_budget:
            break
        head_chars.append(c)
        used += n
    tail_chars: list[str] = []
    used = 0
    for c in reversed(capped):
        n = len(c.encode())
        if used + n > tail_budget:
            break
        tail_chars.append(c)
        used += n
    return "".join(head_chars) + "".join(reversed(tail_chars))


def bind_output(
    text: str,
    *,
    max_lines: int = 2000,
    max_bytes: int = 51200,
    spill_dir: Path | None,
    rel_to: Path | None = None,
) -> str:
    lines = text.splitlines()
    n_bytes = sum(len(c.encode()) for c in text)
    if len(lines) <= max_lines and n_bytes <= max_bytes:
        return text
    capped = _cap_text(text, max_lines, max_bytes)
    if spill_dir is None:
        return capped
    try:
        spill_dir.mkdir(parents=True, exist_ok=True)
        target = spill_dir / f"tool_{uuid.uuid4().hex[:12]}"
        target.write_text(text)
    except OSError:
        return capped
    try:
        display = str(target.relative_to(rel_to)) if rel_to is not None else target.name
    except ValueError:
        display = target.name
    return f"{capped}\n... output truncated; full content saved to {display} ..."


def prune_spills(spill_dir: Path, older_than_days: int = 7) -> None:
    try:
        entries = list(spill_dir.iterdir())
    except OSError:
        return
    cutoff = time.time() - older_than_days * 86400
    for entry in entries:
        try:
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                entry.unlink()
        except OSError:
            continue


def resolve_output_limits(tool_output: dict | None) -> tuple[int, int]:
    max_lines = DEFAULT_MAX_LINES
    max_bytes = DEFAULT_MAX_BYTES
    if isinstance(tool_output, dict):
        v = tool_output.get("max_lines")
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            max_lines = v
        v = tool_output.get("max_bytes")
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            max_bytes = v
    return (max_lines, max_bytes)


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
