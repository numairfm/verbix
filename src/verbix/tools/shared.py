from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

from verbix.tools.base import ToolContext

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 51200


def _cap_text(text: str, max_lines: int, max_bytes: int) -> str:
    lines = text.splitlines()
    if len(lines) > max_lines:
        h = max_lines // 2
        lines = lines[:h] + lines[-(max_lines - h) :]
    text = "\n".join(lines)
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    h_bytes = max_bytes // 2
    head = raw[:h_bytes].decode("utf-8", errors="ignore")
    tail = raw[-(max_bytes - h_bytes) :].decode("utf-8", errors="ignore")
    return head + tail


def bind_output(
    text: str,
    *,
    max_lines: int = 2000,
    max_bytes: int = 51200,
    spill_dir: Path | None,
    rel_to: Path | None = None,
) -> str:
    lines = text.splitlines()
    n_bytes = len(text.encode("utf-8"))
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


def _is_secret_basename(name: str) -> bool:
    return name == ".env" or name == "auth.json" or name.startswith("auth.")


def resolve_contained_path(path_str: str, workspace: Path) -> Path:
    """Resolves a path and ensures it stays strictly within the workspace boundary."""
    if _is_secret_basename(Path(path_str).name):
        raise ValueError(f"refusing to access secret file '{path_str}'")
    norm_workspace = workspace.resolve()
    target = (workspace / path_str).resolve()
    if _is_secret_basename(target.name):
        raise ValueError(f"refusing to access secret file '{path_str}'")
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


def find_ripgrep_binary() -> str:
    """Finds the ripgrep binary bundled in the venv or available in PATH."""
    venv_bin = Path(sys.executable).parent / ("rg.exe" if sys.platform == "win32" else "rg")
    if venv_bin.is_file() and os.access(venv_bin, os.X_OK):
        return str(venv_bin)
    found = shutil.which("rg")
    if found:
        return found
    return "rg"

