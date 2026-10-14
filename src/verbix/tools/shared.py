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
        display = str(target.relative_to(rel_to)) if rel_to is not None else str(target)
    except ValueError:
        display = str(target)
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
    """Resolves a path and ensures it stays strictly within the workspace or safe ~/.verbix subdirs."""
    if _is_secret_basename(Path(path_str).name):
        raise ValueError(f"refusing to access secret file '{path_str}'")
    norm_workspace = workspace.resolve()
    raw_path = Path(path_str).expanduser()
    target = raw_path.resolve() if raw_path.is_absolute() else (workspace / raw_path).resolve()
    if _is_secret_basename(target.name):
        raise ValueError(f"refusing to access secret file '{path_str}'")
    allowed_roots = [
        norm_workspace,
        *((Path.home() / ".verbix" / sub).resolve() for sub in ("tasks", "logs", "screenshots")),
    ]
    for root in allowed_roots:
        try:
            target.relative_to(root)
            return target
        except ValueError:
            continue
    raise ValueError(
        f"Path '{path_str}' escapes configured workspace '{workspace}'"
    )


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


def find_ripgrep_binary() -> str | None:
    """Finds the ripgrep binary bundled in the venv or available in PATH, or None."""
    venv_bin = Path(sys.executable).parent / ("rg.exe" if sys.platform == "win32" else "rg")
    if venv_bin.is_file() and os.access(venv_bin, os.X_OK):
        return str(venv_bin)
    return shutil.which("rg")


_PRUNED_DIRS = frozenset({".git", ".verbix", ".venv", "node_modules", "__pycache__"})


def _read_dir_spec(d: Path):
    import pathspec

    lines: list[str] = []
    for fname in (".gitignore", ".ignore"):
        ig = d / fname
        if ig.is_file():
            try:
                for raw_line in ig.read_text(encoding="utf-8", errors="replace").splitlines():
                    s = raw_line.strip()
                    if s and not s.startswith("#"):
                        lines.append(s)
            except OSError:
                pass
    return pathspec.GitIgnoreSpec.from_lines(lines) if lines else None


def _is_ignored(
    path: Path, is_dir: bool, ws: Path, spec_cache: dict[Path, object]
) -> bool:
    rel_parts = path.relative_to(ws).parts
    ancestors = [ws]
    cur = ws
    for part in rel_parts[:-1]:
        cur = cur / part
        ancestors.append(cur)

    ignored = False
    for anc in ancestors:
        if anc not in spec_cache:
            spec_cache[anc] = _read_dir_spec(anc)
        spec = spec_cache[anc]
        if spec is None:
            continue
        rel_to_anc = path.relative_to(anc).as_posix() + ("/" if is_dir else "")
        res = spec.check_file(rel_to_anc)  # type: ignore[attr-defined]
        if res.include is not None:
            ignored = bool(res.include)
    return ignored


def iter_workspace_files(
    target: Path, workspace: Path, glob_pattern: str | None = None
) -> list[Path]:
    """Walks target in pure Python, pruning ignored directories and honoring .gitignore via pathspec."""
    import pathspec

    ws = workspace.resolve()
    tgt = target.resolve()
    if tgt.is_file():
        return [tgt]

    spec_cache: dict[Path, object] = {}
    glob_spec = (
        pathspec.GitIgnoreSpec.from_lines([glob_pattern.strip()])
        if glob_pattern and glob_pattern.strip()
        else None
    )
    results: list[Path] = []
    for root_str, dirnames, filenames in os.walk(tgt):
        root_path = Path(root_str)
        kept_dirs: list[str] = []
        for d in sorted(dirnames):
            if d in _PRUNED_DIRS or d.startswith("."):
                continue
            d_path = root_path / d
            if _is_ignored(d_path, True, ws, spec_cache):
                continue
            kept_dirs.append(d)
        dirnames[:] = kept_dirs

        for fname in sorted(filenames):
            if fname.startswith(".") or _is_secret_basename(fname):
                continue
            fpath = root_path / fname
            try:
                resolved = fpath.resolve()
                resolved.relative_to(ws)
            except (OSError, ValueError):
                continue
            if resolved.name.startswith(".") or _is_secret_basename(resolved.name):
                continue
            if _is_ignored(fpath, False, ws, spec_cache):
                continue
            if glob_spec is not None:
                rel_tgt = fpath.relative_to(tgt).as_posix()
                if not glob_spec.match_file(rel_tgt):
                    continue
            results.append(fpath)
    return sorted(results)

