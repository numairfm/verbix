from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import time
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = 14
_ALLOWED_KEYS = ("provider", "stt_model", "retention_days")
_GUIDANCE_BLOCK = re.compile(r"\A\s*<!--.*?-->\s*", re.DOTALL)

_CONFIG_TEMPLATE = """\
# Verbix global config (~/.verbix/config.yaml).
# VERBIX_PROVIDER env and CLI flags win over this file.
# Uncomment and edit to set a default speech-to-text model:
# stt_model: /path/to/model.gguf
# Days to retain old tool logs and screenshots (default: 14):
# retention_days: 14
"""

_PERSONA_TEMPLATE = """\
<!-- Persona guidance: write how you want Verbix to sound below this block. This comment is stripped; the rest is appended to the system prompt under "# USER PERSONA". -->

"""

_MEMORY_TEMPLATE = """\
<!-- Memory notes: durable facts and user preferences saved across sessions. This comment is stripped; the rest is appended to the system prompt under "# USER MEMORY". -->

"""


def _read_config_file(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config in {path}: expected a mapping")  # noqa: TRY004 - plan mandates ValueError for all config errors
    for key in data:
        if key not in _ALLOWED_KEYS:
            raise ValueError(f"Unknown config key '{key}' in {path}")
    if "retention_days" in data and data["retention_days"] is not None:
        val = data["retention_days"]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or val <= 0:
            raise ValueError(
                f"Invalid 'retention_days' in {path}: expected a positive number"
            )
    return {k: data[k] for k in _ALLOWED_KEYS if k in data}


def load_config(
    workspace: Path | None = None, args: argparse.Namespace | None = None
) -> dict:
    """Merge config scopes: builtin < global (~/.verbix/config.yaml) < env < CLI."""
    cfg: dict = {"provider": None, "stt_model": None}
    global_file = Path.home() / ".verbix" / "config.yaml"
    if global_file.is_file():
        cfg.update(_read_config_file(global_file))
    env_provider = os.environ.get("VERBIX_PROVIDER")
    if env_provider:
        cfg["provider"] = env_provider
    env_retention = os.environ.get("VERBIX_RETENTION_DAYS")
    if env_retention:
        try:
            val = float(env_retention)
            if val <= 0:
                raise ValueError
            cfg["retention_days"] = int(val) if val.is_integer() else val
        except ValueError as e:
            raise ValueError(
                f"Invalid VERBIX_RETENTION_DAYS='{env_retention}': expected a positive number"
            ) from e
    if args is not None:
        for key in _ALLOWED_KEYS:
            value = getattr(args, key, None)
            if value is not None:
                if key == "retention_days" and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value <= 0
                ):
                    raise ValueError(
                        f"Invalid --retention-days='{value}': expected a positive number"
                    )
                cfg[key] = value
    return cfg


def resolve_retention_days(
    workspace: Path | None = None, args: argparse.Namespace | None = None
) -> float:
    """Resolves the configured retention period in days (default: 14)."""
    cfg = load_config(workspace, args)
    val = cfg.get("retention_days")
    return float(val) if val is not None else float(DEFAULT_RETENTION_DAYS)


def read_persona(workspace: Path | None = None) -> str:
    """Read global ~/.verbix/PERSONA.md with guidance block stripped."""
    path = Path.home() / ".verbix" / "PERSONA.md"
    if not path.is_file():
        return ""
    text = _GUIDANCE_BLOCK.sub("", path.read_text(encoding="utf-8"), count=1)
    return text.strip()


def fill_persona_slot(template: str, persona: str) -> str:
    """Fills the {{persona}} slot with persona text (or "(none)" when empty)."""
    text = persona.strip() if persona else ""
    return template.replace("{{persona}}", text if text else "(none)")


def memory_file() -> Path:
    """Returns the canonical path to ~/.verbix/MEMORY.md."""
    return Path.home() / ".verbix" / "MEMORY.md"


def read_memory() -> str:
    """Reads ~/.verbix/MEMORY.md with guidance block stripped, refusing symlinks."""
    path = memory_file()
    try:
        if path.parent.is_symlink() or path.is_symlink() or not path.is_file():
            return ""
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    body = _GUIDANCE_BLOCK.sub("", raw, count=1).strip()
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    return "\n".join(lines)[:8000]


def ensure_global_scaffold() -> None:
    """Create ~/.verbix/ with commented templates on first run only."""
    root = Path.home() / ".verbix"
    marker = root / ".scaffolded"
    cfg_file = root / "config.yaml"
    persona_file = root / "PERSONA.md"
    mem_file = root / "MEMORY.md"
    if marker.is_file():
        if not mem_file.exists() and not mem_file.is_symlink():
            mem_file.write_text(_MEMORY_TEMPLATE, encoding="utf-8")
        if not cfg_file.is_file() or not persona_file.is_file():
            logger.warning(
                "Verbix global config files missing under %s; see README.md Configuration.",
                root,
            )
        return
    root.mkdir(parents=True, exist_ok=True)
    if not cfg_file.is_file() and not cfg_file.is_symlink():
        cfg_file.write_text(_CONFIG_TEMPLATE, encoding="utf-8")
    if not persona_file.is_file() and not persona_file.is_symlink():
        persona_file.write_text(_PERSONA_TEMPLATE, encoding="utf-8")
    if not mem_file.is_file() and not mem_file.is_symlink():
        mem_file.write_text(_MEMORY_TEMPLATE, encoding="utf-8")
    marker.touch(exist_ok=True)


def default_workspace() -> Path:
    """Returns Verbix's dedicated default workspace directory (~/.verbix/workspace)."""
    return (Path.home() / ".verbix" / "workspace").resolve()


def resolve_workspace(workspace: Path | None = None, *, explicit: bool = False) -> Path:
    """Resolves and validates the workspace directory, refusing $HOME and / unless explicit."""
    if workspace is None:
        ws = default_workspace()
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    ws = Path(workspace).expanduser().resolve()
    home_resolved = Path.home().resolve()
    root_resolved = Path(ws.anchor).resolve() if ws.anchor else Path("/").resolve()
    if not explicit:
        if ws == home_resolved:
            raise ValueError(
                f"Refusing to use home directory '{ws}' as workspace without an explicit --workspace override."
            )
        if ws == root_resolved:
            raise ValueError(
                f"Refusing to use filesystem root '{ws}' as workspace without an explicit --workspace override."
            )
    if not ws.is_dir():
        raise ValueError(f"Workspace directory '{ws}' does not exist.")
    return ws


def _effective_mtime(entry: Path) -> float | None:
    """Returns the effective modification timestamp for a state entry, safely handling symlinks/errors."""
    try:
        st = entry.lstat()
    except OSError:
        return None
    if entry.is_symlink() or not entry.is_dir():
        return st.st_mtime
    mtimes: list[float] = []
    try:
        for child in entry.iterdir():
            child_mtime = _effective_mtime(child)
            if child_mtime is not None:
                mtimes.append(child_mtime)
    except OSError:
        return st.st_mtime
    return max(mtimes) if mtimes else st.st_mtime


def _remove_state_entry(entry: Path, allowed_parent: Path) -> None:
    """Deletes a file, symlink, or directory strictly within allowed_parent without following symlinks."""
    try:
        parent_res = allowed_parent.resolve()
        if entry.parent.resolve() != parent_res:
            return
        if entry.is_symlink() or not entry.is_dir():
            entry.unlink(missing_ok=True)
            return
        entry.resolve().relative_to(parent_res)
        shutil.rmtree(entry, ignore_errors=True)
    except (OSError, ValueError):
        return


def prune_state(
    verbix_root: Path | None = None,
    *,
    retention_days: float = DEFAULT_RETENTION_DAYS,
) -> None:
    """Deletes expired tool logs and screenshots under ~/.verbix."""
    try:
        root = (verbix_root or (Path.home() / ".verbix")).resolve()
        if not root.is_dir() or root.is_symlink():
            return
    except OSError:
        return

    cutoff = time.time() - (float(retention_days) * 86400.0)

    for sub in ("logs", "screenshots"):
        target_dir = root / sub
        try:
            if not target_dir.is_dir() or target_dir.is_symlink():
                continue
            entries = list(target_dir.iterdir())
        except OSError:
            continue

        for entry in entries:
            mtime = _effective_mtime(entry)
            if mtime is not None and mtime < cutoff:
                _remove_state_entry(entry, target_dir)
