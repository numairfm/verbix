from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_ALLOWED_KEYS = ("provider", "stt_model")
_GUIDANCE_BLOCK = re.compile(r"\A\s*<!--.*?-->\s*", re.DOTALL)

_CONFIG_TEMPLATE = """\
# Verbix global config (~/.verbix/config.yaml).
# VERBIX_PROVIDER env and CLI flags win over this file.
# Uncomment and edit to set a default speech-to-text model:
# stt_model: /path/to/model.gguf
"""

_PERSONA_TEMPLATE = """\
<!-- Persona guidance: write how you want Verbix to sound below this block. This comment is stripped; the rest is appended to the system prompt under "# USER PERSONA". -->

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
    if args is not None:
        for key in _ALLOWED_KEYS:
            value = getattr(args, key, None)
            if value is not None:
                cfg[key] = value
    return cfg


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


def ensure_global_scaffold() -> None:
    """Create ~/.verbix/ with commented templates on first run only."""
    root = Path.home() / ".verbix"
    marker = root / ".scaffolded"
    cfg_file = root / "config.yaml"
    persona_file = root / "PERSONA.md"
    if marker.is_file():
        if not cfg_file.is_file() or not persona_file.is_file():
            logger.warning(
                "Verbix global config files missing under %s; see README.md Configuration.",
                root,
            )
        return
    root.mkdir(parents=True, exist_ok=True)
    if not cfg_file.is_file():
        cfg_file.write_text(_CONFIG_TEMPLATE, encoding="utf-8")
    if not persona_file.is_file():
        persona_file.write_text(_PERSONA_TEMPLATE, encoding="utf-8")
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
