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
# Project .verbix/config.yaml overrides this file; VERBIX_PROVIDER env and CLI flags win over both.
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
    workspace: Path | None, args: argparse.Namespace | None = None
) -> dict:
    """Merge config scopes: builtin < global < project < env < CLI."""
    cfg: dict = {"provider": None, "stt_model": None}
    global_file = Path.home() / ".verbix" / "config.yaml"
    if global_file.is_file():
        cfg.update(_read_config_file(global_file))
    if workspace is not None:
        project_file = Path(workspace) / ".verbix" / "config.yaml"
        if project_file.is_file():
            cfg.update(_read_config_file(project_file))
    env_provider = os.environ.get("VERBIX_PROVIDER")
    if env_provider:
        cfg["provider"] = env_provider
    if args is not None:
        for key in _ALLOWED_KEYS:
            value = getattr(args, key, None)
            if value is not None:
                cfg[key] = value
    return cfg


def read_persona(workspace: Path | None) -> str:
    """Global + project PERSONA.md concatenated, guidance block stripped."""
    candidates = [Path.home() / ".verbix" / "PERSONA.md"]
    if workspace is not None:
        candidates.append(Path(workspace) / ".verbix" / "PERSONA.md")
    parts = []
    for path in candidates:
        if path.is_file():
            text = _GUIDANCE_BLOCK.sub("", path.read_text(encoding="utf-8"), count=1)
            text = text.strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


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
