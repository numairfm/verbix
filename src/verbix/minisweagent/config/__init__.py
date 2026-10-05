"""Configuration files and utilities for mini-SWE-agent in Verbix."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

builtin_config_dir = Path(__file__).parent


def get_config_path(config_spec: str | Path) -> Path:
    """Get the path to a config file."""
    config_path = Path(config_spec)
    if config_path.suffix != ".yaml":
        config_path = config_path.with_suffix(".yaml")
    candidates = [
        config_path,
        Path(os.getenv("VERBIX_CONFIG_DIR", ".")) / config_path,
        builtin_config_dir / config_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find config file for {config_spec} (tried: {candidates})")


def get_config_from_spec(config_spec: str | Path) -> dict[str, Any]:
    """Get a config dictionary from a config spec or file path."""
    path = get_config_path(config_spec)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


__all__ = ["builtin_config_dir", "get_config_from_spec", "get_config_path"]
