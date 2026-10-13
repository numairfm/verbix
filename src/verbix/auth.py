from __future__ import annotations

import json
from pathlib import Path

_EXAMPLE = (
    "{\n"
    '  "openrouter": {\n'
    '    "type": "api",\n'
    '    "key": "sk-...",\n'
    '    "model": "openrouter/google/gemini-3.5-flash-lite",\n'
    '    "base_url": "https://openrouter.ai/api/v1"\n'
    "  }\n"
    "}"
)


def auth_file_path() -> Path:
    return Path.home() / ".verbix" / "auth.json"


def resolve_provider(name: str) -> dict:
    """Look up a provider entry by name; fail fast naming the file."""
    path = auth_file_path()
    if not path.is_file():
        raise ValueError(
            f"Unknown provider '{name}': no entry in {path}. "
            f"Add it to {path}, e.g.:\n{_EXAMPLE}"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Invalid auth config in {path}: expected a mapping")  # noqa: TRY004 - task mandates ValueError for all auth errors
    if name not in data:
        raise ValueError(
            f"Unknown provider '{name}': no entry in {path}. "
            f"Add it to {path}, e.g.:\n{_EXAMPLE}"
        )
    entry = data[name]
    if not isinstance(entry, dict):
        raise ValueError(  # noqa: TRY004 - task mandates ValueError for all auth errors
            f"Invalid entry for provider '{name}' in {path}: expected a mapping"
        )
    if entry.get("type") != "api":
        raise ValueError(
            f"Invalid entry for provider '{name}' in {path}: 'type' must be \"api\""
        )
    key = entry.get("key")
    if not isinstance(key, str) or not key:
        raise ValueError(
            f"Invalid entry for provider '{name}' in {path}: 'key' must be a non-empty string"
        )
    model = entry.get("model")
    base_url = entry.get("base_url")
    for field, value in (("model", model), ("base_url", base_url)):
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"Invalid entry for provider '{name}' in {path}: '{field}' must be a string"
            )
    return {"key": key, "model": model, "base_url": base_url}
