from __future__ import annotations

import json
from pathlib import Path

import pytest


def _patch_home(monkeypatch, home: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: home)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _clean_env(monkeypatch):
    for var in (
        "VERBIX_PROVIDER",
        "VERBIX_API_KEY",
        "VERBIX_BASE_URL",
        "VERBIX_MODEL",
        "OPENROUTER_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


def test_select_entry_by_name(tmp_path, monkeypatch):
    from verbix.cli import build_parser, build_runtime

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    _write(
        home / ".verbix" / "auth.json",
        json.dumps(
            {
                "openrouter": {
                    "type": "api",
                    "key": "sk-or-123",
                    "model": "openrouter/google/gemini-3.5-flash-lite",
                    "base_url": "https://openrouter.ai/api/v1",
                },
                "local": {"type": "api", "key": "local-key"},
            }
        ),
    )
    _patch_home(monkeypatch, home)
    _clean_env(monkeypatch)
    args = build_parser().parse_args(["--workspace", str(ws), "--provider", "openrouter"])
    runtime = build_runtime(args)
    assert runtime.provider.config.model_kwargs["api_key"] == "sk-or-123"
    assert runtime.provider.config.model_name == "openrouter/google/gemini-3.5-flash-lite"
    assert runtime.provider.config.model_kwargs["api_base"] == "https://openrouter.ai/api/v1"


def test_missing_entry_fails_startup_with_guidance(tmp_path, monkeypatch):
    from verbix.cli import build_parser, build_runtime

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    _write(home / ".verbix" / "auth.json", json.dumps({"openrouter": {"type": "api", "key": "sk-x"}}))
    _patch_home(monkeypatch, home)
    _clean_env(monkeypatch)
    args = build_parser().parse_args(["--workspace", str(ws), "--provider", "nope"])
    with pytest.raises(ValueError) as exc:
        build_runtime(args)
    msg = str(exc.value)
    assert "auth.json" in msg
    assert '"type": "api"' in msg or "'type'" in msg or "type" in msg


def test_env_keys_still_work_without_auth_file(tmp_path, monkeypatch):
    from verbix.cli import build_parser, build_runtime

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    _patch_home(monkeypatch, home)
    _clean_env(monkeypatch)
    monkeypatch.setenv("VERBIX_API_KEY", "env-key-123")
    args = build_parser().parse_args(["--workspace", str(ws)])
    runtime = build_runtime(args)
    assert runtime.provider.config.model_kwargs["api_key"] == "env-key-123"


def test_flag_model_beats_entry_model(tmp_path, monkeypatch):
    from verbix.cli import build_parser, build_runtime

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    _write(
        home / ".verbix" / "auth.json",
        json.dumps({"openrouter": {"type": "api", "key": "sk-x", "model": "entry-model"}}),
    )
    _patch_home(monkeypatch, home)
    _clean_env(monkeypatch)
    args = build_parser().parse_args(
        ["--workspace", str(ws), "--provider", "openrouter", "--model", "flag-model"]
    )
    runtime = build_runtime(args)
    assert runtime.provider.config.model_name == "flag-model"


def test_malformed_entry_fails_naming_file(tmp_path, monkeypatch):
    from verbix.auth import resolve_provider

    home = tmp_path / "home"
    auth_file = _write(
        home / ".verbix" / "auth.json",
        json.dumps({"openrouter": {"type": "oauth", "key": "sk-x"}}),
    )
    _patch_home(monkeypatch, home)
    _clean_env(monkeypatch)
    with pytest.raises(ValueError) as exc:
        resolve_provider("openrouter")
    assert str(auth_file) in str(exc.value)


def test_provider_from_config_file_and_env(tmp_path, monkeypatch):
    from verbix.cli import build_parser, build_runtime

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    _write(
        home / ".verbix" / "auth.json",
        json.dumps(
            {
                "openrouter": {"type": "api", "key": "sk-or", "model": "entry-model"},
                "local": {"type": "api", "key": "local-key"},
            }
        ),
    )
    _write(home / ".verbix" / "config.yaml", "provider: local\n")
    _patch_home(monkeypatch, home)
    _clean_env(monkeypatch)
    # config file selects entry when no flag/env
    args = build_parser().parse_args(["--workspace", str(ws)])
    assert build_runtime(args).provider.config.model_kwargs["api_key"] == "local-key"
    # env beats config file
    monkeypatch.setenv("VERBIX_PROVIDER", "openrouter")
    args = build_parser().parse_args(["--workspace", str(ws)])
    assert build_runtime(args).provider.config.model_kwargs["api_key"] == "sk-or"
    # flag beats env
    args = build_parser().parse_args(["--workspace", str(ws), "--provider", "local"])
    assert build_runtime(args).provider.config.model_kwargs["api_key"] == "local-key"
