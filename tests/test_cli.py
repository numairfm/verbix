from pathlib import Path

import pytest

from verbix.cli import build_parser, load_env


def test_cli_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.prompt is None
    assert args.model == "google/gemini-2.5-flash"
    assert args.base_url == "https://openrouter.ai/api/v1"
    assert args.api_key is None
    assert args.max_tokens == 2048


def test_cli_parser_custom_args():
    parser = build_parser()
    args = parser.parse_args(
        [
            "hello world",
            "--model",
            "custom-model",
            "--base-url",
            "https://custom.endpoint/v1",
            "--api-key",
            "secret-key",
            "--max-tokens",
            "512",
        ]
    )
    assert args.prompt == "hello world"
    assert args.model == "custom-model"
    assert args.base_url == "https://custom.endpoint/v1"
    assert args.api_key == "secret-key"
    assert args.max_tokens == 512


def test_load_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TEST_KEY_FOO", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_KEY_FOO=bar_value\n")
    load_env(env_file)
    import os

    assert os.environ.get("TEST_KEY_FOO") == "bar_value"
