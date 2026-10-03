from pathlib import Path

import pytest

from verbix.cli import build_parser, load_env


def test_cli_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.prompt is None
    assert args.model == "google/gemini-3.5-flash-lite"
    assert args.base_url == "https://openrouter.ai/api/v1"
    assert args.api_key is None
    assert args.max_tokens == 2048
    assert args.voice is False
    assert args.ptt is None


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


def test_cli_parser_serve_and_direct():
    parser = build_parser()
    args_serve = parser.parse_args(["--serve", "--socket-path", "/tmp/test.sock"])
    assert args_serve.serve is True
    assert str(args_serve.socket_path) == "/tmp/test.sock"

    args_direct = parser.parse_args(["--direct", "direct prompt"])
    assert args_direct.direct is True
    assert args_direct.prompt == "direct prompt"


def test_cli_parser_voice_and_ptt():
    parser = build_parser()
    args_voice = parser.parse_args(["--voice"])
    assert args_voice.voice is True

    args_ptt = parser.parse_args(["--ptt", "down"])
    assert args_ptt.ptt == "down"
