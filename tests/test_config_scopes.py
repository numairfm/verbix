from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytest


def _patch_home(monkeypatch, home: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: home)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_unknown_config_key_fails_loudly(tmp_path, monkeypatch):
    from verbix.config import load_config

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    cfg = _write(home / ".verbix" / "config.yaml", "bogus_key: 1\n")
    _patch_home(monkeypatch, home)
    monkeypatch.delenv("VERBIX_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="bogus_key"):
        load_config(ws)
    try:
        load_config(ws)
    except ValueError as e:
        assert str(cfg) in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_project_overrides_global(tmp_path, monkeypatch):
    from verbix.config import load_config

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    _write(
        home / ".verbix" / "config.yaml",
        "provider: openrouter\nstt_model: /global/model.gguf\n",
    )
    _write(ws / ".verbix" / "config.yaml", "provider: local\n")
    _patch_home(monkeypatch, home)
    monkeypatch.delenv("VERBIX_PROVIDER", raising=False)
    cfg = load_config(ws)
    assert cfg["provider"] == "local"
    assert cfg["stt_model"] == "/global/model.gguf"


def test_missing_files_yield_empty_defaults(tmp_path, monkeypatch):
    from verbix.config import load_config, read_persona

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    _patch_home(monkeypatch, home)
    monkeypatch.delenv("VERBIX_PROVIDER", raising=False)
    assert load_config(ws) == {"provider": None, "stt_model": None}
    assert read_persona(ws) == ""


def test_persona_strips_guidance_block(tmp_path, monkeypatch):
    from verbix.config import read_persona

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    _write(
        home / ".verbix" / "PERSONA.md",
        "<!-- guidance: write persona below -->\nBe chill.\n",
    )
    _patch_home(monkeypatch, home)
    text = read_persona(ws)
    assert "Be chill." in text
    assert "guidance" not in text
    assert "<!--" not in text


def test_env_provider_beats_config_files(tmp_path, monkeypatch):
    from verbix.config import load_config

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    _write(home / ".verbix" / "config.yaml", "provider: openrouter\n")
    _write(ws / ".verbix" / "config.yaml", "provider: local\n")
    _patch_home(monkeypatch, home)
    monkeypatch.setenv("VERBIX_PROVIDER", "env-provider")
    assert load_config(ws)["provider"] == "env-provider"


def test_cli_args_beat_env_and_files(tmp_path, monkeypatch):
    from verbix.config import load_config

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    _write(
        home / ".verbix" / "config.yaml",
        "provider: openrouter\nstt_model: /global/model.gguf\n",
    )
    _patch_home(monkeypatch, home)
    monkeypatch.setenv("VERBIX_PROVIDER", "env-provider")
    args = argparse.Namespace(provider="cli-provider", stt_model="/cli/model.gguf")
    cfg = load_config(ws, args)
    assert cfg == {"provider": "cli-provider", "stt_model": "/cli/model.gguf"}
    # None attrs do not override
    args = argparse.Namespace(provider=None, stt_model=None)
    cfg = load_config(ws, args)
    assert cfg == {"provider": "env-provider", "stt_model": "/global/model.gguf"}


def test_malformed_yaml_fails_naming_file(tmp_path, monkeypatch):
    from verbix.config import load_config

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    cfg = _write(home / ".verbix" / "config.yaml", "provider: [unclosed\n")
    _patch_home(monkeypatch, home)
    monkeypatch.delenv("VERBIX_PROVIDER", raising=False)
    with pytest.raises(ValueError) as exc:
        load_config(ws)
    assert str(cfg) in str(exc.value)


def test_persona_concatenates_global_then_project(tmp_path, monkeypatch):
    from verbix.config import read_persona

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    _write(home / ".verbix" / "PERSONA.md", "Global voice.\n")
    _write(ws / ".verbix" / "PERSONA.md", "Project voice.\n")
    _patch_home(monkeypatch, home)
    text = read_persona(ws)
    assert "Global voice." in text
    assert "Project voice." in text
    assert text.index("Global voice.") < text.index("Project voice.")


def test_persona_appends_as_section_not_replacement(tmp_path, monkeypatch):
    from verbix.cli import build_parser, build_runtime

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    _write(ws / ".verbix" / "PERSONA.md", "Be chill.\n")
    _patch_home(monkeypatch, home)
    monkeypatch.delenv("VERBIX_PROVIDER", raising=False)
    args = build_parser().parse_args(["--workspace", str(ws)])
    runtime = build_runtime(args)
    assert "# USER PERSONA" in runtime.system_prompt
    assert "Be chill." in runtime.system_prompt
    assert "# VOICE & AUDIO CONSTRAINTS" in runtime.system_prompt


def test_cli_system_prompt_flag_beats_persona_file(tmp_path, monkeypatch):
    from verbix.cli import build_parser, build_runtime

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    _write(ws / ".verbix" / "PERSONA.md", "Be chill.\n")
    _patch_home(monkeypatch, home)
    monkeypatch.delenv("VERBIX_PROVIDER", raising=False)
    args = build_parser().parse_args(
        ["--workspace", str(ws), "--system-prompt", "Flag persona."]
    )
    runtime = build_runtime(args)
    assert "Flag persona." in runtime.system_prompt
    assert "Be chill." not in runtime.system_prompt


def test_stt_model_from_config_reaches_transcribe_stt(tmp_path, monkeypatch):
    import verbix.voice.transcribe_stt as stt_mod
    from verbix.cli import _build_voice_pipeline, build_parser, build_runtime

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    model = tmp_path / "custom-stt.gguf"
    model.write_bytes(b"dummy")
    _write(ws / ".verbix" / "config.yaml", f"stt_model: {model}\n")
    _patch_home(monkeypatch, home)
    monkeypatch.delenv("VERBIX_PROVIDER", raising=False)
    monkeypatch.delenv("VERBIX_STT_MODEL", raising=False)

    captured: dict = {}

    class FakeSTT:
        def __init__(self, model_path=None, workspace=None):
            captured["model_path"] = model_path

    monkeypatch.setattr(stt_mod, "TranscribeSTT", FakeSTT)
    args = build_parser().parse_args(["--workspace", str(ws), "--audio-sink", "subprocess"])
    runtime = build_runtime(args)
    _build_voice_pipeline(runtime, args)
    assert str(captured["model_path"]) == str(model)


def test_stt_model_flag_beats_config_file(tmp_path, monkeypatch):
    import verbix.voice.transcribe_stt as stt_mod
    from verbix.cli import _build_voice_pipeline, build_parser, build_runtime

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    _write(ws / ".verbix" / "config.yaml", "stt_model: /config/model.gguf\n")
    _patch_home(monkeypatch, home)
    monkeypatch.delenv("VERBIX_PROVIDER", raising=False)
    monkeypatch.delenv("VERBIX_STT_MODEL", raising=False)

    captured: dict = {}

    class FakeSTT:
        def __init__(self, model_path=None, workspace=None):
            captured["model_path"] = model_path

    monkeypatch.setattr(stt_mod, "TranscribeSTT", FakeSTT)
    args = build_parser().parse_args(
        ["--workspace", str(ws), "--stt-model", "/flag/model.gguf",
         "--audio-sink", "subprocess"]
    )
    runtime = build_runtime(args)
    _build_voice_pipeline(runtime, args)
    assert str(captured["model_path"]) == "/flag/model.gguf"


def test_scaffold_creates_global_once_with_guidance(tmp_path, monkeypatch):
    from verbix.config import ensure_global_scaffold, load_config, read_persona

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    _patch_home(monkeypatch, home)
    monkeypatch.delenv("VERBIX_PROVIDER", raising=False)
    ensure_global_scaffold()
    cfg_file = home / ".verbix" / "config.yaml"
    persona_file = home / ".verbix" / "PERSONA.md"
    assert cfg_file.is_file()
    assert persona_file.is_file()
    assert "stt_model" in cfg_file.read_text(encoding="utf-8")
    assert "<!--" in persona_file.read_text(encoding="utf-8")
    assert load_config(ws) == {"provider": None, "stt_model": None}
    assert read_persona(ws) == ""
    # Second run preserves user edits (creates once).
    _write(cfg_file, "provider: local\n")
    _write(persona_file, "Be chill.\n")
    ensure_global_scaffold()
    assert cfg_file.read_text(encoding="utf-8") == "provider: local\n"
    assert persona_file.read_text(encoding="utf-8") == "Be chill.\n"


def test_scaffold_never_touches_project_dir(tmp_path, monkeypatch):
    from verbix.config import ensure_global_scaffold

    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    _patch_home(monkeypatch, home)
    ensure_global_scaffold()
    assert not (ws / ".verbix").exists()


def test_deleted_scaffold_not_recreated(tmp_path, monkeypatch, caplog):
    from verbix.config import ensure_global_scaffold

    home = tmp_path / "home"
    _patch_home(monkeypatch, home)
    ensure_global_scaffold()
    (home / ".verbix" / "config.yaml").unlink()
    (home / ".verbix" / "PERSONA.md").unlink()
    assert (home / ".verbix" / ".scaffolded").is_file()
    with caplog.at_level(logging.WARNING, logger="verbix.config"):
        ensure_global_scaffold()
    assert not (home / ".verbix" / "config.yaml").exists()
    assert not (home / ".verbix" / "PERSONA.md").exists()
    assert any("README" in r.message for r in caplog.records)
