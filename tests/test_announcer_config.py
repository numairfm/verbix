"""Announcer prompt lives in config/announcer.yaml, not in runtime code."""

from pathlib import Path
from unittest.mock import MagicMock


def _announcer_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "src"
        / "verbix"
        / "minisweagent"
        / "config"
        / "announcer.yaml"
    )


def test_announcer_config_file_exists_with_constraints():
    import yaml

    cfg = yaml.safe_load(_announcer_path().read_text(encoding="utf-8"))
    template = cfg["agent"]["system_template"]
    assert "# VOICE & AUDIO CONSTRAINTS" in template
    assert "Prefer plain words over symbols" in template
    assert "Never recite links, paths, or code" in template


def test_runtime_announcer_uses_config_file(tmp_path):
    from verbix.runtime.runtime import Runtime

    rt = Runtime(workspace=tmp_path, provider=MagicMock())
    prompt = rt.announcer.config.system_template
    assert "Prefer plain words over symbols" in prompt
    assert "Never recite links, paths, or code" in prompt
