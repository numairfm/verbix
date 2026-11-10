from __future__ import annotations

import ctypes.util
from pathlib import Path

import pytest

from verbix.voice import portaudio


@pytest.fixture(autouse=True)
def _reset_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(portaudio, "_patched", False, raising=False)
    monkeypatch.delenv(portaudio.ENV_VAR, raising=False)


def test_bundled_library_is_none_without_the_env_var() -> None:
    assert portaudio.bundled_library() is None
    assert portaudio.ensure_loadable() is None


def test_bundled_library_ignores_a_missing_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(portaudio.ENV_VAR, str(tmp_path / "missing.so.2"))
    assert portaudio.bundled_library() is None
    assert portaudio.ensure_loadable() is None


def test_find_library_returns_the_bundled_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundled = tmp_path / "libportaudio.so.2"
    bundled.write_bytes(b"not really an ELF")
    monkeypatch.setenv(portaudio.ENV_VAR, str(bundled))

    original = ctypes.util.find_library

    assert portaudio.ensure_loadable() == bundled
    assert ctypes.util.find_library("portaudio") == str(bundled)
    # Other lookups keep the host behaviour.
    assert ctypes.util.find_library("m") == original("m")


@pytest.mark.parametrize(
    "relative",
    ["src/verbix/voice/capture.py", "src/verbix/voice/callback_sink.py"],
)
def test_ensure_loadable_runs_before_sounddevice_import(relative: str) -> None:
    source = Path(__file__).resolve().parents[1] / relative
    text = source.read_text(encoding="utf-8")
    ensure_at = text.find("ensure_loadable()")
    sounddevice_at = text.find("import sounddevice")
    assert ensure_at != -1, f"{relative} does not call ensure_loadable()"
    assert sounddevice_at != -1, f"{relative} does not import sounddevice"
    assert ensure_at < sounddevice_at, (
        f"{relative} imports sounddevice before patching find_library"
    )
