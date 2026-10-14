from __future__ import annotations

import os
import plistlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from verbix.tools.base import ToolContext


@pytest.mark.asyncio
async def test_clipboard_tool_read_write_and_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import pyperclip

    from verbix.tools.clipboard import ClipboardTool

    state = {"buf": "initial text"}
    monkeypatch.setattr(pyperclip, "paste", lambda: state["buf"])
    monkeypatch.setattr(pyperclip, "copy", lambda val: state.__setitem__("buf", val))

    tool = ClipboardTool()
    ctx = ToolContext(workspace=tmp_path)

    r1 = await tool.execute({"action": "read"}, ctx)
    assert r1.is_error is False
    assert r1.observation == "initial text"

    r2 = await tool.execute({"action": "write", "text": "hello world"}, ctx)
    assert r2.is_error is False
    assert state["buf"] == "hello world"

    r3 = await tool.execute({"action": "read"}, ctx)
    assert r3.observation == "hello world"

    # Missing text on write
    err_missing = await tool.execute({"action": "write"}, ctx)
    assert err_missing.is_error is True

    # Unknown action
    err_action = await tool.execute({"action": "bogus"}, ctx)
    assert err_action.is_error is True

    # Backend unavailable error
    def _raise_clip():
        raise pyperclip.PyperclipException("no clipboard mechanism")

    monkeypatch.setattr(pyperclip, "paste", _raise_clip)
    err_backend = await tool.execute({"action": "read"}, ctx)
    assert err_backend.is_error is True
    assert "clipboard" in err_backend.observation.lower()


@pytest.mark.asyncio
async def test_media_tool_linux_mpris_and_pulsectl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from verbix.tools import media as media_mod
    from verbix.tools.media import MediaTool

    monkeypatch.setattr(media_mod.sys, "platform", "linux")

    called_mpris: list[str] = []

    async def fake_mpris_action(action: str) -> str:
        called_mpris.append(action)
        if action == "status":
            return "Playing: Daft Punk - Digital Love (spotify)"
        return f"Media {action} sent to spotify."

    monkeypatch.setattr(media_mod, "_linux_mpris_action", fake_mpris_action)

    vol_state = {"vol": 50, "muted": False}

    def fake_pulse_volume(action: str, level: int | None = None) -> str:
        if action == "volume_get":
            return f"Volume: {vol_state['vol']}% ({'muted' if vol_state['muted'] else 'unmuted'})"
        if action == "volume_set":
            assert level is not None
            vol_state["vol"] = max(0, min(100, int(level)))
            return f"Volume set to {vol_state['vol']}%."
        if action == "mute":
            vol_state["muted"] = not vol_state["muted"]
            return f"Audio {'muted' if vol_state['muted'] else 'unmuted'}."
        raise ValueError(action)

    monkeypatch.setattr(media_mod, "_linux_pulse_volume", fake_pulse_volume)

    tool = MediaTool()
    ctx = ToolContext(workspace=tmp_path)

    res_status = await tool.execute({"action": "status"}, ctx)
    assert res_status.is_error is False
    assert "Daft Punk" in res_status.observation

    res_pp = await tool.execute({"action": "play_pause"}, ctx)
    assert res_pp.is_error is False
    assert called_mpris == ["status", "play_pause"]

    res_vset = await tool.execute({"action": "volume_set", "level": 75}, ctx)
    assert res_vset.is_error is False
    assert vol_state["vol"] == 75

    res_vget = await tool.execute({"action": "volume_get"}, ctx)
    assert "75%" in res_vget.observation

    res_mute = await tool.execute({"action": "mute"}, ctx)
    assert res_mute.is_error is False
    assert vol_state["muted"] is True

    # Missing level for volume_set
    res_err = await tool.execute({"action": "volume_set"}, ctx)
    assert res_err.is_error is True


def test_app_catalog_discovers_linux_xdg_desktop_entries_and_filters_hidden_or_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from verbix.tools import app as app_mod

    apps_dir = tmp_path / "applications"
    apps_dir.mkdir(parents=True)

    # 1. Valid GUI app with field codes (%u)
    (apps_dir / "firefox.desktop").write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Firefox Web Browser\n"
        "GenericName=Web Browser\n"
        "Exec=/usr/bin/firefox --new-window %u\n",
        encoding="utf-8",
    )
    # 2. Hidden / NoDisplay entry -> must be excluded
    (apps_dir / "hidden-helper.desktop").write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Hidden Helper\n"
        "NoDisplay=true\n"
        "Exec=/usr/bin/helper\n",
        encoding="utf-8",
    )
    # 3. Terminal CLI entry -> must be excluded
    (apps_dir / "htop.desktop").write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Htop\n"
        "Terminal=true\n"
        "Exec=htop\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(app_mod, "_linux_desktop_dirs", lambda: [apps_dir])
    catalog = app_mod._discover_linux_apps()
    names = [entry.name for entry in catalog]
    assert "Firefox Web Browser" in names
    assert "Hidden Helper" not in names
    assert "Htop" not in names

    ff = next(e for e in catalog if e.name == "Firefox Web Browser")
    assert ff.argv == ("/usr/bin/firefox", "--new-window")


def test_app_catalog_discovers_macos_app_bundles_via_plistlib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from verbix.tools import app as app_mod

    apps_dir = tmp_path / "Applications"
    contents = apps_dir / "Spotify.app" / "Contents"
    macos_dir = contents / "MacOS"
    macos_dir.mkdir(parents=True)
    exe = macos_dir / "Spotify"
    exe.write_text("", encoding="utf-8")
    with (contents / "Info.plist").open("wb") as f:
        plistlib.dump(
            {
                "CFBundleDisplayName": "Spotify",
                "CFBundleExecutable": "Spotify",
                "CFBundleIdentifier": "com.spotify.client",
            },
            f,
        )

    monkeypatch.setattr(app_mod, "_macos_app_dirs", lambda: [apps_dir])
    catalog = app_mod._discover_macos_apps()
    assert len(catalog) == 1
    assert catalog[0].name == "Spotify"
    assert catalog[0].argv == (str(exe),)


@pytest.mark.asyncio
async def test_app_tool_launch_close_and_list_uses_catalog_and_psutil_never_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from verbix.tools import app as app_mod
    from verbix.tools.app import AppEntry, AppTool

    fake_catalog = [
        AppEntry(
            id="firefox",
            name="Firefox Web Browser",
            exe_name="firefox",
            argv=("/usr/bin/firefox",),
            keywords=("browser", "web"),
        ),
        AppEntry(
            id="spotify",
            name="Spotify",
            exe_name="spotify",
            argv=("/usr/bin/spotify",),
            keywords=("music",),
        ),
    ]
    monkeypatch.setattr(app_mod, "discover_installed_apps", lambda: fake_catalog)

    spawned_calls: list[ dict ] = []

    def fake_popen(argv, **kwargs):
        spawned_calls.append({"argv": list(argv), "kwargs": kwargs})
        return SimpleNamespace(pid=4242)

    monkeypatch.setattr(app_mod.subprocess, "Popen", fake_popen)

    terminated_pids: list[int] = []

    class FakeProc:
        def __init__(self, pid: int, name: str, uid: int):
            self.pid = pid
            self.info = {
                "pid": pid,
                "name": name,
                "exe": f"/usr/bin/{name}",
                "uids": SimpleNamespace(real=uid),
            }

        def terminate(self):
            terminated_pids.append(self.pid)

    current_uid = os.getuid() if hasattr(os, "getuid") else 1000
    fake_procs = [
        FakeProc(1111, "firefox", current_uid),
        FakeProc(os.getpid(), "firefox", current_uid),  # Verbix's own PID must NEVER be terminated!
        FakeProc(2222, "root_daemon", 0),
    ]
    monkeypatch.setattr(app_mod.psutil, "process_iter", lambda attrs=None: list(fake_procs))

    tool = AppTool()
    ctx = ToolContext(workspace=tmp_path)

    # 1. List running + installed apps
    res_list = await tool.execute({"action": "list"}, ctx)
    assert res_list.is_error is False
    assert "Firefox Web Browser" in res_list.observation

    # 2. Launch an installed catalog app -> Popen called with shell=False and start_new_session=True
    res_launch = await tool.execute({"action": "launch", "name": "firefox"}, ctx)
    assert res_launch.is_error is False
    assert len(spawned_calls) == 1
    assert spawned_calls[0]["argv"] == ["/usr/bin/firefox"]
    assert spawned_calls[0]["kwargs"].get("shell", False) is False
    assert spawned_calls[0]["kwargs"].get("start_new_session") is True

    # 3. Attempting to launch a shell or unlisted command is rejected!
    res_shell = await tool.execute({"action": "launch", "name": "bash -c 'rm -rf /'"}, ctx)
    assert res_shell.is_error is True
    assert len(spawned_calls) == 1

    res_unknown = await tool.execute({"action": "launch", "name": "nonexistent_app_xyz"}, ctx)
    assert res_unknown.is_error is True
    assert len(spawned_calls) == 1

    # 4. Close a running app -> terminates PID 1111 via psutil, never touches os.getpid()!
    res_close = await tool.execute({"action": "close", "name": "firefox"}, ctx)
    assert res_close.is_error is False
    assert terminated_pids == [1111]


def test_worker_registry_and_runtime_grounding_include_clipboard_media_and_app(tmp_path: Path):
    from verbix.runtime.runtime import Runtime
    from verbix.tools.spawn import SpawnTool

    rt = Runtime(workspace=tmp_path, provider=MagicMock())
    for tool_name in ("clipboard", "media", "app"):
        assert tool_name in rt.worker_tool_names
        assert tool_name in rt.system_prompt
        assert tool_name in SpawnTool(rt).description
