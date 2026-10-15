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


@pytest.mark.asyncio
async def test_linux_mpris_prefers_paused_over_stopped_and_honors_player_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import dbus_fast.aio
    from dbus_fast.constants import MessageType

    from verbix.tools import media as media_mod
    from verbix.tools.media import MediaTool

    monkeypatch.setattr(media_mod.sys, "platform", "linux")

    sent_Commands: list[tuple[str, str]] = []
    chromium_status = {"status": "Stopped"}

    class FakeBus:
        async def connect(self):
            return self

        def disconnect(self):
            pass

        async def call(self, msg):
            if msg.member == "ListNames":
                return SimpleNamespace(
                    message_type=MessageType.METHOD_RETURN,
                    body=[
                        [
                            "org.mpris.MediaPlayer2.chromium.instance3684",
                            "org.mpris.MediaPlayer2.spotify",
                        ]
                    ],
                )
            if msg.member == "GetAll":
                if "chromium" in msg.destination:
                    return SimpleNamespace(
                        message_type=MessageType.METHOD_RETURN,
                        body=[
                            {
                                "PlaybackStatus": SimpleNamespace(
                                    value=chromium_status["status"]
                                ),
                                "Metadata": SimpleNamespace(value={}),
                            }
                        ],
                    )
                return SimpleNamespace(
                    message_type=MessageType.METHOD_RETURN,
                    body=[
                        {
                            "PlaybackStatus": SimpleNamespace(value="Paused"),
                            "Metadata": SimpleNamespace(
                                value={
                                    "xesam:title": SimpleNamespace(value="Harder Better"),
                                    "xesam:artist": SimpleNamespace(value=["Daft Punk"]),
                                }
                            ),
                        }
                    ],
                )
            sent_Commands.append((msg.destination, msg.member))
            return SimpleNamespace(message_type=MessageType.METHOD_RETURN, body=[])

    monkeypatch.setattr(dbus_fast.aio, "MessageBus", lambda bus_type=None: FakeBus())

    tool = MediaTool()
    ctx = ToolContext(workspace=tmp_path)

    # 1. Without player arg: Paused spotify outranks Stopped chromium (alphabetically first)
    r1 = await tool.execute({"action": "play"}, ctx)
    assert r1.is_error is False
    assert "spotify" in r1.observation.lower()
    assert sent_Commands[-1] == ("org.mpris.MediaPlayer2.spotify", "Play")

    # 2. Even if chromium is Playing, explicit player="spotify" targets spotify
    chromium_status["status"] = "Playing"
    r2 = await tool.execute({"action": "play", "player": "spotify"}, ctx)
    assert r2.is_error is False
    assert "spotify" in r2.observation.lower()
    assert sent_Commands[-1] == ("org.mpris.MediaPlayer2.spotify", "Play")

    # 3. Unknown player returns clear error
    r3 = await tool.execute({"action": "play", "player": "vlc"}, ctx)
    assert "vlc" in r3.observation.lower()

    # 4. 'stop' falls back to Pause when a streaming player (like Spotify) ignores Stop and stays Playing
    spotify_state = {"status": "Playing"}

    class StreamingStopBus(FakeBus):
        async def call(self, msg):
            if msg.member == "ListNames":
                return SimpleNamespace(
                    message_type=MessageType.METHOD_RETURN,
                    body=[["org.mpris.MediaPlayer2.spotify"]],
                )
            if msg.member == "GetAll":
                return SimpleNamespace(
                    message_type=MessageType.METHOD_RETURN,
                    body=[
                        {
                            "PlaybackStatus": SimpleNamespace(
                                value=spotify_state["status"]
                            ),
                            "Metadata": SimpleNamespace(
                                value={
                                    "xesam:title": SimpleNamespace(value="Track 1"),
                                    "xesam:artist": SimpleNamespace(value=["Artist"]),
                                }
                            ),
                        }
                    ],
                )
            sent_Commands.append((msg.destination, msg.member))
            if msg.member == "Pause":
                spotify_state["status"] = "Paused"
            return SimpleNamespace(message_type=MessageType.METHOD_RETURN, body=[])

    monkeypatch.setattr(
        dbus_fast.aio, "MessageBus", lambda bus_type=None: StreamingStopBus()
    )
    sent_Commands.clear()
    r4 = await tool.execute({"action": "stop"}, ctx)
    assert r4.is_error is False
    assert ("org.mpris.MediaPlayer2.spotify", "Stop") in sent_Commands
    assert ("org.mpris.MediaPlayer2.spotify", "Pause") in sent_Commands
    assert "Paused" in r4.observation

    # 5. When multiple players are Playing and player is omitted, 'pause'/'stop' pauses all Playing players
    multi_state = {
        "org.mpris.MediaPlayer2.firefox": "Playing",
        "org.mpris.MediaPlayer2.spotify": "Playing",
    }

    class MultiPlayingBus(FakeBus):
        async def call(self, msg):
            if msg.member == "ListNames":
                return SimpleNamespace(
                    message_type=MessageType.METHOD_RETURN,
                    body=[list(multi_state.keys())],
                )
            if msg.member == "GetAll":
                return SimpleNamespace(
                    message_type=MessageType.METHOD_RETURN,
                    body=[
                        {
                            "PlaybackStatus": SimpleNamespace(
                                value=multi_state[msg.destination]
                            ),
                            "Metadata": SimpleNamespace(value={}),
                        }
                    ],
                )
            sent_Commands.append((msg.destination, msg.member))
            if msg.member in ("Pause", "Stop"):
                multi_state[msg.destination] = "Paused"
            return SimpleNamespace(message_type=MessageType.METHOD_RETURN, body=[])

    monkeypatch.setattr(
        dbus_fast.aio, "MessageBus", lambda bus_type=None: MultiPlayingBus()
    )
    sent_Commands.clear()
    r5 = await tool.execute({"action": "pause"}, ctx)
    assert r5.is_error is False
    assert ("org.mpris.MediaPlayer2.firefox", "Pause") in sent_Commands
    assert ("org.mpris.MediaPlayer2.spotify", "Pause") in sent_Commands


@pytest.mark.asyncio
async def test_media_tool_macos_and_windows_cross_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from verbix.tools import media as media_mod
    from verbix.tools.media import MediaTool

    ctx = ToolContext(workspace=tmp_path)
    tool = MediaTool()

    # macOS (darwin) volume + playback via osascript
    monkeypatch.setattr(media_mod.sys, "platform", "darwin")
    osascript_calls: list[str] = []

    def fake_run(cmd, **kwargs):
        script = cmd[-1]
        osascript_calls.append(script)
        if "get volume settings" in script:
            return SimpleNamespace(
                returncode=0,
                stdout="output volume:65, input volume:50, alert volume:100, output muted:false\n",
                stderr="",
            )
        if "Spotify" in script or "Music" in script:
            return SimpleNamespace(
                returncode=0,
                stdout="Playing: Daft Punk - Voyager (Spotify)\n",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(media_mod.subprocess, "run", fake_run)

    r_mac_vol = await tool.execute({"action": "volume_get"}, ctx)
    assert not r_mac_vol.is_error
    assert "65%" in r_mac_vol.observation

    r_mac_set = await tool.execute({"action": "volume_set", "level": 40}, ctx)
    assert not r_mac_set.is_error
    assert "40%" in r_mac_set.observation

    r_mac_status = await tool.execute({"action": "status"}, ctx)
    assert not r_mac_status.is_error
    assert "Spotify" in r_mac_status.observation

    r_mac_play = await tool.execute({"action": "play_pause", "player": "spotify"}, ctx)
    assert not r_mac_play.is_error

    # Windows (win32) status does not error
    monkeypatch.setattr(media_mod.sys, "platform", "win32")
    monkeypatch.setattr(
        media_mod, "_windows_media_or_volume", lambda action, level=None, player=None: f"Win {action} ok"
    )
    r_win_status = await tool.execute({"action": "status"}, ctx)
    assert not r_win_status.is_error
    assert "Win status ok" in r_win_status.observation


@pytest.mark.asyncio
async def test_app_tool_close_kills_stubborn_processes_after_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from verbix.tools import app as app_mod
    from verbix.tools.app import AppEntry, AppTool

    fake_catalog = [
        AppEntry(
            id="kmines",
            name="KMines",
            exe_name="kmines",
            argv=("/usr/bin/kmines",),
        )
    ]
    monkeypatch.setattr(app_mod, "discover_installed_apps", lambda: fake_catalog)

    actions: list[tuple[str, int]] = []

    class StubbornProc:
        def __init__(self):
            self.pid = 7777
            self.info = {"pid": 7777, "name": "kmines", "exe": "/usr/bin/kmines"}

        def terminate(self):
            actions.append(("terminate", self.pid))

        def kill(self):
            actions.append(("kill", self.pid))

    stubborn = StubbornProc()
    monkeypatch.setattr(app_mod, "_matching_procs", lambda entry, procs=None: [stubborn])

    wait_calls = 0

    def fake_wait_procs(procs, timeout):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            return ([], list(procs))  # still alive after SIGTERM!
        return (list(procs), [])  # dead after SIGKILL

    monkeypatch.setattr(app_mod.psutil, "wait_procs", fake_wait_procs)

    res = await AppTool().execute({"action": "close", "name": "KMines"}, ToolContext(workspace=tmp_path))
    assert not res.is_error
    assert actions == [("terminate", 7777), ("kill", 7777)]
    assert "verified stopped" in res.observation


@pytest.mark.asyncio
async def test_system_tool_reports_host_metrics_across_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from verbix.tools import system as sys_mod
    from verbix.tools.system import SystemTool

    monkeypatch.setattr(
        sys_mod.psutil,
        "sensors_battery",
        lambda: SimpleNamespace(percent=82.4, power_plugged=False, secsleft=3660),
    )

    tool = SystemTool()
    ctx = ToolContext(workspace=tmp_path)

    r_overview = await tool.execute({}, ctx)
    assert not r_overview.is_error
    for expected in ("CPU:", "Memory:", "Disk", "Battery: 82%", "Network:"):
        assert expected in r_overview.observation

    r_batt = await tool.execute({"section": "battery"}, ctx)
    assert not r_batt.is_error
    assert "82%" in r_batt.observation
    assert "discharging" in r_batt.observation
    assert "1h 1m" in r_batt.observation

    r_procs = await tool.execute({"section": "processes"}, ctx)
    assert not r_procs.is_error
    assert "Top processes" in r_procs.observation

    r_bad = await tool.execute({"section": "bogus"}, ctx)
    assert r_bad.is_error


@pytest.mark.asyncio
async def test_timer_tool_set_list_cancel_and_fires_announcement(tmp_path: Path):
    import asyncio

    from verbix.tools.timer import TimerTool

    announced: list[tuple[str, str]] = []

    class DummyRuntime:
        def __init__(self):
            self._timers = {}
            self._timer_counter = 0
            self._pending_announcements = []

        def _on_task_finished(self, task_id: str, result: str) -> None:
            announced.append((task_id, result))

    rt = DummyRuntime()
    tool = TimerTool(rt)
    ctx = ToolContext(workspace=tmp_path)

    # 1. Set a fast timer (0.05s) and a slow timer (10s)
    r_fast = await tool.execute(
        {"action": "set", "seconds": 0.05, "label": "check pasta"}, ctx
    )
    assert not r_fast.is_error
    assert "timer-1" in r_fast.observation

    r_slow = await tool.execute(
        {"action": "set", "seconds": 10, "label": "laundry"}, ctx
    )
    assert not r_slow.is_error
    assert "timer-2" in r_slow.observation

    # 2. List shows active timers
    r_list = await tool.execute({"action": "list"}, ctx)
    assert not r_list.is_error
    assert "laundry" in r_list.observation

    # 3. Wait for fast timer to fire -> triggers announcement on runtime
    await asyncio.sleep(0.1)
    assert len(announced) == 1
    assert announced[0][0] == "timer-1"
    assert "check pasta" in announced[0][1]

    # 4. Cancel remaining timer
    r_cancel = await tool.execute({"action": "cancel", "timer_id": "timer-2"}, ctx)
    assert not r_cancel.is_error
    assert "Cancelled timer-2" in r_cancel.observation

    r_list_after = await tool.execute({"action": "list"}, ctx)
    assert "No active timers" in r_list_after.observation


@pytest.mark.asyncio
async def test_memory_tool_remember_list_forget_dedup_and_security(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from verbix.config import (
        ensure_global_scaffold,
        memory_file,
        prune_state,
        read_memory,
    )
    from verbix.tools.memory import MemoryTool

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Pre-existing .scaffolded install still gets MEMORY.md created on ensure_global_scaffold()
    root = tmp_path / ".verbix"
    root.mkdir(parents=True)
    (root / ".scaffolded").touch()
    (root / "config.yaml").write_text("# cfg\n", encoding="utf-8")
    (root / "PERSONA.md").write_text("<!-- p -->\n", encoding="utf-8")
    ensure_global_scaffold()
    assert memory_file().is_file()
    assert read_memory() == ""

    tool = MemoryTool()
    ctx = ToolContext(workspace=tmp_path)

    # 1. Empty list
    r0 = await tool.execute({"action": "list"}, ctx)
    assert r0.observation == "(memory is empty)"

    # 2. Remember normalizes whitespace and strips leading '#' / '['
    r1 = await tool.execute(
        {"action": "remember", "text": "  # Door code\n is 4812  "}, ctx
    )
    assert not r1.is_error
    assert "Door code is 4812" in r1.observation
    assert "- Door code is 4812" in read_memory()

    # 3. Case-insensitive deduplication
    r_dup = await tool.execute(
        {"action": "remember", "text": "door code is 4812"}, ctx
    )
    assert not r_dup.is_error
    assert "Already in memory" in r_dup.observation
    assert read_memory().splitlines() == ["- Door code is 4812"]

    # 4. prune_state never deletes MEMORY.md even when mtime is old
    old_ts = 1000.0
    os.utime(memory_file(), (old_ts, old_ts))
    prune_state(retention_days=14)
    assert memory_file().is_file()
    assert "- Door code is 4812" in read_memory()

    # 5. Rejects secrets and oversized (>500 chars) entries
    r_sec = await tool.execute(
        {"action": "remember", "text": "OPENROUTER_API_KEY=sk-secret123"}, ctx
    )
    assert r_sec.is_error
    r_long = await tool.execute({"action": "remember", "text": "x" * 501}, ctx)
    assert r_long.is_error

    # 6. Forget removes matching entry and preserves HTML comment header
    r_forget = await tool.execute({"action": "forget", "text": "door code"}, ctx)
    assert not r_forget.is_error
    assert read_memory() == ""
    assert "<!--" in memory_file().read_text(encoding="utf-8")

    # 7. Refuses symlinked MEMORY.md (and ensure_global_scaffold does not write through dangling symlink)
    memory_file().unlink()
    dangling_target = tmp_path / "dangling_outside.txt"
    memory_file().symlink_to(dangling_target)
    ensure_global_scaffold()
    assert not dangling_target.exists()
    outside = tmp_path / "outside.txt"
    outside.write_text("- secret", encoding="utf-8")
    memory_file().unlink()
    memory_file().symlink_to(outside)
    assert read_memory() == ""
    r_sym = await tool.execute({"action": "list"}, ctx)
    assert r_sym.is_error


@pytest.mark.asyncio
async def test_runtime_dynamically_grounds_memory_across_foreground_announcer_and_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from verbix.runtime.announcer import _snapshot_foreground_history
    from verbix.runtime.runtime import Runtime
    from verbix.runtime.worker import WorkerTask
    from verbix.tools.memory import MemoryTool

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(WorkerTask, "start", lambda self: None)
    ws = tmp_path / "ws"
    ws.mkdir()

    rt = Runtime(
        workspace=ws,
        provider=MagicMock(),
        system_prompt="Talk like a caveman",
    )
    fg_defs = rt.foreground.get_tool_definitions() or []
    fg_tool_names = {d["function"]["name"] for d in fg_defs}
    assert "memory" in fg_tool_names
    assert "memory" in rt.worker_tool_names
    assert "Saved notes about the human user" not in rt.system_prompt

    # Save a memory mid-session via MemoryTool
    await MemoryTool().execute(
        {"action": "remember", "text": "Preferred browser is Helium"},
        ToolContext(workspace=ws),
    )

    # Refresh grounding as _step_inner and _snapshot_foreground_history do
    rt._refresh_grounded_prompts()
    assert "Saved notes about the human user" in rt.system_prompt
    assert "your own name is Verbix" in rt.system_prompt
    assert "Preferred browser is Helium" in rt.system_prompt
    # Memory appears inside ENVIRONMENT & GROUNDING before CUSTOM USER INSTRUCTIONS
    assert rt.system_prompt.index("Preferred browser is Helium") < rt.system_prompt.index(
        "# CUSTOM USER INSTRUCTIONS"
    )

    rt.foreground.messages = [
        {"role": "system", "content": "old"},
        {"role": "user", "content": "hi"},
    ]
    rt._refresh_grounded_prompts()
    assert "Preferred browser is Helium" in rt.foreground.messages[0]["content"]
    snap = _snapshot_foreground_history(rt)
    assert "Preferred browser is Helium" in snap[0]["content"]

    # Spawned worker also receives USER MEMORY in its system prompt
    task_id = await rt.spawn("open browser")
    worker_sys = rt.tasks[task_id].session.config.system_template
    assert "Preferred browser is Helium" in worker_sys

    # Forget removes memory from foreground and announcer grounding immediately
    await MemoryTool().execute(
        {"action": "forget", "text": "Helium"},
        ToolContext(workspace=ws),
    )
    rt._refresh_grounded_prompts()
    assert "Saved notes about the human user" not in rt.system_prompt
    assert "Preferred browser is Helium" not in rt.foreground.messages[0]["content"]
    snap_after = _snapshot_foreground_history(rt)
    assert "Preferred browser is Helium" not in snap_after[0]["content"]


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
    assert "verified" in res_close.observation.lower()


@pytest.mark.asyncio
async def test_app_tool_list_includes_all_installed_apps_with_generic_names_not_truncated_at_40(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from verbix.tools import app as app_mod
    from verbix.tools.app import AppTool

    apps_dir = tmp_path / "applications"
    apps_dir.mkdir(parents=True)
    for i in range(55):
        (apps_dir / f"app-{i:02d}.desktop").write_text(
            f"[Desktop Entry]\nType=Application\nName=App {i:02d}\nExec=/usr/bin/app{i:02d}\n",
            encoding="utf-8",
        )
    (apps_dir / "z-firefox.desktop").write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Firefox\n"
        "GenericName=Web Browser\n"
        "Exec=/usr/bin/firefox %u\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(app_mod, "_linux_desktop_dirs", lambda: [apps_dir])
    monkeypatch.setattr(app_mod, "_running_user_procs", list)
    monkeypatch.setattr(app_mod.sys, "platform", "linux")

    res = await AppTool().execute({"action": "list"}, ToolContext(workspace=tmp_path))
    assert res.is_error is False
    assert "Firefox (Web Browser)" in res.observation



def test_worker_registry_and_runtime_grounding_include_clipboard_media_and_app(tmp_path: Path):
    from verbix.runtime.runtime import Runtime
    from verbix.tools.spawn import SpawnTool

    rt = Runtime(workspace=tmp_path, provider=MagicMock())
    for tool_name in ("clipboard", "media", "app", "open"):
        assert tool_name in rt.worker_tool_names
        assert tool_name in rt.system_prompt
        assert tool_name in SpawnTool(rt).description


@pytest.mark.asyncio
async def test_open_tool_launches_safe_files_and_rejects_missing_or_outside_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from verbix.tools import open as open_mod
    from verbix.tools.open import OpenTool

    opened_targets: list[Path] = []

    def fake_dispatch(target: Path) -> str:
        opened_targets.append(target)
        return f"Opened '{target.name}' with Okular."

    monkeypatch.setattr(open_mod, "_dispatch_open", fake_dispatch)

    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.7 sample pdf")
    sub = tmp_path / "docs_folder"
    sub.mkdir()

    tool = OpenTool()
    ctx = ToolContext(workspace=tmp_path)

    r_file = await tool.execute({"path": "report.pdf"}, ctx)
    assert r_file.is_error is False
    assert "Okular" in r_file.observation
    assert opened_targets == [doc.resolve()]

    r_dir = await tool.execute({"path": "docs_folder"}, ctx)
    assert r_dir.is_error is False
    assert opened_targets == [doc.resolve(), sub.resolve()]

    # Missing file
    r_missing = await tool.execute({"path": "nope.pdf"}, ctx)
    assert r_missing.is_error is True

    # Workspace escape
    r_escape = await tool.execute({"path": "../outside.pdf"}, ctx)
    assert r_escape.is_error is True

    # Secret file (.env)
    (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
    r_secret = await tool.execute({"path": ".env"}, ctx)
    assert r_secret.is_error is True


@pytest.mark.asyncio
async def test_open_tool_blocks_executable_extensions_and_magic_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from verbix.tools import open as open_mod
    from verbix.tools.open import OpenTool

    monkeypatch.setattr(
        open_mod, "_dispatch_open", lambda target: (_ for _ in ()).throw(AssertionError("should not dispatch"))
    )

    tool = OpenTool()
    ctx = ToolContext(workspace=tmp_path)

    # 1. Blocked script/binary extensions
    for name in ("run.sh", "payload.exe", "launch.desktop", "script.bat", "app.AppImage"):
        p = tmp_path / name
        p.write_text("echo hello\n", encoding="utf-8")
        res = await tool.execute({"path": name}, ctx)
        assert res.is_error is True
        assert "executable" in res.observation.lower()

    # 2. Extensionless shebang script
    shebang = tmp_path / "innocent_note"
    shebang.write_bytes(b"#!/bin/sh\nrm -rf /\n")
    res_sh = await tool.execute({"path": "innocent_note"}, ctx)
    assert res_sh.is_error is True

    # 3. Extensionless ELF binary
    elf = tmp_path / "binary_blob"
    elf.write_bytes(b"\x7fELF\x02\x01\x01\x00")
    res_elf = await tool.execute({"path": "binary_blob"}, ctx)
    assert res_elf.is_error is True

    # 4. macOS .app directory bundle
    app_bundle = tmp_path / "Evil.app"
    app_bundle.mkdir()
    res_app = await tool.execute({"path": "Evil.app"}, ctx)
    assert res_app.is_error is True


@pytest.mark.asyncio
async def test_app_tool_launch_with_paths_and_open_tool_with_app_and_multiple_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from verbix.tools import app as app_mod
    from verbix.tools import open as open_mod
    from verbix.tools.app import AppEntry, AppTool
    from verbix.tools.open import OpenTool

    fake_catalog = [
        AppEntry(
            id="firefox",
            name="Firefox",
            exe_name="firefox",
            argv=("/usr/bin/firefox",),
            keywords=("browser",),
        ),
    ]
    monkeypatch.setattr(app_mod, "discover_installed_apps", lambda: fake_catalog)
    monkeypatch.setattr(app_mod, "_matching_procs", lambda entry, procs=None: [])

    spawned_argvs: list[list[str]] = []

    def fake_popen(argv, **kwargs):
        spawned_argvs.append(list(argv))
        return SimpleNamespace(pid=9999)

    monkeypatch.setattr(app_mod.subprocess, "Popen", fake_popen)

    f1 = tmp_path / "index.html"
    f2 = tmp_path / "second.html"
    f1.write_text("<h1>1</h1>", encoding="utf-8")
    f2.write_text("<h1>2</h1>", encoding="utf-8")
    evil = tmp_path / "evil.sh"
    evil.write_text("#!/bin/sh\n", encoding="utf-8")

    ctx = ToolContext(workspace=tmp_path)
    app_tool = AppTool()
    open_tool = OpenTool()

    # 1. AppTool launch with multiple paths
    r_app_multi = await app_tool.execute(
        {"action": "launch", "name": "firefox", "paths": ["index.html", "second.html"]},
        ctx,
    )
    assert not r_app_multi.is_error
    assert spawned_argvs[-1] == ["/usr/bin/firefox", str(f1.resolve()), str(f2.resolve())]

    # 2. AppTool launch with single path rejects executable scripts and workspace escapes
    r_app_evil = await app_tool.execute(
        {"action": "launch", "name": "firefox", "path": "evil.sh"}, ctx
    )
    assert r_app_evil.is_error
    r_app_escape = await app_tool.execute(
        {"action": "launch", "name": "firefox", "path": "../outside.html"}, ctx
    )
    assert r_app_escape.is_error

    # 3. OpenTool with app="firefox" and multiple paths launches Firefox with those paths
    r_open_app = await open_tool.execute(
        {"paths": ["index.html", "second.html"], "app": "firefox"}, ctx
    )
    assert not r_open_app.is_error
    assert "Firefox" in r_open_app.observation
    assert spawned_argvs[-1] == ["/usr/bin/firefox", str(f1.resolve()), str(f2.resolve())]

    # 4. OpenTool with unknown app fails cleanly
    r_open_bad_app = await open_tool.execute(
        {"path": "index.html", "app": "nonexistent_browser"}, ctx
    )
    assert r_open_bad_app.is_error

    # 5. OpenTool with multiple paths and default app dispatches each path
    dispatched: list[Path | str] = []
    monkeypatch.setattr(
        open_mod, "_dispatch_open", lambda t: (dispatched.append(t), f"Opened '{t}'.")[1]
    )
    r_open_multi = await open_tool.execute({"paths": ["index.html", "second.html"]}, ctx)
    assert not r_open_multi.is_error
    assert dispatched == [f1.resolve(), f2.resolve()]

    # 6. Both AppTool and OpenTool accept http:// and https:// URLs and reject unsafe URI schemes
    r_app_url = await app_tool.execute(
        {
            "action": "launch",
            "name": "firefox",
            "path": "https://en.wikipedia.org/wiki/Albert_Einstein",
        },
        ctx,
    )
    assert not r_app_url.is_error
    assert spawned_argvs[-1] == [
        "/usr/bin/firefox",
        "https://en.wikipedia.org/wiki/Albert_Einstein",
    ]

    dispatched.clear()
    r_open_url = await open_tool.execute(
        {"path": "https://en.wikipedia.org/wiki/Albert_Einstein"}, ctx
    )
    assert not r_open_url.is_error
    assert dispatched == ["https://en.wikipedia.org/wiki/Albert_Einstein"]

    for unsafe_uri in (
        "javascript:alert(1)",
        "file:///etc/passwd",
        "data:text/html,<script>alert(1)</script>",
    ):
        res_bad_open = await open_tool.execute({"path": unsafe_uri}, ctx)
        assert res_bad_open.is_error
        res_bad_app = await app_tool.execute(
            {"action": "launch", "name": "firefox", "path": unsafe_uri}, ctx
        )
        assert res_bad_app.is_error


def test_flatpak_desktop_entry_extracts_real_exe_name_strips_forwarding_markers_and_matches_precisely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from verbix.tools import app as app_mod

    apps_dir = tmp_path / "applications"
    apps_dir.mkdir(parents=True)
    (apps_dir / "com.spotify.Client.desktop").write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Spotify\n"
        "GenericName=Music Player\n"
        "Exec=/usr/bin/flatpak run --branch=stable --arch=x86_64 --command=spotify --file-forwarding com.spotify.Client @@u %U @@\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app_mod, "_linux_desktop_dirs", lambda: [apps_dir])
    catalog = app_mod._discover_linux_apps()
    assert len(catalog) == 1
    spotify = catalog[0]
    assert spotify.exe_name == "spotify"
    assert spotify.argv == (
        "/usr/bin/flatpak",
        "run",
        "--branch=stable",
        "--arch=x86_64",
        "--command=spotify",
        "com.spotify.Client",
    )

    # Process matching must NOT match an unrelated flatpak process
    other_flatpak = SimpleNamespace(
        pid=101,
        info={
            "pid": 101,
            "name": "flatpak",
            "exe": "/usr/bin/flatpak",
            "cmdline": ["/usr/bin/flatpak", "run", "org.gimp.GIMP"],
        },
    )
    spotify_flatpak = SimpleNamespace(
        pid=202,
        info={
            "pid": 202,
            "name": "bwrap",
            "exe": "/usr/bin/bwrap",
            "cmdline": ["/usr/bin/bwrap", "--args", "com.spotify.Client"],
        },
    )
    matched = app_mod._matching_procs(spotify, [other_flatpak, spotify_flatpak])
    assert matched == [spotify_flatpak]

