from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from typing import Any

from verbix.tools.base import ToolContext, ToolResult


async def _linux_notify(title: str, message: str) -> str:
    """Sends a desktop notification via org.freedesktop.Notifications DBus (or notify-send fallback)."""
    try:
        from dbus_fast.aio import MessageBus
        from dbus_fast.constants import BusType, MessageType
        from dbus_fast.message import Message

        bus = await MessageBus(bus_type=BusType.SESSION).connect()
        try:
            reply = await bus.call(
                Message(
                    destination="org.freedesktop.Notifications",
                    path="/org/freedesktop/Notifications",
                    interface="org.freedesktop.Notifications",
                    member="Notify",
                    signature="susssasa{sv}i",
                    body=[
                        "Verbix",
                        0,
                        "",
                        title,
                        message,
                        [],
                        {},
                        5000,
                    ],
                )
            )
            if reply.message_type != MessageType.ERROR:
                return f"Sent desktop notification '{title}'."
        finally:
            bus.disconnect()
    except Exception:  # noqa: BLE001, S110
        pass

    notify_bin = shutil.which("notify-send")
    if notify_bin:
        await asyncio.to_thread(
            subprocess.run,
            [notify_bin, "--app-name=Verbix", "--", title, message],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        )
        return f"Sent desktop notification '{title}'."
    raise RuntimeError("No desktop notification service available on session DBus")


def _macos_notify(title: str, message: str) -> str:
    """Sends a macOS notification via osascript passing title and message safely as argv."""
    script = (
        "on run argv\n"
        "  display notification (item 2 of argv) with title (item 1 of argv)\n"
        "end run"
    )
    subprocess.run(
        ["/usr/bin/osascript", "-e", script, "--", title, message],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=True,
    )
    return f"Sent desktop notification '{title}'."


def _windows_notify(title: str, message: str) -> str:
    """Sends a Windows toast/balloon notification passing strings safely via environment variables."""
    ps_script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$n = New-Object System.Windows.Forms.NotifyIcon; "
        "$n.Icon = [System.Drawing.SystemIcons]::Information; "
        "$n.Visible = $true; "
        "$n.ShowBalloonTip(5000, $env:VERBIX_NOTIFY_TITLE, $env:VERBIX_NOTIFY_MSG, "
        "[System.Windows.Forms.ToolTipIcon]::Info)"
    )
    env = {
        **os.environ,
        "VERBIX_NOTIFY_TITLE": title,
        "VERBIX_NOTIFY_MSG": message,
    }
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
        env=env,
        capture_output=True,
        text=True,
        timeout=8.0,
        check=True,
    )
    return f"Sent desktop notification '{title}'."


class NotifyTool:
    """Displays a native desktop notification toast across Linux, macOS, and Windows."""

    name: str = "notify"
    side_effecting: bool = True
    description: str = (
        "Displays a native desktop notification toast with a title and message across Linux, macOS, and Windows."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short notification title (e.g. 'Verbix').",
                },
                "message": {
                    "type": "string",
                    "description": "Notification body text to display on screen.",
                },
            },
            "required": ["title", "message"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        title = str(args.get("title") or "").strip()
        message = str(args.get("message") or "").strip()
        if not title or not message:
            return ToolResult(
                "error: both 'title' and 'message' are required",
                is_error=True,
            )
        try:
            if sys.platform.startswith("linux"):
                obs = await _linux_notify(title, message)
                return ToolResult(obs)
            if sys.platform == "darwin":
                obs = await asyncio.to_thread(_macos_notify, title, message)
                return ToolResult(obs)
            if sys.platform == "win32":
                obs = await asyncio.to_thread(_windows_notify, title, message)
                return ToolResult(obs)
            return ToolResult(
                f"error: notify tool is not supported on platform '{sys.platform}'",
                is_error=True,
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(f"error: notification failed: {e}", is_error=True)
