from __future__ import annotations

import asyncio
import sys
from typing import Any

from verbix.tools.base import ToolContext, ToolResult

_PLAYBACK_ACTIONS = ("play_pause", "play", "pause", "next", "previous", "stop", "status")
_VOLUME_ACTIONS = ("volume_get", "volume_set", "mute")
_ALL_ACTIONS = (*_PLAYBACK_ACTIONS, *_VOLUME_ACTIONS)

_MPRIS_METHODS = {
    "play_pause": "PlayPause",
    "play": "Play",
    "pause": "Pause",
    "next": "Next",
    "previous": "Previous",
    "stop": "Stop",
}


async def _linux_mpris_action(action: str) -> str:
    """Talks directly to org.mpris.MediaPlayer2.Player on the session DBus via dbus-fast."""
    from dbus_fast.aio import MessageBus
    from dbus_fast.constants import BusType, MessageType
    from dbus_fast.message import Message

    bus = await MessageBus(bus_type=BusType.SESSION).connect()
    try:
        reply = await bus.call(
            Message(
                destination="org.freedesktop.DBus",
                path="/org/freedesktop/DBus",
                interface="org.freedesktop.DBus",
                member="ListNames",
            )
        )
        if reply.message_type == MessageType.ERROR:
            raise RuntimeError(f"DBus ListNames error: {reply.body}")
        names: list[str] = reply.body[0] if reply.body else []
        players = sorted(n for n in names if n.startswith("org.mpris.MediaPlayer2."))
        if not players:
            return "No active media players found on the session bus."

        # Prefer a player whose PlaybackStatus is "Playing"
        chosen = players[0]
        statuses: list[str] = []
        for p in players:
            short = p.removeprefix("org.mpris.MediaPlayer2.")
            prop_reply = await bus.call(
                Message(
                    destination=p,
                    path="/org/mpris/MediaPlayer2",
                    interface="org.freedesktop.DBus.Properties",
                    member="GetAll",
                    signature="s",
                    body=["org.mpris.MediaPlayer2.Player"],
                )
            )
            if prop_reply.message_type == MessageType.ERROR or not prop_reply.body:
                continue
            props = prop_reply.body[0]
            status_var = props.get("PlaybackStatus")
            status_str = getattr(status_var, "value", str(status_var or "Unknown"))
            if status_str == "Playing":
                chosen = p
            meta_var = props.get("Metadata")
            meta = getattr(meta_var, "value", {}) if meta_var else {}
            title_var = meta.get("xesam:title") if isinstance(meta, dict) else None
            title = getattr(title_var, "value", "") if title_var else ""
            artist_var = meta.get("xesam:artist") if isinstance(meta, dict) else None
            artists = getattr(artist_var, "value", []) if artist_var else []
            artist = ", ".join(str(a) for a in artists) if isinstance(artists, list) else str(artists or "")
            track = f"{artist} - {title}".strip(" -") or "(unknown track)"
            statuses.append(f"{status_str}: {track} ({short})")

        if action == "status":
            return "\n".join(statuses) if statuses else "No active media player metadata available."

        method = _MPRIS_METHODS[action]
        call_reply = await bus.call(
            Message(
                destination=chosen,
                path="/org/mpris/MediaPlayer2",
                interface="org.mpris.MediaPlayer2.Player",
                member=method,
            )
        )
        if call_reply.message_type == MessageType.ERROR:
            raise RuntimeError(f"MPRIS {method} error: {call_reply.body}")
        short_chosen = chosen.removeprefix("org.mpris.MediaPlayer2.")
        return f"Sent '{action}' to {short_chosen}."
    finally:
        bus.disconnect()


def _linux_pulse_volume(action: str, level: int | None = None) -> str:
    """Reads or sets system audio volume via libpulse (pulsectl)."""
    import pulsectl

    with pulsectl.Pulse("verbix-media") as pulse:
        server_info = pulse.server_info()
        default_sink_name = getattr(server_info, "default_sink_name", None)
        sinks = pulse.sink_list()
        if not sinks:
            raise RuntimeError("No audio output sinks found")
        sink = next((s for s in sinks if s.name == default_sink_name), sinks[0])
        if action == "volume_get":
            pct = round(pulse.volume_get_all_chans(sink) * 100)
            state = "muted" if sink.mute else "unmuted"
            return f"Volume: {pct}% ({state})"
        if action == "volume_set":
            assert level is not None
            clamped = max(0, min(100, int(level)))
            pulse.volume_set_all_chans(sink, clamped / 100.0)
            if clamped > 0 and sink.mute:
                pulse.mute(sink, False)
            return f"Volume set to {clamped}%."
        if action == "mute":
            new_mute = not bool(sink.mute)
            pulse.mute(sink, new_mute)
            return f"Audio {'muted' if new_mute else 'unmuted'}."
        raise ValueError(f"Unsupported volume action: {action}")


def _windows_media_or_volume(action: str, level: int | None = None) -> str:
    """Uses Win32 keybd_event for media keys and pycaw for volume on Windows."""
    import ctypes

    vk_map = {
        "play_pause": 0xB3,
        "play": 0xB3,
        "pause": 0xB3,
        "next": 0xB0,
        "previous": 0xB1,
        "stop": 0xB2,
        "mute": 0xAD,
    }
    if action in vk_map and action != "mute":
        vk = vk_map[action]
        ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
        ctypes.windll.user32.keybd_event(vk, 0, 2, 0)
        return f"Sent media key '{action}'."

    if action in ("volume_get", "volume_set", "mute"):
        try:
            from ctypes import POINTER, cast

            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            if action == "volume_get":
                pct = round(volume.GetMasterVolumeLevelScalar() * 100)
                muted = bool(volume.GetMute())
                return f"Volume: {pct}% ({'muted' if muted else 'unmuted'})"
            if action == "volume_set":
                assert level is not None
                clamped = max(0, min(100, int(level)))
                volume.SetMasterVolumeLevelScalar(clamped / 100.0, None)
                return f"Volume set to {clamped}%."
            if action == "mute":
                new_mute = not bool(volume.GetMute())
                volume.SetMute(int(new_mute), None)
                return f"Audio {'muted' if new_mute else 'unmuted'}."
        except Exception:
            if action == "mute":
                vk = vk_map["mute"]
                ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
                ctypes.windll.user32.keybd_event(vk, 0, 2, 0)
                return "Toggled audio mute."
            raise
    raise RuntimeError(f"Action '{action}' is not supported on Windows")


class MediaTool:
    """Controls desktop media playback and system audio volume via native IPC/C libraries."""

    name: str = "media"
    side_effecting: bool = True
    description: str = (
        "Controls desktop media playback (play_pause, play, pause, next, previous, stop, status) "
        "and system audio volume (volume_get, volume_set with level 0-100, mute)."
    )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(_ALL_ACTIONS),
                    "description": "Media or volume action to perform.",
                },
                "level": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Target volume percentage (0-100) when action='volume_set'.",
                },
            },
            "required": ["action"],
        }

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        action = (args.get("action") or "").strip().lower()
        if action not in _ALL_ACTIONS:
            return ToolResult(
                f"error: invalid action '{action}'; valid actions: {', '.join(_ALL_ACTIONS)}",
                is_error=True,
            )
        level = args.get("level")
        if action == "volume_set":
            if level is None or isinstance(level, bool):
                return ToolResult(
                    "error: integer 'level' (0-100) is required when action='volume_set'",
                    is_error=True,
                )
            try:
                level = int(level)
            except (TypeError, ValueError):
                return ToolResult(
                    "error: 'level' must be an integer between 0 and 100",
                    is_error=True,
                )

        try:
            if sys.platform.startswith("linux"):
                if action in _PLAYBACK_ACTIONS:
                    obs = await _linux_mpris_action(action)
                    return ToolResult(obs)
                obs = await asyncio.to_thread(_linux_pulse_volume, action, level)
                return ToolResult(obs)
            if sys.platform == "win32":
                obs = await asyncio.to_thread(_windows_media_or_volume, action, level)
                return ToolResult(obs)
            return ToolResult(
                f"error: media tool is not supported on platform '{sys.platform}'",
                is_error=True,
            )
        except Exception as e:  # noqa: BLE001 — DBus, PulseAudio, and Win32 COM raise backend-specific exceptions
            return ToolResult(f"error: media {action} failed: {e}", is_error=True)
